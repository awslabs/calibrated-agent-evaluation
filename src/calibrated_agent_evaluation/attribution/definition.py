# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The crew Define artifact builder.

Validates and constructs a :class:`~calibrated_agent_evaluation.attribution.types.CrewDefinition`:
the agents axis, the handoff (seam) axis, the per-scenario expected process, and the named
mandated-follow-up rules. A skipped or mis-routed agent is only detectable as *expected minus
actual*, so an ``ExpectedProcess`` must declare a non-empty ordered agent set before any trace is
judged. An undeclared responsibility boundary is a defect, not an annotator judgment call; so every
consequent must name a real agent and every handoff must be a well-formed ``a->b`` over declared
agents.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from calibrated_agent_evaluation.attribution.errors import CrewDefinitionError
from calibrated_agent_evaluation.attribution.types import (
    CrewDefinition,
    ExpectedProcess,
    MandatedRule,
)

_HANDOFF_SEP = "->"


def build_crew_definition(
    *,
    agents: Iterable[str],
    handoffs: Iterable[str],
    scenarios: Mapping[str, ExpectedProcess],
    rules: Iterable[MandatedRule],
) -> CrewDefinition:
    """Build a validated crew Define artifact.

    Raises :class:`CrewDefinitionError` if the crew declares no agents, a scenario declares an empty
    expected agent set, a scenario references an agent not in ``agents``, a mandated rule's
    ``consequent_agent`` is not a declared agent, or a handoff is not a well-formed ``a->b`` over
    two declared agents. No silent fall-back; an invalid Define artifact refuses loudly.
    """
    agent_tuple = tuple(agents)
    if not agent_tuple:
        raise CrewDefinitionError("A crew Define artifact must declare at least one agent.")
    agent_set = set(agent_tuple)

    scenario_map = dict(scenarios)
    for name, proc in scenario_map.items():
        _validate_expected_process(name, proc, agent_set)

    rule_tuple = tuple(rules)
    for rule in rule_tuple:
        if rule.consequent_agent not in agent_set:
            raise CrewDefinitionError(
                f"rule {rule.rule_id!r} names consequent agent {rule.consequent_agent!r}, "
                f"which is not a declared crew agent {sorted(agent_set)}."
            )

    handoff_tuple = tuple(handoffs)
    for handoff in handoff_tuple:
        _validate_handoff(handoff, agent_set)

    return CrewDefinition(
        agents=agent_tuple,
        handoffs=handoff_tuple,
        scenarios=scenario_map,
        rules=rule_tuple,
    )


def require_expected_process(crew: CrewDefinition, scenario: str) -> ExpectedProcess:
    """Return the ``ExpectedProcess`` for ``scenario``, or refuse.

    Raises :class:`CrewDefinitionError` if the scenario declares no expected process; the process
    judge cannot diff *expected minus actual* without it, and an undeclared boundary is a defect.
    """
    proc = crew.scenarios.get(scenario)
    if proc is None:
        raise CrewDefinitionError(
            f"scenario {scenario!r} declares no expected process; the process judge cannot "
            f"evaluate an undeclared crew route."
        )
    return proc


def _validate_expected_process(name: str, proc: ExpectedProcess, agent_set: set[str]) -> None:
    if not proc.expected_agents:
        raise CrewDefinitionError(
            f"scenario {name!r} declares an empty expected agent set; the expected route must be "
            f"declared before a trace can be judged against it."
        )
    unknown = [a for a in proc.expected_agents if a not in agent_set]
    if unknown:
        raise CrewDefinitionError(
            f"scenario {name!r} expects agents {unknown} not in the declared crew "
            f"{sorted(agent_set)}."
        )


def _validate_handoff(handoff: str, agent_set: set[str]) -> None:
    parts = handoff.split(_HANDOFF_SEP)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise CrewDefinitionError(
            f"handoff {handoff!r} is not a well-formed 'source->target' seam."
        )
    unknown = [p for p in parts if p not in agent_set]
    if unknown:
        raise CrewDefinitionError(
            f"handoff {handoff!r} names agent(s) {unknown} not in the declared crew "
            f"{sorted(agent_set)}."
        )
