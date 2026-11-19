# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crew failure counts for a store-derived crew evaluation.

A crew fix is gated on the workflow outcome rate and guarded per agent and per declared handoff.
This module turns one store's labels over the paired traces into those counts, and pairs the two
sides into the per-trace ``(failed before, failed after)`` tables the guard slices test:

  * :func:`crew_definition_from_file` reads the crew file (one JSON object) into a
    :class:`~calibrated_agent_evaluation.attribution.CrewDefinition` through
    ``build_crew_definition``.
  * :func:`count_crew_side` counts, for one side, each paired trace at most once per agent and
    once per handoff. Which labels are the source's verdicts and which agent a label blames are
    injected, so the caller applies the same attribution rule its per-agent rates use.
  * :func:`compare_crew_sides` pairs the two sides, refuses a slice measured on one side only and
    lists the slices measured on neither, which stay in the panel at a regression p-value of 1.
  * :func:`crew_outcomes` builds the two
    :class:`~calibrated_agent_evaluation.attribution.CrewOutcomes` for ``relocation_check`` from the
    workflow pairs and the per-agent counts.

A FAIL whose culprit is ``"disputed"`` blames no agent; its trace is named on the side's
``disputed`` list and still counts toward the workflow outcome, which comes from the pairs.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, ClassVar

from pydantic import ConfigDict, Field

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.attribution import (
    CrewDefinition,
    CrewOutcomes,
    ExpectedProcess,
    MandatedRule,
    build_crew_definition,
)
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.improve.errors import ImproveError
from calibrated_agent_evaluation.improve.report import PairedObs
from calibrated_agent_evaluation.schema.types import Label

__all__ = [
    "DISPUTED_CULPRIT",
    "METHOD_NOT_MEASURED",
    "CrewComparison",
    "CrewFile",
    "CrewSide",
    "compare_crew_sides",
    "count_crew_side",
    "crew_definition_from_file",
    "crew_outcomes",
    "not_measured_rate",
]

DISPUTED_CULPRIT = "disputed"
METHOD_NOT_MEASURED = "not_measured"
_SHOW = 5


class _Strict(CalibratedAgentEvaluationModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    invalid_error: ClassVar[type[Exception]] = ImproveError


class CrewFileScenario(_Strict):
    """One scenario's expected process in a crew file."""

    expected_agents: list[str]
    terminal_decision: str
    payout_allowed: bool
    escalation_required: bool


class CrewFileRule(_Strict):
    """One mandated follow-up rule in a crew file."""

    rule_id: str
    antecedent: str
    consequent_agent: str
    deterministic: bool = True


class CrewFile(_Strict):
    """The crew file ``calibrated-agent-evaluation improve --crew-file`` reads.

    ``agents`` and ``handoffs`` drive the guard panel. ``scenarios`` and ``rules`` are optional,
    validated by ``build_crew_definition`` when present, and not read by the counting.
    """

    agents: list[str]
    handoffs: list[str] = Field(default_factory=list)
    scenarios: dict[str, CrewFileScenario] = Field(default_factory=dict)
    rules: list[CrewFileRule] = Field(default_factory=list)


def crew_definition_from_file(doc: Mapping[str, Any]) -> CrewDefinition:
    """Validate a parsed crew file and build its :class:`CrewDefinition`.

    Raises :class:`ImproveError` for an unknown field, a value of the wrong type or a repeated
    agent or handoff, and :class:`~calibrated_agent_evaluation.attribution.CrewDefinitionError` for
    whatever ``build_crew_definition`` refuses (no agents, a malformed or undeclared handoff, a
    scenario or rule naming an undeclared agent).
    """
    crew_file = CrewFile.model_validate(dict(doc))
    for what, values in (("agent", crew_file.agents), ("handoff", crew_file.handoffs)):
        repeated = sorted({v for v in values if values.count(v) > 1})
        if repeated:
            raise ImproveError(f"the crew file lists {what}(s) more than once: {repeated}")
    return build_crew_definition(
        agents=crew_file.agents,
        handoffs=crew_file.handoffs,
        scenarios={
            name: ExpectedProcess(
                scenario=name,
                expected_agents=tuple(s.expected_agents),
                terminal_decision=s.terminal_decision,
                payout_allowed=s.payout_allowed,
                escalation_required=s.escalation_required,
            )
            for name, s in crew_file.scenarios.items()
        },
        rules=[
            MandatedRule(
                rule_id=r.rule_id,
                antecedent=r.antecedent,
                consequent_agent=r.consequent_agent,
                deterministic=r.deterministic,
            )
            for r in crew_file.rules
        ],
    )


class CrewSide(CalibratedAgentEvaluationModel):
    """One side's crew failures over the paired traces.

    ``agent_failed`` and ``handoff_failed`` map every declared agent and handoff to the paired
    traces that failed on it, each trace once. ``agents_measured`` and ``handoffs_measured`` are
    the ones the source spoke to on this side. ``disputed`` are the traces with a FAIL whose
    culprit is disputed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ImproveError

    agent_failed: dict[str, tuple[str, ...]]
    handoff_failed: dict[str, tuple[str, ...]]
    agents_measured: tuple[str, ...]
    handoffs_measured: tuple[str, ...]
    disputed: tuple[str, ...]


def count_crew_side(
    crew: CrewDefinition,
    labels: Mapping[str, Sequence[Label]],
    paired: Collection[str],
    *,
    is_verdict: Callable[[Label], bool],
    attribute: Callable[[Label], str | None],
    ran: Mapping[str, Collection[str]] | None = None,
    side: str,
) -> CrewSide:
    """Count one side's agent and handoff failures over the ``paired`` traces.

    ``is_verdict`` says which labels are the chosen source's verdicts; ``attribute`` names the
    agent a verdict blames (``None`` for none). A trace counts once for an agent when at least one
    FAIL verdict on it is attributed to the agent, and once for a handoff when one names it in
    ``handoff_implicated``. A FAIL whose ``culprit_span`` is ``"disputed"`` blames no agent and
    puts its trace on ``disputed``.

    An agent is measured when any verdict on a paired trace is attributed to it. ``ran`` (trace id
    -> the agents with a span on it) is given when the source's verdicts speak for every agent that
    ran, as a human's trace-level verdict does: an agent is then also measured when it ran on a
    paired trace, and a handoff when both its agents ran on one. Otherwise a handoff is measured
    only when a verdict names it.

    Raises :class:`ImproveError` when a verdict is attributed to an agent, or names a handoff, that
    ``crew`` does not declare: the crew file and the store describe different crews.
    """
    agent_failed: dict[str, set[str]] = {a: set() for a in crew.agents}
    handoff_failed: dict[str, set[str]] = {h: set() for h in crew.handoffs}
    agents_seen: set[str] = set()
    handoffs_seen: set[str] = set()
    disputed: set[str] = set()
    undeclared: dict[str, set[str]] = {}
    for trace_id in sorted(paired):
        for label in labels.get(trace_id, ()):
            if not is_verdict(label):
                continue
            failed = label.outcome == "FAIL"
            agent = attribute(label)
            if agent is not None:
                if agent not in agent_failed:
                    undeclared.setdefault(f"agent {agent!r}", set()).add(trace_id)
                    continue
                agents_seen.add(agent)
                if failed:
                    agent_failed[agent].add(trace_id)
            elif failed and label.culprit_span == DISPUTED_CULPRIT:
                disputed.add(trace_id)
            seam = label.handoff_implicated
            if seam is not None:
                if seam not in handoff_failed:
                    undeclared.setdefault(f"handoff {seam!r}", set()).add(trace_id)
                    continue
                handoffs_seen.add(seam)
                if failed:
                    handoff_failed[seam].add(trace_id)
        if ran is not None:
            here = set(ran.get(trace_id, ())) & set(agent_failed)
            agents_seen |= here
            handoffs_seen |= {h for h in crew.handoffs if set(h.split("->")) <= here}
    if undeclared:
        named = "; ".join(f"{what} on {_names(ids)}" for what, ids in sorted(undeclared.items()))
        raise ImproveError(
            f"the {side} store blames {named}, which the crew file does not declare (agents "
            f"{list(crew.agents)}, handoffs {list(crew.handoffs)}); the crew file and the store "
            "describe different crews"
        )
    return CrewSide(
        agent_failed={a: tuple(sorted(ids)) for a, ids in agent_failed.items()},
        handoff_failed={h: tuple(sorted(ids)) for h, ids in handoff_failed.items()},
        agents_measured=tuple(a for a in crew.agents if a in agents_seen),
        handoffs_measured=tuple(h for h in crew.handoffs if h in handoffs_seen),
        disputed=tuple(sorted(disputed)),
    )


class CrewComparison(CalibratedAgentEvaluationModel):
    """Both sides' crew failures, paired per trace for the guard slices.

    ``agent_pairs`` and ``handoff_pairs`` hold, for each agent and handoff measured on both
    sides, every paired trace's ``(failed before, failed after)``. ``not_measured_agents`` and
    ``not_measured_handoffs`` were measured on neither side.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ImproveError

    before: CrewSide
    after: CrewSide
    agent_pairs: dict[str, dict[str, tuple[bool, bool]]]
    handoff_pairs: dict[str, dict[str, tuple[bool, bool]]]
    not_measured_agents: tuple[str, ...]
    not_measured_handoffs: tuple[str, ...]


def compare_crew_sides(
    crew: CrewDefinition,
    before: CrewSide,
    after: CrewSide,
    paired: Collection[str],
    *,
    source: str,
) -> CrewComparison:
    """Pair ``before`` and ``after`` per trace over ``paired``, for every agent and handoff.

    Raises :class:`ImproveError` naming each agent or handoff measured on one side only: its
    other side has no evidence, and reading that as no failures would hide a regression.
    ``source`` names the verdicts in that message.
    """
    one_sided = [
        f"{what} {name!r} ({'before' if name in b else 'after'} only)"
        for what, names, b, a in (
            ("agent", crew.agents, before.agents_measured, after.agents_measured),
            ("handoff", crew.handoffs, before.handoffs_measured, after.handoffs_measured),
        )
        for name in names
        if (name in b) != (name in a)
    ]
    if one_sided:
        raise ImproveError(
            f"the {source} measure {', '.join(one_sided)} on one side only, so a change there "
            "cannot be paired; give it verdicts in both stores or fix the crew file"
        )
    ids = sorted(paired)

    def table(b: tuple[str, ...], a: tuple[str, ...]) -> dict[str, tuple[bool, bool]]:
        sb, sa = set(b), set(a)
        return {t: (t in sb, t in sa) for t in ids}

    return CrewComparison(
        before=before,
        after=after,
        agent_pairs={
            g: table(before.agent_failed[g], after.agent_failed[g])
            for g in crew.agents
            if g in before.agents_measured
        },
        handoff_pairs={
            h: table(before.handoff_failed[h], after.handoff_failed[h])
            for h in crew.handoffs
            if h in before.handoffs_measured
        },
        not_measured_agents=tuple(g for g in crew.agents if g not in before.agents_measured),
        not_measured_handoffs=tuple(h for h in crew.handoffs if h not in before.handoffs_measured),
    )


def crew_outcomes(
    comparison: CrewComparison, workflow: Sequence[PairedObs]
) -> tuple[CrewOutcomes, CrewOutcomes]:
    """The before and after :class:`CrewOutcomes` for ``relocation_check``.

    The workflow failures are the FAILs in the (adjudicated) workflow pairs, over all of them;
    each agent's failures are its counted traces on that side.
    """
    total = len(workflow)
    return (
        CrewOutcomes(
            workflow_failures=sum(p.before_fail for p in workflow),
            workflow_total=total,
            per_agent_failures={a: len(t) for a, t in comparison.before.agent_failed.items()},
        ),
        CrewOutcomes(
            workflow_failures=sum(p.after_fail for p in workflow),
            workflow_total=total,
            per_agent_failures={a: len(t) for a, t in comparison.after.agent_failed.items()},
        ),
    )


def not_measured_rate(source: str) -> Interval:
    """The rate of a slice the source did not measure: no point, and the whole [0, 1] range."""
    return Interval(
        point=None,
        lo=0.0,
        hi=1.0,
        refusal_reason=f"not measured by the {source}",
        method=METHOD_NOT_MEASURED,
        level=0.95,
        sided="two-sided",
        n=0,
    )


def _names(ids: Collection[str]) -> str:
    shown = sorted(ids)
    head = ", ".join(repr(i) for i in shown[:_SHOW])
    return head + (f" and {len(shown) - _SHOW} more" if len(shown) > _SHOW else "")
