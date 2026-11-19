# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The deterministic process judge.

The process judge answers the structural question no span judge can: *right agents, valid order, no
skips/loops?* It is **deterministic** wherever possible -- a set-difference of the scenario's
expected agent set against the trace's actual agents, plus an ordering check -- so it needs no LLM
and no calibration. Only a **mandated-follow-up rule** whose antecedent depends on content
(e.g. R2: "a fraud flag mandates an escalation span") cannot be decided by set-difference; such a
rule is marked ``deterministic=False`` and reported ``undecided`` -- never a silent deterministic
pass. Nothing ships that decides one, so none is routed anywhere: the caller-supplied sub-judge
seam was removed because no implementation existed and no planned feature
owned one.

The scenario is resolved from ``trace.taxonomy_cell["scenario"]``; a trace whose scenario declares
no expected process is refused -- an undeclared boundary is a defect, not a judgment call.
"""

from __future__ import annotations

from calibrated_agent_evaluation.attribution.definition import require_expected_process
from calibrated_agent_evaluation.attribution.types import CrewDefinition, MandatedRule
from calibrated_agent_evaluation.judges.types import ProcessVerdict
from calibrated_agent_evaluation.schema.types import Trace

_SCENARIO_KEY = "scenario"


def process_judge(trace: Trace, crew: CrewDefinition) -> ProcessVerdict:
    """Deterministically judge a crew trace's structure against its scenario.

    Resolves the scenario from ``trace.taxonomy_cell["scenario"]`` and reports the skipped agents
    (expected minus actual) and unexpected agents (actual minus expected) -- the order-independent
    set differences a normalized trace reliably supports. Each deterministic rule is checked by
    agent presence; each non-deterministic rule is reported as undecided. ``passed`` is True only
    when nothing was skipped or unexpected and no rule was violated.

    **Ordering is NOT asserted deterministically.** A supervisor-fanout crew emits every specialist
    as a sibling of the supervisor, so the pipeline order (intake -> policy-checker -> ...) is not
    in the span tree; and the normalized ``seq`` is span *emission* order (the specialists start
    ~1 ms of each other), not the logical route -- the trace simply does not encode it (verified on
    the ClaimsCrew Strands ``Graph`` capture: no edge attributes, near-simultaneous starts). A
    ``seq``-based ordering assertion therefore false-positives on real traces, so
    ``ordering_undetermined=True`` records that order could not be verified rather than claiming a
    violation. A mis-route is still caught by the set differences + the rule clauses.

    Raises :class:`~calibrated_agent_evaluation.attribution.errors.CrewDefinitionError` if the
    scenario declares no expected process.
    """
    scenario = trace.taxonomy_cell.get(_SCENARIO_KEY, "")
    expected = require_expected_process(crew, scenario)

    actual_order = [s.agent for s in sorted(trace.spans, key=lambda s: s.seq)]
    actual_set = set(actual_order)
    expected_set = set(expected.expected_agents)

    skipped = tuple(a for a in expected.expected_agents if a not in actual_set)
    unexpected = tuple(a for a in actual_order if a not in expected_set)

    rule_violations: list[str] = []
    undecided_rules: list[str] = []
    for rule in crew.rules:
        _evaluate_rule(rule, actual_set, rule_violations, undecided_rules)

    passed = not skipped and not unexpected and not rule_violations
    return ProcessVerdict(
        skipped_agents=skipped,
        unexpected_agents=unexpected,
        ordering_undetermined=True,
        rule_violations=tuple(rule_violations),
        undecided_rules=tuple(undecided_rules),
        passed=passed,
    )


def _evaluate_rule(
    rule: MandatedRule,
    actual_set: set[str],
    rule_violations: list[str],
    undecided_rules: list[str],
) -> None:
    """Fold one rule into the violation/undecided lists."""
    if rule.deterministic:
        # A deterministic mandate is satisfied iff its consequent agent actually ran.
        if rule.consequent_agent not in actual_set:
            rule_violations.append(rule.rule_id)
        return
    # Content-dependent: nothing here can decide it, so say so rather than pass it.
    undecided_rules.append(rule.rule_id)
