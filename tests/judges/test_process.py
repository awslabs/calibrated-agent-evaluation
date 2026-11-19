# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic process judge.

Every test embeds its FR and asserts behavior (the ``ProcessVerdict``, or the typed refusal), never
mock calls. ``CLM-4471`` (S2) ran the right agents in the right order — its failure is a span-level
intake defect, not a structural one — so the process judge PASSES it structurally.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.attribution import CrewDefinitionError, MandatedRule
from calibrated_agent_evaluation.judges import process_judge
from calibrated_agent_evaluation.schema.types import Trace

# --- CLM-4471 ran the expected agents in order → passed, no LLM call ---


def test_clm4471_structure_passes(corpus_clm: Trace, claimscrew) -> None:
    verdict = process_judge(corpus_clm, claimscrew)
    assert verdict.skipped_agents == ()
    assert verdict.unexpected_agents == ()
    assert verdict.rule_violations == ()
    assert verdict.passed is True


# --- skipped / unexpected agent reported (order-independent set difference) ---


def test_skipped_expected_agent_reported(corpus_clm: Trace, claimscrew) -> None:
    # Drop the policy-checker span: the expected set now has a member the trace doesn't.
    without_policy = replace(
        corpus_clm, spans=[s for s in corpus_clm.spans if s.agent != "policy-checker"]
    )
    verdict = process_judge(without_policy, claimscrew)
    assert "policy-checker" in verdict.skipped_agents
    assert verdict.passed is False


def test_seq_reorder_is_not_a_false_violation(corpus_clm: Trace, claimscrew) -> None:
    # Reordering sibling spans' seq must NOT fail the judge: a supervisor-fanout crew's pipeline
    # order is not encoded in a normalized trace (siblings under the supervisor; seq is emission
    # order, not the route), so the deterministic judge records order as undetermined rather than
    # inventing a violation. The set of agents is unchanged, so nothing is skipped/unexpected.
    spans = list(corpus_clm.spans)
    intake = next(s for s in spans if s.agent == "intake")
    payout = next(s for s in spans if s.agent == "payout")
    reordered = []
    for s in spans:
        if s.agent == "intake":
            reordered.append(replace(s, seq=payout.seq))
        elif s.agent == "payout":
            reordered.append(replace(s, seq=intake.seq))
        else:
            reordered.append(s)
    verdict = process_judge(replace(corpus_clm, spans=reordered), claimscrew)
    assert verdict.ordering_undetermined is True
    assert verdict.skipped_agents == ()
    assert verdict.unexpected_agents == ()
    assert verdict.passed is True  # same agent SET, no rule violated → structurally sound


def test_unexpected_agent_reported(corpus_clm: Trace, claimscrew) -> None:
    # A genuine mis-route IS caught: an agent that ran but is not in the expected set. (This is the
    # order-independent structural catch that survives on a real normalized trace.)
    spans = list(corpus_clm.spans)
    rogue = replace(spans[-1], span_id="CLM-4471/auditor", agent="auditor", seq=9)
    verdict = process_judge(replace(corpus_clm, spans=[*spans, rogue]), claimscrew)
    assert "auditor" in verdict.unexpected_agents
    assert verdict.passed is False


def test_deterministic_rule_violation_reported(corpus_clm: Trace, claimscrew) -> None:
    # Add a deterministic rule mandating an 'auditor' agent that never ran → R9 violated.
    crew = replace(
        claimscrew,
        agents=(*claimscrew.agents, "auditor"),
        rules=(
            *claimscrew.rules,
            MandatedRule(rule_id="R9", antecedent="always", consequent_agent="auditor"),
        ),
    )
    verdict = process_judge(corpus_clm, crew)
    assert "R9" in verdict.rule_violations
    assert verdict.passed is False


# --- a non-deterministic rule is reported undecided, never a silent pass ---


def test_nondeterministic_rule_is_undecided(corpus_clm: Trace, claimscrew) -> None:
    verdict = process_judge(corpus_clm, claimscrew)
    # R2 depends on content; nothing decides it, so it is undecided, NOT a silent pass.
    assert "R2" in verdict.undecided_rules
    assert "R2" not in verdict.rule_violations


# --- (negative) scenario with no declared expected process → refused ---


def test_undeclared_scenario_refused(corpus_clm: Trace, claimscrew) -> None:
    unknown_scenario = replace(
        corpus_clm, taxonomy_cell={**corpus_clm.taxonomy_cell, "scenario": "S_unknown"}
    )
    with pytest.raises(CrewDefinitionError, match="declares no expected process"):
        process_judge(unknown_scenario, claimscrew)
