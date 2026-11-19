# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crew Define artifact — build + guards.

Every test embeds its FR; tests assert behavior (the built artifact's fields, or the typed refusal
that fires), never mock calls. The ClaimsCrew scenario S2 (vacancy exclusion) is the worked example.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.attribution import (
    CrewDefinitionError,
    ExpectedProcess,
    MandatedRule,
    build_crew_definition,
    require_expected_process,
)

_AGENTS = ("supervisor", "intake", "policy-checker", "fraud-scorer", "payout", "escalation")
_HANDOFFS = (
    "intake->policy-checker",
    "policy-checker->fraud-scorer",
    "fraud-scorer->payout",
    "fraud-scorer->escalation",
)
_S2 = ExpectedProcess(
    scenario="S2_vacancy_exclusion",
    expected_agents=("supervisor", "intake", "policy-checker"),
    terminal_decision="deny",
    payout_allowed=False,
    escalation_required=True,
)
_RULES = (
    MandatedRule(rule_id="R1", antecedent="deny", consequent_agent="supervisor"),
    MandatedRule(rule_id="R2", antecedent="fraud flag raised", consequent_agent="escalation"),
    MandatedRule(
        rule_id="R3",
        antecedent="occupancy narrative present",
        consequent_agent="intake",
    ),
)


def _crew():
    return build_crew_definition(
        agents=_AGENTS, handoffs=_HANDOFFS, scenarios={"S2_vacancy_exclusion": _S2}, rules=_RULES
    )


# --- a valid ClaimsCrew Define artifact builds and round-trips ---


def test_valid_claimscrew_definition_builds() -> None:
    crew = _crew()
    assert crew.agents == _AGENTS
    assert crew.handoffs == _HANDOFFS
    assert crew.scenarios["S2_vacancy_exclusion"].terminal_decision == "deny"
    assert {r.rule_id for r in crew.rules} == {"R1", "R2", "R3"}


# --- S2 records terminal deny, payout not allowed, escalation required ---


def test_S2_expected_process_fields() -> None:
    crew = _crew()
    proc = require_expected_process(crew, "S2_vacancy_exclusion")
    assert proc.expected_agents == ("supervisor", "intake", "policy-checker")
    assert proc.terminal_decision == "deny"
    assert proc.payout_allowed is False
    assert proc.escalation_required is True


# --- rules R1-R3 register with antecedent + consequent; R3 retrievable by id ---


def test_mandated_rules_retrievable_by_id() -> None:
    crew = _crew()
    r3 = next(r for r in crew.rules if r.rule_id == "R3")
    assert r3.consequent_agent == "intake"
    assert r3.antecedent
    assert r3.deterministic is True


# --- (negative) bad rule / malformed handoff refused ---


def test_rule_with_unknown_consequent_agent_refused() -> None:
    bad_rule = MandatedRule(rule_id="RX", antecedent="x", consequent_agent="auditor")
    with pytest.raises(CrewDefinitionError, match="not a declared crew agent"):
        build_crew_definition(
            agents=_AGENTS,
            handoffs=_HANDOFFS,
            scenarios={"S2_vacancy_exclusion": _S2},
            rules=(bad_rule,),
        )


def test_malformed_handoff_refused() -> None:
    with pytest.raises(CrewDefinitionError, match="well-formed"):
        build_crew_definition(
            agents=_AGENTS,
            handoffs=("intake->",),
            scenarios={"S2_vacancy_exclusion": _S2},
            rules=_RULES,
        )


def test_handoff_over_unknown_agent_refused() -> None:
    with pytest.raises(CrewDefinitionError, match="not in the declared crew"):
        build_crew_definition(
            agents=_AGENTS,
            handoffs=("intake->auditor",),
            scenarios={"S2_vacancy_exclusion": _S2},
            rules=_RULES,
        )


# --- (negative) empty / undeclared expected agent set refused ---


def test_empty_expected_agent_set_refused() -> None:
    empty = ExpectedProcess(
        scenario="S9",
        expected_agents=(),
        terminal_decision="deny",
        payout_allowed=False,
        escalation_required=False,
    )
    with pytest.raises(CrewDefinitionError, match="empty expected agent set"):
        build_crew_definition(
            agents=_AGENTS, handoffs=_HANDOFFS, scenarios={"S9": empty}, rules=_RULES
        )


def test_require_expected_process_refuses_undeclared_scenario() -> None:
    crew = _crew()
    with pytest.raises(CrewDefinitionError, match="declares no expected process"):
        require_expected_process(crew, "S_unknown")
