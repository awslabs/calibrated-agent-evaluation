# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the auditable EnforcementDecision.

Proves the tiering invariant (exactly one of three tiers) and the no-empty-inputs guard that keeps
an audit record from being un-auditable.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.enforce import EnforcementDecision, EnforcementError


def _decision(**over: object) -> EnforcementDecision:
    base: dict[str, object] = {
        "rule": "ENF-2",
        "tier": "hard_refusal",
        "inputs": {"judge_id": "j1"},
        "outcome": "refused",
        "message": "refused for test",
    }
    base.update(over)
    return EnforcementDecision(**base)  # type: ignore[arg-type]


# --- the decision records its inputs; empty inputs is refused ---------------


def test_decision_records_rule_tier_inputs_outcome() -> None:
    d = _decision()
    assert (d.rule, d.tier, d.outcome) == ("ENF-2", "hard_refusal", "refused")
    assert d.inputs == {"judge_id": "j1"}


def test_decision_with_empty_inputs_is_refused() -> None:
    # Negative: an audit record with nothing to audit is a silent-failure in disguise.
    with pytest.raises(EnforcementError):
        _decision(inputs={})


# --- every decision carries exactly one valid tier --------------------------


@pytest.mark.parametrize("tier", ["hard_refusal", "warning", "display_default"])
def test_decision_accepts_each_declared_tier(tier: str) -> None:
    assert _decision(tier=tier).tier == tier


def test_decision_rejects_out_of_set_tier() -> None:
    # Negative: an unrecognized tier value cannot be persisted (one behavior, one tier).
    with pytest.raises(EnforcementError):
        _decision(tier="soft_block")
