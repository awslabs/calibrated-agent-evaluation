# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for checklist collapse.

Any critical "no" collapses to overall FAIL; non-critical items are tracked, never gating.
"""

from __future__ import annotations

from calibrated_agent_evaluation.enforce import (
    Axis,
    ChecklistItem,
    JudgeDefinition,
    collapse_checklist,
)


def _defn() -> JudgeDefinition:
    return JudgeDefinition(
        judge_id="orderbot-refund",
        axes=(Axis(name="resolved"),),
        output_fields=("critique", "verdict"),
        checklist=(
            ChecklistItem(name="policy_cited", critical=True),
            ChecklistItem(name="greeting", critical=False),
        ),
    )


def test_critical_no_collapses_to_fail() -> None:
    result = collapse_checklist(_defn(), {"policy_cited": "FAIL", "greeting": "PASS"})
    assert result.overall == "FAIL"


def test_non_critical_no_does_not_gate() -> None:
    # Negative/edge: only a non-critical item failed -> overall PASS, item tracked as sub-metric.
    result = collapse_checklist(_defn(), {"policy_cited": "PASS", "greeting": "FAIL"})
    assert result.overall == "PASS"
    assert result.non_gating == {"greeting": "FAIL"}


def test_all_pass_is_pass() -> None:
    result = collapse_checklist(_defn(), {"policy_cited": "PASS", "greeting": "PASS"})
    assert result.overall == "PASS"


def test_missing_critical_verdict_fails_closed() -> None:
    # Edge: a critical item with no recorded verdict cannot certify a PASS.
    result = collapse_checklist(_defn(), {"greeting": "PASS"})
    assert result.overall == "FAIL"


def test_empty_checklist_is_pass() -> None:
    # Edge: a definition with no checklist items collapses to PASS with no sub-metrics.
    defn = JudgeDefinition(
        judge_id="j", axes=(Axis(name="resolved"),), output_fields=("critique", "verdict")
    )
    result = collapse_checklist(defn, {})
    assert result.overall == "PASS"
    assert result.non_gating == {}
