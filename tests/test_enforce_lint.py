# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the judge-definition linter and its tiered rules.

One positive and one negative or edge case per rule, asserting the tier of returned decisions and
the typed refusal on hard cases.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.enforce import (
    Axis,
    ChecklistItem,
    CritiqueOrderError,
    EnforcementContext,
    GenericMetricInCIError,
    JudgeDefinition,
    LikertNotAllowedError,
    MultiAxisLikertError,
    WeightedCollapseError,
    lint_judge_definition,
)


def _defn(**over: object) -> JudgeDefinition:
    base: dict[str, object] = {
        "judge_id": "orderbot-refund",
        "axes": (Axis(name="resolved"),),
        "output_fields": ("critique", "verdict"),
    }
    base.update(over)
    return JudgeDefinition(**base)  # type: ignore[arg-type]


_OFF = EnforcementContext()
_CI = EnforcementContext(ci=True)
_LIKERT_OK = EnforcementContext(allow_likert=True)


# --- multiple non-binary axes in one call -----------------------------


def test_two_likert_axes_refused() -> None:
    defn = _defn(axes=(Axis(name="helpfulness", scale="likert"), Axis(name="tone", scale="likert")))
    with pytest.raises(MultiAxisLikertError):
        lint_judge_definition(defn, _LIKERT_OK)  # refused even when likert is allowed


def test_single_binary_axis_lints_clean() -> None:
    assert lint_judge_definition(_defn(), _OFF) == []


# --- generic generation-quality metric banned in CI, else exploration -


def test_generic_metric_in_ci_refused() -> None:
    with pytest.raises(GenericMetricInCIError):
        lint_judge_definition(_defn(scorer="rouge"), _CI)


def test_generic_metric_outside_ci_allowed_as_exploration_only() -> None:
    decisions = lint_judge_definition(_defn(scorer="rouge"), _OFF)
    enf5 = [d for d in decisions if d.rule == "ENF-5"]
    assert len(enf5) == 1
    assert enf5[0].tier == "display_default"


# --- general multi-criteria -> warning + nudge, never refuse ----------


def test_multi_criteria_warns_with_nudge() -> None:
    defn = _defn(axes=(Axis(name="resolved"), Axis(name="policy_cited")))
    decisions = lint_judge_definition(defn, _OFF)
    enf3 = [d for d in decisions if d.rule == "ENF-3"]
    assert len(enf3) == 1
    assert enf3[0].tier == "warning"
    assert enf3[0].nudge is not None


def test_single_axis_emits_no_multi_criteria_warning() -> None:
    # Negative/edge: one axis is not multi-criteria.
    assert [d for d in lint_judge_definition(_defn(), _OFF) if d.rule == "ENF-3"] == []


# --- Likert behind the opt-in ------------------------


def test_single_likert_axis_allowed_with_warning() -> None:
    defn = _defn(axes=(Axis(name="tone", scale="likert"),))
    decisions = lint_judge_definition(defn, _LIKERT_OK)
    enf6 = [d for d in decisions if d.rule == "ENF-6"]
    assert len(enf6) == 1
    assert enf6[0].tier == "display_default"


def test_default_context_disables_likert() -> None:
    # Edge: the safe default is likert-disabled.
    assert EnforcementContext().allow_likert is False


def test_single_likert_axis_refused_when_disabled() -> None:
    # Negative: a Likert axis with allow_likert off is refused.
    defn = _defn(axes=(Axis(name="tone", scale="likert"),))
    with pytest.raises(LikertNotAllowedError):
        lint_judge_definition(defn, _OFF)


# --- critique before verdict -----------------------------------------


def test_correct_output_order_lints_clean() -> None:
    assert lint_judge_definition(_defn(output_fields=("critique", "verdict")), _OFF) == []


def test_verdict_before_critique_refused() -> None:
    with pytest.raises(CritiqueOrderError):
        lint_judge_definition(_defn(output_fields=("verdict", "critique")), _OFF)


def test_missing_critique_refused() -> None:
    # Edge: no critique field at all.
    with pytest.raises(CritiqueOrderError):
        lint_judge_definition(_defn(output_fields=("verdict",)), _OFF)


# --- weighted-threshold collapse is refused ---------------------------


def test_weighted_collapse_refused() -> None:
    defn = _defn(
        checklist=(ChecklistItem(name="policy_cited", critical=True),),
        weighted_collapse=True,
    )
    with pytest.raises(WeightedCollapseError):
        lint_judge_definition(defn, _OFF)


def test_tagged_checklist_lints_clean() -> None:
    defn = _defn(
        checklist=(
            ChecklistItem(name="policy_cited", critical=True),
            ChecklistItem(name="greeting", critical=False),
        ),
    )
    assert lint_judge_definition(defn, _OFF) == []
