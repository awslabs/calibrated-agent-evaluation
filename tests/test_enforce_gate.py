# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the score gate + error-analysis-first guard.

: no certifying calibration -> UncalibratedScoreError, reading the verdict.
: scoring before any human labels warns.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from _helpers import TEST_STAMP
from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord
from calibrated_agent_evaluation.enforce import (
    UncalibratedScoreError,
    enforce_score_gate,
    warn_error_analysis_first,
)


@dataclass(frozen=True)
class FakeJudge:
    judge_id: str
    model: str
    prompt_version: str


def _record(
    *, result: str = "CERTIFIED", model: str = "m1", prompt_version: str = "v1"
) -> CalibrationRecord:
    return CalibrationRecord(
        provenance=TEST_STAMP,
        calib_id="c",
        judge_id="orderbot-refund",
        model=model,
        prompt_version=prompt_version,
        hunted_set_id="h",
        n_pos=80,
        n_neg=80,
        tpr=Bound(point=0.90, lb95=0.83, n=80),
        tnr=Bound(point=0.90, lb95=0.81, n=80),
        youden_j=0.80,
        result=result,  # type: ignore[arg-type]
    )


# --- the score gate reads the verdict -----------------------------------


def test_certified_judge_at_pinned_version_passes() -> None:
    judge = FakeJudge("orderbot-refund", "m1", "v1")
    assert enforce_score_gate(_record(), judge) is None  # returns cleanly, no refusal


def test_uncertified_record_refuses_score() -> None:
    # Negative: a REFUSED calibration blocks the score.
    judge = FakeJudge("orderbot-refund", "m1", "v1")
    with pytest.raises(UncalibratedScoreError) as exc:
        enforce_score_gate(_record(result="REFUSED"), judge)
    assert exc.value.decision is not None
    assert exc.value.decision.tier == "hard_refusal"


def test_version_drift_refuses_score() -> None:
    # Negative: a CERTIFIED record no longer matching the judge's prompt_version requires
    # re-calibration -> the gate refuses.
    judge = FakeJudge("orderbot-refund", "m1", "v2")
    with pytest.raises(UncalibratedScoreError):
        enforce_score_gate(_record(prompt_version="v1"), judge)


# --- error-analysis-first warning -------------------------------------------


def test_zero_human_labels_warns() -> None:
    decision = warn_error_analysis_first("orderbot-refund", human_label_count=0)
    assert decision is not None
    assert decision.tier == "warning"
    assert decision.nudge is not None


def test_existing_labels_do_not_warn() -> None:
    # Negative: once error analysis has produced labels, no warning fires.
    assert warn_error_analysis_first("orderbot-refund", human_label_count=12) is None
