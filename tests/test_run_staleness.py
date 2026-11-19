# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the passive staleness warning.

Behavioural: check_staleness warns (warning tier) on version drift OR input-drift-exceeded, returns
None when the version matches and drift is sub-threshold, and never produces a gate-failing tier.
"""

from __future__ import annotations

from dataclasses import dataclass

from _helpers import TEST_STAMP
from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord
from calibrated_agent_evaluation.run.report import DriftMetric, InputDriftReport
from calibrated_agent_evaluation.run.staleness import check_staleness


@dataclass(frozen=True)
class _Judge:
    judge_id: str
    model: str
    prompt_version: str


def _record(judge: _Judge, *, result: str = "CERTIFIED") -> CalibrationRecord:
    return CalibrationRecord(
        provenance=TEST_STAMP,
        calib_id="calib-1",
        judge_id=judge.judge_id,
        model=judge.model,
        prompt_version=judge.prompt_version,
        hunted_set_id="hunted",
        n_pos=80,
        n_neg=80,
        tpr=Bound(point=0.95, lb95=0.90, n=80),
        tnr=Bound(point=0.95, lb95=0.90, n=80),
        youden_j=0.90,
        result=result,  # type: ignore[arg-type]
    )


def _drift(*, exceeded: bool) -> InputDriftReport:
    value, threshold = (0.5, 0.25) if exceeded else (0.05, 0.25)
    return InputDriftReport(
        metrics=(
            DriftMetric(
                name="psi",
                value=value,
                threshold=threshold,
                exceeded=exceeded,
                p_value=0.001,
                method="test",
            ),
        ),
        dimension="taxonomy_cell",
    )


def test_version_drift_emits_warning() -> None:
    """A judge whose (model, prompt_version) drifted from its record → warning decision."""
    record = _record(_Judge("j", "bedrock/claude", "v1"))
    drifted = _Judge("j", "bedrock/claude", "v2")

    decision = check_staleness(drifted, record)

    assert decision is not None
    assert decision.tier == "warning"
    assert decision.rule == "RUN-4"


def test_input_drift_exceeded_emits_warning() -> None:
    """A matching-version judge whose input drift crossed threshold → warning decision."""
    judge = _Judge("j", "bedrock/claude", "v1")
    decision = check_staleness(judge, _record(judge), _drift(exceeded=True))

    assert decision is not None
    assert decision.tier == "warning"


def test_matching_version_and_subthreshold_drift_no_warning() -> None:
    """A matching version + sub-threshold drift → None (no staleness warning)."""
    judge = _Judge("j", "bedrock/claude", "v1")
    assert check_staleness(judge, _record(judge), _drift(exceeded=False)) is None


def test_matching_version_no_drift_report_no_warning() -> None:
    """A matching version and no drift report supplied → None."""
    judge = _Judge("j", "bedrock/claude", "v1")
    assert check_staleness(judge, _record(judge)) is None


def test_uncertified_record_is_stale() -> None:
    """A REFUSED record never certifies the judge → treated as stale (warning)."""
    judge = _Judge("j", "bedrock/claude", "v1")
    decision = check_staleness(judge, _record(judge, result="REFUSED"))
    assert decision is not None
    assert decision.tier == "warning"


def test_staleness_decision_is_never_a_gate_failing_tier() -> None:
    """Whatever fires, the staleness decision tier is 'warning' — never a hard refusal."""
    judge = _Judge("j", "bedrock/claude", "v1")
    for decision in (
        check_staleness(_Judge("j", "bedrock/claude", "v2"), _record(judge)),
        check_staleness(judge, _record(judge), _drift(exceeded=True)),
    ):
        assert decision is not None
        assert decision.tier == "warning"
