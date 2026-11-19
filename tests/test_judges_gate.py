# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the certification gate + level guard at run time."""

from __future__ import annotations

import pytest

from _judge_helpers import FakeJudgeModel, certified_record, judge
from calibrated_agent_evaluation.enforce import UncalibratedScoreError
from calibrated_agent_evaluation.judges.errors import JudgeLevelError
from calibrated_agent_evaluation.judges.runtime import JudgeTarget, run_judge


def _target() -> JudgeTarget:
    return JudgeTarget(trace_id="ORD-1001", span_id=None, content="issued a refund out of window")


def test_refused_calibration_blocks_run_before_model_call() -> None:
    """A REFUSED calibration record → UncalibratedScoreError, model never called,
    no Label written."""
    j = judge()
    model = FakeJudgeModel()
    record = certified_record(j, result="REFUSED")
    with pytest.raises(UncalibratedScoreError):
        run_judge(j, [_target()], calib_record=record, model_backend=model)
    assert model.call_count == 0  # gate fired before any model call


def test_version_drift_blocks_run() -> None:
    """A CERTIFIED record whose prompt_version drifts from the judge is refused."""
    j = judge(prompt_version="v2")
    model = FakeJudgeModel()
    # record pinned to v1 while the judge is v2 → certifies() is False → refuse.
    record = certified_record(j, prompt_version="v1")
    with pytest.raises(UncalibratedScoreError):
        run_judge(j, [_target()], calib_record=record, model_backend=model)
    assert model.call_count == 0


def test_certified_version_matched_record_proceeds() -> None:
    """A CERTIFIED, version-matched record lets the run proceed."""
    j = judge()
    model = FakeJudgeModel()
    result = run_judge(j, [_target()], calib_record=certified_record(j), model_backend=model)
    assert len(result.labels) == 1
    assert model.call_count == 1


def test_workflow_judge_refuses_per_agent_rate() -> None:
    """A per-agent rate from a workflow-level judge → JudgeLevelError."""
    j = judge(level="workflow")
    with pytest.raises(JudgeLevelError):
        run_judge(
            j,
            [_target()],
            calib_record=certified_record(j),
            model_backend=FakeJudgeModel(),
            per_agent_rate=True,
        )


def test_span_judge_allows_per_agent_rate() -> None:
    """A span-level judge may compute a per-agent rate (no refusal)."""
    j = judge(level="span")
    span_target = JudgeTarget(
        trace_id="ORD-1001", span_id="ORD-1001/orderbot", content="the span input+output"
    )
    result = run_judge(
        j,
        [span_target],
        calib_record=certified_record(j),
        model_backend=FakeJudgeModel(),
        per_agent_rate=True,
    )
    assert result.labels[0].span_id == "ORD-1001/orderbot"
