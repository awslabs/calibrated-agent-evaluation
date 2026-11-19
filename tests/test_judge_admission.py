# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the three-gate judge-trust chain.

admit_judge admits a judge only after certify -> bias -> score all pass; an
uncertified, biased, or version-drifted judge is refused (no Score written from it). A refused
GateDecisionRef drives the judge stage's ok=False, so the harness never scores with an untrusted
judge.
"""

from __future__ import annotations

from _helpers import TEST_STAMP
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.bias import BiasBatteryRecord
from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord
from calibrated_agent_evaluation.enforce.admission import admit_judge
from calibrated_agent_evaluation.judges import ExecutableJudge


def _judge() -> ExecutableJudge:
    return ExecutableJudge(
        judge_id="orderbot-refund-resolved",
        model="bedrock/global.anthropic.claude-sonnet-4-6",
        prompt_version="demo-v1",
        level="workflow",
        prompt_template="grade the refund",
    )


def _certified_record(judge: ExecutableJudge) -> CalibrationRecord:
    return CalibrationRecord(
        provenance=TEST_STAMP,
        calib_id="calib-orderbot",
        judge_id=judge.judge_id,
        model=judge.model,
        prompt_version=judge.prompt_version,
        judge_fingerprint=judge.fingerprint,
        hunted_set_id="hunted",
        n_pos=80,
        n_neg=80,
        tpr=Bound(point=0.95, lb95=0.90, n=80),
        tnr=Bound(point=0.95, lb95=0.90, n=80),
        youden_j=0.90,
        result="CERTIFIED",
    )


def _clear_battery(judge: ExecutableJudge, verdict: str = "CLEAR") -> BiasBatteryRecord:
    return BiasBatteryRecord(
        provenance=TEST_STAMP,
        battery_id="bias-orderbot",
        judge_id=judge.judge_id,
        model=judge.model,
        prompt_version=judge.prompt_version,
        judge_fingerprint=judge.fingerprint,
        is_pairwise=False,
        results=(),
        verdict=verdict,  # type: ignore[arg-type]
        failed_biases=() if verdict == "CLEAR" else ("length",),
    )


def test_certified_clear_judge_is_admitted() -> None:
    """A certified + bias-CLEAR judge clears all three gates and is admitted to score."""
    judge = _judge()
    ref = admit_judge(judge, _certified_record(judge), _clear_battery(judge))
    assert ref.admitted is True
    assert ref.certified is True
    assert ref.bias_verdict == "CLEAR"
    assert ref.score_gate_admitted is True
    assert ref.refusal_reason is None


def test_uncertified_judge_is_refused_no_score() -> None:
    """A REFUSED calibration record fails the first gate -> refused ref, no Score."""
    judge = _judge()
    uncertified = replace(_certified_record(judge), result="REFUSED")
    ref = admit_judge(judge, uncertified, _clear_battery(judge))
    assert ref.admitted is False
    assert ref.certified is False
    assert "calibration gate refused" in (ref.refusal_reason or "")


def test_biased_judge_is_refused_no_score() -> None:
    """A BIASED battery fails the bias gate even for a certified judge -> refused ref."""
    judge = _judge()
    ref = admit_judge(judge, _certified_record(judge), _clear_battery(judge, verdict="BIASED"))
    assert ref.admitted is False
    assert ref.certified is True  # calibration passed
    assert ref.bias_verdict == "BIASED"
    assert "bias gate refused" in (ref.refusal_reason or "")


def test_version_drifted_judge_is_refused_no_score() -> None:
    """A judge whose (model, prompt_version) drifted from its record is refused."""
    judge = _judge()
    record = _certified_record(judge)
    drifted = replace(judge, prompt_version="demo-v2")  # record pins demo-v1
    ref = admit_judge(drifted, record, _clear_battery(judge))
    assert ref.admitted is False
    assert ref.certified is False  # certifies() rejects the version drift
    assert "calibration gate refused" in (ref.refusal_reason or "")
