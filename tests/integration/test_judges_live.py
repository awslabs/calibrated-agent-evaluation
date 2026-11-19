# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in LIVE judge run against real Bedrock Claude — both-worlds parity.

Skipped by default (marked ``integration``, deselected in the default pytest run). Requires AWS
Bedrock access (the same creds the live executor uses):

    export CALIBRATED_AGENT_EVALUATION_JUDGE_LIVE=1
    export AWS_REGION=us-east-1            # and valid Bedrock creds on the default chain
    uv run pytest -m integration -k judges_live

The "does the judge runtime actually force a structured verdict off a real model and write it back"
gate: exercises the real Inspect Bedrock provider + the response_schema path + a real trace-store
write-back. Point ``store`` at a local backend here; the Langfuse leg is exercised in acceptance.
"""

from __future__ import annotations

import os

import pytest
from _bedrock import inspect_bedrock_model

from _helpers import TEST_STAMP
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord
from calibrated_agent_evaluation.execute import InspectJudgeModel
from calibrated_agent_evaluation.judges.executable import ExecutableJudge
from calibrated_agent_evaluation.judges.runtime import JudgeTarget, run_judge

pytestmark = pytest.mark.integration

_LIVE = os.environ.get("CALIBRATED_AGENT_EVALUATION_JUDGE_LIVE")


@pytest.mark.skipif(
    not _LIVE, reason="set CALIBRATED_AGENT_EVALUATION_JUDGE_LIVE=1 to run the live judge"
)
def test_live_judge_forces_structured_verdict_and_writes_back() -> None:
    """Live: real Bedrock Claude forces a structured verdict, writes back."""
    judge = ExecutableJudge(
        judge_id="orderbot-refund-resolved",
        model=inspect_bedrock_model(),
        prompt_version="live-v1",
        level="workflow",
        prompt_template=(
            "You are grading whether an order agent resolved a refund correctly. "
            "Return a binary verdict with a critique explaining your reasoning first."
        ),
    )
    record = CalibrationRecord(
        provenance=TEST_STAMP,
        calib_id="calib-live",
        judge_id=judge.judge_id,
        model=judge.model,
        prompt_version=judge.prompt_version,
        judge_fingerprint=judge.fingerprint,
        hunted_set_id="hunted-live",
        n_pos=80,
        n_neg=80,
        tpr=Bound(point=0.95, lb95=0.90, n=80),
        tnr=Bound(point=0.95, lb95=0.90, n=80),
        youden_j=0.90,
        result="CERTIFIED",
    )
    target = JudgeTarget(
        trace_id="ORD-1001",
        span_id=None,
        content=(
            "The agent issued an $89.99 refund on a return 51 days after delivery, "
            "against a 30-day return window."
        ),
    )
    with LocalBackend(":memory:") as backend:
        result = run_judge(
            judge,
            [target],
            calib_record=record,
            model_backend=InspectJudgeModel(judge.model),
            store=backend,
        )
        assert len(result.labels) == 1
        label = result.labels[0]
        assert label.kind == "llm-judge"
        assert label.outcome in ("PASS", "FAIL")
        assert label.critique.strip()
        # written back and re-readable
        assert backend.get_scores("ORD-1001")[0].critique == label.critique
