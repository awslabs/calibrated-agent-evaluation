# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What an online run cost: the evaluations Calibrated Agent Evaluation can count, priced per
evaluation."""

from __future__ import annotations

from decimal import Decimal

from calibrated_agent_evaluation.agentcore.online_record import (
    AgentCoreSessionLog,
    AttachedEvaluator,
    OnlineEvaluationRecord,
)
from calibrated_agent_evaluation.agentcore.pricing import EvaluationRates

_RATES = EvaluationRates(region="us-east-1", custom_per_evaluation=Decimal("0.0015"))


def _record() -> OnlineEvaluationRecord:
    return OnlineEvaluationRecord(
        config_id="calibrated_agent_evaluation_live_rate-AbCdEf1234",
        config_name="calibrated_agent_evaluation_live_rate",
        runtime_service_name="calibrated_agent_evaluation_orderbot.DEFAULT",
        evaluators=(
            AttachedEvaluator(
                evaluator_id="calibrated_agent_evaluation_live_v2-AbCdEf1234",
                evaluator_arn="arn:aws:bedrock-agentcore:us-east-1:111122223333:evaluator/v2",
                updated_at="2026-11-06T17:06:54+00:00",
                fingerprint="2262bd6bd613198f",
                calib_id="calib-v2",
                battery_id="bias-v2",
            ),
        ),
        excluded_session_ids=("c1",),
        sampling_percentage=100.0,
        session_timeout_minutes=2,
        created_at="2026-11-06T18:15:00+00:00",
    )


def _log(kind: str, run_id: str, calls: int) -> AgentCoreSessionLog:
    return AgentCoreSessionLog(
        run_kind=kind,  # type: ignore[arg-type]
        run_id=run_id,
        judge_id="orderbot-refund-policy",
        evaluator_fingerprint="2262bd6bd613198f",
        session_ids=tuple(f"{run_id}-{i}" for i in range(calls)),
    )


def test_the_cost_counts_the_attached_runs_and_the_results_at_the_per_evaluation_rate() -> None:
    from calibrated_agent_evaluation.agentcore.cost import online_run_cost

    logs = [_log("calibration", "calib-v2", 324), _log("bias", "bias-v2", 420)]
    # An earlier calibration of the same judge is not the one the config was attached with.
    logs.append(_log("calibration", "calib-v2-first", 324))

    report = online_run_cost(_record(), logs, results=200, error_results=3, rates=_RATES)

    (judge,) = report.judges
    assert (judge.calibration_calls, judge.bias_calls) == (324, 420)
    assert report.results == 200
    assert report.error_results == 3
    assert report.priced_evaluations == 324 + 420 + 200
    assert report.cost == Decimal("0.0015") * (324 + 420 + 200)
    text = "\n".join(report.describe())
    assert "not included" in text  # the evaluator's own model usage
    assert "3 error result" in text
    assert "not documented" in text  # whether AWS bills an error result


def test_with_no_rate_the_cost_report_shows_the_counts_and_says_why() -> None:
    from calibrated_agent_evaluation.agentcore.cost import online_run_cost

    logs = [_log("calibration", "calib-v2", 324), _log("bias", "bias-v2", 420)]

    report = online_run_cost(_record(), logs, results=200, error_results=0, rates=None)

    assert report.cost is None
    text = "\n".join(report.describe())
    assert "944" in text
    assert "Price List" in text


def test_a_run_with_no_session_log_is_counted_as_none_and_named() -> None:
    from calibrated_agent_evaluation.agentcore.cost import online_run_cost

    report = online_run_cost(_record(), [], results=0, error_results=0, rates=_RATES)

    (judge,) = report.judges
    assert (judge.calibration_calls, judge.bias_calls) == (0, 0)
    assert "no session log" in "\n".join(report.describe())
