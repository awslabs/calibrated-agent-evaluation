# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publishing an online judge's calibrated pass rate to CloudWatch, against stubbed AWS calls."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("boto3")

from botocore.stub import Stubber

from calibrated_agent_evaluation.agentcore.errors import AgentCoreMetricNotPublished
from calibrated_agent_evaluation.agentcore.judge import judge_from_evaluator
from calibrated_agent_evaluation.agentcore.online_record import (
    AttachedEvaluator,
    OnlineEvaluationRecord,
)
from calibrated_agent_evaluation.calibrate.engine import corrected_rate
from calibrated_agent_evaluation.calibrate.plan import MIN_RATE_SAMPLE
from calibrated_agent_evaluation.judges.verdict import verdict_rater_id
from calibrated_agent_evaluation.schema.types import Label
from cli_store_helpers import certified_calibration

_ACCOUNT = "111122223333"
_EVALUATOR_ID = "calibrated_agent_evaluation_live_refunds-AbCdEf1234"
_ARN = f"arn:aws:bedrock-agentcore:us-east-1:{_ACCOUNT}:evaluator/{_EVALUATOR_ID}"
_WHEN = "2026-11-05T16:19:46+00:00"
_MODE = "trusted_broken_policy_tool"
_DOC = {
    "evaluatorArn": _ARN,
    "evaluatorId": _EVALUATOR_ID,
    "evaluatorName": "calibrated_agent_evaluation_live_refunds",
    "evaluatorType": "Custom",
    "level": "TRACE",
    "status": "ACTIVE",
    "createdAt": _WHEN,
    "updatedAt": _WHEN,
    "evaluatorConfig": {
        "llmAsAJudge": {
            "instructions": "Judge the refund. {context} {assistant_turn}",
            "ratingScale": {
                "numerical": [
                    {"value": 0.0, "label": "FAIL", "definition": "The refund breaks policy."},
                    {"value": 1.0, "label": "PASS", "definition": "The refund follows policy."},
                ]
            },
            "modelConfig": {
                "bedrockEvaluatorModelConfig": {
                    "modelId": "global.anthropic.claude-haiku-4-5-20251001-v1:0"
                }
            },
        }
    },
}


_NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)


def _clients() -> Any:
    import boto3

    from calibrated_agent_evaluation.agentcore.session import AwsClients

    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    return AwsClients(session)


def _judge() -> Any:
    return judge_from_evaluator("refunds", _DOC)


def _calib(judge: Any) -> Any:
    return certified_calibration(
        "refunds",
        model=judge.model,
        prompt_version=judge.prompt_version,
        fingerprint=judge.fingerprint,
    )


def _record(judge: Any, calib: Any) -> OnlineEvaluationRecord:
    return OnlineEvaluationRecord(
        config_id="calibrated_agent_evaluation_live_attach-AbCdEf1234",
        config_name="calibrated_agent_evaluation_live_attach",
        runtime_service_name="calibrated_agent_evaluation_orderbot.DEFAULT",
        evaluators=(
            AttachedEvaluator(
                evaluator_id=judge.evaluator_id,
                evaluator_arn=judge.evaluator_arn,
                updated_at=judge.updated_at,
                fingerprint=judge.fingerprint,
                calib_id=calib.calib_id,
                battery_id="bias-refunds-clear",
            ),
        ),
        excluded_session_ids=(),
        sampling_percentage=100.0,
        session_timeout_minutes=15,
        created_at=_WHEN,
    )


def _verdicts(fails: int, passes: int) -> list[Label]:
    """Online verdicts as the read-back stores them: advisory judge labels, one per session."""
    outcomes = ["FAIL"] * fails + ["PASS"] * passes
    return [
        Label(
            trace_id=f"session-{i:03d}",
            span_id=None,
            kind="llm-judge",
            outcome=outcome,  # type: ignore[arg-type]
            rater_id=verdict_rater_id(_judge()),
            failure_mode=_MODE if outcome == "FAIL" else None,
            critique="online verdict",
        )
        for i, outcome in enumerate(outcomes)
    ]


def _publish(
    labels: list[Label],
    *,
    document: dict[str, Any] | None = None,
    put: dict[str, Any] | None = None,
    put_error: str | None = None,
    calib: Any = None,
    deleted: bool = False,
) -> Any:
    from calibrated_agent_evaluation.agentcore.metrics import publish_pass_rate

    judge = _judge()
    attached_calib = _calib(judge)
    calib = calib or attached_calib
    clients = _clients()
    with (
        Stubber(clients.client("bedrock-agentcore-control")) as control,
        Stubber(clients.client("cloudwatch")) as cloudwatch,
    ):
        if deleted:
            control.add_client_error(
                "get_evaluator",
                service_error_code="ResourceNotFoundException",
                http_status_code=404,
            )
        else:
            control.add_response("get_evaluator", document or _DOC, {"evaluatorId": _EVALUATOR_ID})
        if put_error is not None:
            cloudwatch.add_client_error(
                "put_metric_data", service_error_code=put_error, http_status_code=400
            )
        elif put is not None:
            cloudwatch.add_response("put_metric_data", {}, put)
        report = publish_pass_rate(
            _record(judge, attached_calib),
            judge,
            calib,
            clients=clients,
            rate_labels=labels,
            gold_source="synthetic-rule",
            clock=lambda: _NOW,
        )
        control.assert_no_pending_responses()
        cloudwatch.assert_no_pending_responses()
    return report


def _expected_put(labels: list[Label]) -> dict[str, Any]:
    """The request the corrected rate implies: one minus it, with its bounds swapped."""
    judge = _judge()
    scored = corrected_rate(
        _calib(judge),
        [1 if label.outcome == "FAIL" else 0 for label in labels],
        "online-calibrated_agent_evaluation_live_attach-AbCdEf1234",
        rate_trace_ids=[label.trace_id for label in labels],
    ).corrected_rate
    assert scored is not None
    assert scored.point is not None
    dimensions = [
        {"Name": "JudgeId", "Value": "refunds"},
        {"Name": "EvaluatorFingerprint", "Value": judge.fingerprint},
        {"Name": "GoldSource", "Value": "synthetic-rule"},
    ]
    values = [
        ("CorrectedPassRate", 1 - scored.point),
        ("CorrectedPassRateLower95", 1 - scored.hi),
        ("CorrectedPassRateUpper95", 1 - scored.lo),
    ]
    return {
        "Namespace": "CalibratedAgentEvaluation/Judges",
        "MetricData": [
            {
                "MetricName": name,
                "Dimensions": dimensions,
                "Value": value,
                "Unit": "None",
                "Timestamp": _NOW,
            }
            for name, value in values
        ],
    }


def test_the_published_values_are_one_minus_the_corrected_rate_with_its_bounds_swapped() -> None:
    labels = _verdicts(30, 170)
    put = _expected_put(labels)

    report = _publish(labels, put=put)

    assert report.refusal is None
    published = dict(report.published)
    assert published == {m["MetricName"]: m["Value"] for m in put["MetricData"]}
    # Rogan-Gladen by hand: (0.15 - (1 - 0.95)) / (0.95 + 0.95 - 1) = 0.111 corrected failure
    # rate, so 0.889, not the raw 170 of 200 = 0.85 that AgentCore's own metrics carry.
    assert published["CorrectedPassRate"] == pytest.approx(1 - (0.15 - 0.05) / 0.9)
    assert published["CorrectedPassRateLower95"] <= published["CorrectedPassRate"]
    assert published["CorrectedPassRate"] <= published["CorrectedPassRateUpper95"]
    assert report.gold_source == "synthetic-rule"


def test_too_small_a_rate_sample_publishes_nothing_and_names_the_refusal() -> None:
    report = _publish(_verdicts(3, 17))  # no PutMetricData is stubbed: one would fail the test

    assert report.published == ()
    assert report.refusal is not None
    assert str(MIN_RATE_SAMPLE) in report.refusal


def test_a_rate_the_judge_cannot_correct_publishes_nothing_and_names_the_refusal() -> None:
    # 1 FAIL in 200 sits below the judge's false-positive floor, so the corrected rate has no
    # point: there is no honest pass rate to publish.
    report = _publish(_verdicts(1, 199))

    assert report.published == ()
    assert report.refusal is not None
    # Plain words for the user; the statistics are kept for --explain.
    assert "nothing was published" in report.refusal
    assert report.detail is not None
    assert "transfer" in report.detail


def test_a_changed_evaluator_is_reported_uncertified_and_nothing_is_published() -> None:
    edited = {
        **_DOC,
        "evaluatorConfig": {
            "llmAsAJudge": {
                **_DOC["evaluatorConfig"]["llmAsAJudge"],
                "instructions": "Judge it strictly. {context}",
            }
        },
    }

    report = _publish(_verdicts(30, 170), document=edited)

    assert report.published == ()
    assert report.status.state == "changed"
    assert report.refusal == report.status.reason


def test_a_rejected_publish_names_the_metric_and_is_not_reported_as_published() -> None:
    with pytest.raises(AgentCoreMetricNotPublished) as raised:
        _publish(_verdicts(30, 170), put_error="InvalidParameterValue")

    # One call carries all three values, so a rejected call published none of them.
    for name in ("CorrectedPassRate", "CorrectedPassRateLower95", "CorrectedPassRateUpper95"):
        assert name in raised.value.metric
    assert raised.value.error_code == "InvalidParameterValue"


def test_a_calibration_other_than_the_one_attached_publishes_nothing() -> None:
    # The rate must be corrected with the error rates measured for the evaluator the config runs,
    # not with another record for the same judge.
    judge = _judge()
    other = _calib(judge).model_copy(update={"calib_id": "calib-other"})

    report = _publish(_verdicts(30, 170), calib=other)

    assert report.published == ()
    assert report.refusal is not None
    assert "calib-other" in report.refusal


def test_verdicts_from_another_rater_or_repeated_for_a_trace_are_refused() -> None:
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError

    stranger = _verdicts(30, 170)
    stranger[0] = stranger[0].model_copy(update={"rater_id": "someone-else"})
    with pytest.raises(AgentCoreError, match="someone-else"):
        _publish(stranger)

    repeated = _verdicts(30, 170)
    repeated[1] = repeated[1].model_copy(update={"trace_id": repeated[0].trace_id})
    with pytest.raises(AgentCoreError, match=repeated[0].trace_id):
        _publish(repeated)


def test_a_lock_alone_still_publishes() -> None:
    locked = {**_DOC, "updatedAt": "2026-11-06T10:00:00+00:00", "lockedForModification": True}
    labels = _verdicts(30, 170)

    report = _publish(labels, document=locked, put=_expected_put(labels))

    assert report.status.state == "certified"
    assert len(report.published) == 3


def test_a_deleted_evaluator_is_reported_missing_and_nothing_is_published() -> None:
    report = _publish(_verdicts(30, 170), deleted=True)

    assert report.published == ()
    assert report.status.state == "missing"
    assert report.refusal == report.status.reason


def test_a_publish_that_may_or_may_not_have_landed_is_not_reported_as_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from botocore.exceptions import ReadTimeoutError

    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
    from calibrated_agent_evaluation.agentcore.metrics import publish_pass_rate

    judge = _judge()
    calib = _calib(judge)
    clients = _clients()

    def timed_out(**_: Any) -> Any:
        raise ReadTimeoutError(endpoint_url="https://monitoring.us-east-1.amazonaws.com")

    monkeypatch.setattr(clients.client("cloudwatch"), "put_metric_data", timed_out)
    with Stubber(clients.client("bedrock-agentcore-control")) as control:
        control.add_response("get_evaluator", _DOC, {"evaluatorId": _EVALUATOR_ID})
        with pytest.raises(AgentCoreError, match="may or may not have been recorded") as raised:
            publish_pass_rate(
                _record(judge, calib),
                judge,
                calib,
                clients=clients,
                rate_labels=_verdicts(30, 170),
                gold_source="synthetic-rule",
                clock=lambda: _NOW,
            )
    assert not isinstance(raised.value, AgentCoreMetricNotPublished)
    assert "CorrectedPassRateLower95" in str(raised.value)


def test_a_judge_the_config_did_not_attach_is_refused_before_any_aws_call() -> None:
    from dataclasses import replace

    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
    from calibrated_agent_evaluation.agentcore.metrics import publish_pass_rate

    judge = _judge()
    calib = _calib(judge)
    other = replace(judge, evaluator_id="calibrated_agent_evaluation_live_other-AbCdEf1234")
    with pytest.raises(AgentCoreError, match="not attached"):
        publish_pass_rate(
            _record(judge, calib),
            other,
            calib,
            clients=_clients(),
            rate_labels=[],
            gold_source="synthetic-rule",
        )


def test_verdicts_from_another_version_of_the_judge_are_refused() -> None:
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError

    older = [
        label.model_copy(update={"rater_id": "refunds@0000000000000000"})
        for label in _verdicts(30, 170)
    ]
    with pytest.raises(AgentCoreError, match="refunds@0000000000000000"):
        _publish(older)
