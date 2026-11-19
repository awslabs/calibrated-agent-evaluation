# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AgentCore online-evaluation adapter: its errors and its import boundary."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import textwrap
import typing
from typing import Any

import pytest

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreError,
    AgentCoreMetricNotPublished,
    AgentCoreNotCertified,
    AgentCoreResourceExists,
    AgentCoreResourceNotActive,
    AgentCoreRuntimeNotFound,
)

_AWS_SDK_ROOTS = frozenset({"boto3", "botocore", "bedrock_agentcore", "strands_evals"})

# The modules this adapter adds; each must load without an AWS SDK. Grows as modules land.
_ONLINE_MODULES = (
    "calibrated_agent_evaluation.agentcore.bias",
    "calibrated_agent_evaluation.agentcore.calibrate",
    "calibrated_agent_evaluation.agentcore.cost",
    "calibrated_agent_evaluation.agentcore.evaluator_create",
    "calibrated_agent_evaluation.agentcore.metrics",
    "calibrated_agent_evaluation.agentcore.online",
    "calibrated_agent_evaluation.agentcore.online_record",
    "calibrated_agent_evaluation.agentcore.online_version",
    "calibrated_agent_evaluation.agentcore.readback",
)


@pytest.mark.parametrize(
    ("error", "says"),
    [
        pytest.param(
            AgentCoreResourceExists("evaluator", "calibrated_agent_evaluation_live_refunds"),
            ["evaluator calibrated_agent_evaluation_live_refunds already exists", "not created"],
            id="name-taken",
        ),
        pytest.param(
            AgentCoreResourceNotActive(
                "online config",
                "calibrated_agent_evaluation_live_refunds-ZE8rieGKWt",
                "CREATE_FAILED",
                "no role",
            ),
            ["reported CREATE_FAILED: no role", "not recorded as created"],
            id="create-failed",
        ),
        pytest.param(
            AgentCoreResourceNotActive(
                "online config", "calibrated_agent_evaluation_live_refunds-ZE8rieGKWt", "ERROR"
            ),
            ["reported ERROR: no reason given"],
            id="error-without-a-reason",
        ),
        pytest.param(
            AgentCoreMetricNotPublished("CorrectedPassRateLower95", "AccessDenied"),
            ["CorrectedPassRateLower95 (AccessDenied)", "not published"],
            id="metric-refused",
        ),
        pytest.param(
            AgentCoreRuntimeNotFound("orderbot-AbCdEf1234"),
            ["runtime orderbot-AbCdEf1234 was not found", "no online config was created"],
            id="no-runtime",
        ),
        pytest.param(
            AgentCoreNotCertified(
                "orderbot-refund-policy",
                ("a CLEAR bias battery (the latest is BIASED)", "the session log of its bias run"),
            ),
            [
                "missing a CLEAR bias battery (the latest is BIASED); the session log of its",
                "Nothing was sent to AgentCore",
            ],
            id="not-certified",
        ),
    ],
)
def test_each_online_refusal_names_what_it_refused(error: AgentCoreError, says: list[str]) -> None:
    assert isinstance(error, AgentCoreError)
    for phrase in says:
        assert phrase in str(error)


def test_the_metric_error_carries_the_aws_error_code_not_cloudwatchs_message() -> None:
    error = AgentCoreMetricNotPublished("CorrectedPassRateLower95", "InvalidParameterValue")

    assert error.error_code == "InvalidParameterValue"
    assert not hasattr(error, "reason")
    assert str(error).count("(") == 1  # the code alone, in one parenthesis; no message after it


def test_a_resource_still_creating_says_the_wait_ran_out_even_with_a_reason() -> None:
    error = AgentCoreResourceNotActive(
        "evaluator",
        "calibrated_agent_evaluation_live_refunds-AbCdEf1234",
        "CREATING",
        "still provisioning",
    )

    assert error.status == "CREATING"
    assert "was still CREATING when the wait ran out (still provisioning)" in str(error)


def test_an_uncertified_judge_with_nothing_missing_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="missing"):
        AgentCoreNotCertified("orderbot-refund-policy", ())


def test_the_online_modules_load_no_aws_sdk() -> None:
    code = f"""
        import json, sys
        for name in {list(_ONLINE_MODULES)!r}:
            __import__(name)
        print(json.dumps(sorted(sys.modules)))
    """
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, check=True
    ).stdout
    loaded = set(json.loads(out.strip().splitlines()[-1]))

    assert {name for name in loaded if name.partition(".")[0] in _AWS_SDK_ROOTS} == set()


def _classes_named(hint: object) -> set[type]:
    """Every class a type hint names, through unions and generics."""
    found = {hint} if isinstance(hint, type) else set()
    for arg in typing.get_args(hint):
        found |= _classes_named(arg)
    return found


def _hints_of(obj: object) -> list[object]:
    # These are imported for type checking only (AwsClients pulls boto3, BiasBatteryRecord the
    # stats extra); resolve them here, where the test may import them. All are the library's own
    # types, so the walk checks what they name.
    from calibrated_agent_evaluation.agentcore.guards import ReadOnlyGuard
    from calibrated_agent_evaluation.agentcore.pricing import RunEstimate
    from calibrated_agent_evaluation.agentcore.session import AwsClients
    from calibrated_agent_evaluation.bias import BiasBatteryRecord

    if inspect.isclass(obj) or inspect.isfunction(obj):
        return list(
            typing.get_type_hints(
                obj,
                localns={
                    "AwsClients": AwsClients,
                    "BiasBatteryRecord": BiasBatteryRecord,
                    "ReadOnlyGuard": ReadOnlyGuard,
                    "RunEstimate": RunEstimate,
                },
            ).values()
        )
    return []


def test_the_online_modules_name_no_aws_type_on_their_surface() -> None:
    exported = [
        getattr(module, name)
        for module in (__import__(m, fromlist=["__all__"]) for m in _ONLINE_MODULES)
        for name in module.__all__
    ]
    named = {cls for obj in exported for hint in _hints_of(obj) for cls in _classes_named(hint)}

    assert named, "the walk found no annotated names, so it checked nothing"
    assert not [cls for cls in named if cls.__module__.partition(".")[0] in _AWS_SDK_ROOTS]


# --- the preview-and-confirm step and the AWS error wrapping -----------------------------------

_ACCOUNT = "111122223333"
_EVALUATOR_ID = "calibrated_agent_evaluation_live_refunds-AbCdEf1234"
_EVALUATOR_ARN = f"arn:aws:bedrock-agentcore:us-east-1:{_ACCOUNT}:evaluator/{_EVALUATOR_ID}"
_CREATE_EVALUATOR = {
    "evaluatorName": "calibrated_agent_evaluation_live_refunds",
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
    "level": "TRACE",
}
_CREATE_ONLINE_CONFIG = {
    "onlineEvaluationConfigName": "calibrated_agent_evaluation_live_refunds",
    "rule": {
        "samplingConfig": {"samplingPercentage": 100.0},
        "sessionConfig": {"sessionTimeoutMinutes": 15},
    },
    "dataSourceConfig": {
        "cloudWatchLogs": {"serviceNames": ["orderbot.DEFAULT"], "logGroupNames": []}
    },
    "evaluators": [{"evaluatorId": _EVALUATOR_ID}],
    "evaluationExecutionRoleArn": (
        f"arn:aws:iam::{_ACCOUNT}:role/calibrated-agent-evaluation-live-agentcore-evaluation"
    ),
    "enableOnCreate": True,
}
_PUT_METRIC = {
    "Namespace": "CalibratedAgentEvaluation/Judges",
    "MetricData": [{"MetricName": "CorrectedPassRate", "Value": 0.9}],
}


def _clients() -> Any:
    boto3 = pytest.importorskip("boto3")
    from calibrated_agent_evaluation.agentcore.session import AwsClients

    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    return AwsClients(session)


@pytest.mark.parametrize(
    ("service", "operation", "request_"),
    [
        ("bedrock-agentcore-control", "CreateEvaluator", _CREATE_EVALUATOR),
        ("cloudwatch", "PutMetricData", _PUT_METRIC),
    ],
    ids=["agentcore", "cloudwatch"],
)
def test_a_declined_preview_sends_nothing(
    service: str, operation: str, request_: dict[str, Any]
) -> None:
    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.errors import AgentCoreRunDeclined
    from calibrated_agent_evaluation.agentcore.online import send_confirmed

    clients = _clients()
    seen: list[Any] = []
    # Nothing is queued, so any request the step sent would fail as unstubbed, not as declined.
    with Stubber(clients.client(service)), pytest.raises(AgentCoreRunDeclined, match=operation):
        send_confirmed(
            clients,
            service,
            operation,
            request_,
            summary="one write",
            confirm=lambda preview: seen.append(preview) or False,
        )
    assert [p.operation for p in seen] == [operation]


def test_an_approved_preview_sends_exactly_the_previewed_request() -> None:
    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.online import send_confirmed

    clients = _clients()
    seen: list[Any] = []
    response = {
        "evaluatorArn": _EVALUATOR_ARN,
        "evaluatorId": _EVALUATOR_ID,
        "createdAt": "2026-11-05T10:00:00Z",
        "status": "CREATING",
    }
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        stub.add_response("create_evaluator", response, expected_params=_CREATE_EVALUATOR)
        got = send_confirmed(
            clients,
            "bedrock-agentcore-control",
            "CreateEvaluator",
            _CREATE_EVALUATOR,
            summary="creates the evaluator calibrated_agent_evaluation_live_refunds",
            confirm=lambda preview: seen.append(preview) or True,
        )
        stub.assert_no_pending_responses()

    assert got["evaluatorId"] == _EVALUATOR_ID
    (preview,) = seen
    assert preview.request == _CREATE_EVALUATOR
    assert "CreateEvaluator" in preview.describe()
    assert "calibrated_agent_evaluation_live_refunds" in preview.describe()


def test_a_failed_call_after_a_created_resource_names_the_call_and_lists_what_was_created() -> None:
    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.online import call_aws

    clients = _clients()
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        stub.add_client_error(
            "create_online_evaluation_config",
            service_error_code="ValidationException",
            service_message="the evaluation role cannot be assumed",
        )
        with pytest.raises(AgentCoreError) as refused:
            call_aws(
                clients,
                "bedrock-agentcore-control",
                "CreateOnlineEvaluationConfig",
                _CREATE_ONLINE_CONFIG,
                created=(f"evaluator {_EVALUATOR_ID}",),
            )

    message = str(refused.value)
    assert "CreateOnlineEvaluationConfig failed with ValidationException" in message
    assert f"evaluator {_EVALUATOR_ID}" in message
    assert "still exists" in message


def test_a_call_that_was_never_sent_says_so() -> None:
    from botocore.exceptions import NoCredentialsError
    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.online import call_aws

    clients = _clients()
    client = clients.client("bedrock-agentcore-control")

    def no_credentials(**_: Any) -> None:
        raise NoCredentialsError()

    with Stubber(client), pytest.MonkeyPatch.context() as patch:
        patch.setattr(client, "get_evaluator", no_credentials)
        with pytest.raises(AgentCoreError, match="GetEvaluator was not sent") as refused:
            call_aws(clients, "bedrock-agentcore-control", "GetEvaluator", {"evaluatorId": "x"})

    assert "created" not in str(refused.value)


def _raising(client: Any, method: str, error: BaseException, patch: pytest.MonkeyPatch) -> None:
    def fail(**_: Any) -> None:
        raise error

    patch.setattr(client, method, fail)


def test_a_call_that_timed_out_may_have_taken_effect() -> None:
    from botocore.exceptions import ReadTimeoutError

    from calibrated_agent_evaluation.agentcore.online import call_aws

    clients = _clients()
    client = clients.client("bedrock-agentcore-control")
    with pytest.MonkeyPatch.context() as patch:
        _raising(client, "create_evaluator", ReadTimeoutError(endpoint_url="https://x"), patch)
        with pytest.raises(AgentCoreError, match="may have taken effect") as refused:
            call_aws(clients, "bedrock-agentcore-control", "CreateEvaluator", _CREATE_EVALUATOR)

    assert "was not sent" not in str(refused.value)


def test_a_guard_refusal_is_raised_as_it_is_when_nothing_was_created() -> None:
    from calibrated_agent_evaluation.agentcore.errors import BilledCallBudgetExceeded
    from calibrated_agent_evaluation.agentcore.online import call_aws

    clients = _clients()
    refusal = BilledCallBudgetExceeded("the billed-call budget of 0 is spent")
    with pytest.MonkeyPatch.context() as patch:
        _raising(clients.client("bedrock-agentcore-control"), "create_evaluator", refusal, patch)
        with pytest.raises(BilledCallBudgetExceeded) as refused:
            call_aws(clients, "bedrock-agentcore-control", "CreateEvaluator", _CREATE_EVALUATOR)

    assert refused.value is refusal


def test_a_guard_refusal_after_resources_were_created_lists_them_all() -> None:
    from calibrated_agent_evaluation.agentcore.errors import BilledCallBudgetExceeded
    from calibrated_agent_evaluation.agentcore.online import call_aws

    clients = _clients()
    refusal = BilledCallBudgetExceeded("the billed-call budget of 1 is spent")
    created = (
        f"evaluator {_EVALUATOR_ID}",
        "evaluator calibrated_agent_evaluation_live_second-AbCdEf1234",
    )
    with pytest.MonkeyPatch.context() as patch:
        _raising(
            clients.client("bedrock-agentcore-control"),
            "create_online_evaluation_config",
            refusal,
            patch,
        )
        with pytest.raises(AgentCoreError) as refused:
            call_aws(
                clients,
                "bedrock-agentcore-control",
                "CreateOnlineEvaluationConfig",
                _CREATE_ONLINE_CONFIG,
                created=created,
            )

    message = str(refused.value)
    assert "BilledCallBudgetExceeded" in message
    assert "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS" in message
    assert "they still exist" in message
    assert all(resource in message for resource in created)
    assert refused.value.__cause__ is refusal


def test_the_not_sent_errors_match_the_runtime_executors() -> None:
    pytest.importorskip("boto3")
    from calibrated_agent_evaluation.agentcore.online import _not_sent_errors
    from calibrated_agent_evaluation.agentcore.runtime import NOT_SENT_ERRORS

    assert _not_sent_errors() == NOT_SENT_ERRORS


# --- attaching certified judges to an online evaluation config ----------------------------------

_RUNTIME_ID = "calibrated_agent_evaluation_orderbot-AbCdEf1234"
_RUNTIME_ARN = f"arn:aws:bedrock-agentcore:us-east-1:{_ACCOUNT}:runtime/{_RUNTIME_ID}"
_ROLE = f"arn:aws:iam::{_ACCOUNT}:role/calibrated-agent-evaluation-live-agentcore-evaluation"
_CONFIG_ID = "calibrated_agent_evaluation_live_attach-AbCdEf1234"
_CONFIG_ARN = (
    f"arn:aws:bedrock-agentcore:us-east-1:{_ACCOUNT}:online-evaluation-config/{_CONFIG_ID}"
)
_WHEN = "2026-11-05T16:19:46+00:00"


def _evaluator_doc(evaluator_type: str = "Custom") -> dict[str, Any]:
    return {
        "evaluatorArn": _EVALUATOR_ARN,
        "evaluatorId": _EVALUATOR_ID,
        "evaluatorName": "calibrated_agent_evaluation_live_refunds",
        "evaluatorType": evaluator_type,
        "level": "TRACE",
        "status": "ACTIVE",
        "createdAt": _WHEN,
        "updatedAt": _WHEN,
        "evaluatorConfig": _CREATE_EVALUATOR["evaluatorConfig"],
    }


def _online_judge(
    *, calibrated: bool = True, battery: str | None = "CLEAR", result: str = "CERTIFIED"
) -> Any:
    from calibrated_agent_evaluation.agentcore.judge import judge_from_evaluator
    from calibrated_agent_evaluation.agentcore.online import OnlineJudge
    from cli_store_helpers import certified_calibration, clear_battery

    judge = judge_from_evaluator("refunds", _evaluator_doc(), level="workflow")
    pin = {
        "model": judge.model,
        "prompt_version": judge.prompt_version,
        "fingerprint": judge.fingerprint,
    }
    calib = (
        certified_calibration("refunds", calib_id="calib-r", result=result, **pin)
        if calibrated
        else None
    )
    record = clear_battery("refunds", verdict=battery, **pin) if battery else None
    return OnlineJudge(judge=judge, calib=calib, battery=record)


def _session_logs(
    online: Any,
    *,
    bias: bool = True,
    calibration: bool = True,
    gold_source: str | None = "synthetic-rule",
) -> list[Any]:
    from calibrated_agent_evaluation.agentcore.online_record import AgentCoreSessionLog

    judge = online.judge
    logs = []
    if calibration:
        logs.append(
            AgentCoreSessionLog(
                run_kind="calibration",
                run_id=online.calib.calib_id,
                judge_id=judge.judge_id,
                evaluator_fingerprint=judge.fingerprint,
                session_ids=("c1", "c2"),
                gold_source=gold_source,
            )
        )
    if bias and online.battery is not None:
        logs.append(
            AgentCoreSessionLog(
                run_kind="bias",
                run_id=online.battery.battery_id,
                judge_id=judge.judge_id,
                evaluator_fingerprint=judge.fingerprint,
                session_ids=("b1", "c2"),
            )
        )
    return logs


@pytest.fixture
def _rates(monkeypatch: pytest.MonkeyPatch) -> None:
    from decimal import Decimal

    from calibrated_agent_evaluation.agentcore.pricing import EvaluationRates

    rates = EvaluationRates(region="us-east-1", custom_per_evaluation=Decimal("0.0015"))
    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.pricing.lookup_evaluation_rates", lambda _c: rates
    )


def _runtime(stub: Any) -> None:
    stub.add_response(
        "get_agent_runtime",
        {
            "agentRuntimeArn": _RUNTIME_ARN,
            "agentRuntimeName": "calibrated_agent_evaluation_orderbot",
            "agentRuntimeId": _RUNTIME_ID,
            "agentRuntimeVersion": "1",
            "createdAt": _WHEN,
            "lastUpdatedAt": _WHEN,
            "roleArn": _ROLE,
            "networkConfiguration": {"networkMode": "PUBLIC"},
            "lifecycleConfiguration": {"idleRuntimeSessionTimeout": 900, "maxLifetime": 28800},
            "status": "READY",
        },
        {"agentRuntimeId": _RUNTIME_ID},
    )


def _listed(stub: Any, *pages: list[str]) -> None:
    for index, names in enumerate(pages):
        response: dict[str, Any] = {
            "onlineEvaluationConfigs": [
                {
                    "onlineEvaluationConfigArn": f"{_CONFIG_ARN.rsplit('/', 1)[0]}/{n}-AbCdEf1234",
                    "onlineEvaluationConfigId": f"{n}-AbCdEf1234",
                    "onlineEvaluationConfigName": n,
                    "status": "ACTIVE",
                    "executionStatus": "ENABLED",
                    "createdAt": _WHEN,
                    "updatedAt": _WHEN,
                }
                for n in names
            ]
        }
        expected: dict[str, Any] = {}
        if index:
            expected["nextToken"] = f"page-{index}"
        if index + 1 < len(pages):
            response["nextToken"] = f"page-{index + 1}"
        stub.add_response("list_online_evaluation_configs", response, expected)


def _status(stub: Any, status: str, reason: str | None = None) -> None:
    response: dict[str, Any] = {
        "onlineEvaluationConfigArn": _CONFIG_ARN,
        "onlineEvaluationConfigId": _CONFIG_ID,
        "onlineEvaluationConfigName": "calibrated_agent_evaluation_live_attach",
        "rule": {"samplingConfig": {"samplingPercentage": 100.0}},
        "dataSourceConfig": {
            "cloudWatchLogs": {"serviceNames": ["calibrated_agent_evaluation_orderbot.DEFAULT"]}
        },
        "evaluators": [{"evaluatorId": _EVALUATOR_ID}],
        "evaluationExecutionRoleArn": _ROLE,
        "status": status,
        "executionStatus": "ENABLED",
        "createdAt": _WHEN,
        "updatedAt": _WHEN,
    }
    if reason:
        response["failureReason"] = reason
    stub.add_response(
        "get_online_evaluation_config", response, {"onlineEvaluationConfigId": _CONFIG_ID}
    )


def _create(
    stub: Any,
    *,
    endpoint: str = "DEFAULT",
    filters: list[Any] | None = None,
    evaluators: tuple[str, ...] = (_EVALUATOR_ID,),
) -> None:
    from botocore.stub import ANY

    rule: dict[str, Any] = {
        "samplingConfig": {"samplingPercentage": 100.0},
        "sessionConfig": {"sessionTimeoutMinutes": 15},
    }
    if filters:
        rule["filters"] = filters
    stub.add_response(
        "create_online_evaluation_config",
        {
            "onlineEvaluationConfigArn": _CONFIG_ARN,
            "onlineEvaluationConfigId": _CONFIG_ID,
            "createdAt": _WHEN,
            "status": "CREATING",
            "executionStatus": "ENABLED",
        },
        {
            "onlineEvaluationConfigName": "calibrated_agent_evaluation_live_attach",
            "rule": rule,
            "dataSourceConfig": {
                "cloudWatchLogs": {
                    "serviceNames": [f"calibrated_agent_evaluation_orderbot.{endpoint}"],
                    "logGroupNames": [f"/aws/bedrock-agentcore/runtimes/{_RUNTIME_ID}-{endpoint}"],
                }
            },
            "evaluators": [{"evaluatorId": e} for e in evaluators],
            "evaluationExecutionRoleArn": _ROLE,
            "enableOnCreate": True,
            "clientToken": ANY,
            "description": ANY,
        },
    )


def _attach(clients: Any, judges: list[Any], logs: list[Any], **over: Any) -> Any:
    from calibrated_agent_evaluation.agentcore.online import create_online_config

    shown: list[Any] = over.pop("shown", [])

    def confirm(preview: Any) -> bool:
        shown.append(preview)
        return True

    kwargs: dict[str, Any] = {
        "name": "calibrated_agent_evaluation_live_attach",
        "runtime_arn": _RUNTIME_ARN,
        "sampling_percentage": 100.0,
        "execution_role_arn": _ROLE,
        "session_logs": logs,
        "clients": clients,
        "confirm": confirm,
        "sleep": lambda _s: None,
    }
    kwargs.update(over)
    return create_online_config(judges, **kwargs)


@pytest.mark.usefixtures("_rates")
@pytest.mark.parametrize("endpoint", ["DEFAULT", "beta"])
def test_an_attached_config_records_the_judges_pins_and_the_sessions_to_skip(
    endpoint: str,
) -> None:
    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.online import OnlineFilter

    clients = _clients()
    online = _online_judge()
    typed = [
        {"key": "gen_ai.agent.name", "operator": "Equals", "value": {"stringValue": "orderbot"}}
    ]
    shown: list[Any] = []
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        _runtime(stub)
        _listed(stub, ["something_else"])
        _create(stub, endpoint=endpoint, filters=typed)
        _status(stub, "CREATING")
        _status(stub, "ACTIVE")
        record = _attach(
            clients,
            [online],
            _session_logs(online),
            endpoint=endpoint,
            filters=[OnlineFilter("gen_ai.agent.name", "Equals", "orderbot")],
            shown=shown,
        )
        stub.assert_no_pending_responses()

    (preview,) = shown
    text = preview.describe()
    assert f"calibrated_agent_evaluation_orderbot.{endpoint}" in text
    assert f"/aws/bedrock-agentcore/runtimes/{_RUNTIME_ID}-{endpoint}" in text
    assert "$0.0015" in text
    assert record.config_id == _CONFIG_ID
    assert record.runtime_service_name == f"calibrated_agent_evaluation_orderbot.{endpoint}"
    (attached,) = record.evaluators
    assert attached.fingerprint == online.judge.fingerprint
    assert attached.calib_id == "calib-r"
    assert attached.battery_id == online.battery.battery_id
    assert record.excluded_session_ids == ("c1", "c2", "b1")
    assert record.sampling_percentage == 100.0
    assert record.session_timeout_minutes == 15


@pytest.mark.parametrize(
    ("make", "logs", "says"),
    [
        ({"calibrated": False}, {}, "a calibration record"),
        ({"result": "REFUSED"}, {}, "trust gates"),
        ({"battery": None}, {}, "a bias battery"),
        ({"battery": "BIASED"}, {}, "BIASED"),
        ({}, {"bias": False}, "session log of its bias run"),
        ({}, {"calibration": False}, "session log of its calibration run"),
    ],
    ids=[
        "no-calibration",
        "refused-calibration",
        "no-battery",
        "biased",
        "no-bias-session-log",
        "no-calibration-session-log",
    ],
)
def test_a_judge_not_ready_for_online_use_is_refused_with_no_aws_call(
    make: dict[str, Any], logs: dict[str, Any], says: str
) -> None:
    from botocore.stub import Stubber

    clients = _clients()
    online = _online_judge(**make)
    stored = _session_logs(online, **logs) if online.calib is not None else []
    with (
        Stubber(clients.client("bedrock-agentcore-control")),
        pytest.raises(AgentCoreNotCertified, match=says),
    ):
        _attach(clients, [online], stored)


def test_an_advisory_only_judge_is_refused_with_no_aws_call() -> None:
    from dataclasses import replace

    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.errors import AgentCoreAdvisoryOnly

    clients = _clients()
    online = _online_judge()
    builtin = replace(online, judge=replace(online.judge, evaluator_type="Builtin"))
    with Stubber(clients.client("bedrock-agentcore-control")), pytest.raises(AgentCoreAdvisoryOnly):
        _attach(clients, [builtin], _session_logs(online))


@pytest.mark.usefixtures("_rates")
def test_a_config_name_already_in_use_on_a_later_page_is_refused_before_any_write() -> None:
    from botocore.stub import Stubber

    clients = _clients()
    online = _online_judge()
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        _runtime(stub)
        _listed(stub, ["first"], ["calibrated_agent_evaluation_live_attach"])
        with pytest.raises(
            AgentCoreResourceExists, match="calibrated_agent_evaluation_live_attach"
        ):
            _attach(clients, [online], _session_logs(online))
        stub.assert_no_pending_responses()


def test_an_unknown_runtime_is_refused_naming_it_before_any_write() -> None:
    from botocore.stub import Stubber

    clients = _clients()
    online = _online_judge()
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        stub.add_client_error(
            "get_agent_runtime",
            service_error_code="ResourceNotFoundException",
            http_status_code=404,
        )
        with pytest.raises(AgentCoreRuntimeNotFound, match=_RUNTIME_ID):
            _attach(clients, [online], _session_logs(online))
        stub.assert_no_pending_responses()


@pytest.mark.usefixtures("_rates")
def test_a_config_that_fails_or_never_becomes_active_is_refused_naming_its_status() -> None:
    from botocore.stub import Stubber

    clients = _clients()
    online = _online_judge()
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        _runtime(stub)
        _listed(stub, [])
        _create(stub)
        _status(stub, "CREATE_FAILED", "the role cannot be assumed")
        with pytest.raises(
            AgentCoreResourceNotActive, match=f"{_CONFIG_ID} reported CREATE_FAILED: the role"
        ):
            _attach(clients, [online], _session_logs(online))

    ticks = iter([0.0, 400.0])
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        _runtime(stub)
        _listed(stub, [])
        _create(stub)
        _status(stub, "CREATING")
        with pytest.raises(
            AgentCoreResourceNotActive,
            match=f"{_CONFIG_ID} was still CREATING when the wait ran out",
        ):
            _attach(clients, [online], _session_logs(online), clock=lambda: next(ticks))


@pytest.mark.parametrize(
    ("over", "says"),
    [
        ({"sampling_percentage": 0.0}, "0.0"),
        ({"sampling_percentage": 150.0}, "150"),
        ({"session_timeout_minutes": 2000}, "2000"),
        ({"name": "1st_config"}, "1st_config"),
    ],
    ids=["sampling-0", "sampling-150", "timeout", "name"],
)
def test_a_value_agentcore_would_refuse_is_refused_before_any_aws_call(
    over: dict[str, Any], says: str
) -> None:
    from botocore.stub import Stubber

    clients = _clients()
    online = _online_judge()
    with (
        Stubber(clients.client("bedrock-agentcore-control")),
        pytest.raises(AgentCoreError, match=says),
    ):
        _attach(clients, [online], _session_logs(online), **over)


@pytest.mark.parametrize(
    ("filters", "says"),
    [
        ([("k", "Equals", "v")] * 6, "five"),
        ([("k", "Like", "v")], "Like"),
        ([("k", "Equals", float("nan"))], "nan"),
    ],
    ids=["too-many", "operator", "nan"],
)
def test_a_filter_agentcore_would_refuse_is_refused_before_any_aws_call(
    filters: list[tuple[str, str, Any]], says: str
) -> None:
    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.online import OnlineFilter

    clients = _clients()
    online = _online_judge()
    with (
        Stubber(clients.client("bedrock-agentcore-control")),
        pytest.raises(AgentCoreError, match=says),
    ):
        _attach(
            clients,
            [online],
            _session_logs(online),
            filters=[OnlineFilter(*f) for f in filters],
        )


def test_a_judge_whose_evaluator_is_not_trace_level_is_refused_with_no_aws_call() -> None:
    from dataclasses import replace

    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.errors import AgentCoreEvaluatorLevelUnsupported

    clients = _clients()
    online = _online_judge()
    session = replace(online, judge=replace(online.judge, evaluator_level="SESSION"))
    with (
        Stubber(clients.client("bedrock-agentcore-control")),
        pytest.raises(AgentCoreEvaluatorLevelUnsupported, match="SESSION"),
    ):
        _attach(clients, [session], _session_logs(online))


@pytest.mark.parametrize(
    ("over", "says"),
    [
        ({"runtime_arn": f"{_RUNTIME_ARN}/runtime-endpoint/DEFAULT"}, "not an AgentCore runtime"),
        ({"runtime_arn": _RUNTIME_ARN.replace("us-east-1", "us-west-2")}, "us-west-2"),
    ],
    ids=["endpoint-arn", "other-region"],
)
def test_a_runtime_arn_that_cannot_be_read_here_is_refused_with_no_aws_call(
    over: dict[str, Any], says: str
) -> None:
    from botocore.stub import Stubber

    clients = _clients()
    online = _online_judge()
    with (
        Stubber(clients.client("bedrock-agentcore-control")),
        pytest.raises(AgentCoreError, match=says),
    ):
        _attach(clients, [online], _session_logs(online), **over)


def test_the_same_judge_given_twice_is_refused_with_no_aws_call() -> None:
    from botocore.stub import Stubber

    clients = _clients()
    online = _online_judge()
    with (
        Stubber(clients.client("bedrock-agentcore-control")),
        pytest.raises(AgentCoreError, match="more than once"),
    ):
        _attach(clients, [online, online], _session_logs(online))


@pytest.mark.parametrize(
    ("bad", "says"),
    [
        (("bad key!", "Equals", "v"), "bad key!"),
        (("k", "Equals", ""), "1 to 1024"),
        (("k", "Equals", "x" * 1025), "1 to 1024"),
    ],
    ids=["key", "empty-value", "long-value"],
)
def test_a_filter_key_or_string_agentcore_would_refuse_is_refused(
    bad: tuple[str, str, Any], says: str
) -> None:
    from calibrated_agent_evaluation.agentcore.online import OnlineFilter, check_config_request

    with pytest.raises(AgentCoreError, match=says):
        check_config_request(
            "calibrated_agent_evaluation_live_attach", 100.0, 15, [OnlineFilter(*bad)]
        )


@pytest.mark.usefixtures("_rates")
def test_two_judges_skip_the_union_of_their_sessions_once_each() -> None:
    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.judge import judge_from_evaluator
    from calibrated_agent_evaluation.agentcore.online import OnlineJudge
    from calibrated_agent_evaluation.agentcore.online_record import AgentCoreSessionLog
    from cli_store_helpers import certified_calibration, clear_battery

    first = _online_judge()
    other_id = "calibrated_agent_evaluation_live_other-ZyXwVu9876"
    doc = {
        **_evaluator_doc(),
        "evaluatorId": other_id,
        "evaluatorArn": _EVALUATOR_ARN.replace(_EVALUATOR_ID, other_id),
    }
    judge = judge_from_evaluator("other", doc, level="workflow")
    pin = {
        "model": judge.model,
        "prompt_version": judge.prompt_version,
        "fingerprint": judge.fingerprint,
    }
    second = OnlineJudge(
        judge=judge,
        calib=certified_calibration("other", calib_id="calib-o", **pin),
        battery=clear_battery("other", **pin).model_copy(update={"battery_id": "bias-other"}),
    )
    logs = [
        *_session_logs(first),
        AgentCoreSessionLog(
            run_kind="calibration",
            run_id="calib-o",
            judge_id="other",
            evaluator_fingerprint=judge.fingerprint,
            session_ids=("c2", "o1"),
        ),
        AgentCoreSessionLog(
            run_kind="bias",
            run_id="bias-other",
            judge_id="other",
            evaluator_fingerprint=judge.fingerprint,
            session_ids=("b1", "o2"),
        ),
    ]
    clients = _clients()
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        _runtime(stub)
        _listed(stub, [])
        _create(stub, evaluators=(_EVALUATOR_ID, other_id))
        _status(stub, "ACTIVE")
        record = _attach(clients, [first, second], logs)

    assert [e.evaluator_id for e in record.evaluators] == [_EVALUATOR_ID, other_id]
    assert record.excluded_session_ids == ("c1", "c2", "b1", "o1", "o2")


def test_more_than_twenty_five_judges_are_refused_with_no_aws_call() -> None:
    from dataclasses import replace

    from botocore.stub import Stubber

    clients = _clients()
    online = _online_judge()
    many = [
        replace(
            online,
            judge=replace(
                online.judge, evaluator_id=f"calibrated_agent_evaluation_live_{i:02d}-AbCdEf1234"
            ),
        )
        for i in range(26)
    ]
    with (
        Stubber(clients.client("bedrock-agentcore-control")),
        pytest.raises(AgentCoreError, match="at most 25"),
    ):
        _attach(clients, many, _session_logs(online))


# --- whether each attached evaluator is still the one that was certified ------------------------


def _attached_record(online: Any) -> Any:
    from calibrated_agent_evaluation.agentcore.online_record import (
        AttachedEvaluator,
        OnlineEvaluationRecord,
    )

    judge = online.judge
    return OnlineEvaluationRecord(
        config_id=_CONFIG_ID,
        config_name="calibrated_agent_evaluation_live_attach",
        runtime_service_name="calibrated_agent_evaluation_orderbot.DEFAULT",
        evaluators=(
            AttachedEvaluator(
                evaluator_id=judge.evaluator_id,
                evaluator_arn=judge.evaluator_arn,
                updated_at=judge.updated_at,
                fingerprint=judge.fingerprint,
                calib_id=online.calib.calib_id,
                battery_id=online.battery.battery_id,
            ),
        ),
        excluded_session_ids=("c1",),
        sampling_percentage=100.0,
        session_timeout_minutes=15,
        created_at=_WHEN,
    )


def _derived_docs() -> tuple[dict[str, Any], dict[str, Any]]:
    """A derived evaluator running Helpfulness on its own model, and that built-in base."""
    base = {
        "evaluatorArn": "arn:aws:bedrock-agentcore:::evaluator/Builtin.Helpfulness",
        "evaluatorId": "Builtin.Helpfulness",
        "evaluatorName": "Builtin.Helpfulness",
        "evaluatorType": "Builtin",
        "level": "TRACE",
        "status": "ACTIVE",
        "createdAt": _WHEN,
        "updatedAt": _WHEN,
        "evaluatorConfig": _CREATE_EVALUATOR["evaluatorConfig"],
    }
    derived = {
        **_evaluator_doc("CustomDerived"),
        "evaluatorConfig": {
            "derived": {
                "baseEvaluatorId": "Builtin.Helpfulness",
                "modelConfig": {
                    "bedrockEvaluatorModelConfig": {
                        "modelId": "global.anthropic.claude-haiku-4-5-20251001-v1:0"
                    }
                },
            }
        },
    }
    return derived, base


def _check(document: dict[str, Any] | None, *, error: str | None = None) -> Any:
    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.online import check_attached

    clients = _clients()
    record = _attached_record(_online_judge())
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        if error is not None:
            stub.add_client_error("get_evaluator", service_error_code=error, http_status_code=400)
        else:
            stub.add_response("get_evaluator", document, {"evaluatorId": _EVALUATOR_ID})
        statuses = check_attached(record, clients=clients)
        stub.assert_no_pending_responses()
    return statuses


def test_an_attached_evaluator_whose_only_change_is_the_lock_is_still_certified() -> None:
    # Attaching locks the evaluator and moves its updatedAt; nothing that decides verdicts moved.
    locked = {**_evaluator_doc(), "updatedAt": "2026-11-06T10:00:00+00:00"}
    locked["lockedForModification"] = True

    (status,) = _check(locked)

    assert status.evaluator_id == _EVALUATOR_ID
    assert status.state == "certified"
    assert status.certified


def test_an_attached_evaluator_whose_instructions_changed_is_reported_uncertified() -> None:
    edited = _evaluator_doc()
    llm = dict(edited["evaluatorConfig"]["llmAsAJudge"])
    llm["instructions"] = "Judge the refund strictly. {context} {assistant_turn}"
    edited["evaluatorConfig"] = {"llmAsAJudge": llm}

    (status,) = _check(edited)

    assert status.state == "changed"
    assert not status.certified
    assert _EVALUATOR_ID in status.reason
    assert "no longer applies" in status.reason


def test_an_attached_evaluator_that_no_longer_exists_is_reported_missing() -> None:
    (status,) = _check(None, error="ResourceNotFoundException")

    assert status.state == "missing"
    assert not status.certified
    assert "no longer exists" in status.reason


def test_an_evaluator_that_cannot_be_read_is_raised_not_reported_missing() -> None:
    # Only a not-found answer means the evaluator is gone; a denied read says nothing about it.
    with pytest.raises(AgentCoreError, match="GetEvaluator failed with AccessDeniedException"):
        _check(None, error="AccessDeniedException")


def test_a_derived_evaluator_is_checked_with_its_base_and_missing_when_its_base_is_gone() -> None:
    from dataclasses import replace

    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.judge import judge_from_evaluator
    from calibrated_agent_evaluation.agentcore.online import check_attached

    derived, base = _derived_docs()
    judge = judge_from_evaluator(_EVALUATOR_ID, derived, base_evaluator=base)
    online = _online_judge()
    record = _attached_record(replace(online, judge=judge))
    clients = _clients()
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        stub.add_response("get_evaluator", derived, {"evaluatorId": _EVALUATOR_ID})
        stub.add_response("get_evaluator", base, {"evaluatorId": "Builtin.Helpfulness"})
        stub.add_response("get_evaluator", derived, {"evaluatorId": _EVALUATOR_ID})
        stub.add_client_error(
            "get_evaluator", service_error_code="ResourceNotFoundException", http_status_code=404
        )
        (present,) = check_attached(record, clients=clients)
        (gone,) = check_attached(record, clients=clients)
        stub.assert_no_pending_responses()

    assert present.state == "certified"
    assert gone.state == "missing"
    assert "Builtin.Helpfulness" in gone.reason


def test_an_evaluator_that_can_no_longer_be_a_judge_is_reported_changed() -> None:
    # Moved to session level since it was certified: it is no longer the judge that was measured.
    (status,) = _check({**_evaluator_doc(), "level": "SESSION"})

    assert status.state == "changed"
    assert "SESSION" in status.reason


def test_an_evaluator_read_without_its_instructions_cannot_be_checked() -> None:
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreEvaluatorUnreadable

    stripped = _evaluator_doc()
    stripped["evaluatorConfig"] = {
        "llmAsAJudge": {
            k: v
            for k, v in _CREATE_EVALUATOR["evaluatorConfig"]["llmAsAJudge"].items()
            if k != "instructions"
        }
    }
    with pytest.raises(AgentCoreEvaluatorUnreadable):
        _check_unvalidated(stripped)


def _check_unvalidated(document: dict[str, Any]) -> Any:
    """``check_attached`` against a ``GetEvaluator`` reply botocore's model would not let a stub
    return (it marks the instructions required), stripped after the call as a reply parses."""
    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.online import check_attached

    clients = _clients()
    client = clients.client("bedrock-agentcore-control")
    full = _evaluator_doc()

    def strip(parsed: dict[str, Any], **_: Any) -> None:
        parsed["evaluatorConfig"] = document["evaluatorConfig"]

    client.meta.events.register("after-call.bedrock-agentcore-control.GetEvaluator", strip)
    with Stubber(client) as stub:
        stub.add_response("get_evaluator", full, {"evaluatorId": _EVALUATOR_ID})
        return check_attached(_attached_record(_online_judge()), clients=clients)


def test_each_attached_evaluator_gets_its_own_status_in_the_records_order() -> None:
    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.online import check_attached

    record = _attached_record(_online_judge())
    other = record.evaluators[0].model_copy(
        update={"evaluator_id": "calibrated_agent_evaluation_live_other-AbCdEf1234"}
    )
    record = record.model_copy(update={"evaluators": (record.evaluators[0], other)})
    clients = _clients()
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        stub.add_response("get_evaluator", _evaluator_doc(), {"evaluatorId": _EVALUATOR_ID})
        stub.add_client_error(
            "get_evaluator", service_error_code="ResourceNotFoundException", http_status_code=404
        )
        statuses = check_attached(record, clients=clients)

    assert [(s.evaluator_id, s.state) for s in statuses] == [
        (_EVALUATOR_ID, "certified"),
        ("calibrated_agent_evaluation_live_other-AbCdEf1234", "missing"),
    ]


def test_a_failed_aws_call_is_told_apart_from_another_error() -> None:
    from botocore.exceptions import ClientError, ReadTimeoutError

    from calibrated_agent_evaluation.agentcore.online import aws_call_failed

    def wrapped(cause: BaseException) -> AgentCoreError:
        exc = AgentCoreError("the call failed")
        exc.__cause__ = cause
        return exc

    throttled = ClientError({"Error": {"Code": "ThrottlingException"}}, "GetEvaluator")
    timed_out = ReadTimeoutError(endpoint_url="https://example.invalid")

    assert aws_call_failed(wrapped(throttled))
    assert aws_call_failed(wrapped(timed_out))
    assert not aws_call_failed(wrapped(ValueError("not an AWS call")))
    assert not aws_call_failed(AgentCoreError("no cause"))
