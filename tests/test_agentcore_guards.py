# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Call guards refuse disallowed and over-budget AWS calls before anything is sent."""

from __future__ import annotations

import io
from typing import Any, ClassVar

import pytest

pytest.importorskip("boto3")

import boto3
from botocore.response import StreamingBody
from botocore.stub import Stubber

from calibrated_agent_evaluation.agentcore import AwsClients, BilledCallBudget, ReadOnlyGuard
from calibrated_agent_evaluation.agentcore.doctor import (
    DEFAULT_CHECKS,
    allowed_operations,
    run_checks,
)
from calibrated_agent_evaluation.agentcore.errors import BilledCallBudgetExceeded, ReadOnlyViolation
from calibrated_agent_evaluation.agentcore.guards import BILLED_OPERATIONS

_EVALUATE_PARAMS: dict[str, Any] = {
    "evaluatorId": "Builtin.Helpfulness",
    "evaluationInput": {"sessionSpans": [{}]},
}


def _session() -> Any:
    return boto3.Session(
        region_name="us-west-2", aws_access_key_id="testing", aws_secret_access_key="testing"
    )


def _evaluate_response() -> dict[str, Any]:
    return {"evaluationResults": []}


def _stubbed_data_plane(handlers: list[Any]) -> tuple[Any, Stubber]:
    client = AwsClients(_session(), handlers=handlers).client("bedrock-agentcore")
    return client, Stubber(client)


def test_billed_calls_within_the_budget_go_through_and_the_next_is_refused_unsent() -> None:
    budget = BilledCallBudget(limit=2)
    client, stub = _stubbed_data_plane([budget])
    for _ in range(3):
        stub.add_response("evaluate", _evaluate_response())

    with stub:
        client.evaluate(**_EVALUATE_PARAMS)
        client.evaluate(**_EVALUATE_PARAMS)
        with pytest.raises(BilledCallBudgetExceeded) as refused:
            client.evaluate(**_EVALUATE_PARAMS)

    assert budget.used == 2
    assert "budget of 2" in str(refused.value)
    assert "Evaluate" in str(refused.value)
    with pytest.raises(AssertionError):
        stub.assert_no_pending_responses()


def test_a_zero_budget_refuses_the_first_billed_call() -> None:
    client, stub = _stubbed_data_plane([BilledCallBudget(limit=0)])
    stub.add_response("evaluate", _evaluate_response())

    with stub, pytest.raises(BilledCallBudgetExceeded):
        client.evaluate(**_EVALUATE_PARAMS)


def test_without_a_limit_billed_calls_are_counted_but_never_refused() -> None:
    budget = BilledCallBudget()
    client, stub = _stubbed_data_plane([budget])
    for _ in range(50):
        stub.add_response("evaluate", _evaluate_response())

    with stub:
        for _ in range(50):
            client.evaluate(**_EVALUATE_PARAMS)

    assert budget.used == 50


def test_unbilled_calls_never_count_against_the_budget() -> None:
    budget = BilledCallBudget(limit=0)
    xray = AwsClients(_session(), handlers=[budget]).client("xray")
    stub = Stubber(xray)
    stub.add_response(
        "get_trace_segment_destination", {"Destination": "CloudWatchLogs", "Status": "ACTIVE"}
    )

    with stub:
        xray.get_trace_segment_destination()

    assert budget.used == 0


def test_a_negative_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        BilledCallBudget(limit=-1)


_DEFAULT_RULE: dict[str, Any] = {
    "Name": "Default",
    "Rule": {"Probabilistic": {"DesiredSamplingPercentage": 1.0}},
}


def _guarded_xray(*responses: tuple[str, dict[str, Any]]) -> tuple[Any, Stubber]:
    guard = ReadOnlyGuard({("xray", "GetIndexingRules")})
    xray = AwsClients(_session(), handlers=[guard]).client("xray")
    stub = Stubber(xray)
    for method, response in responses:
        stub.add_response(method, response)
    return xray, stub


def test_the_read_only_guard_lets_a_listed_call_through() -> None:
    xray, stub = _guarded_xray(("get_indexing_rules", {"IndexingRules": [_DEFAULT_RULE]}))

    with stub:
        assert xray.get_indexing_rules()["IndexingRules"][0]["Name"] == "Default"


@pytest.mark.parametrize(
    ("method", "params", "response", "refused"),
    [
        (
            "update_indexing_rule",
            {"Name": "Default", "Rule": {"Probabilistic": {"DesiredSamplingPercentage": 100.0}}},
            {"IndexingRule": _DEFAULT_RULE},
            "xray:UpdateIndexingRule",
        ),
        ("get_trace_summaries", {"StartTime": 0, "EndTime": 1}, {}, "xray:GetTraceSummaries"),
    ],
)
def test_the_read_only_guard_refuses_an_unlisted_call_before_it_is_sent(
    method: str, params: dict[str, Any], response: dict[str, Any], refused: str
) -> None:
    # The queued response would answer the call if the guard let it through.
    xray, stub = _guarded_xray((method, response))

    with stub, pytest.raises(ReadOnlyViolation, match=refused):
        getattr(xray, method)(**params)

    with pytest.raises(AssertionError):
        stub.assert_no_pending_responses()


def test_doctor_allows_exactly_its_declared_reads_and_no_billed_call() -> None:
    declared = {op for check in DEFAULT_CHECKS for op in check.operations}
    allowed = allowed_operations(DEFAULT_CHECKS)

    assert allowed == {*declared, ("sts", "GetCallerIdentity")}
    assert not allowed & BILLED_OPERATIONS
    assert all(op.startswith(("Get", "List", "Describe")) for _, op in allowed)


def test_a_check_that_trips_the_guard_makes_the_run_raise_instead_of_reporting_unknown() -> None:
    from calibrated_agent_evaluation.agentcore.doctor import DoctorTarget

    class Rogue:
        name: ClassVar[str] = "rogue"
        operations: ClassVar[frozenset[tuple[str, str]]] = frozenset({("xray", "GetIndexingRules")})

        def run(self, clients: AwsClients, target: DoctorTarget) -> Any:
            return clients.client("xray").update_indexing_rule(  # pyright: ignore[reportAttributeAccessIssue]
                Name="Default", Rule={"Probabilistic": {"DesiredSamplingPercentage": 100.0}}
            )

    guard = ReadOnlyGuard(allowed_operations([Rogue()]))
    clients = AwsClients(_session(), handlers=[guard])
    sts = clients.client("sts")
    stub = Stubber(sts)
    stub.add_response(
        "get_caller_identity",
        {"UserId": "AIDA", "Account": "000000000000", "Arn": "arn:aws:iam::000000000000:user/t"},
    )

    xray_stub = Stubber(clients.client("xray"))
    xray_stub.add_response("update_indexing_rule", {"IndexingRule": _DEFAULT_RULE})

    with stub, xray_stub, pytest.raises(ReadOnlyViolation, match="UpdateIndexingRule"):
        run_checks([Rogue()], clients, DoctorTarget(region="us-west-2"))


_INVOKE = ("bedrock-agentcore", "InvokeAgentRuntime")
_TELEMETRY_QUERY = ("cloudwatchomni", "StartTelemetryQuery")
_ANSWER = b'{"response": "hi"}'
_QUERY_SESSION = "calibrated-agent-evaluation-guard-test-query"


def _invoke(client: Any) -> dict[str, Any]:
    return client.invoke_agent_runtime(
        agentRuntimeArn="arn:aws:bedrock-agentcore:us-west-2:000000000000:runtime/guarded-test",
        runtimeSessionId="calibrated-agent-evaluation-0123456789abcdef0123456789abcdef",
        payload=b'{"prompt": "hi"}',
    )


def _queue_invoke(stub: Stubber) -> None:
    stub.add_response(
        "invoke_agent_runtime",
        {
            "contentType": "application/json",
            "statusCode": 200,
            "response": StreamingBody(io.BytesIO(_ANSWER), len(_ANSWER)),
        },
    )


def _query(client: Any) -> dict[str, Any]:
    return client.start_telemetry_query(queryString="fields @message", sessionId=_QUERY_SESSION)


def _queue_query(stub: Stubber) -> None:
    stub.add_response(
        "start_telemetry_query", {"queryId": "guard-test-query", "sessionId": _QUERY_SESSION}
    )


def test_both_guards_carry_the_refused_operation() -> None:
    data_plane, data_stub = _stubbed_data_plane([BilledCallBudget(limit=0)])
    _queue_invoke(data_stub)
    guard = ReadOnlyGuard({("cloudwatchomni", "ListDomains")})
    omni = AwsClients(_session(), handlers=[guard]).client("cloudwatchomni")
    omni_stub = Stubber(omni)
    _queue_query(omni_stub)

    with data_stub, pytest.raises(BilledCallBudgetExceeded) as over_budget:
        _invoke(data_plane)
    with omni_stub, pytest.raises(ReadOnlyViolation) as not_allowed:
        _query(omni)

    assert over_budget.value.operation == _INVOKE
    assert not_allowed.value.operation == _TELEMETRY_QUERY
    # The pair is the one each guard's message already names.
    assert str(over_budget.value).startswith("refused bedrock-agentcore:InvokeAgentRuntime: ")
    assert str(not_allowed.value).startswith("refused cloudwatchomni:StartTelemetryQuery: ")


def test_an_allowed_call_passes_both_guards_untouched() -> None:
    budget = BilledCallBudget(limit=2)
    guard = ReadOnlyGuard({_INVOKE, _TELEMETRY_QUERY})
    clients = AwsClients(_session(), handlers=[guard, budget])
    data_plane, omni = clients.client("bedrock-agentcore"), clients.client("cloudwatchomni")
    data_stub, omni_stub = Stubber(data_plane), Stubber(omni)
    _queue_invoke(data_stub)
    _queue_query(omni_stub)

    with data_stub, omni_stub:
        answer = _invoke(data_plane)
        query = _query(omni)

    assert answer["contentType"] == "application/json"
    assert answer["response"].read() == _ANSWER
    assert query == {"queryId": "guard-test-query", "sessionId": _QUERY_SESSION}
    assert budget.used == 2
    data_stub.assert_no_pending_responses()
    omni_stub.assert_no_pending_responses()
