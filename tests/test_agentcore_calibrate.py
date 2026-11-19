# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrating an AgentCore judge: its locked test, its refusals, and the gold's source."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

pytest.importorskip("boto3")

import boto3
from botocore.config import Config
from botocore.stub import Stubber

from calibrated_agent_evaluation.agentcore.calibrate import agentcore_locked_test_set, gold_source
from calibrated_agent_evaluation.agentcore.errors import AgentCoreAdvisoryOnly, AgentCoreError
from calibrated_agent_evaluation.agentcore.evaluate_payload import aws_trace_id
from calibrated_agent_evaluation.agentcore.judge import AgentCoreJudge, ScaleValue
from calibrated_agent_evaluation.agentcore.pricing import EvaluationRates
from calibrated_agent_evaluation.agentcore.session import AwsClients
from calibrated_agent_evaluation.calibrate import InMemoryLockedTestLedger
from calibrated_agent_evaluation.calibrate.engine import certify
from calibrated_agent_evaluation.calibrate.errors import (
    DuplicateGoldError,
    UnpairedObservationError,
    UnratifiedGoldError,
)
from calibrated_agent_evaluation.schema.types import Label, Span, ToolCall, Trace

_ACCOUNT = "111122223333"
_ID = "orderbot_refund_policy_v2-AbCdEf1234"
_ARN = f"arn:aws:bedrock-agentcore:us-east-1:{_ACCOUNT}:evaluator/{_ID}"
_SCALE = (ScaleValue(0.0, "FAIL"), ScaleValue(1.0, "PASS"))


@pytest.fixture(autouse=True)
def _published_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    rates = EvaluationRates(
        region="us-east-1",
        builtin_input_per_million=Decimal("2.40"),
        builtin_output_per_million=Decimal("12.00"),
        custom_per_evaluation=Decimal("0.0015"),
    )
    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.evaluate.lookup_evaluation_rates", lambda _c: rates
    )


def _clients() -> AwsClients:
    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    return AwsClients(session).with_config(Config(retries={"total_max_attempts": 1}))


def _judge(evaluator_type: str = "Custom") -> AgentCoreJudge:
    return AgentCoreJudge(
        judge_id="orderbot-refund-policy",
        evaluator_id=_ID if evaluator_type == "Custom" else "Builtin.Correctness",
        evaluator_arn=_ARN,
        evaluator_type=evaluator_type,
        updated_at="2026-11-05T10:19:46.771000-06:00",
        config_digest="0123456789abcdef",
        pinned_model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        rating_scale=_SCALE,
        evaluator_level="TRACE",
    )


def _trace(trace_id: str) -> Trace:
    span = Span(
        span_id=f"{trace_id}/agent",
        trace_id=trace_id,
        agent="orderbot",
        role="specialist",
        parent=None,
        seq=0,
        status="completed",
        input={"message": "refund please"},
        output={"answer": "refunded"},
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00",
    )
    return Trace(
        trace_id=trace_id,
        frame="discovery",
        ownership="owned",
        input={"message": "refund please"},
        output={"answer": "refunded"},
        taxonomy_cell={},
        spans=[span],
    )


def _gold(trace_id: str, outcome: str, **over: Any) -> Label:
    fields: dict[str, Any] = {
        "trace_id": trace_id,
        "span_id": None,
        "kind": "human",
        "outcome": outcome,
        "rater_id": "synthetic-rule-oracle",
        "failure_mode": "trusted_broken_policy_tool" if outcome == "FAIL" else None,
        "critique": "rule",
    }
    fields.update(over)
    return Label(**fields)


def _result(trace_id: str, value: float) -> dict[str, Any]:
    derived = aws_trace_id(trace_id)
    return {
        "evaluatorArn": _ARN,
        "evaluatorId": _ID,
        "evaluatorName": "orderbot_refund_policy_v2",
        "explanation": "graded",
        "value": value,
        "label": "PASS" if value else "FAIL",
        "context": {"spanContext": {"sessionId": derived, "traceId": derived}},
    }


def _scored(stub: Stubber, *values: tuple[str, float]) -> None:
    for trace_id, value in values:
        stub.add_response("evaluate", {"evaluationResults": [_result(trace_id, value)]})


def test_a_locked_test_scored_by_agentcore_certifies_into_a_record_pinned_to_the_evaluator() -> (
    None
):
    clients = _clients()
    gold = [(_trace(f"f-{i}"), _gold(f"f-{i}", "FAIL")) for i in range(40)]
    gold += [(_trace(f"p-{i}"), _gold(f"p-{i}", "PASS")) for i in range(40)]
    with Stubber(clients.client("bedrock-agentcore")) as stub:
        _scored(stub, *((t.trace_id, 0.0 if g.outcome == "FAIL" else 1.0) for t, g in gold))
        run = agentcore_locked_test_set(
            _judge(), gold, clients=clients, confirm=lambda _: True, set_id="gold-test"
        )
        stub.assert_no_pending_responses()

    record = certify(_judge(), run.calibration_set, ledger=InMemoryLockedTestLedger())
    assert record.result == "CERTIFIED"
    assert record.model == _ARN
    # The record is pinned to the evaluator's configuration digest, not its updatedAt.
    assert record.prompt_version == "0123456789abcdef"
    assert record.judge_fingerprint == _judge().fingerprint
    assert run.session_ids == tuple(aws_trace_id(t.trace_id) for t, _ in gold)


def test_a_judge_that_misses_most_failures_is_refused() -> None:
    clients = _clients()
    gold = [(_trace(f"f-{i}"), _gold(f"f-{i}", "FAIL")) for i in range(40)]
    gold += [(_trace(f"p-{i}"), _gold(f"p-{i}", "PASS")) for i in range(40)]
    with Stubber(clients.client("bedrock-agentcore")) as stub:
        _scored(stub, *((t.trace_id, 1.0) for t, _ in gold))  # it passes everything
        run = agentcore_locked_test_set(
            _judge(), gold, clients=clients, confirm=lambda _: True, set_id="gold-test"
        )

    assert certify(_judge(), run.calibration_set, ledger=InMemoryLockedTestLedger()).result == (
        "REFUSED"
    )


def test_a_builtin_evaluator_is_refused_before_any_billed_call() -> None:
    clients = _clients()
    with Stubber(clients.client("bedrock-agentcore")):
        with pytest.raises(AgentCoreAdvisoryOnly, match="Builtin"):
            agentcore_locked_test_set(
                _judge("Builtin"),
                [(_trace("f-1"), _gold("f-1", "FAIL"))],
                clients=clients,
                confirm=lambda _: True,
                set_id="gold-test",
            )


@pytest.mark.parametrize(
    ("gold", "error"),
    [
        pytest.param(
            [(_trace("f-1"), _gold("f-1", "FAIL", kind="llm-judge"))],
            UnratifiedGoldError,
            id="not-human",
        ),
        pytest.param(
            [(_trace("f-1"), _gold("f-1", "FAIL")), (_trace("f-1"), _gold("f-1", "PASS"))],
            DuplicateGoldError,
            id="graded-twice",
        ),
        pytest.param([], UnpairedObservationError, id="empty"),
        pytest.param(
            [(_trace("f-1"), _gold("f-1", "FAIL", span_id="f-1/agent"))],
            UnpairedObservationError,
            id="span-level",
        ),
        pytest.param(
            [(_trace("f-1"), _gold("f-2", "FAIL"))],
            UnpairedObservationError,
            id="names-another-trace",
        ),
    ],
)
def test_bad_gold_is_refused_before_any_billed_call(
    gold: list[tuple[Trace, Label]], error: type[Exception]
) -> None:
    clients = _clients()
    # No Evaluate response is queued: a billed call would fail as unstubbed, not as bad gold.
    with Stubber(clients.client("bedrock-agentcore")):
        with pytest.raises(error):
            agentcore_locked_test_set(
                _judge(), gold, clients=clients, confirm=lambda _: True, set_id="gold-test"
            )


def test_a_trace_without_exactly_one_trace_level_verdict_is_not_paired() -> None:
    clients = _clients()
    gold = [(_trace("f-1"), _gold("f-1", "FAIL"))]
    with Stubber(clients.client("bedrock-agentcore")) as stub:
        stub.add_response(
            "evaluate", {"evaluationResults": [_result("f-1", 0.0), _result("f-1", 1.0)]}
        )
        with pytest.raises(AgentCoreError, match="2 verdict"):
            agentcore_locked_test_set(
                _judge(), gold, clients=clients, confirm=lambda _: True, set_id="gold-test"
            )


def test_a_span_level_verdict_is_not_paired_with_trace_level_gold() -> None:
    clients = _clients()
    result = _result("f-1", 0.0)
    result["context"]["spanContext"]["spanId"] = __import__(
        "calibrated_agent_evaluation.agentcore.evaluate_payload", fromlist=["aws_span_id"]
    ).aws_span_id("f-1/agent")
    with Stubber(clients.client("bedrock-agentcore")) as stub:
        stub.add_response("evaluate", {"evaluationResults": [result]})
        with pytest.raises(AgentCoreError, match="one of them for a span"):
            agentcore_locked_test_set(
                _judge(),
                [(_trace("f-1"), _gold("f-1", "FAIL"))],
                clients=clients,
                confirm=lambda _: True,
                set_id="gold-test",
            )


def test_the_quickstart_gold_is_named_rule_derived_and_mixed_gold_names_both() -> None:
    rule = _gold("f-1", "FAIL")
    person = _gold("p-1", "PASS", rater_id="alice")

    assert gold_source([rule, rule]) == "synthetic-rule"
    assert gold_source([person]) == "human"
    assert gold_source([rule, person]) == "human and synthetic-rule"


# --- the session ids recorded are the ones the requests carried -----------------------------------


def _real_sessions(monkeypatch: pytest.MonkeyPatch, sessions: dict[str, Any]) -> None:
    """Make the serializer name a session per trace other than the one derived from its id.

    ``sessions`` maps a trace id to the session id its documents get, a list of ids to spread
    over its documents, or ``None`` to send none.
    """
    from calibrated_agent_evaluation.agentcore import evaluate as evaluate_module
    from calibrated_agent_evaluation.agentcore.evaluate_payload import trace_to_evaluation_input

    def serialize(trace: Trace) -> dict[str, Any]:
        payload = trace_to_evaluation_input(trace)
        wanted = sessions[trace.trace_id]
        documents = payload["sessionSpans"]
        for index, document in enumerate(documents):
            if wanted is None:
                document["attributes"].pop("session.id", None)
            elif isinstance(wanted, list):
                document["attributes"]["session.id"] = wanted[index % len(wanted)]
            else:
                document["attributes"]["session.id"] = wanted
        return payload

    monkeypatch.setattr(evaluate_module, "trace_to_evaluation_input", serialize)


def _echoing(clients: AwsClients, *, session_only: bool = False, other: bool = False) -> None:
    """Answer each Evaluate call with a result naming the session its own request carried."""
    from botocore.awsrequest import AWSResponse

    events = clients.client("bedrock-agentcore").meta.events

    def capture(params: dict[str, Any], context: dict[str, Any], **_: Any) -> None:
        context["calibrated_agent_evaluation_target"] = params["evaluationTarget"]["traceIds"][0]
        context["calibrated_agent_evaluation_session"] = params["evaluationInput"]["sessionSpans"][
            0
        ]["attributes"]["session.id"]

    def respond(context: dict[str, Any], **_: Any) -> Any:
        span_context = {
            "sessionId": "some-other-session"
            if other
            else context["calibrated_agent_evaluation_session"]
        }
        if not session_only:
            span_context["traceId"] = context["calibrated_agent_evaluation_target"]
        result = {
            "evaluatorArn": _ARN,
            "evaluatorId": _ID,
            "evaluatorName": "orderbot_refund_policy_v2",
            "explanation": "graded",
            "value": 1.0,
            "label": "PASS",
            "context": {"spanContext": span_context},
        }
        return AWSResponse("https://stub", 200, {}, None), {"evaluationResults": [result]}

    events.register("before-parameter-build.bedrock-agentcore.Evaluate", capture)
    events.register("before-call.bedrock-agentcore.Evaluate", respond)


_TWO = [(_trace("p-1"), _gold("p-1", "PASS")), (_trace("p-2"), _gold("p-2", "PASS"))]


@pytest.mark.parametrize("session_only", [False, True], ids=["trace-and-session", "session-only"])
def test_the_session_ids_recorded_are_the_ones_the_requests_carried(
    monkeypatch: pytest.MonkeyPatch, session_only: bool
) -> None:
    _real_sessions(monkeypatch, {"p-1": "conversation-a", "p-2": "conversation-b"})
    clients = _clients()
    _echoing(clients, session_only=session_only)

    run = agentcore_locked_test_set(
        _judge(), _TWO, clients=clients, confirm=lambda _: True, set_id="gold-test"
    )

    assert run.session_ids == ("conversation-a", "conversation-b")
    assert [pair.verdict.trace_id for pair in run.calibration_set.pairs] == ["p-1", "p-2"]


def test_a_result_naming_a_session_its_call_did_not_send_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreResultUnaddressable

    _real_sessions(monkeypatch, {"p-1": "conversation-a", "p-2": "conversation-b"})
    clients = _clients()
    _echoing(clients, other=True)

    with pytest.raises(AgentCoreResultUnaddressable, match="some-other-session"):
        agentcore_locked_test_set(
            _judge(), _TWO, clients=clients, confirm=lambda _: True, set_id="gold-test"
        )


@pytest.mark.parametrize(
    ("sessions", "error"),
    [
        (["conversation-a", "conversation-b"], "AgentCoreMixedSession"),
        (None, "AgentCoreContentUnserializable"),
    ],
    ids=["two-sessions", "no-session"],
)
def test_a_request_without_exactly_one_session_is_refused_before_any_billed_call(
    monkeypatch: pytest.MonkeyPatch, sessions: Any, error: str
) -> None:
    from calibrated_agent_evaluation.agentcore import errors

    # Two documents per trace (the agent span and a tool span), so two sessions can be spread.
    trace = _trace("p-1")
    tooled = trace.model_copy(
        update={
            "spans": [
                trace.spans[0].model_copy(
                    update={"tool_calls": [ToolCall(name="lookup_order", args={}, result={})]}
                )
            ]
        }
    )
    _real_sessions(monkeypatch, {"p-1": sessions})
    clients = _clients()
    with (
        Stubber(clients.client("bedrock-agentcore")) as stub,
        pytest.raises(getattr(errors, error)),
    ):
        agentcore_locked_test_set(
            _judge(),
            [(tooled, _gold("p-1", "PASS"))],
            clients=clients,
            confirm=lambda _: True,
            set_id="gold-test",
        )
    stub.assert_no_pending_responses()
