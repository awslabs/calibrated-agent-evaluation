# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running an AgentCore evaluator over a batch of targets as an advisory judge."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("boto3")

import boto3
from botocore.config import Config
from botocore.stub import Stubber

from _helpers import load_corpus_trace, load_quickstart_trace
from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreContentUnserializable,
    AgentCoreError,
    AgentCoreEvaluatorLevelUnsupported,
    AgentCoreEvaluatorResultError,
    AgentCoreNoPassDirection,
    AgentCoreReferenceDropped,
    AgentCoreResultUnaddressable,
    AgentCoreRunDeclined,
    AgentCoreScoreIncomplete,
    AgentCoreScoreOffScale,
    BilledCallBudgetExceeded,
)
from calibrated_agent_evaluation.agentcore.evaluate import run_agentcore_judge
from calibrated_agent_evaluation.agentcore.evaluate_payload import (
    aws_span_id,
    aws_trace_id,
    trace_to_evaluation_input,
)
from calibrated_agent_evaluation.agentcore.guards import BilledCallBudget
from calibrated_agent_evaluation.agentcore.judge import AgentCoreJudge, ScaleValue
from calibrated_agent_evaluation.agentcore.pricing import EvaluationRates, RunEstimate
from calibrated_agent_evaluation.agentcore.session import AwsClients
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.schema.types import Label, Span, Trace

_ARN = "arn:aws:bedrock-agentcore:us-east-1:111122223333:evaluator/correctness"


_RATES = EvaluationRates(
    region="us-east-1",
    builtin_input_per_million=Decimal("2.40"),
    builtin_output_per_million=Decimal("12.00"),
    custom_per_evaluation=Decimal("0.0015"),
)


def _approve(_estimate: RunEstimate) -> bool:
    return True


@pytest.fixture(autouse=True)
def _published_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every run here prices against fixed rates; the Price List lookup is tested on its own."""
    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.evaluate.lookup_evaluation_rates", lambda _c: _RATES
    )


def _clients() -> AwsClients:
    # Already set to one attempt, so the run sends on these clients and the Stubbers below answer.
    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    return AwsClients(session).with_config(Config(retries={"total_max_attempts": 1}))


def _judge() -> AgentCoreJudge:
    return AgentCoreJudge(
        judge_id="correctness",
        evaluator_id="correctness",
        evaluator_arn=_ARN,
        evaluator_type="Custom",
        updated_at="2026-10-01T00:00:00+00:00",
        config_digest="0123456789abcdef",
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
        frame="production-representative",
        ownership="owned",
        input={"message": "refund please"},
        output={"answer": "refunded"},
        taxonomy_cell={},
        spans=[span],
    )


def _result(trace_id: str, *, value: float = 1.0, label: str = "correct") -> dict[str, Any]:
    derived = aws_trace_id(trace_id)
    return {
        "evaluatorArn": _ARN,
        "evaluatorId": "correctness",
        "evaluatorName": "correctness",
        "explanation": "the answer is grounded",
        "value": value,
        "label": label,
        "context": {"spanContext": {"sessionId": derived, "traceId": derived}},
    }


def test_a_run_over_two_targets_returns_two_labels_carrying_the_input_ids() -> None:
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    stub.add_response("evaluate", {"evaluationResults": [_result("t-1")]})
    stub.add_response("evaluate", {"evaluationResults": [_result("t-2")]})
    with stub:
        labels = run_agentcore_judge(
            _judge(), [_trace("t-1"), _trace("t-2")], clients=clients, confirm=_approve
        )
    # The labels carry our own trace ids, mapped back from the conforming ids on the request.
    assert sorted(label.trace_id for label in labels) == ["t-1", "t-2"]
    assert all(label.kind == "llm-judge" for label in labels)
    stub.assert_no_pending_responses()


def test_twenty_five_traces_issue_twenty_five_calls_each_with_only_its_own_spans() -> None:
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    traces = [_trace(f"t-{i}") for i in range(25)]
    for trace in traces:
        own_spans = trace_to_evaluation_input(trace)["sessionSpans"]
        # Every span in the call belongs to this trace and its one session.
        own_id = aws_trace_id(trace.trace_id)
        assert {(s["traceId"], s["attributes"]["session.id"]) for s in own_spans} == {
            (own_id, own_id)
        }
        expected = {
            "evaluatorId": "correctness",
            "evaluationInput": {"sessionSpans": own_spans},
            "evaluationTarget": {"traceIds": [aws_trace_id(trace.trace_id)]},
        }
        stub.add_response("evaluate", {"evaluationResults": [_result(trace.trace_id)]}, expected)
    with stub:
        labels = run_agentcore_judge(_judge(), traces, clients=clients, confirm=_approve)
    assert [label.trace_id for label in labels] == [trace.trace_id for trace in traces]
    stub.assert_no_pending_responses()  # exactly 25 calls, one per trace


class _OneSessionEvaluate:
    """A stand-in for the Evaluate service that, like the real one, refuses an input whose spans
    belong to more than one session, and otherwise scores the call's one target."""

    # A real client's config surface, read by the runner: already set to one attempt.
    meta = SimpleNamespace(config=SimpleNamespace(retries={"total_max_attempts": 1}))

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, **request: Any) -> dict[str, Any]:
        from botocore.exceptions import ClientError

        self.calls += 1
        sessions = {
            span["attributes"]["session.id"] for span in request["evaluationInput"]["sessionSpans"]
        }
        if len(sessions) > 1:
            raise ClientError(
                {
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "EvaluationInput has spans from more than one session, please "
                        "submit each session separately",
                    }
                },
                "Evaluate",
            )
        targets = request["evaluationTarget"]["traceIds"]
        assert len(targets) == 1, "one target per call"
        (target,) = targets
        result = _result("unused")
        result["context"] = {"spanContext": {"sessionId": target, "traceId": target}}
        return {"evaluationResults": [result]}


class _ServiceClients:
    billed_call_budget = None
    region = "us-east-1"

    def __init__(self, service: _OneSessionEvaluate) -> None:
        self._service = service

    def client(self, _name: str) -> Any:
        return self._service


def test_a_multi_trace_run_is_accepted_by_a_service_that_scores_one_session_per_call() -> None:
    service = _OneSessionEvaluate()
    clients = cast("AwsClients", _ServiceClients(service))
    labels = run_agentcore_judge(
        _judge(), [_trace("t-1"), _trace("t-2")], clients=clients, confirm=_approve
    )
    assert [label.trace_id for label in labels] == ["t-1", "t-2"]
    assert service.calls == 2


def test_an_empty_run_makes_no_call_and_returns_no_label() -> None:
    service = _OneSessionEvaluate()
    clients = cast("AwsClients", _ServiceClients(service))
    assert run_agentcore_judge(_judge(), [], clients=clients, confirm=_approve) == []
    assert service.calls == 0


def test_the_stored_label_is_advisory_with_the_raw_verdict_preserved(tmp_path: Any) -> None:
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    stub.add_response(
        "evaluate",
        {"evaluationResults": [_result("t-1", value=1.0, label="Perfectly Correct")]},
    )
    backend = LocalBackend(root=str(tmp_path / "store"))
    with stub:
        (label,) = run_agentcore_judge(
            _judge(), [_trace("t-1")], clients=clients, confirm=_approve, store=backend
        )
    assert label.kind == "llm-judge"
    assert label.outcome == "PASS"  # derived from the numeric value
    assert "value=1.0" in label.critique  # the raw verdict rides in the critique
    assert "Perfectly Correct" in label.critique
    assert backend.get_scores("t-1")[0].kind == "llm-judge"


def test_an_unreachable_account_raises_and_writes_no_label(tmp_path: Any) -> None:
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    stub.add_client_error("evaluate", service_error_code="ServiceUnavailableException")
    backend = LocalBackend(root=str(tmp_path / "store"))
    with stub, pytest.raises(AgentCoreError):
        run_agentcore_judge(
            _judge(), [_trace("t-1")], clients=clients, confirm=_approve, store=backend
        )
    assert backend.get_scores("t-1") == []


def test_an_off_scale_result_on_a_later_call_stores_none_of_the_runs_labels(tmp_path: Any) -> None:
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    scaled = replace(
        _judge(),
        rating_scale=(ScaleValue(0.0, "Incorrect"), ScaleValue(1.0, "Correct")),
    )
    traces = [_trace("t-0"), _trace("t-1")]
    # The first trace's result is on the scale; the second trace's, on a later call, is not.
    stub.add_response("evaluate", {"evaluationResults": [_result("t-0")]})
    stub.add_response("evaluate", {"evaluationResults": [_result("t-1", value=0.7)]})
    backend = LocalBackend(root=str(tmp_path / "store"))
    with stub, pytest.raises(AgentCoreScoreOffScale):
        run_agentcore_judge(scaled, traces, clients=clients, confirm=_approve, store=backend)
    assert all(backend.get_scores(t.trace_id) == [] for t in traces)


def test_an_inverted_judge_turns_a_high_score_into_a_fail() -> None:
    # A judge derived from an inverted base carries the direction itself: its results name only the
    # derived evaluator, so the runner must apply the judge's direction rather than look it up.
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    inverted = replace(
        _judge(),
        rating_scale=(ScaleValue(0.0, "clean"), ScaleValue(1.0, "toxic")),
        inverted=True,
    )
    stub.add_response("evaluate", {"evaluationResults": [_result("t-1", value=1.0)]})
    stub.add_response("evaluate", {"evaluationResults": [_result("t-2", value=0.0)]})
    with stub:
        labels = run_agentcore_judge(
            inverted, [_trace("t-1"), _trace("t-2")], clients=clients, confirm=_approve
        )
    assert {label.trace_id: label.outcome for label in labels} == {"t-1": "FAIL", "t-2": "PASS"}


def test_a_judge_on_an_evaluator_with_no_pass_direction_is_refused_before_any_billed_call() -> None:
    # Built directly rather than through the factory, the judge must still be refused: the runner
    # checks the direction itself, so no Evaluate call is made (the stub has no response queued).
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    refusal = replace(_judge(), evaluator_id="Builtin.Refusal")
    with stub, pytest.raises(AgentCoreNoPassDirection):
        run_agentcore_judge(refusal, [_trace("t-1")], clients=clients, confirm=_approve)
    stub.assert_no_pending_responses()


def _multi_span_trace(trace_id: str, *agents: str) -> Trace:
    base = _trace(trace_id)
    spans = [
        base.spans[0].model_copy(
            update={"span_id": f"{trace_id}/{agent}", "agent": agent, "seq": i}
        )
        for i, agent in enumerate(agents)
    ]
    return base.model_copy(update={"spans": spans})


def _span_result(trace_id: str, span_id: str, *, of_trace: str | None = None) -> dict[str, Any]:
    # A span-level result names a span by its derived id; ``of_trace`` lets a test pair a trace
    # with a span that belongs to another trace.
    result = _result(trace_id)
    derived_trace = aws_trace_id(of_trace or trace_id)
    result["context"] = {
        "spanContext": {
            "sessionId": derived_trace,
            "traceId": derived_trace,
            "spanId": aws_span_id(span_id),
        }
    }
    return result


def test_span_verdicts_come_back_under_their_own_ids_with_their_trace_as_the_cluster(
    tmp_path: Any,
) -> None:
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    one = _multi_span_trace("t-1", "orderbot", "refunds")
    two = _multi_span_trace("t-2", "orderbot")
    session_only = _result("t-2")
    session_only["context"] = {"spanContext": {"sessionId": aws_trace_id("t-2")}}
    t1_results = [
        _span_result("t-1", "t-1/orderbot"),
        _span_result("t-1", "t-1/refunds"),
        _result("t-1"),  # trace-level: no span id
    ]
    t2_results = [_span_result("t-2", "t-2/orderbot"), session_only]  # session-only: no trace id
    stub.add_response("evaluate", {"evaluationResults": t1_results})
    stub.add_response("evaluate", {"evaluationResults": t2_results})
    backend = LocalBackend(root=str(tmp_path / "store"))
    with stub:
        labels = run_agentcore_judge(
            _judge(), [one, two], clients=clients, confirm=_approve, store=backend
        )
    # The session-only verdict is stored under the original Calibrated Agent Evaluation trace id,
    # not the derived hash.
    assert [lb.span_id for lb in backend.get_scores("t-2")].count(None) == 1
    assert [(label.trace_id, label.span_id) for label in labels] == [
        ("t-1", "t-1/orderbot"),
        ("t-1", "t-1/refunds"),
        ("t-1", None),
        ("t-2", "t-2/orderbot"),
        ("t-2", None),
    ]


def _refused(tmp_path: Any, traces: list[Trace], responses: list[list[dict[str, Any]]]) -> None:
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    for results in responses:
        stub.add_response("evaluate", {"evaluationResults": results})
    backend = LocalBackend(root=str(tmp_path / "store"))
    with stub, pytest.raises(AgentCoreResultUnaddressable):
        run_agentcore_judge(_judge(), traces, clients=clients, confirm=_approve, store=backend)
    assert all(backend.get_scores(trace.trace_id) == [] for trace in traces)


def test_a_result_for_a_trace_the_run_never_sent_is_refused(tmp_path: Any) -> None:
    _refused(tmp_path, [_trace("t-1")], [[_result("t-1"), _result("ghost")]])


def test_a_result_naming_a_trace_from_another_call_of_the_run_is_refused(tmp_path: Any) -> None:
    traces = [_trace("t-0"), _trace("t-1")]
    # The second call targeted only t-1; a result naming t-0, the first call's trace, does not
    # belong to it.
    _refused(tmp_path, traces, [[_result("t-0")], [_result("t-0")]])


def test_a_result_pairing_a_trace_with_another_traces_span_is_refused(tmp_path: Any) -> None:
    traces = [_multi_span_trace("t-1", "orderbot"), _multi_span_trace("t-2", "orderbot")]
    _refused(tmp_path, traces, [[_span_result("t-1", "t-2/orderbot")]])


def test_a_result_naming_an_untargeted_session_and_no_trace_is_refused(tmp_path: Any) -> None:
    session_only = _result("t-1")
    session_only["context"] = {"spanContext": {"sessionId": aws_trace_id("ghost")}}
    _refused(tmp_path, [_trace("t-1")], [[session_only]])


def test_a_result_whose_trace_and_session_ids_disagree_is_refused(tmp_path: Any) -> None:
    mismatched = _result("t-1")
    mismatched["context"] = {
        "spanContext": {"traceId": aws_trace_id("t-1"), "sessionId": aws_trace_id("t-2")}
    }
    _refused(tmp_path, [_trace("t-1"), _trace("t-2")], [[mismatched]])


def _stored_verdict(tmp_path: Any, value: float = 1.0) -> tuple[Trace, Label]:
    """One AgentCore verdict, run and stored the way `calibrated-agent-evaluation` stores any
    judge's verdict."""
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    stub.add_response("evaluate", {"evaluationResults": [_result("t-1", value=value)]})
    backend = LocalBackend(root=str(tmp_path / "store"))
    trace = _trace("t-1")
    with stub:
        run_agentcore_judge(_judge(), [trace], clients=clients, confirm=_approve, store=backend)
    (stored,) = backend.get_scores("t-1")
    return trace, stored


@pytest.mark.parametrize("value", [1.0, 0.0], ids=["pass-verdict", "fail-verdict"])
def test_a_stored_agentcore_verdict_is_never_picked_as_calibration_gold(
    tmp_path: Any, value: float
) -> None:
    from calibrated_agent_evaluation.cli.judgefile import load_judge_file
    from calibrated_agent_evaluation.cli.judging import gold_for
    from cli_store_helpers import JUDGE, write_judge_file

    trace, verdict = _stored_verdict(tmp_path, value)
    jf = load_judge_file(write_judge_file(tmp_path / "judge.json"))
    assert gold_for(jf, [trace], {"t-1": [verdict]}).pairs == []
    # The same label from a human (a FAIL naming the judge's mode) would be gold, so it is the
    # advisory kind that keeps it out, for a negative and a positive alike.
    human = verdict.model_copy(update={"kind": "human", "failure_mode": JUDGE["failure_mode"]})
    assert len(gold_for(jf, [trace], {"t-1": [human]}).pairs) == 1


def test_an_agentcore_verdict_offered_as_gold_is_refused_at_pairing(tmp_path: Any) -> None:
    from calibrated_agent_evaluation.calibrate.errors import UnratifiedGoldError
    from calibrated_agent_evaluation.calibrate.pairing import pair_observations
    from calibrated_agent_evaluation.calibrate.record import LabelPair

    _, verdict = _stored_verdict(tmp_path)
    with pytest.raises(UnratifiedGoldError):
        pair_observations([LabelPair(gold=verdict, verdict=verdict)])
    # A human gold label for the same target pairs, so the refusal is about who produced the gold.
    human = verdict.model_copy(update={"kind": "human"})
    assert pair_observations([LabelPair(gold=human, verdict=verdict)]).n_neg == 1


@pytest.mark.parametrize(
    ("level", "named"),
    [("TOOL_CALL", "TOOL_CALL"), ("SESSION", "SESSION"), (None, "no level")],
    ids=["tool-call", "session", "missing"],
)
def test_a_judge_built_directly_without_a_trace_level_is_refused_before_any_billed_call(
    level: str | None, named: str
) -> None:
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))  # no response queued: a call would fail
    judge = replace(_judge(), evaluator_level=level)
    with stub, pytest.raises(AgentCoreEvaluatorLevelUnsupported, match=named):
        run_agentcore_judge(judge, [_trace("t-1")], clients=clients, confirm=_approve)
    stub.assert_no_pending_responses()


def test_a_judge_built_from_a_trace_level_evaluator_runs() -> None:
    from calibrated_agent_evaluation.agentcore.judge import judge_from_evaluator

    document = {
        "evaluatorId": "correctness",
        "evaluatorArn": _ARN,
        "evaluatorType": "Custom",
        "level": "TRACE",
        "updatedAt": "2026-10-01T00:00:00+00:00",
        "evaluatorConfig": {
            "llmAsAJudge": {
                "instructions": "judge",
                "ratingScale": {
                    "numerical": [{"value": 0.0, "label": "FAIL"}, {"value": 1.0, "label": "PASS"}]
                },
            }
        },
    }
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    stub.add_response("evaluate", {"evaluationResults": [_result("t-1")]})
    with stub:
        labels = run_agentcore_judge(
            judge_from_evaluator("correctness", document),
            [_trace("t-1")],
            clients=clients,
            confirm=_approve,
        )
    assert [label.trace_id for label in labels] == ["t-1"]


# The shapes below are the ones the live service returned on an AWS test account,
# re-addressed to this module's evaluator and traces.
_LOG_EVENT_MISSING = (
    "Session span data is incomplete. Span with ID: 74137f618a9b017f and name: invoke_agent "
    "Strands Agents is missing a corresponding log event. Please retry the request with the "
    "complete span data"
)


def _error_code_result(trace_id: str) -> dict[str, Any]:
    """An HTTP-200 result the service could not score: an error code and message, no verdict."""
    derived = aws_trace_id(trace_id)
    return {
        "evaluatorArn": _ARN,
        "evaluatorId": "correctness",
        "evaluatorName": "correctness",
        "context": {"spanContext": {"sessionId": derived, "traceId": derived}},
        "errorMessage": _LOG_EVENT_MISSING,
        "errorCode": "LogEventMissingException",
    }


def _dropped_reference_result(trace_id: str) -> dict[str, Any]:
    """A scored result whose evaluator ignored the reference input it was given."""
    result = _result(trace_id, value=0.67, label="Somewhat Helpful")
    result["ignoredReferenceInputFields"] = ["expectedResponse"]
    return result


class _QueuedEvaluate:
    """An Evaluate client that returns one queued result per call, as given: the botocore stubber
    refuses shapes the live service can still send, such as a result with no address."""

    billed_call_budget = None
    region = "us-east-1"
    # The part of a real client's surface the runner reads: it is already set to one attempt.
    meta = SimpleNamespace(config=SimpleNamespace(retries={"total_max_attempts": 1}))

    def __init__(self, *results: dict[str, Any]) -> None:
        self._results = list(results)

    def client(self, _name: str) -> Any:
        return self

    def evaluate(self, **_request: Any) -> dict[str, Any]:
        return {"evaluationResults": [self._results.pop(0)]}


def _timed_corpus_trace(trace_id: str) -> Trace:
    """A shared-corpus trace given span timing, which the corpus does not carry: the root span
    encloses the run and each child gets its own slot inside it, in ``seq`` order."""
    trace = load_corpus_trace(trace_id)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    children = sorted((span for span in trace.spans if span.parent), key=lambda span: span.seq)
    timed = []
    for span in trace.spans:
        if span.parent is None:
            began, ended = start, start + timedelta(seconds=len(children) + 1)
        else:
            began = start + timedelta(seconds=children.index(span) + 1)
            ended = began + timedelta(milliseconds=500)
        timed.append(
            span.model_copy(update={"started_at": began.isoformat(), "ended_at": ended.isoformat()})
        )
    return trace.model_copy(update={"spans": timed})


def _refused_on_second_call(
    tmp_path: Any, second: dict[str, Any], error: type[Exception], match: str
) -> None:
    """ORD-1002 scores cleanly, then CLM-4471's call returns ``second``: the run raises ``error``
    and the store holds neither trace's verdict."""
    clients = cast("AwsClients", _QueuedEvaluate(_result("ORD-1002"), second))
    backend = LocalBackend(root=str(tmp_path / "store"))
    traces = [_timed_corpus_trace("ORD-1002"), _timed_corpus_trace("CLM-4471")]
    with pytest.raises(error, match=match):
        run_agentcore_judge(_judge(), traces, clients=clients, confirm=_approve, store=backend)
    assert all(backend.get_scores(trace.trace_id) == [] for trace in traces)


def test_an_error_code_result_refuses_the_run_with_the_services_reason_and_stores_nothing(
    tmp_path: Any,
) -> None:
    _refused_on_second_call(
        tmp_path,
        _error_code_result("CLM-4471"),
        AgentCoreEvaluatorResultError,
        "LogEventMissingException.*missing a corresponding log event",
    )


def _without_context(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "context"}


def _naming(trace_id: str, result: dict[str, Any]) -> dict[str, Any]:
    derived = aws_trace_id(trace_id)
    return {**result, "context": {"spanContext": {"sessionId": derived, "traceId": derived}}}


@pytest.mark.parametrize(
    "result",
    [
        _without_context(_error_code_result("t-1")),
        {**_error_code_result("t-1"), "value": 1.0, "label": "correct"},
        {**_error_code_result("t-1"), "ignoredReferenceInputFields": ["expectedResponse"]},
        _naming("ghost", _error_code_result("t-1")),
    ],
    ids=["no-address", "also-a-value", "also-a-dropped-reference", "an-untargeted-trace"],
)
def test_an_error_code_is_reported_ahead_of_whatever_else_is_wrong_with_the_result(
    result: dict[str, Any],
) -> None:
    clients = cast("AwsClients", _QueuedEvaluate(result))
    with pytest.raises(AgentCoreEvaluatorResultError, match="LogEventMissingException"):
        run_agentcore_judge(_judge(), [_trace("t-1")], clients=clients, confirm=_approve)


def test_a_dropped_reference_refuses_the_run_even_though_the_result_carries_a_score(
    tmp_path: Any,
) -> None:
    _refused_on_second_call(
        tmp_path,
        _dropped_reference_result("CLM-4471"),
        AgentCoreReferenceDropped,
        "expectedResponse",
    )


def test_a_result_with_an_empty_ignored_reference_list_scores_normally() -> None:
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    stub.add_response(
        "evaluate", {"evaluationResults": [{**_result("t-1"), "ignoredReferenceInputFields": []}]}
    )
    with stub:
        (label,) = run_agentcore_judge(_judge(), [_trace("t-1")], clients=clients, confirm=_approve)
    assert (label.trace_id, label.outcome) == ("t-1", "PASS")


@pytest.mark.parametrize(
    ("result", "match"),
    [
        (
            {k: v for k, v in _result("CLM-4471").items() if k not in ("value", "label")},
            "neither a value nor a label",
        ),
        ({**_result("CLM-4471"), "context": {}}, "no traceId or sessionId"),
    ],
    ids=["no-verdict", "no-address"],
)
def test_an_incomplete_result_refuses_the_run_and_stores_nothing(
    tmp_path: Any, result: dict[str, Any], match: str
) -> None:
    _refused_on_second_call(tmp_path, result, AgentCoreScoreIncomplete, match)


class _NoCallAllowed:
    """An Evaluate client that fails the test on any call, so a refusal provably cost nothing."""

    billed_call_budget = None
    region = "us-east-1"

    def client(self, _name: str) -> Any:
        return self

    def evaluate(self, **_request: Any) -> dict[str, Any]:
        raise AssertionError("a billed Evaluate call was made")


def _partly_timed(trace_id: str) -> Trace:
    trace = _trace(trace_id)
    (span,) = trace.spans
    return trace.model_copy(update={"spans": [span.model_copy(update={"ended_at": None})]})


def _with_lost_parent(trace_id: str) -> Trace:
    trace = load_corpus_trace(trace_id)
    lost = trace.spans[-1].model_copy(update={"parent": f"{trace_id}/ghost"})
    return trace.model_copy(update={"spans": [*trace.spans[:-1], lost]})


@pytest.mark.parametrize(
    ("second", "match"),
    [(_partly_timed("t-2"), "partial"), (_with_lost_parent("CLM-4471"), "ghost")],
    ids=["partly-timed", "a-parent-outside-the-trace"],
)
def test_a_run_that_cannot_be_given_synthetic_timing_makes_no_billed_call(
    second: Trace, match: str
) -> None:
    clients = cast("AwsClients", _NoCallAllowed())
    with pytest.raises(AgentCoreContentUnserializable, match=match):
        run_agentcore_judge(_judge(), [_trace("t-1"), second], clients=clients, confirm=_approve)


def test_synthetic_timing_marks_only_verdicts_on_traces_that_used_it_and_warns_once(
    tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    quickstart = load_quickstart_trace()
    corpus = load_corpus_trace("CLM-4471")
    backend = LocalBackend(root=str(tmp_path / "store"))
    backend.put_trace(corpus)
    results = (_result("t-1"), _result(quickstart.trace_id), _result("CLM-4471"))
    clients = cast("AwsClients", _QueuedEvaluate(*results))
    with caplog.at_level("WARNING", logger="calibrated_agent_evaluation.agentcore.evaluate"):
        labels = run_agentcore_judge(
            _judge(),
            [_trace("t-1"), quickstart, corpus],
            clients=clients,
            confirm=_approve,
            store=backend,
        )
    marks = {label.trace_id: "timing=synthetic" in label.critique for label in labels}
    # Real span times and no tool calls: nothing made up. Tool calls, or no span times: marked.
    assert marks == {"t-1": False, quickstart.trace_id: True, "CLM-4471": True}
    assert "timing=synthetic" in backend.get_scores("CLM-4471")[0].critique
    (warning,) = [r for r in caplog.records if r.levelname == "WARNING"]
    assert "2 of 3" in warning.getMessage()
    stored = backend.get_trace("CLM-4471")
    assert all(span.started_at is None and span.ended_at is None for span in stored.spans)


def test_a_run_with_no_synthetic_timing_logs_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    clients = cast("AwsClients", _QueuedEvaluate(_result("t-1")))
    with caplog.at_level("WARNING", logger="calibrated_agent_evaluation.agentcore.evaluate"):
        (label,) = run_agentcore_judge(_judge(), [_trace("t-1")], clients=clients, confirm=_approve)
    assert "timing=synthetic" not in label.critique
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def _builtin() -> AgentCoreJudge:
    return replace(_judge(), evaluator_type="Builtin")


def _estimate_of(judge: AgentCoreJudge, traces: list[Trace]) -> RunEstimate:
    seen: list[RunEstimate] = []

    def confirm(estimate: RunEstimate) -> bool:
        seen.append(estimate)
        return False

    clients = cast("AwsClients", _NoCallAllowed())
    with pytest.raises(AgentCoreRunDeclined):
        run_agentcore_judge(judge, traces, clients=clients, confirm=confirm)
    (estimate,) = seen
    return estimate


def test_a_builtin_judge_estimate_names_the_calls_the_tokens_and_the_cost() -> None:
    estimate = _estimate_of(_builtin(), [_trace("t-1"), _trace("t-2")])
    # Each trace: "refund please" (13) + "refunded" (8) = 21 characters -> ceil(21/4) + 500 = 506.
    assert (estimate.calls, estimate.input_tokens, estimate.output_tokens) == (2, 1012, 600)
    assert estimate.cost_usd == Decimal("0.0096288")
    assert "2 billed" in estimate.describe()
    assert "$0.0096" in estimate.describe()


def test_a_custom_judge_estimate_is_per_evaluation_and_leaves_out_model_usage() -> None:
    estimate = _estimate_of(_judge(), [_trace("t-1"), _trace("t-2")])
    assert estimate.cost_usd == Decimal("0.0030")
    assert "model usage" in estimate.describe()
    assert "not included" in estimate.describe()


def test_an_approved_estimate_lets_the_run_go_ahead() -> None:
    seen: list[RunEstimate] = []

    def confirm(estimate: RunEstimate) -> bool:
        seen.append(estimate)
        return True

    clients = cast("AwsClients", _QueuedEvaluate(_result("t-1")))
    labels = run_agentcore_judge(_judge(), [_trace("t-1")], clients=clients, confirm=confirm)
    assert [label.trace_id for label in labels] == ["t-1"]
    assert [e.calls for e in seen] == [1]


def test_a_declined_estimate_makes_no_billed_call() -> None:
    clients = cast("AwsClients", _NoCallAllowed())
    with pytest.raises(AgentCoreRunDeclined):
        run_agentcore_judge(_judge(), [_trace("t-1")], clients=clients, confirm=lambda _e: False)


def _no_rates(_clients: Any) -> EvaluationRates:
    raise AgentCoreError("pricing:GetProducts was denied (AccessDeniedException)")


def _unlisted(_clients: Any) -> EvaluationRates:
    return EvaluationRates(region="eu-west-9")


def _not_looked_up(_clients: Any) -> EvaluationRates:
    raise AssertionError("rates were looked up for an evaluator type that has none")


@pytest.mark.parametrize(
    ("judge", "lookup", "reason"),
    [
        (_builtin(), _no_rates, "AccessDeniedException"),
        (_builtin(), _unlisted, "no AgentCore Evaluations rate"),
        (replace(_judge(), evaluator_type="ThirdParty"), _not_looked_up, "third-party"),
        (replace(_judge(), evaluator_type="Mystery"), _not_looked_up, "no known AgentCore pricing"),
    ],
    ids=["lookup-denied", "no-listing", "third-party", "unknown-type"],
)
def test_an_estimate_without_a_rate_still_shows_calls_and_tokens_and_asks(
    monkeypatch: pytest.MonkeyPatch, judge: AgentCoreJudge, lookup: Any, reason: str
) -> None:
    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.evaluate.lookup_evaluation_rates", lookup
    )
    estimate = _estimate_of(judge, [_trace("t-1")])
    assert (estimate.calls, estimate.input_tokens, estimate.output_tokens) == (1, 506, 300)
    assert estimate.cost_usd is None
    assert reason in (estimate.no_rate_reason or "")
    assert "no cost" in estimate.describe()


def _budgeted(limit: int | None) -> tuple[AwsClients, BilledCallBudget, Stubber]:
    budget = BilledCallBudget(limit)
    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    clients = AwsClients(session, handlers=[budget]).with_config(
        Config(retries={"total_max_attempts": 1})
    )
    return clients, budget, Stubber(clients.client("bedrock-agentcore"))


def _never_asked(_estimate: RunEstimate) -> bool:
    raise AssertionError("confirmation was asked for a run the budget cannot fit")


def test_a_run_over_budget_is_refused_before_any_call_and_before_confirmation() -> None:
    clients, budget, stub = _budgeted(1)
    traces = [_trace("t-1"), _trace("t-2")]
    with stub, pytest.raises(BilledCallBudgetExceeded):
        run_agentcore_judge(_judge(), traces, clients=clients, confirm=_never_asked)
    stub.assert_no_pending_responses()
    assert budget.used == 0


@pytest.mark.parametrize("limit", [2, None], ids=["exactly-enough-budget", "no-budget-limit"])
def test_a_run_within_budget_goes_ahead(limit: int | None) -> None:
    clients, budget, stub = _budgeted(limit)
    stub.add_response("evaluate", {"evaluationResults": [_result("t-1")]})
    stub.add_response("evaluate", {"evaluationResults": [_result("t-2")]})
    traces = [_trace("t-1"), _trace("t-2")]
    with stub:
        labels = run_agentcore_judge(_judge(), traces, clients=clients, confirm=_approve)
    assert len(labels) == 2
    assert budget.used == 2


def test_reading_the_budget_charges_nothing_and_a_region_client_carries_it() -> None:
    clients, budget, _ = _budgeted(3)
    assert clients.billed_call_budget is budget
    assert (budget.remaining, budget.remaining) == (3, 3)
    other = clients.client("bedrock-agentcore", region="us-west-2")
    assert other is clients.client("bedrock-agentcore", region="us-west-2")
    assert other is not clients.client("bedrock-agentcore")
    stub = Stubber(other)
    stub.add_response("evaluate", {"evaluationResults": [_result("t-1")]})
    with stub:
        other.evaluate(  # pyright: ignore[reportAttributeAccessIssue]
            evaluatorId="correctness",
            evaluationInput=trace_to_evaluation_input(_trace("t-1")),
            evaluationTarget={"traceIds": [aws_trace_id("t-1")]},
        )
    assert budget.remaining == 2
    assert AwsClients(boto3.Session(region_name="us-east-1")).billed_call_budget is None
    assert BilledCallBudget().remaining is None


def _with_usage(trace_id: str, inputs: int, outputs: int) -> dict[str, Any]:
    usage = {"inputTokens": inputs, "outputTokens": outputs, "totalTokens": inputs + outputs}
    return {**_result(trace_id), "tokenUsage": usage}


def _usage_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "reported" in r.getMessage()]


def test_after_a_run_the_estimate_is_followed_by_the_reported_token_usage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clients = cast("AwsClients", _QueuedEvaluate(_with_usage("t-1", 900, 200), _result("t-2")))
    traces = [_trace("t-1"), _trace("t-2")]
    with caplog.at_level("INFO", logger="calibrated_agent_evaluation.agentcore.evaluate"):
        run_agentcore_judge(_builtin(), traces, clients=clients, confirm=_approve)
    # 900 * 2.40 / 1M + 200 * 12.00 / 1M = $0.00456; the result with no tokenUsage adds nothing.
    (line,) = _usage_lines(caplog)
    assert "900 input" in line
    assert "200 output" in line
    assert "$0.0046" in line


def test_a_run_refused_part_way_still_logs_the_usage_after_its_estimate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    results = (_with_usage("t-1", 900, 200), _error_code_result("t-2"))
    clients = cast("AwsClients", _QueuedEvaluate(*results))
    traces = [_trace("t-1"), _trace("t-2")]
    with (
        caplog.at_level("INFO", logger="calibrated_agent_evaluation.agentcore.evaluate"),
        pytest.raises(AgentCoreEvaluatorResultError),
    ):
        run_agentcore_judge(_builtin(), traces, clients=clients, confirm=_approve)
    (line,) = _usage_lines(caplog)
    assert "900 input" in line


def test_usage_after_an_estimate_with_no_rate_is_logged_without_a_cost(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.evaluate.lookup_evaluation_rates", _no_rates
    )
    clients = cast("AwsClients", _QueuedEvaluate(_with_usage("t-1", 900, 200)))
    with caplog.at_level("INFO", logger="calibrated_agent_evaluation.agentcore.evaluate"):
        run_agentcore_judge(_builtin(), [_trace("t-1")], clients=clients, confirm=_approve)
    (line,) = _usage_lines(caplog)
    assert "900 input" in line
    assert "$" not in line


def test_each_verdict_names_the_fingerprint_of_the_judge_version_that_produced_it(
    tmp_path: Any,
) -> None:
    before = _judge()
    after = replace(before, config_digest="fedcba9876543210")  # the evaluator was edited
    backend = LocalBackend(root=str(tmp_path / "store"))
    for judge in (before, after):
        clients = _clients()
        stub = Stubber(clients.client("bedrock-agentcore"))
        stub.add_response("evaluate", {"evaluationResults": [_result("t-1")]})
        with stub:
            run_agentcore_judge(
                judge, [_trace("t-1")], clients=clients, confirm=_approve, store=backend
            )
    critiques = [label.critique for label in backend.get_scores("t-1")]
    # Same evaluator ARN, value and explanation: only the fingerprint tells the two versions apart.
    assert before.fingerprint != after.fingerprint
    for judge in (before, after):
        tag = f"[fingerprint={judge.fingerprint}]"
        # The trace has real timing and no tool calls, so the tag is the critique's last mark.
        assert [critique.endswith(tag) for critique in critiques].count(True) == 1


def test_the_fingerprint_comes_before_the_synthetic_timing_mark() -> None:
    trace = load_corpus_trace("ORD-1002")  # no span timing, so its verdict uses synthetic times
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    stub.add_response("evaluate", {"evaluationResults": [_result("ORD-1002")]})
    with stub:
        (label,) = run_agentcore_judge(_judge(), [trace], clients=clients, confirm=_approve)
    assert label.critique.endswith(f"[fingerprint={_judge().fingerprint}] [timing=synthetic]")


def test_a_failed_evaluate_call_is_sent_once_so_the_budget_counts_every_billed_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from _agentcore_runtime_fakes import SendCounter, zero_retry_backoff

    zero_retry_backoff(monkeypatch)
    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    # Answers every request on the wire with a 500 and counts the requests. A Stubber would
    # answer before the SDK's retry loop, so it could not show a retry.
    failing = SendCounter.error("InternalServerException", 500)
    session.events.register("before-send.*.*", failing)
    budget = BilledCallBudget()
    backend = LocalBackend(root=str(tmp_path / "store"))
    with AwsClients(session, handlers=[budget]) as clients, pytest.raises(AgentCoreError):
        run_agentcore_judge(
            _judge(), [_trace("t-1")], clients=clients, confirm=_approve, store=backend
        )
    assert len(failing.sends) == 1  # a 500 can come after the billed work, so it is not resent
    assert budget.used == 1
    assert backend.get_scores("t-1") == []


def test_a_call_with_no_result_for_its_trace_is_refused_as_incomplete_and_stores_nothing(
    tmp_path: Any,
) -> None:
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    stub.add_response("evaluate", {"evaluationResults": [_result("t-1")]})
    stub.add_response("evaluate", {"evaluationResults": []})
    backend = LocalBackend(root=str(tmp_path / "store"))
    with stub, pytest.raises(AgentCoreScoreIncomplete, match="'t-2'"):
        run_agentcore_judge(
            _judge(),
            [_trace("t-1"), _trace("t-2")],
            clients=clients,
            confirm=_approve,
            store=backend,
        )
    assert backend.get_scores("t-1") == []
    assert backend.get_scores("t-2") == []


def test_a_price_listed_in_an_unexpected_unit_gives_no_cost_and_names_the_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from calibrated_agent_evaluation.agentcore import pricing

    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.evaluate.lookup_evaluation_rates",
        pricing.lookup_evaluation_rates,
    )

    def entry(usagetype: str, unit: str, usd: str) -> str:
        dimension = {"unit": unit, "pricePerUnit": {"USD": usd}}
        return json.dumps(
            {
                "product": {"attributes": {"usagetype": usagetype, "regionCode": "us-east-1"}},
                "terms": {"OnDemand": {"T1": {"priceDimensions": {"D1": dimension}}}},
            }
        )

    prefix = "USE1-Evaluations:Consumption-based:"
    listing = [
        entry(f"{prefix}BuiltIn-Input:Tier1", "1K Input Tokens", "0.0024000000"),
        entry(f"{prefix}BuiltIn-Output:Tier1", "1M Output Tokens", "12.0000000000"),
    ]
    clients = _clients()
    prices = Stubber(clients.client("pricing", region="us-east-1"))
    prices.add_response("get_products", {"PriceList": listing, "FormatVersion": "aws_v1"})
    builtin = replace(_judge(), evaluator_type="Builtin")
    seen: list[RunEstimate] = []

    def decline(estimate: RunEstimate) -> bool:
        seen.append(estimate)
        return False

    with prices, pytest.raises(AgentCoreRunDeclined):
        run_agentcore_judge(builtin, [_trace("t-1")], clients=clients, confirm=decline)
    (estimate,) = seen
    assert estimate.cost_usd is None
    assert estimate.calls == 1
    assert estimate.input_tokens > 0
    assert estimate.no_rate_reason is not None
    assert "1K Input Tokens" in estimate.no_rate_reason


def test_a_span_time_without_a_utc_offset_refuses_the_run_with_no_billed_call() -> None:
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))  # no response queued: a call would fail
    trace = _trace("t-1")
    untimed = trace.model_copy(
        update={"spans": [trace.spans[0].model_copy(update={"started_at": "2026-01-01T00:00:00"})]}
    )
    with stub, pytest.raises(AgentCoreContentUnserializable, match="'t-1/agent'"):
        run_agentcore_judge(_judge(), [untimed], clients=clients, confirm=_approve)
    stub.assert_no_pending_responses()


def test_a_call_the_service_answered_with_an_error_is_counted_in_the_usage_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    stub.add_client_error(
        "evaluate", service_error_code="InternalServerException", http_status_code=500
    )
    with (
        caplog.at_level("INFO", logger="calibrated_agent_evaluation.agentcore.evaluate"),
        stub,
        pytest.raises(AgentCoreError),
    ):
        run_agentcore_judge(_judge(), [_trace("t-1")], clients=clients, confirm=_approve)
    assert any("over 1 billed call" in record.getMessage() for record in caplog.records)


def test_a_call_whose_response_timed_out_is_counted_in_the_usage_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from botocore.exceptions import ReadTimeoutError

    clients = _clients()

    def timed_out(**_kwargs: Any) -> None:
        raise ReadTimeoutError(endpoint_url="https://bedrock-agentcore.us-east-1.amazonaws.com")

    # Raised where the request has already been sent, as a read timeout is.
    clients.client("bedrock-agentcore").meta.events.register("before-send.*.*", timed_out)
    with (
        caplog.at_level("INFO", logger="calibrated_agent_evaluation.agentcore.evaluate"),
        pytest.raises(AgentCoreError),
    ):
        run_agentcore_judge(_judge(), [_trace("t-1")], clients=clients, confirm=_approve)
    assert any("over 1 billed call" in record.getMessage() for record in caplog.records)


# --- several calls at a time, a failure or throttle mid-run ---------------------------------------


class _Gate:
    """Answers ``Evaluate`` per request from botocore's own hooks, as the real service would.

    Each call is held until ``min(limit, traces not yet answered)`` calls are open, so a runner
    that sends up to ``limit`` at once is seen doing so; ``peak`` is the most calls ever open.
    ``fail`` maps a trace id to the error code its call is answered with; with ``hold_others``,
    a successful call is answered only after every failing call has been answered, so the order
    in which the runner sees them is fixed.
    """

    def __init__(
        self,
        clients: AwsClients,
        *,
        limit: int,
        total: int,
        fail: dict[str, str] | None = None,
        hold_others: bool = False,
        pause: float = 0.0,
        failure_seen: Any = None,
    ) -> None:
        import threading

        self.limit = limit
        self.total = total
        self.fail = fail or {}
        self.hold_others = hold_others
        self.pause = pause
        # Set once the failing call's error has left the runner's worker, so a call held behind
        # it is answered only after the runner can see the failure.
        self.failure_seen = failure_seen
        self.cond = threading.Condition()
        self.open = 0
        self.peak = 0
        self.answered = 0
        self.failed_answered = 0
        # Bumped each time the open calls reach their target, so a call waiting when it happened
        # is released even if another call ends before it wakes up.
        self.released = 0
        self.started: list[str] = []
        events = clients.client("bedrock-agentcore").meta.events
        events.register("before-parameter-build.bedrock-agentcore.Evaluate", self._capture)
        events.register("before-call.bedrock-agentcore.Evaluate", self._respond)
        self.by_aws_id: dict[str, str] = {}

    def expect(self, traces: list[Trace]) -> None:
        self.by_aws_id = {aws_trace_id(t.trace_id): t.trace_id for t in traces}

    def _capture(self, params: dict[str, Any], context: dict[str, Any], **_: Any) -> None:
        context["calibrated_agent_evaluation_target"] = params["evaluationTarget"]["traceIds"][0]

    def _respond(self, context: dict[str, Any], **_: Any) -> Any:
        import time

        from botocore.awsrequest import AWSResponse

        target = context["calibrated_agent_evaluation_target"]
        trace_id = self.by_aws_id[target]
        with self.cond:
            self.open += 1
            self.started.append(trace_id)
            self.peak = max(self.peak, self.open)
            entered = self.released
            if self.open >= min(self.limit, self.total - self.answered):
                self.released += 1
            self.cond.notify_all()
            reached = self.cond.wait_for(lambda: self.released > entered, timeout=5)
            assert reached, "the gate waited 5 s for calls that never came"
            if self.hold_others and trace_id not in self.fail:
                assert self.cond.wait_for(
                    lambda: self.failed_answered == len(self.fail), timeout=5
                ), "the failing call was never answered"
        if self.hold_others and trace_id not in self.fail and self.failure_seen is not None:
            assert self.failure_seen.wait(timeout=5), "the failing call never raised"
            # The worker has raised; give the pool a moment to mark its future done.
            time.sleep(0.05)
        time.sleep(self.pause)
        code = self.fail.get(trace_id)
        with self.cond:
            self.open -= 1
            self.answered += 1
            if code is not None:
                self.failed_answered += 1
            self.cond.notify_all()
        if code is not None:
            error = {"Error": {"Code": code, "Message": "refused by the gate"}}
            return AWSResponse("https://stub", 400, {}, None), error
        result = _result(trace_id)
        return AWSResponse("https://stub", 200, {}, None), {"evaluationResults": [result]}


def _budgeted_clients(budget: BilledCallBudget | None = None) -> AwsClients:
    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    handlers = [budget] if budget is not None else []
    return AwsClients(session, handlers=handlers).with_config(
        Config(retries={"total_max_attempts": 1})
    )


def test_a_limit_of_three_keeps_three_calls_in_flight_and_returns_verdicts_in_trace_order() -> None:
    budget = BilledCallBudget()
    clients = _budgeted_clients(budget)
    traces = [_trace(f"t-{i}") for i in range(7)]
    gate = _Gate(clients, limit=3, total=7)
    gate.expect(traces)
    seen: list[tuple[int, int]] = []

    labels = run_agentcore_judge(
        _judge(),
        traces,
        clients=clients,
        confirm=_approve,
        max_concurrency=3,
        progress=lambda scored, total: seen.append((scored, total)),
    )

    assert gate.peak == 3
    assert [label.trace_id for label in labels] == [t.trace_id for t in traces]
    assert seen == [(n, 7) for n in range(1, 8)]
    # Every call is counted once by the budget, though seven were sent from three threads.
    assert budget.used == 7


def test_with_no_limit_the_runner_sends_one_call_at_a_time() -> None:
    clients = _budgeted_clients()
    traces = [_trace(f"t-{i}") for i in range(4)]
    gate = _Gate(clients, limit=1, total=4, pause=0.05)
    gate.expect(traces)

    labels = run_agentcore_judge(_judge(), traces, clients=clients, confirm=_approve)

    assert gate.peak == 1
    assert [label.trace_id for label in labels] == [t.trace_id for t in traces]


def test_a_failed_call_stops_new_calls_counts_the_ones_in_flight_and_stores_nothing(
    caplog: pytest.LogCaptureFixture, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from calibrated_agent_evaluation.agentcore import evaluate as evaluate_module

    failure_seen = threading.Event()
    score_one = evaluate_module._score_one

    def signalling(judge: Any, request: Any, **kwargs: Any) -> Any:
        try:
            return score_one(judge, request, **kwargs)
        finally:
            if request[0].trace_id == "t-1":
                failure_seen.set()

    monkeypatch.setattr(evaluate_module, "_score_one", signalling)
    clients = _budgeted_clients()
    traces = [_trace(f"t-{i}") for i in range(5)]
    gate = _Gate(
        clients,
        limit=2,
        total=5,
        fail={"t-1": "ValidationException"},
        hold_others=True,
        failure_seen=failure_seen,
    )
    gate.expect(traces)
    backend = LocalBackend(root=str(tmp_path / "store"))

    with (
        caplog.at_level("INFO", logger="calibrated_agent_evaluation.agentcore.evaluate"),
        pytest.raises(AgentCoreError, match="ValidationException"),
    ):
        run_agentcore_judge(
            _judge(), traces, clients=clients, confirm=_approve, store=backend, max_concurrency=2
        )

    # The first two calls were in flight together; nothing started after the failure was seen.
    assert sorted(gate.started) == ["t-0", "t-1"]
    assert any("over 2 billed call" in record.getMessage() for record in caplog.records)
    assert all(backend.get_scores(t.trace_id) == [] for t in traces)


def test_a_throttled_single_call_says_to_wait_rather_than_lower_the_concurrency() -> None:
    clients = _budgeted_clients()
    gate = _Gate(clients, limit=1, total=1, fail={"t-1": "ThrottlingException"})
    gate.expect([_trace("t-1")])

    with pytest.raises(AgentCoreError) as raised:
        run_agentcore_judge(_judge(), [_trace("t-1")], clients=clients, confirm=_approve)

    message = str(raised.value)
    assert "ThrottlingException" in message
    assert "wait" in message
    assert "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_CONCURRENCY" not in message


def test_a_throttled_call_refuses_the_run_naming_the_concurrency_to_lower() -> None:
    clients = _budgeted_clients()
    traces = [_trace(f"t-{i}") for i in range(3)]
    gate = _Gate(clients, limit=3, total=3, fail={"t-2": "ThrottlingException"})
    gate.expect(traces)

    with pytest.raises(AgentCoreError) as raised:
        run_agentcore_judge(_judge(), traces, clients=clients, confirm=_approve, max_concurrency=3)

    message = str(raised.value)
    assert "ThrottlingException" in message
    assert "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_CONCURRENCY" in message
    assert "3 at a time" in message


@pytest.mark.parametrize("limit", [0, -1, 11])
def test_a_concurrency_outside_one_to_ten_is_refused_before_any_call(limit: int) -> None:
    clients = _budgeted_clients()
    gate = _Gate(clients, limit=1, total=1)
    gate.expect([_trace("t-1")])

    with pytest.raises(AgentCoreError, match=rf"max_concurrency={limit} .*1 to 10"):
        run_agentcore_judge(
            _judge(), [_trace("t-1")], clients=clients, confirm=_approve, max_concurrency=limit
        )
    assert gate.started == []


def _evaluate_views(clients: Any) -> list[Any]:
    """Every client view the run opens, each answering one ``Evaluate`` from a stub."""
    views: list[Any] = []
    real = clients.with_config

    def spy(config: Any) -> Any:
        view = real(config)
        views.append(view)
        stub = Stubber(view.client("bedrock-agentcore"))
        stub.add_response("evaluate", {"evaluationResults": [_result("t-1")]})
        stub.activate()
        return view

    clients.with_config = spy
    return views


def test_evaluate_waits_up_to_two_minutes_for_a_response_and_sends_each_call_once() -> None:
    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    with AwsClients(session) as clients:
        views = _evaluate_views(clients)
        (label,) = run_agentcore_judge(_judge(), [_trace("t-1")], clients=clients, confirm=_approve)

    assert label.trace_id == "t-1"
    (view,) = views
    config = view.client("bedrock-agentcore").meta.config
    assert config.read_timeout == 120
    assert config.retries.get("total_max_attempts") == 1


def test_clients_already_set_to_one_attempt_are_used_as_they_are_wait_included() -> None:
    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    own = Config(read_timeout=45, retries={"total_max_attempts": 1})
    with AwsClients(session).with_config(own) as clients:
        views = _evaluate_views(clients)
        stub = Stubber(clients.client("bedrock-agentcore"))
        stub.add_response("evaluate", {"evaluationResults": [_result("t-1")]})
        with stub:
            (label,) = run_agentcore_judge(
                _judge(), [_trace("t-1")], clients=clients, confirm=_approve
            )
            stub.assert_no_pending_responses()
        assert clients.client("bedrock-agentcore").meta.config.read_timeout == 45

    assert label.trace_id == "t-1"
    assert views == []
