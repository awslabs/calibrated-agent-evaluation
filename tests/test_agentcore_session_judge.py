# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judging simulated conversations with an AgentCore session-level evaluator."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from botocore.stub import ANY, Stubber

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreAmbiguousSessionResult,
    AgentCoreBlankAssertion,
    AgentCoreError,
    AgentCoreEvaluateLimitExceeded,
    AgentCoreEvaluatorLevelUnsupported,
    AgentCoreEvaluatorResultError,
    AgentCoreNotAConversation,
    AgentCoreReferenceDropped,
    AgentCoreResultUnaddressable,
    AgentCoreRunDeclined,
    AgentCoreScoreIncomplete,
    AgentCoreScoreOffScale,
    BilledCallBudgetExceeded,
)
from calibrated_agent_evaluation.agentcore.evaluate import run_agentcore_judge
from calibrated_agent_evaluation.agentcore.evaluate_payload import (
    MAX_ASSERTIONS,
    aws_trace_id,
    conversation_session_id,
    conversation_to_evaluation_input,
)
from calibrated_agent_evaluation.agentcore.guards import BilledCallBudget
from calibrated_agent_evaluation.agentcore.judge import (
    AgentCoreJudge,
    judge_from_evaluator,
    load_agentcore_judge_file,
)
from calibrated_agent_evaluation.agentcore.pricing import EvaluationRates, RunEstimate
from calibrated_agent_evaluation.agentcore.session import AwsClients
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.schema.types import Span, ToolCall, Trace

_REPO = Path(__file__).resolve().parent.parent
_GOAL_SUCCESS = json.loads(
    (
        _REPO / "fixtures" / "raw" / "agentcore" / "evaluate" / "goal-success-evaluator.json"
    ).read_text()
)
_JUDGE_FILE = _REPO / "fixtures" / "judge-files" / "agentcore-goal-success.json"
_ASSERTION = "Agent denies a refund outside the return window"
_RATES = EvaluationRates(
    region="us-west-2",
    builtin_input_per_million=Decimal("2.40"),
    builtin_output_per_million=Decimal("12.00"),
    custom_per_evaluation=Decimal("0.0015"),
)


@pytest.fixture(autouse=True)
def _published_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.evaluate.lookup_evaluation_rates", lambda _c: _RATES
    )


def _approve(_estimate: RunEstimate) -> bool:
    return True


def _clients(budget: BilledCallBudget | None = None) -> AwsClients:
    session = boto3.Session(
        region_name="us-west-2", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    handlers = [budget] if budget is not None else []
    return AwsClients(session, handlers=handlers).with_config(
        Config(retries={"total_max_attempts": 1})
    )


def _judge() -> AgentCoreJudge:
    return judge_from_evaluator("agentcore-goal-success", _GOAL_SUCCESS, level="session")


def _conversation(trace_id: str, *, assertions: list[str] | None = None, turns: int = 2) -> Trace:
    """A simulated conversation of ``turns`` agent turns, each one span re-keyed ``t<n>:``."""
    spans = []
    transcript: list[dict[str, str]] = []
    for n in range(1, turns + 1):
        user, agent = f"customer message {n}", f"agent reply {n}"
        transcript += [{"role": "user", "text": user}, {"role": "assistant", "text": agent}]
        tools = (
            [ToolCall(name="lookup_order", args={"order_id": "A-7781"}, result={"days": 51})]
            if n == 1
            else []
        )
        spans.append(
            Span(
                span_id=f"t{n}:orderbot",
                trace_id=trace_id,
                agent="orderbot",
                role="supervisor",
                parent=None,
                seq=n - 1,
                status="completed",
                input={"prompt": user},
                output={"text": agent},
                tool_calls=tools,
            )
        )
    return Trace(
        trace_id=trace_id,
        frame="discovery",
        ownership="owned",
        input={
            "conversation": transcript,
            "scenario_id": trace_id,
            "goal": "Get a refund for order A-7781.",
            "assertions": [_ASSERTION] if assertions is None else assertions,
            "stop_reason": "goal_completed",
            "turns": turns,
            "simulator": "strands",
        },
        output={"text": transcript[-1]["text"]},
        taxonomy_cell={},
        spans=spans,
    )


def _result(trace: Trace, *, value: float, label: str, **extra: Any) -> dict[str, Any]:
    return {
        "evaluatorArn": "arn:aws:bedrock-agentcore:::evaluator/Builtin.GoalSuccessRate",
        "evaluatorId": "Builtin.GoalSuccessRate",
        "evaluatorName": "Builtin.GoalSuccessRate",
        "explanation": "the customer's goal was judged against the assertion",
        "value": value,
        "label": label,
        "context": {"spanContext": {"sessionId": conversation_session_id(trace)}},
        "tokenUsage": {"inputTokens": 900, "outputTokens": 150, "totalTokens": 1050},
        **extra,
    }


def _expected(trace: Trace) -> dict[str, Any]:
    """The request a session call must be: no target, the assertions on the session."""
    params: dict[str, Any] = {"evaluatorId": "Builtin.GoalSuccessRate", "evaluationInput": ANY}
    if trace.input["assertions"]:
        params["evaluationReferenceInputs"] = [
            {
                "context": {"spanContext": {"sessionId": conversation_session_id(trace)}},
                "assertions": [{"text": a} for a in trace.input["assertions"]],
            }
        ]
    return params


# --- the judge file and the evaluator's level --------------------------------------------------


def test_a_session_judge_file_builds_over_a_session_evaluator() -> None:
    judge_file = load_agentcore_judge_file(_JUDGE_FILE)
    assert judge_file.level == "session"
    judge = judge_from_evaluator(judge_file.judge_id, _GOAL_SUCCESS, level=judge_file.level)
    assert judge.level == "session"
    assert judge.evaluator_level == "SESSION"
    # A workflow judge over a TRACE evaluator still builds.
    trace_level = {**_GOAL_SUCCESS, "evaluatorId": "Builtin.Correctness", "level": "TRACE"}
    assert judge_from_evaluator("c", trace_level).level == "workflow"


@pytest.mark.parametrize(
    ("level", "evaluator_level", "named"),
    [
        ("session", "TRACE", "a TRACE evaluator, and a session judge runs SESSION"),
        ("workflow", "SESSION", "a SESSION evaluator, and a workflow judge runs TRACE"),
        ("span", "SESSION", "judge level 'span' runs no AgentCore evaluator"),
    ],
)
def test_a_judge_file_level_must_match_its_evaluator(
    level: Any, evaluator_level: str, named: str
) -> None:
    with pytest.raises(AgentCoreEvaluatorLevelUnsupported, match=named):
        judge_from_evaluator("j", {**_GOAL_SUCCESS, "level": evaluator_level}, level=level)


@pytest.mark.parametrize("level", ["span", ["session"], 3])
def test_a_judge_file_naming_another_level_is_refused(tmp_path: Path, level: Any) -> None:
    path = tmp_path / "bad.json"
    doc = json.loads(_JUDGE_FILE.read_text())
    path.write_text(json.dumps({**doc, "level": level}))
    with pytest.raises(AgentCoreEvaluatorLevelUnsupported, match="'workflow'.*or 'session'"):
        load_agentcore_judge_file(path)


# --- the session payload -----------------------------------------------------------------------


def test_a_conversation_is_sent_as_one_session_of_turns() -> None:
    trace = _conversation("ORD-1001-sim")
    payload, references = conversation_to_evaluation_input(trace)
    documents = payload["sessionSpans"]
    session = conversation_session_id(trace)
    assert {d["attributes"]["session.id"] for d in documents} == {session}
    turn_1 = aws_trace_id("ORD-1001-sim#t1")
    turn_2 = aws_trace_id("ORD-1001-sim#t2")
    # Turn 1 is its span and its one tool call's document; turn 2 is its span alone.
    assert [d["traceId"] for d in documents] == [turn_1, turn_1, turn_2]
    assert references == [
        {"context": {"spanContext": {"sessionId": session}}, "assertions": [{"text": _ASSERTION}]}
    ]


def test_a_conversation_over_an_evaluate_limit_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exactly = _conversation("c", assertions=[f"assertion {i}" for i in range(MAX_ASSERTIONS)])
    _payload, [reference] = conversation_to_evaluation_input(exactly)
    assert len(reference["assertions"]) == MAX_ASSERTIONS
    over = _conversation("c", assertions=[f"assertion {i}" for i in range(MAX_ASSERTIONS + 1)])
    with pytest.raises(AgentCoreEvaluateLimitExceeded, match="101 assertions"):
        conversation_to_evaluation_input(over)
    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.evaluate_payload.MAX_SESSION_SPANS", 2
    )
    with pytest.raises(AgentCoreEvaluateLimitExceeded, match="3 span documents"):
        conversation_to_evaluation_input(_conversation("c"))


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_a_blank_assertion_is_refused_before_any_call(blank: str) -> None:
    trace = _conversation("ORD-1001-sim", assertions=[_ASSERTION, blank])
    with pytest.raises(AgentCoreBlankAssertion, match="'ORD-1001-sim' has a blank assertion"):
        conversation_to_evaluation_input(trace)
    clients = _clients()
    with (
        Stubber(clients.client("bedrock-agentcore")) as stub,
        pytest.raises(AgentCoreBlankAssertion),
    ):
        run_agentcore_judge(_judge(), [trace], clients=clients, confirm=_approve)
    stub.assert_no_pending_responses()
    # A blank assertion is not mistaken for a trace that is not a conversation.
    assert not issubclass(AgentCoreBlankAssertion, AgentCoreNotAConversation)


# --- running the session judge ----------------------------------------------------------------


def test_a_session_judge_gives_one_verdict_per_conversation(tmp_path: Path) -> None:
    met, missed = _conversation("ORD-1002-sim"), _conversation("ORD-1001-sim")
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    stub.add_response(
        "evaluate", {"evaluationResults": [_result(met, value=1.0, label="Yes")]}, _expected(met)
    )
    stub.add_response(
        "evaluate",
        {"evaluationResults": [_result(missed, value=0.0, label="No")]},
        _expected(missed),
    )
    with stub, LocalBackend(root=tmp_path) as store:
        labels = run_agentcore_judge(
            _judge(), [met, missed], clients=clients, confirm=_approve, store=store
        )
        stored = {t: store.get_scores(t) for t in ("ORD-1002-sim", "ORD-1001-sim")}
    stub.assert_no_pending_responses()
    by_trace = {label.trace_id: label for label in labels}
    assert by_trace["ORD-1002-sim"].outcome == "PASS"
    assert by_trace["ORD-1001-sim"].outcome == "FAIL"
    assert all(label.span_id is None and label.kind == "llm-judge" for label in labels)
    assert all(f"[fingerprint={_judge().fingerprint}]" in label.critique for label in labels)
    assert all(len(found) == 1 for found in stored.values())


def test_a_declined_session_run_makes_no_call() -> None:
    clients = _clients()
    with Stubber(clients.client("bedrock-agentcore")) as stub, pytest.raises(AgentCoreRunDeclined):
        run_agentcore_judge(
            _judge(), [_conversation("c")], clients=clients, confirm=lambda _e: False
        )
    stub.assert_no_pending_responses()


def test_a_conversation_with_no_assertions_is_sent_without_a_reference_input() -> None:
    trace = _conversation("ORD-1002-sim", assertions=[])
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    stub.add_response(
        "evaluate",
        {"evaluationResults": [_result(trace, value=1.0, label="Yes")]},
        _expected(trace),
    )
    with stub:
        [label] = run_agentcore_judge(_judge(), [trace], clients=clients, confirm=_approve)
    assert label.outcome == "PASS"


def test_a_session_judge_refuses_a_trace_that_is_not_a_conversation() -> None:
    single = _conversation("single")
    single = single.model_copy(update={"input": {"prompt": "refund please"}})
    clients = _clients()
    with (
        Stubber(clients.client("bedrock-agentcore")) as stub,
        pytest.raises(AgentCoreNotAConversation, match="'single' is not a simulated conversation"),
    ):
        run_agentcore_judge(_judge(), [single], clients=clients, confirm=_approve)
    stub.assert_no_pending_responses()


@pytest.mark.parametrize(
    ("extra", "error"),
    [
        (
            {"errorCode": "LogEventMissingException", "errorMessage": "x"},
            AgentCoreEvaluatorResultError,
        ),
        ({"ignoredReferenceInputFields": ["assertions"]}, AgentCoreReferenceDropped),
        (
            {"context": {"spanContext": {"sessionId": "f" * 32}}},
            AgentCoreResultUnaddressable,
        ),
        ({"value": None, "label": None}, AgentCoreScoreIncomplete),
    ],
)
def test_a_bad_session_result_stores_nothing(
    tmp_path: Path, extra: dict[str, Any], error: type[Exception]
) -> None:
    good, bad = _conversation("good"), _conversation("bad")
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    stub.add_response("evaluate", {"evaluationResults": [_result(good, value=1.0, label="Yes")]})
    # A field the service leaves out (here: no value and no label) is dropped, not sent as null.
    result = {
        k: v for k, v in {**_result(bad, value=0.0, label="No"), **extra}.items() if v is not None
    }
    stub.add_response("evaluate", {"evaluationResults": [result]})
    with stub, LocalBackend(root=tmp_path) as store, pytest.raises(error):
        run_agentcore_judge(_judge(), [good, bad], clients=clients, confirm=_approve, store=store)
    with LocalBackend(root=tmp_path) as store:
        assert store.get_scores("good") == []


def test_a_session_call_answered_with_two_results_stores_nothing(tmp_path: Path) -> None:
    good, doubled = _conversation("good"), _conversation("doubled")
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    stub.add_response("evaluate", {"evaluationResults": [_result(good, value=1.0, label="Yes")]})
    stub.add_response(
        "evaluate",
        {
            "evaluationResults": [
                _result(doubled, value=1.0, label="Yes"),
                _result(doubled, value=0.0, label="No"),
            ]
        },
    )
    with (
        stub,
        LocalBackend(root=tmp_path) as store,
        pytest.raises(AgentCoreAmbiguousSessionResult, match="'doubled' returned 2 results"),
    ):
        run_agentcore_judge(
            _judge(), [good, doubled], clients=clients, confirm=_approve, store=store
        )
    with LocalBackend(root=tmp_path) as store:
        assert store.get_scores("good") == []
        assert store.get_scores("doubled") == []


def test_an_evaluate_failure_mid_run_stores_nothing(tmp_path: Path) -> None:
    good, bad = _conversation("good"), _conversation("bad")
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    stub.add_response("evaluate", {"evaluationResults": [_result(good, value=1.0, label="Yes")]})
    stub.add_client_error(
        "evaluate", service_error_code="ThrottlingException", http_status_code=429
    )
    with (
        stub,
        LocalBackend(root=tmp_path) as store,
        pytest.raises(AgentCoreError, match="throttled"),
    ):
        run_agentcore_judge(_judge(), [good, bad], clients=clients, confirm=_approve, store=store)
    with LocalBackend(root=tmp_path) as store:
        assert store.get_scores("good") == []


def test_a_session_run_over_budget_is_refused_before_confirmation() -> None:
    asked: list[RunEstimate] = []

    def confirm(estimate: RunEstimate) -> bool:
        asked.append(estimate)
        return True

    clients = _clients(BilledCallBudget(1))
    with (
        Stubber(clients.client("bedrock-agentcore")) as stub,
        pytest.raises(BilledCallBudgetExceeded),
    ):
        run_agentcore_judge(
            _judge(), [_conversation("a"), _conversation("b")], clients=clients, confirm=confirm
        )
    assert asked == []
    stub.assert_no_pending_responses()


def test_a_failed_session_call_is_sent_once() -> None:
    session = boto3.Session(
        region_name="us-west-2", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    clients = AwsClients(session)
    sent: list[str] = []

    def count(**kwargs: Any) -> None:
        sent.append(kwargs["model"].name)
        raise ClientError({"Error": {"Code": "InternalServerException"}}, "Evaluate")

    with clients.with_config(Config(retries={"total_max_attempts": 1})) as view:
        view.client("bedrock-agentcore").meta.events.register("before-call.*.Evaluate", count)
        with pytest.raises(AgentCoreError):
            run_agentcore_judge(_judge(), [_conversation("c")], clients=view, confirm=_approve)
    assert sent == ["Evaluate"]


# --- the trace-level judge and the scale ------------------------------------------------------


def test_trace_level_judging_is_unchanged() -> None:
    trace = _conversation("c")
    judge = AgentCoreJudge(
        judge_id="correctness",
        evaluator_id="correctness",
        evaluator_arn="arn:aws:bedrock-agentcore:us-west-2:111122223333:evaluator/correctness",
        evaluator_type="Custom",
        updated_at="2026-10-01T00:00:00+00:00",
        config_digest="0123456789abcdef",
        evaluator_level="TRACE",
    )
    derived = aws_trace_id("c")
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    stub.add_response(
        "evaluate",
        {
            "evaluationResults": [
                {
                    "evaluatorArn": judge.evaluator_arn,
                    "evaluatorId": "correctness",
                    "evaluatorName": "correctness",
                    "value": 1.0,
                    "label": "correct",
                    "context": {"spanContext": {"sessionId": derived, "traceId": derived}},
                }
            ]
        },
        {
            "evaluatorId": "correctness",
            "evaluationInput": ANY,
            "evaluationTarget": {"traceIds": [derived]},
        },
    )
    with stub:
        [label] = run_agentcore_judge(judge, [trace], clients=clients, confirm=_approve)
    assert label.trace_id == "c"


def test_tool_call_evaluators_are_still_refused() -> None:
    with pytest.raises(AgentCoreEvaluatorLevelUnsupported, match="TOOL_CALL"):
        judge_from_evaluator("t", {**_GOAL_SUCCESS, "level": "TOOL_CALL"}, level="session")


def test_goal_success_values_map_to_pass_and_fail() -> None:
    trace = _conversation("c")
    for value, label, outcome in ((1.0, "Yes", "PASS"), (0.0, "No", "FAIL")):
        clients = _clients()
        stub = Stubber(clients.client("bedrock-agentcore"))
        stub.add_response(
            "evaluate", {"evaluationResults": [_result(trace, value=value, label=label)]}
        )
        with stub:
            [verdict] = run_agentcore_judge(_judge(), [trace], clients=clients, confirm=_approve)
        assert verdict.outcome == outcome


def test_an_off_scale_goal_success_value_is_refused() -> None:
    trace = _conversation("c")
    clients = _clients()
    stub = Stubber(clients.client("bedrock-agentcore"))
    stub.add_response("evaluate", {"evaluationResults": [_result(trace, value=0.5, label="Maybe")]})
    with stub, pytest.raises(AgentCoreScoreOffScale):
        run_agentcore_judge(_judge(), [trace], clients=clients, confirm=_approve)
