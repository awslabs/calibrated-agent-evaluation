# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading an online config's verdicts back into a store, over a fake reader and stubbed logs."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")

from botocore.stub import ANY, Stubber

from calibrated_agent_evaluation.agentcore.judge import AgentCoreJudgeFile, judge_from_evaluator
from calibrated_agent_evaluation.agentcore.online import AttachedStatus
from calibrated_agent_evaluation.agentcore.online_record import (
    AttachedEvaluator,
    OnlineEvaluationRecord,
)
from calibrated_agent_evaluation.agentcore.settle import SettlePolicy
from calibrated_agent_evaluation.agentcore.sources import RawTrace
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.judges.verdict import verdict_rater_id
from calibrated_agent_evaluation.schema.errors import TraceNotFoundError
from calibrated_agent_evaluation.schema.types import Trace

_ACCOUNT = "111122223333"
_EVALUATOR_ID = "calibrated_agent_evaluation_live_refunds-AbCdEf1234"
_ARN = f"arn:aws:bedrock-agentcore:us-east-1:{_ACCOUNT}:evaluator/{_EVALUATOR_ID}"
_CONFIG_ID = "calibrated_agent_evaluation_live_rate-AbCdEf1234"
_RESULTS_GROUP = f"/aws/bedrock-agentcore/evaluations/results/{_CONFIG_ID}"
_MODE = "trusted_broken_policy_tool"
_SINCE = "2026-11-05T18:00:00+00:00"
# The time the tests read at: a day after the fixed window, so it is long past.
_NOW = datetime(2026, 11, 6, 19, 0, tzinfo=UTC)
_UNTIL = "2026-11-05T19:00:00+00:00"
_DOC = {
    "evaluatorArn": _ARN,
    "evaluatorId": _EVALUATOR_ID,
    "evaluatorName": "calibrated_agent_evaluation_live_refunds",
    "evaluatorType": "Custom",
    "level": "TRACE",
    "status": "ACTIVE",
    "createdAt": _SINCE,
    "updatedAt": _SINCE,
    "evaluatorConfig": {
        "llmAsAJudge": {
            "instructions": "Judge the refund. {context} {assistant_turn}",
            "ratingScale": {
                "numerical": [
                    {"value": 0.0, "label": "FAIL", "definition": "The refund breaks policy."},
                    {"value": 1.0, "label": "PASS", "definition": "The refund follows policy."},
                ]
            },
            "modelConfig": {"bedrockEvaluatorModelConfig": {"modelId": "anthropic.claude-x"}},
        }
    },
}


def _ns(iso: str) -> str:
    return str(int(datetime.fromisoformat(iso).timestamp() * 1_000_000_000))


def _raw(trace_id: str, session: str, ended: str = "2026-11-05T18:10:00+00:00") -> RawTrace:
    """One OrderBot turn as the span reader decodes it: an agent span with one chat child."""
    end = _ns(ended)
    return RawTrace(
        trace_id=trace_id,
        span_records=[
            {
                "spanId": f"{trace_id[:12]}0001",
                "traceId": trace_id,
                "startTimeUnixNano": end,
                "endTimeUnixNano": end,
                "attributes": {"gen_ai.operation.name": "invoke_agent", "session.id": session},
            },
            {
                "spanId": f"{trace_id[:12]}0002",
                "traceId": trace_id,
                "parentSpanId": f"{trace_id[:12]}0001",
                "startTimeUnixNano": end,
                "endTimeUnixNano": end,
                "attributes": {"gen_ai.operation.name": "chat", "session.id": session},
            },
        ],
        log_records=[
            {
                "spanId": f"{trace_id[:12]}0002",
                "eventName": "gen_ai.user.message",
                "body": {"content": [{"text": "a refund for Q-30000, please"}]},
            },
            {
                "spanId": f"{trace_id[:12]}0002",
                "eventName": "gen_ai.choice",
                "body": {"message": {"content": [{"text": "refund issued"}]}},
            },
        ],
        session_ids=(session,),
    )


class _Reader:
    def __init__(self, traces: list[RawTrace]) -> None:
        self._traces = traces

    def read_traces(self) -> Iterator[RawTrace]:
        yield from self._traces

    def read_scores(self) -> Iterator[dict[str, Any]]:
        return iter(())


def _result(trace_id: str, session: str, label: str = "FAIL") -> dict[str, Any]:
    """An online result record as AgentCore writes it to the config's result group (seen live)."""
    return {
        "name": "gen_ai.evaluation.result",
        "traceId": trace_id,
        "attributes": {
            "gen_ai.evaluation.name": "calibrated_agent_evaluation_live_refunds",
            "session.id": session,
            "gen_ai.response.id": trace_id,
            "gen_ai.evaluation.score.value": 0.0 if label == "FAIL" else 1.0,
            "gen_ai.evaluation.score.label": label,
            "gen_ai.evaluation.explanation": f"the refund was a {label}",
            "aws.bedrock_agentcore.evaluator.arn": _ARN,
            "aws.bedrock_agentcore.online_evaluation_config.name": (
                "calibrated_agent_evaluation_live_rate"
            ),
            "aws.bedrock_agentcore.evaluation_level": "Trace",
        },
        "onlineEvaluationConfigId": _CONFIG_ID,
    }


def _error(trace_id: str, session: str, error_type: str | None) -> dict[str, Any]:
    """A result AgentCore could not score, as it wrote one live."""
    attributes: dict[str, Any] = {
        "gen_ai.evaluation.name": "calibrated_agent_evaluation_live_refunds",
        "session.id": session,
        "gen_ai.response.id": session,
        "error": 1,
        "aws.bedrock_agentcore.evaluator.arn": _ARN,
    }
    if error_type is not None:
        attributes["error.type"] = error_type
        attributes["error.message"] = "none of the spans contain the required agent invocation"
    return {"name": "gen_ai.evaluation.result", "traceId": trace_id, "attributes": attributes}


def _judge() -> Any:
    return judge_from_evaluator("refunds", _DOC)


def _record(*, excluded: tuple[str, ...] = (), filtered: bool = False) -> OnlineEvaluationRecord:
    judge = _judge()
    return OnlineEvaluationRecord(
        config_id=_CONFIG_ID,
        config_name="calibrated_agent_evaluation_live_rate",
        runtime_service_name="calibrated_agent_evaluation_orderbot.DEFAULT",
        evaluators=(
            AttachedEvaluator(
                evaluator_id=_EVALUATOR_ID,
                evaluator_arn=_ARN,
                updated_at=judge.updated_at,
                fingerprint=judge.fingerprint,
                calib_id="calib-refunds",
                battery_id="bias-refunds-clear",
            ),
        ),
        excluded_session_ids=excluded,
        sampling_percentage=100.0,
        session_timeout_minutes=2,
        created_at=_SINCE,
        filtered=filtered,
    )


def _read_judges(state: str = "certified") -> dict[str, Any]:
    from calibrated_agent_evaluation.agentcore.readback import ReadJudge

    judge_file = AgentCoreJudgeFile(
        judge_id="refunds",
        evaluator_id=_EVALUATOR_ID,
        failure_mode=_MODE,
        agent=None,
        level="workflow",
        path=Path("judge.agentcore.json"),
    )
    status = AttachedStatus(_EVALUATOR_ID, state, f"evaluator {_EVALUATOR_ID} is {state}")  # type: ignore[arg-type]
    judge = None if state == "missing" else _judge()
    return {_EVALUATOR_ID: ReadJudge(judge_file=judge_file, judge=judge, status=status)}


def _rows(records: list[dict[str, Any]]) -> list[list[dict[str, str]]]:
    return [
        [{"field": "@message", "value": json.dumps(r)}, {"field": "@ptr", "value": f"ptr-{i}"}]
        for i, r in enumerate(records)
    ]


def _read(
    tmp_path: Path,
    traces: list[RawTrace],
    results: list[dict[str, Any]],
    *,
    record: OnlineEvaluationRecord | None = None,
    judges: dict[str, Any] | None = None,
    wait_minutes: int = 15,
    since: str = _SINCE,
    until: str = _UNTIL,
    backend: Any = None,
) -> tuple[Any, Any]:
    import boto3

    from calibrated_agent_evaluation.agentcore.readback import read_back
    from calibrated_agent_evaluation.agentcore.session import AwsClients

    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    clients = AwsClients(session)
    backend = backend if backend is not None else LocalBackend(root=str(tmp_path / "store"))
    with Stubber(clients.client("logs")) as logs:
        # The config's own result group, and no other group, is queried.
        logs.add_response(
            "start_query",
            {"queryId": "q-1"},
            {
                "logGroupName": _RESULTS_GROUP,
                "startTime": ANY,
                "endTime": ANY,
                "queryString": ANY,
            },
        )
        logs.add_response(
            "get_query_results",
            {
                "status": "Complete",
                "results": _rows(results),
                "statistics": {"recordsMatched": float(len(results))},
            },
            {"queryId": "q-1"},
        )
        if not results:
            # An empty Insights read is confirmed against the group itself before it is trusted.
            logs.add_response("filter_log_events", {"events": []}, _ANY_FILTER)
        report = read_back(
            record or _record(),
            judges or _read_judges(),
            clients=clients,
            backend=backend,
            since=since,
            until=until,
            unscored_after_minutes=wait_minutes,
            reader=_Reader(traces),
            settle=SettlePolicy(lag_seconds=0),
            clock=lambda: _NOW,
        )
        logs.assert_no_pending_responses()
    return report, backend


_T1, _T2, _T3 = ("1" * 32, "2" * 32, "3" * 32)
_ANY_FILTER = {"logGroupName": _RESULTS_GROUP, "startTime": ANY, "endTime": ANY}


@pytest.fixture(params=["local", "langfuse"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    if request.param == "local":
        return LocalBackend(root=str(tmp_path / "store"))
    from typing import cast

    from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend, LangfuseLike
    from test_backends_langfuse import StubLangfuseClient

    return LangfuseBackend(cast("LangfuseLike", StubLangfuseClient()))


def test_calibration_and_bias_sessions_are_skipped_and_the_rest_are_stored_as_judge_verdicts(
    tmp_path: Path, store: Any
) -> None:
    traces = [_raw(_T1, "s-live"), _raw(_T2, "s-calib"), _raw(_T3, "s-bias")]
    results = [_result(_T1, "s-live"), _result(_T2, "s-calib"), _result(_T3, "s-bias", "PASS")]

    report, backend = _read(
        tmp_path, traces, results, record=_record(excluded=("s-calib", "s-bias")), backend=store
    )

    assert report.skipped_sessions == 2
    # The library's own sessions are not the agent's traffic: their traces are not stored either.
    assert report.skipped_traces == 2
    assert [t.trace_id for t in report.traces] == [_T1]
    assert backend.get_trace(_T1).trace_id == _T1
    (label,) = report.stored
    assert label.trace_id == _T1
    assert label.kind == "llm-judge"
    assert label.outcome == "FAIL"
    # Stamped as the judge that made it, so the publish step accepts it.
    assert label.rater_id == verdict_rater_id(_judge())
    assert label.failure_mode == _MODE
    assert backend.get_scores(_T1) == [label]
    assert backend.get_scores(_T2) == []


def test_the_stored_verdicts_are_the_ones_publish_accepts(tmp_path: Path) -> None:
    import boto3

    from calibrated_agent_evaluation.agentcore.metrics import publish_pass_rate
    from calibrated_agent_evaluation.agentcore.session import AwsClients
    from cli_store_helpers import certified_calibration

    report, _ = _read(tmp_path, [_raw(_T1, "s-1")], [_result(_T1, "s-1")])
    judge = _judge()
    calib = certified_calibration(
        "refunds",
        model=judge.model,
        prompt_version=judge.prompt_version,
        fingerprint=judge.fingerprint,
        calib_id="calib-refunds",
    )
    clients = AwsClients(
        boto3.Session(
            region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
        )
    )
    with Stubber(clients.client("bedrock-agentcore-control")) as control:
        control.add_response("get_evaluator", _DOC, {"evaluatorId": _EVALUATOR_ID})
        published = publish_pass_rate(
            _record(),
            judge,
            calib,
            clients=clients,
            rate_labels=report.stored,
            gold_source="synthetic-rule",
        )

    # The verdicts are accepted as the judge's own; one verdict is just too few to publish yet.
    assert published.refusal is not None
    assert "too few online verdicts" in published.refusal


def test_traces_read_from_an_unfiltered_config_are_production_representative(
    tmp_path: Path,
) -> None:
    report, backend = _read(tmp_path, [_raw(_T1, "s-1")], [_result(_T1, "s-1")])

    assert [t.frame for t in report.traces] == ["production-representative"]
    assert backend.get_trace(_T1).frame == "production-representative"


def test_a_result_agentcore_could_not_score_stores_no_verdict_and_is_counted_by_type(
    tmp_path: Path,
) -> None:
    traces = [_raw(_T1, "s-1"), _raw(_T2, "s-2"), _raw(_T3, "s-3")]
    results = [
        _result(_T1, "s-1"),
        _error(_T2, "s-2", "ValidationException"),
        _error(_T3, "s-3", None),
    ]

    report, backend = _read(tmp_path, traces, results)

    assert [label.trace_id for label in report.stored] == [_T1]
    assert report.errors == {"ValidationException": 1, "unknown": 1}
    assert backend.get_scores(_T2) == []
    assert backend.get_scores(_T3) == []


@pytest.mark.parametrize("state", ["missing", "changed"])
def test_an_uncertified_evaluators_results_are_counted_and_not_stored(
    tmp_path: Path, state: str
) -> None:
    report, backend = _read(
        tmp_path, [_raw(_T1, "s-1")], [_result(_T1, "s-1")], judges=_read_judges(state)
    )

    assert report.stored == ()
    assert report.not_stored == {_EVALUATOR_ID: 1}
    assert backend.get_scores(_T1) == []


def _ago(minutes: int) -> str:
    return (_NOW - timedelta(minutes=minutes)).isoformat()


def test_a_session_past_its_timeout_and_wait_with_no_result_is_counted_unscored(
    tmp_path: Path,
) -> None:
    # A 2-minute session timeout and a 15-minute wait: a session that ended 50 minutes ago should
    # have its result by now; one that ended 3 minutes ago still has time.
    traces = [
        _raw(_T1, "s-scored", ended=_ago(55)),
        _raw(_T2, "s-late", ended=_ago(50)),
        _raw(_T3, "s-recent", ended=_ago(3)),
    ]

    report, _ = _read(tmp_path, traces, [_result(_T1, "s-scored")], since=_ago(60), until=_ago(1))

    assert report.unscored_sessions == ("s-late",)
    assert report.pending_sessions == ("s-recent",)
    assert report.every_session_scored


def test_a_session_near_a_past_windows_end_is_judged_on_results_after_the_window(
    tmp_path: Path,
) -> None:
    # The window ended at 19:00; a session that ended at 18:55 is scored after 19:00, so its
    # result is read past the window's end and its absence counts.
    traces = [_raw(_T1, "s-late", ended="2026-11-05T18:55:00+00:00")]

    report, _ = _read(tmp_path, traces, [])

    assert report.unscored_sessions == ("s-late",)
    assert report.pending_sessions == ()


def test_re_reading_the_same_window_adds_no_duplicate_verdict(tmp_path: Path) -> None:
    _read(tmp_path, [_raw(_T1, "s-1")], [_result(_T1, "s-1")])

    report, backend = _read(tmp_path, [_raw(_T1, "s-1")], [_result(_T1, "s-1")])

    assert report.stored == ()
    assert report.already_stored == 1
    assert len(backend.get_scores(_T1)) == 1
    # The verdict found already stored is still named, so a publish counts it.
    from calibrated_agent_evaluation.agentcore.online_record import verdict_digest

    (named,) = report.verdicts
    (held,) = backend.get_scores(_T1)
    assert (named.evaluator_id, named.trace_id) == (_EVALUATOR_ID, _T1)
    assert named.verdict_digest == verdict_digest(held)


def test_a_config_with_no_results_yet_reads_back_nothing_rather_than_failing(
    tmp_path: Path,
) -> None:
    import boto3

    from calibrated_agent_evaluation.agentcore.readback import read_back
    from calibrated_agent_evaluation.agentcore.session import AwsClients

    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    clients = AwsClients(session)
    with Stubber(clients.client("logs")) as logs:
        # AgentCore creates the result group with the config's first result.
        logs.add_client_error(
            "start_query", service_error_code="ResourceNotFoundException", http_status_code=400
        )
        report = read_back(
            _record(),
            _read_judges(),
            clients=clients,
            backend=LocalBackend(root=str(tmp_path / "store")),
            since=_ago(30),
            until=_ago(1),
            reader=_Reader([_raw(_T1, "s-1", ended=_ago(2))]),
            settle=SettlePolicy(lag_seconds=0),
            clock=lambda: _NOW,
        )

    assert report.stored == ()
    assert report.pending_sessions == ("s-1",)


def test_traces_read_from_a_filtered_config_keep_the_frame_their_source_gave(
    tmp_path: Path,
) -> None:
    # A filtered config scores only the sessions its filters admit: not a random sample.
    report, _ = _read(
        tmp_path, [_raw(_T1, "s-1")], [_result(_T1, "s-1")], record=_record(filtered=True)
    )

    # The fake reader's traces carry no frame, so they keep the ingest's default.
    assert [t.frame for t in report.traces] == ["discovery"]


def test_a_result_that_cannot_be_read_does_not_count_its_session_as_scored(
    tmp_path: Path,
) -> None:
    stranger = _result(_T1, "s-1")
    stranger["attributes"]["aws.bedrock_agentcore.evaluator.arn"] = "arn:aws:other"

    report, _ = _read(tmp_path, [_raw(_T1, "s-1")], [stranger])

    assert report.unreadable == 1
    assert report.unscored_sessions == ("s-1",)


class _LaggingStore(LocalBackend):
    """A store whose written traces are not readable yet, as Langfuse's are for a few seconds."""

    def get_trace(self, trace_id: str) -> Trace:
        raise TraceNotFoundError(f"no trace with trace_id {trace_id!r} yet")


def test_a_store_that_shows_written_traces_only_later_still_reads_them_all_back(
    tmp_path: Path,
) -> None:
    traces = [_raw(_T1, "s-live"), _raw(_T2, "s-other")]
    results = [_result(_T1, "s-live"), _result(_T2, "s-other", "PASS")]

    report, _ = _read(
        tmp_path, traces, results, backend=_LaggingStore(root=str(tmp_path / "lagging"))
    )

    assert sorted(t.trace_id for t in report.traces) == sorted([_T1, _T2])
    assert sorted((lb.trace_id, lb.outcome) for lb in report.stored) == [
        (_T1, "FAIL"),
        (_T2, "PASS"),
    ]
