# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading one AgentCore Runtime run's span tree by id, as the executor's live span reader."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import pytest

from _agentcore_capture import captured_run
from calibrated_agent_evaluation.agentcore import span_reader
from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreContentMissing,
    AgentCoreError,
    AgentCoreMixedSession,
    AgentCorePartialResult,
    AgentCoreSettlingLag,
    AgentCoreTruncatedResult,
    BilledCallBudgetExceeded,
)
from calibrated_agent_evaluation.agentcore.run_spans import RunSpanQuery
from calibrated_agent_evaluation.agentcore.sources import AgentCoreSourceConfig, TimeWindow
from calibrated_agent_evaluation.agentcore.span_reader import AgentCoreSpanReader

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.session import AwsClients

_RUN = captured_run()["trace_id"]  # the captured run's id (also a valid 32-hex W3C id)
# The captured run's one agent and its last model call, which carries its answer.
_AGENT_SPAN = "78f80f5902b08d48"
_FINAL_CALL = "6752ac8ae589d9a1"


@pytest.fixture(autouse=True)
def _fake_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the poll loop on a fake clock: each sleep advances time, so deadlines come at once."""
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + max(s, 0.5)))


def _fixture_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The captured run's span and content records, its content only in the content records.

    The runtime also wrote each model call's messages as span events. Those are dropped here, so a
    test that holds back content records holds back the content itself.
    """
    run = captured_run()
    spans = [{k: v for k, v in s.items() if k != "events"} for s in run["span_records"]]
    return spans, run["log_records"]


def _query(*, w3c: str = _RUN, trace_id: str = "RUN-1", session_id: str = "sess-1") -> RunSpanQuery:
    return RunSpanQuery(
        trace_id=trace_id,
        w3c_trace_id=w3c,
        session_id=session_id,
        runtime_arn="arn:aws:bedrock-agentcore:us-east-1:111122223333:runtime/agent-1",
        qualifier="DEFAULT",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T01:00:00+00:00",
    )


def _without_final_answer(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The run's content with the agent's final model call not yet indexed (observed live)."""
    return [r for r in content if r.get("spanId") != _FINAL_CALL]


class _Scripted:
    """A by-id read that returns each scripted result in turn, then repeats the last one."""

    def __init__(self, reads: list[tuple[Any, Any, set[str]]]) -> None:
        self._reads = list(reads)
        self.calls = 0

    def __call__(self, _config: Any, _w3c: str) -> tuple[Any, Any, set[str]]:
        self.calls += 1
        return self._reads.pop(0) if len(self._reads) > 1 else self._reads[0]


def _http_only(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only the runtime's HTTP span: what lands first, before any agent span."""
    return [s for s in spans if s.get("name") == "POST /invocations"]


def _config() -> AgentCoreSourceConfig:
    return AgentCoreSourceConfig(
        runtime="agent-1",
        window=TimeWindow(start="2026-01-01T00:00:00+00:00", end="2026-01-01T01:00:00+00:00"),
    )


class _FakeClients:
    """Records which services were asked for and hands back one stand-in client per service."""

    def __init__(self) -> None:
        self.requested: list[str] = []
        self._client = object()

    def client(self, service_name: str) -> Any:
        self.requested.append(service_name)
        return self._client


def _reader(clients: Any = None, **kwargs: Any) -> AgentCoreSpanReader:
    return AgentCoreSpanReader(cast("AwsClients", clients or _FakeClients()), **kwargs)


# --- reading one run's whole tree by id ---------------------------------------------------


def test_it_returns_the_whole_normalized_tree_keyed_to_the_library_trace_id() -> None:
    spans, content = _fixture_records()
    reader = _reader()
    # The AWS rows carry the W3C id; the returned spans must all carry the Calibrated Agent
    # Evaluation trace id.
    reader._read_once = lambda _config, _w3c: (spans, content, {"sess-1"})  # type: ignore[method-assign]

    result = reader.read_spans(_query())

    assert len(result) == 1
    assert {s.trace_id for s in result} == {"RUN-1"}
    # An owned tree with recovered content proves the decode + content re-home + normalize reuse.
    assert any(span.input or span.output for span in result)


def test_the_logs_insights_read_goes_through_the_injected_clients_with_the_trace_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClients()
    reader = _reader(fake)
    seen: dict[str, Any] = {}

    def fake_li(
        client: Any, group: str, _config: Any, *, trace_id: str | None = None, **_kw: Any
    ) -> list[dict[str, Any]]:
        seen.setdefault("clients", []).append(client)
        seen.setdefault("trace_ids", []).append(trace_id)
        if group != span_reader._SPANS_LOG_GROUP:
            return []
        return [{"traceId": _RUN, "attributes": {"session.id": "sess-1"}}]

    def fake_store(client: Any, group: str, _config: Any, trace_id: str) -> list[dict[str, Any]]:
        seen.setdefault("store_clients", []).append(client)
        seen.setdefault("store_trace_ids", []).append(trace_id)
        return fake_li(client, group, _config, trace_id=trace_id)

    monkeypatch.setattr(span_reader, "run_logs_insights", fake_li)
    monkeypatch.setattr(span_reader, "read_trace_events", fake_store)

    _spans, _content, sessions = reader._read_logs_insights(_config(), _RUN)

    assert sessions == {"sess-1"}
    assert all(c is fake._client for c in seen["clients"])  # every read used the injected clients
    assert set(seen["trace_ids"]) == {_RUN}  # and every read was a by-id query
    assert all(c is fake._client for c in seen["store_clients"])
    assert set(seen["store_trace_ids"]) == {_RUN}
    assert "logs" in fake.requested


def test_a_malformed_w3c_trace_id_is_refused_before_any_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("no AWS read should happen for a malformed id")

    monkeypatch.setattr(span_reader, "run_logs_insights", explode)
    monkeypatch.setattr(span_reader, "run_omni_query", explode)

    with pytest.raises(AgentCoreError):
        _reader().read_spans(_query(w3c="not-a-32-hex-id'; DROP"))


# --- path selection ------------------------------------------------------------------------


def test_use_omni_reads_through_omni_and_the_default_reads_through_logs_insights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    used: list[str] = []
    monkeypatch.setattr(
        span_reader, "run_omni_query", lambda *_a, **_k: (used.append("omni"), ([], {}))[1]
    )
    monkeypatch.setattr(
        span_reader, "run_logs_insights", lambda *_a, **_k: (used.append("logs"), [])[1]
    )
    monkeypatch.setattr(
        span_reader, "read_trace_events", lambda *_a, **_k: (used.append("store"), [])[1]
    )

    _reader(use_omni=True)._read_once(_config(), _RUN)
    assert set(used) == {"omni"}  # the Omni path reads only through Omni
    used.clear()
    _reader(use_omni=False)._read_once(_config(), _RUN)
    # The default path reads through Logs Insights, checked against the CloudWatch Logs store.
    assert set(used) == {"logs", "store"}


# --- a whole tree or a typed refusal, never partial ---------------------------------------


def test_a_tree_still_growing_is_not_returned_until_it_is_stable() -> None:
    spans, content = _fixture_records()
    reads = iter(
        [
            (spans[:7], content, {"sess-1"}),  # partial — must not be returned
            (spans, content, {"sess-1"}),  # grew
            (spans, content, {"sess-1"}),  # stable → whole
        ]
    )
    reader = _reader(poll_interval_seconds=1.0)
    reader._read_once = lambda _config, _w3c: next(reads)  # type: ignore[method-assign]

    result = reader.read_spans(_query())

    assert len(result) == 1
    assert len(result[0].tool_calls) == 3  # the complete tree, not the 7-span partial


def test_a_tree_that_never_appears_is_refused_with_settling_lag() -> None:
    reader = _reader(settle_deadline_seconds=1.0, poll_interval_seconds=1.0)
    reader._read_once = lambda _config, _w3c: ([], [], set())  # type: ignore[method-assign]

    with pytest.raises(AgentCoreSettlingLag):
        reader.read_spans(_query())


def test_no_spans_within_the_deadline_raises_rather_than_returning_an_empty_list() -> None:
    reader = _reader(settle_deadline_seconds=1.0, poll_interval_seconds=1.0)
    reader._read_once = lambda _config, _w3c: ([], [], set())  # type: ignore[method-assign]

    with pytest.raises(AgentCoreError):  # a typed error, never an empty return
        reader.read_spans(_query())


def test_a_run_that_normalizes_to_no_agent_span_tree_is_refused() -> None:
    # A short-circuited run (e.g. an auth-rejected invocation) writes only a POST /invocations
    # span; it normalizes to a probed trace with no spans. The reader must raise, not return [].
    http_only = {
        "traceId": _RUN,
        "spanId": "aa" * 8,
        "parentSpanId": "",
        "name": "POST /invocations",
        "startTimeUnixNano": "1",
        "endTimeUnixNano": "2",
        "attributes": {"session.id": "sess-1"},
    }
    reads = _Scripted([([http_only], [], {"sess-1"})])
    reader = _reader()
    reader._read_once = reads  # type: ignore[method-assign]

    with pytest.raises(AgentCoreError, match="no agent span tree"):
        reader.read_spans(_query())
    assert reads.calls > 2


def test_content_that_lags_the_settled_spans_is_waited_for() -> None:
    # Live: the spans had settled while the agent's final answer was not yet in the index.
    # A span-count-only check returned the tree on the second poll with that answer missing.
    spans, content = _fixture_records()
    only_choice_missing = [
        r
        for r in content
        if not (r.get("spanId") == _FINAL_CALL and r.get("eventName") == "gen_ai.choice")
    ]
    reader = _reader(poll_interval_seconds=1.0)
    reader._read_once = _Scripted(  # type: ignore[method-assign]
        [
            (spans, _without_final_answer(content), {"s"}),
            (spans, only_choice_missing, {"s"}),
            (spans, content, {"s"}),
            (spans, content, {"s"}),
        ]
    )

    result = reader.read_spans(_query())

    assert len(result) == 1
    assert all(span.input and span.output for span in result)


@pytest.mark.parametrize("growing", ["spans", "content"])
def test_a_tree_still_changing_at_the_deadline_is_refused(
    monkeypatch: pytest.MonkeyPatch, growing: str
) -> None:
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    spans, content = _fixture_records()
    polls = {"n": 0}

    def ever_growing(_config: Any, _w3c: str) -> tuple[Any, Any, set[str]]:
        polls["n"] += 1
        n = 2 + polls["n"]
        if growing == "spans":
            return spans[:n], content, {"s"}
        return spans, content[:n], {"s"}

    reader = _reader(settle_deadline_seconds=10.0, poll_interval_seconds=3.0)
    reader._read_once = ever_growing  # type: ignore[method-assign]

    with pytest.raises(AgentCoreSettlingLag):
        reader.read_spans(_query())


def test_a_completed_agent_with_input_but_no_output_is_refused() -> None:
    spans, content = _fixture_records()
    reads = _Scripted([(spans, _without_final_answer(content), {"s"})])
    reader = _reader()
    reader._read_once = reads  # type: ignore[method-assign]

    with pytest.raises(AgentCoreContentMissing, match="output"):
        reader.read_spans(_query())
    assert reads.calls > 2  # polled until the deadline, not refused on the first stable read


def test_a_completed_agent_with_output_but_no_input_is_refused() -> None:
    spans, content = _fixture_records()
    # Every model call replays the user's turn, so the input is gone only when none of them has it.
    no_agent_input = [r for r in content if r.get("eventName") != "gen_ai.user.message"]
    reads = _Scripted([(spans, no_agent_input, {"s"})])
    reader = _reader()
    reader._read_once = reads  # type: ignore[method-assign]

    with pytest.raises(AgentCoreContentMissing, match="input"):
        reader.read_spans(_query())
    assert reads.calls > 2  # polled until the deadline, not refused on the first stable read


def test_a_half_joined_agent_that_becomes_whole_on_a_later_poll_is_returned() -> None:
    # The live case: the spans had settled and the counts were stable, but the agent's final
    # answer was not yet readable. Waiting returns the run once that answer arrives.
    spans, content = _fixture_records()
    half = (spans, _without_final_answer(content), {"s"})
    whole = (spans, content, {"s"})
    reader = _reader()
    reader._read_once = _Scripted([half, half, half, whole, whole])  # type: ignore[method-assign]

    result = reader.read_spans(_query())

    assert len(result) == 1
    assert all(span.input and span.output for span in result)


def test_spans_that_form_no_agent_tree_yet_are_polled_until_the_agents_arrive() -> None:
    # The runtime's HTTP span can land before any agent span; that is not yet a tree.
    spans, content = _fixture_records()
    early = (_http_only(spans), [], {"s"})
    whole = (spans, content, {"s"})
    reader = _reader()
    reader._read_once = _Scripted([early, early, early, whole, whole])  # type: ignore[method-assign]

    result = reader.read_spans(_query())

    assert len(result) == 1


def test_spans_that_never_form_an_agent_tree_are_refused_at_the_deadline() -> None:
    spans, _content = _fixture_records()
    reads = _Scripted([(_http_only(spans), [], {"s"})])
    reader = _reader()
    reader._read_once = reads  # type: ignore[method-assign]

    with pytest.raises(AgentCoreError, match="no agent span tree"):
        reader.read_spans(_query())
    assert reads.calls > 2


def test_an_empty_content_index_is_polled_through_until_the_content_arrives() -> None:
    spans, content = _fixture_records()
    empty = (spans, [], {"s"})
    whole = (spans, content, {"s"})
    reader = _reader()
    reader._read_once = _Scripted([empty, empty, empty, whole, whole])  # type: ignore[method-assign]

    result = reader.read_spans(_query())

    assert all(span.input and span.output for span in result)


def test_content_that_never_arrives_is_refused_at_the_deadline() -> None:
    spans, _content = _fixture_records()
    reads = _Scripted([(spans, [], {"s"})])
    reader = _reader()
    reader._read_once = reads  # type: ignore[method-assign]

    with pytest.raises(AgentCoreContentMissing):
        reader.read_spans(_query())
    assert reads.calls > 2


def _insights_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        [{"field": "@message", "value": json.dumps(r)}, {"field": "@ptr", "value": f"p{i}"}]
        for i, r in enumerate(records)
    ]
    return {
        "results": rows,
        "statistics": {"recordsMatched": float(len(rows))},
        "status": "Complete",
    }


def test_the_logs_insights_reads_filter_by_trace_id_and_an_empty_content_read_is_not_an_error() -> (
    None
):
    # Driven through the real CloudWatch Logs model: each query must carry the trace-id filter, and
    # an empty content read the log store agrees with must come back empty. Only the trace-filtered
    # FilterLogEvents reads are stubbed, so the window ingest's unfiltered empty-read cross-check
    # would fail the test.
    import boto3
    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.session import AwsClients

    spans, _content = _fixture_records()
    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    clients = AwsClients(session)
    stub = Stubber(clients.client("logs"))
    window = _config().window
    start = int(datetime.fromisoformat(window.start).timestamp())
    end = int(datetime.fromisoformat(window.end).timestamp())
    query = f"fields @message, @ptr | filter traceId = '{_RUN}' | limit 10000"
    content_group = spans[0]["resource"]["attributes"]["aws.log.group.names"]
    content_group = content_group[0] if isinstance(content_group, list) else content_group
    for group, records in (("aws/spans", spans), (content_group, [])):
        stub.add_response(
            "start_query",
            {"queryId": "q"},
            {"logGroupName": group, "startTime": start, "endTime": end, "queryString": query},
        )
        stub.add_response("get_query_results", _insights_results(records))
    for group, records in (("aws/spans", spans), (content_group, [])):
        stub.add_response(
            "filter_log_events",
            _store_events(records),
            {
                "logGroupName": group,
                "startTime": start * 1000,
                "endTime": end * 1000,
                "filterPattern": f'{{ $.traceId = "{_RUN}" }}',
            },
        )

    with stub:
        got_spans, got_content, sessions = _reader(clients)._read_logs_insights(_config(), _RUN)

    assert len(got_spans) == len(spans)
    assert got_content == []
    assert sessions
    stub.assert_no_pending_responses()


# The agent's refund tool call, which the index served in a later batch than the rest.
_LATE_SPAN = "450f07db19bf53c5"


def _store_events(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "events": [{"eventId": f"e{i}", "message": json.dumps(r)} for i, r in enumerate(records)]
    }


def _without_late_batch(
    spans: list[dict[str, Any]], content: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The run with one batch of its spans not yet served by the index (the live failure)."""
    dropped = {_LATE_SPAN}
    while True:
        more = {s["spanId"] for s in spans if s.get("parentSpanId") in dropped} - dropped
        if not more:
            break
        dropped |= more
    return (
        [s for s in spans if s["spanId"] not in dropped],
        [r for r in content if r.get("spanId") not in dropped],
    )


class _BatchedIndex:
    """Logs Insights serving a run's records poll by poll while the log store already holds all."""

    def __init__(
        self,
        index_reads: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]],
        store: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    ) -> None:
        self._index_reads = index_reads
        self._store_spans, self._store_content = store
        self._results: dict[str, list[dict[str, Any]]] = {}
        self._serving: tuple[list[dict[str, Any]], list[dict[str, Any]]] = ([], [])
        self.polls = 0

    def client(self, service_name: str) -> Any:
        assert service_name == "logs"
        return self

    def start_query(self, *, logGroupName: str, **_kwargs: Any) -> dict[str, str]:
        if logGroupName == "aws/spans":
            self.polls += 1
            self._serving = self._index_reads[min(self.polls, len(self._index_reads)) - 1]
            records = self._serving[0]
        else:
            records = self._serving[1]
        query_id = f"q{len(self._results)}"
        self._results[query_id] = records
        return {"queryId": query_id}

    def get_query_results(self, *, queryId: str) -> dict[str, Any]:
        return _insights_results(self._results[queryId])

    def filter_log_events(self, *, logGroupName: str, **_kwargs: Any) -> dict[str, Any]:
        records = self._store_spans if logGroupName == "aws/spans" else self._store_content
        return _store_events(records)


def test_a_partial_tree_the_index_serves_twice_in_a_row_is_polled_until_it_matches_the_store() -> (
    None
):
    # Live, the index served 9 of a run's 25 spans on two polls in a row while the log store held
    # all 25; two equal reads alone returned that partial tree. The whole tree must come back.
    spans, content = _fixture_records()
    partial = _without_late_batch(spans, content)
    index = _BatchedIndex([partial, partial, (spans, content)], (spans, content))

    result = _reader(index).read_spans(_query(w3c=_RUN))

    whole = _reader()
    whole._read_once = lambda _config, _w3c: (spans, content, {"sess-1"})  # type: ignore[method-assign]
    expected = whole.read_spans(_query(w3c=_RUN))
    assert len(result) == len(expected)
    assert {s.span_id for s in result} == {s.span_id for s in expected}
    # Two matched polls in a row: the first full read and the one after it.
    assert index.polls == 4


def test_a_content_record_the_index_has_not_served_yet_is_waited_for() -> None:
    spans, content = _fixture_records()
    lagging = (spans, content[:-1])
    index = _BatchedIndex([lagging, lagging, (spans, content)], (spans, content))

    result = _reader(index).read_spans(_query(w3c=_RUN))

    assert result
    assert index.polls == 4


def test_a_run_the_index_never_catches_up_with_is_refused_at_the_deadline_naming_both_counts() -> (
    None
):
    spans, content = _fixture_records()
    partial = _without_late_batch(spans, content)
    index = _BatchedIndex([partial], (spans, content))

    with pytest.raises(AgentCoreSettlingLag) as refused:
        _reader(index, settle_deadline_seconds=60.0).read_spans(_query(w3c=_RUN))

    message = str(refused.value)
    assert f"held {len(partial[0])} of the {len(spans)} spans" in message
    assert f"{len(partial[1])} of the {len(content)} content records" in message
    assert "partial tree" in message


def test_an_errored_agent_with_its_input_but_no_output_is_returned() -> None:
    # An agent that failed can end without an answer; that is a real failure trace, not a
    # half-read one, so it is returned for error analysis rather than refused.
    spans, content = _fixture_records()
    errored = [
        {**s, "status": {"code": "ERROR"}} if s.get("spanId") == _AGENT_SPAN else s for s in spans
    ]
    reader = _reader()
    reader._read_once = _Scripted([(errored, _without_final_answer(content), {"s"})])  # type: ignore[method-assign]

    result = reader.read_spans(_query())

    [agent] = result
    assert agent.status == "errored"
    assert agent.input
    assert not agent.output


def test_the_no_spans_refusal_points_at_an_unsampled_trace() -> None:
    reader = _reader(settle_deadline_seconds=1.0, poll_interval_seconds=1.0)
    reader._read_once = lambda _config, _w3c: ([], [], set())  # type: ignore[method-assign]

    with pytest.raises(AgentCoreSettlingLag, match="sampled"):
        reader.read_spans(_query())


def test_a_normalization_failure_surfaces_as_an_agentcore_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calibrated_agent_evaluation.normalize.errors import NormalizationError

    def fail(*_a: Any, **_k: Any) -> Any:
        raise NormalizationError("the trace failed schema validation")

    monkeypatch.setattr(span_reader, "normalize", fail)
    spans, content = _fixture_records()
    reader = _reader()
    reader._read_once = _Scripted([(spans, content, {"s"})])  # type: ignore[method-assign]

    with pytest.raises(AgentCoreError) as excinfo:
        reader.read_spans(_query())
    assert isinstance(excinfo.value.__cause__, NormalizationError)  # chained, not swallowed


def test_a_tree_that_settles_on_the_last_allowed_poll_is_returned() -> None:
    # Deadline 2s, a poll every second: polls at t=0, 1 and 2. The tree only holds still on the
    # poll at the deadline, and that poll still counts.
    spans, content = _fixture_records()
    reads = _Scripted(
        [(spans[:7], content, {"s"}), (spans, content, {"s"}), (spans, content, {"s"})]
    )
    reader = _reader(settle_deadline_seconds=2.0, poll_interval_seconds=1.0)
    reader._read_once = reads  # type: ignore[method-assign]

    result = reader.read_spans(_query())

    assert len(result) == 1
    assert reads.calls == 3


def test_a_malformed_invocation_time_is_refused_as_an_agentcore_error() -> None:
    bad = _query().model_copy(update={"started_at": "yesterday"})
    with pytest.raises(AgentCoreError, match="ISO-8601"):
        _reader().read_spans(bad)


def test_a_record_that_is_not_a_json_object_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(span_reader, "run_logs_insights", lambda *_a, **_k: ["not an object"])
    with pytest.raises(AgentCoreError, match="not a JSON object"):
        _reader()._read_logs_insights(_config(), _RUN)


@pytest.mark.parametrize(("deadline", "interval"), [(300.0, 0.0), (300.0, -1.0), (5.0, 10.0)])
def test_unusable_poll_settings_are_refused(deadline: float, interval: float) -> None:
    with pytest.raises(AgentCoreError, match="poll_interval_seconds"):
        _reader(settle_deadline_seconds=deadline, poll_interval_seconds=interval)


def test_the_omni_queries_filter_on_the_trace_id_inside_the_padded_window() -> None:
    sent: list[str] = []

    class _OmniClient:
        def start_telemetry_query_session(self, **_kw: Any) -> dict[str, Any]:
            return {"sessionId": "s-1"}

        def start_telemetry_query(self, *, sessionId: str, queryString: str) -> dict[str, Any]:
            sent.append(queryString)
            return {"queryId": "q"}

        def get_telemetry_query_results(self, **_kw: Any) -> dict[str, Any]:
            return {"rows": [], "statistics": {}, "status": "Complete"}

        def stop_telemetry_query_session(self, **_kw: Any) -> dict[str, Any]:
            return {}

    omni = _OmniClient()

    class _Clients:
        def client(self, _service_name: str) -> Any:
            return omni

    reader = _reader(
        _Clients(), use_omni=True, settle_deadline_seconds=1.0, poll_interval_seconds=1.0
    )
    with pytest.raises(AgentCoreSettlingLag):
        reader.read_spans(_query())

    spans_sql, content_sql = sent[0], sent[1]
    assert "FROM traces.default" in spans_sql
    assert "FROM logs.default" in content_sql
    for sql in (spans_sql, content_sql):
        assert f"traceId = '{_RUN}'" in sql
        # The 2026-01-01T00:00..01:00 invocation window, padded by a minute on each side.
        assert "to_timestamp('2025-12-31T23:59:00+00:00')" in sql
        assert "to_timestamp('2026-01-01T01:01:00+00:00')" in sql


def test_a_run_with_more_than_one_session_id_is_refused() -> None:
    spans, content = _fixture_records()
    reader = _reader()
    reader._read_once = lambda _config, _w3c: (spans, content, {"sess-1", "sess-2"})  # type: ignore[method-assign]

    with pytest.raises(AgentCoreMixedSession):
        reader.read_spans(_query())


def test_a_content_missing_run_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    spans, content = _fixture_records()

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise AgentCoreContentMissing("span carries no handoff content")

    monkeypatch.setattr(span_reader, "adot_to_otlp", refuse)
    reader = _reader()
    reader._read_once = lambda _config, _w3c: (spans, content, {"sess-1"})  # type: ignore[method-assign]

    with pytest.raises(AgentCoreContentMissing):
        reader.read_spans(_query())


@pytest.mark.parametrize("error", [AgentCorePartialResult, AgentCoreTruncatedResult])
def test_a_partial_or_truncated_read_propagates_its_typed_error(
    monkeypatch: pytest.MonkeyPatch, error: type[AgentCoreError]
) -> None:
    def refuse(*_a: Any, **_k: Any) -> Any:
        raise error("the read was incomplete")

    monkeypatch.setattr(span_reader, "run_logs_insights", refuse)

    with pytest.raises(error):
        _reader().read_spans(_query())


# --- reads honor the injected clients' guards and budget ----------------------------------


def test_billed_calls_stay_within_the_documented_bound_across_the_poll_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    calls = {"n": 0}

    def counting_omni(*_a: Any, **_k: Any) -> tuple[list[Any], dict[str, Any]]:
        calls["n"] += 1
        return [], {}

    monkeypatch.setattr(span_reader, "run_omni_query", counting_omni)
    reader = _reader(use_omni=True, settle_deadline_seconds=2.0, poll_interval_seconds=1.0)

    with pytest.raises(AgentCoreSettlingLag):
        reader.read_spans(_query())

    bound = 2 * (math.ceil(2.0 / 1.0) + 1)  # two StartTelemetryQuery calls per poll
    assert calls["n"] == bound  # the most polls the deadline allows, and not one more


def test_a_budget_guard_refusal_on_the_clients_propagates_and_stops_the_read() -> None:
    # The injected clients carry a billed-call budget; when it is exhausted the guard raises on the
    # billed StartTelemetryQuery. The reader must surface that refusal (it is an AgentCoreError, so
    # the reused driver re-raises it as itself), not swallow it or keep polling.
    class _GuardedOmniClient:
        def __init__(self) -> None:
            self.billed_calls = 0

        def start_telemetry_query_session(self, **_kw: Any) -> dict[str, Any]:
            return {"sessionId": "s-1"}

        def start_telemetry_query(self, **_kw: Any) -> dict[str, Any]:
            self.billed_calls += 1
            raise BilledCallBudgetExceeded(
                "refused cloudwatchomni:StartTelemetryQuery: the billed-call budget is used up"
            )

        def stop_telemetry_query_session(self, **_kw: Any) -> dict[str, Any]:
            return {}

    omni = _GuardedOmniClient()

    class _GuardedClients:
        def client(self, _service_name: str) -> Any:
            return omni

    reader = _reader(
        _GuardedClients(), use_omni=True, settle_deadline_seconds=1.0, poll_interval_seconds=1.0
    )

    with pytest.raises(BilledCallBudgetExceeded):
        reader.read_spans(_query())

    assert omni.billed_calls == 1  # refused on the first billed call; it did not keep polling
