# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Logs Insights fallback reader: the two-query read, runtime scoping, and completeness.

Stubbed with botocore's ``Stubber`` over the real CloudWatch Logs model (a wrong operation or
parameter name fails the test); the assertions are on the decoded :class:`RawTrace` and the typed
refusal, grounded in the shapes a live ``aws/spans`` + ``otel-rt-logs`` read returns.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

import pytest

pytest.importorskip("boto3")

import boto3
from botocore.stub import Stubber

from calibrated_agent_evaluation.agentcore.errors import AgentCoreError, AgentCoreTruncatedResult
from calibrated_agent_evaluation.agentcore.readers import (
    LogsInsightsReader,
    read_trace_events,
    run_logs_insights,
)
from calibrated_agent_evaluation.agentcore.session import AwsClients
from calibrated_agent_evaluation.agentcore.sources import AgentCoreSourceConfig, TimeWindow

_WINDOW = TimeWindow(start="2026-10-01T00:00:00+00:00", end="2026-10-01T01:00:00+00:00")
_RUNTIME = "SupportAgent.DEFAULT"
_CONTENT_GROUP = "/aws/bedrock-agentcore/runtimes/Tok-abc-DEFAULT"
_QUERY = "fields @message, @ptr | filter ispresent(traceId) | limit 10000"


def _clients() -> AwsClients:
    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    return AwsClients(session)


def _config(**over: Any) -> AgentCoreSourceConfig:
    base: dict[str, Any] = {"runtime": _RUNTIME, "window": _WINDOW, "omni": False}
    base.update(over)
    return AgentCoreSourceConfig(**base)


def _epoch(iso: str) -> int:
    return int(datetime.datetime.fromisoformat(iso).timestamp())


def _start_params(group: str) -> dict[str, Any]:
    return {
        "logGroupName": group,
        "startTime": _epoch(_WINDOW.start),
        "endTime": _epoch(_WINDOW.end),
        "queryString": _QUERY,
    }


def _span(span_id: str, *, runtime: str = _RUNTIME, trace_id: str = "t-1") -> dict[str, Any]:
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "resource": {
            "attributes": {"service.name": runtime, "aws.log.group.names": _CONTENT_GROUP}
        },
        "attributes": {"gen_ai.operation.name": "invoke_agent", "session.id": "sess-1"},
    }


def _content(span_id: str, trace_id: str = "t-1") -> dict[str, Any]:
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "eventName": "gen_ai.user.message",
        "body": {"content": [{"text": "hi"}]},
    }


def _rows(*docs: dict[str, Any]) -> list[list[dict[str, str]]]:
    # @ptr is the record's unique id; using the message as the ptr means the same record dedups
    # across split windows, which is what the real @ptr guarantees.
    out: list[list[dict[str, str]]] = []
    for doc in docs:
        msg = json.dumps(doc)
        out.append([{"field": "@message", "value": msg}, {"field": "@ptr", "value": msg}])
    return out


def _results(rows: list[list[dict[str, str]]], *, matched: int | None = None) -> dict[str, Any]:
    return {
        "results": rows,
        "statistics": {"recordsMatched": float(matched if matched is not None else len(rows))},
        "status": "Complete",
    }


def _stub_insights(
    stub: Stubber, group: str, rows: list[list[dict[str, str]]], *, matched: int | None = None
) -> None:
    """Queue one full Logs Insights round (start_query + get_query_results) for ``group``."""
    stub.add_response("start_query", {"queryId": "q"}, _start_params(group))
    stub.add_response("get_query_results", _results(rows, matched=matched))


def test_reads_spans_then_content_and_returns_the_trace_with_both() -> None:
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    _stub_insights(stub, "aws/spans", _rows(_span("sp-1")))
    _stub_insights(stub, _CONTENT_GROUP, _rows(_content("sp-1")))
    with stub:
        traces = list(LogsInsightsReader(clients, _config()).read_traces())
    assert len(traces) == 1
    assert traces[0].trace_id == "t-1"
    assert traces[0].span_records[0]["spanId"] == "sp-1"
    assert traces[0].log_records[0]["eventName"] == "gen_ai.user.message"
    assert traces[0].session_ids == ("sess-1",)
    stub.assert_no_pending_responses()


def test_an_array_form_log_group_names_attribute_is_read_not_crashed() -> None:
    # The OTel convention types aws.log.group.names as a string array; a span that carries the
    # array form must still resolve its content group (a list element would be unhashable in the
    # group set and the wrong shape for a scalar logGroupName).
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    span = _span("sp-1")
    span["resource"]["attributes"]["aws.log.group.names"] = [_CONTENT_GROUP]
    _stub_insights(stub, "aws/spans", _rows(span))
    _stub_insights(stub, _CONTENT_GROUP, _rows(_content("sp-1")))
    with stub:
        traces = list(LogsInsightsReader(clients, _config()).read_traces())
    assert len(traces) == 1
    assert traces[0].log_records[0]["eventName"] == "gen_ai.user.message"
    stub.assert_no_pending_responses()


def test_a_span_from_another_runtime_is_filtered_out() -> None:
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    # Two spans in the shared aws/spans group; only the configured runtime's span is ours.
    _stub_insights(
        stub, "aws/spans", _rows(_span("sp-1"), _span("other", runtime="SomeOtherAgent.DEFAULT"))
    )
    _stub_insights(stub, _CONTENT_GROUP, _rows(_content("sp-1")))
    with stub:
        traces = list(LogsInsightsReader(clients, _config()).read_traces())
    assert len(traces) == 1
    assert [s["spanId"] for s in traces[0].span_records] == ["sp-1"]
    stub.assert_no_pending_responses()


def test_an_empty_span_read_yields_no_trace_and_runs_no_content_query() -> None:
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    _stub_insights(stub, "aws/spans", _rows())  # no spans in the window
    with stub:
        traces = list(LogsInsightsReader(clients, _config()).read_traces())
    assert traces == []
    # No content query was issued (nothing left pending means only the one round was consumed).
    stub.assert_no_pending_responses()


def _stub_any(stub: Stubber, rows: list[list[dict[str, str]]], matched: int) -> None:
    """Queue one query round, matching any window bounds (sub-window epochs vary when splitting)."""
    stub.add_response("start_query", {"queryId": "q"})
    stub.add_response("get_query_results", _results(rows, matched=matched))


def test_a_window_matching_more_than_one_query_is_split_and_stitched() -> None:
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    # Full window: 1 row returned but 5 matched -> cannot be held in one query, so it is split.
    _stub_any(stub, _rows(_span("cap")), matched=5)
    # The two half-windows each return a complete slice; stitched, they are the real result.
    _stub_any(stub, _rows(_span("sp-1")), matched=1)
    _stub_any(stub, _rows(_span("sp-2")), matched=1)
    with stub:
        records = run_logs_insights(clients.client("logs"), "aws/spans", _config())
    span_ids = sorted(r["spanId"] for r in records)
    # The capped parent read is discarded; the two leaf slices are stitched (no duplicate).
    assert span_ids == ["sp-1", "sp-2"]
    stub.assert_no_pending_responses()


def test_an_empty_read_that_filter_log_events_contradicts_is_a_lag_error() -> None:
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    _stub_any(stub, _rows(), matched=0)  # Insights index says empty...
    stub.add_response("filter_log_events", {"events": [{"message": "x"}]})  # ...but events exist
    with stub, pytest.raises(AgentCoreError, match="lagging"):
        run_logs_insights(
            clients.client("logs"),
            _CONTENT_GROUP,
            _config(),
            cross_check=True,
            stream="otel-rt-logs",
        )


def test_a_genuinely_empty_window_cross_checks_clean_and_returns_empty() -> None:
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    _stub_any(stub, _rows(), matched=0)  # Insights empty...
    stub.add_response("filter_log_events", {"events": []})  # ...and FilterLogEvents agrees
    with stub:
        records = run_logs_insights(
            clients.client("logs"), _CONTENT_GROUP, _config(), cross_check=True
        )
    assert records == []
    stub.assert_no_pending_responses()


def test_a_filter_log_events_failure_fails_closed_not_open() -> None:
    # The cross-check exists to catch Insights index lag; if FilterLogEvents itself fails, refusing
    # (raising) is correct — failing open (treating it as "empty") would defeat the whole purpose.
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    _stub_any(stub, _rows(), matched=0)  # Insights index says empty...
    stub.add_client_error("filter_log_events", service_error_code="ServiceUnavailable")
    with stub, pytest.raises(AgentCoreError, match="cross-check"):
        run_logs_insights(
            clients.client("logs"),
            _CONTENT_GROUP,
            _config(),
            cross_check=True,
            stream="otel-rt-logs",
        )


def test_the_cross_check_pages_past_an_empty_first_page_before_trusting_absence() -> None:
    # FilterLogEvents can return an empty page with a nextToken before a page that holds events, so
    # the cross-check pages to exhaustion; events on a later page are still found (a lag error), not
    # mistaken for "nothing produced" after the first empty page.
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    _stub_any(stub, _rows(), matched=0)  # Insights index says empty...
    stub.add_response("filter_log_events", {"events": [], "nextToken": "p2"})  # empty first page
    stub.add_response("filter_log_events", {"events": [{"message": "x"}]})  # ...events on page 2
    with stub, pytest.raises(AgentCoreError, match="lagging"):
        run_logs_insights(
            clients.client("logs"),
            _CONTENT_GROUP,
            _config(),
            cross_check=True,
            stream="otel-rt-logs",
        )


def test_a_single_second_window_that_still_overflows_is_refused() -> None:
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    # A sub-second window floors to one epoch second: it cannot be split any finer.
    narrow = _config(
        window=TimeWindow(start="2026-10-01T00:00:00+00:00", end="2026-10-01T00:00:00.800000+00:00")
    )
    _stub_any(stub, _rows(_span("sp-1")), matched=5)  # matched exceeds what one query returned
    with stub, pytest.raises(AgentCoreTruncatedResult, match="incomplete"):
        run_logs_insights(clients.client("logs"), "aws/spans", narrow)


def test_a_non_complete_query_status_is_a_typed_error() -> None:
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    stub.add_response("start_query", {"queryId": "q"}, _start_params("aws/spans"))
    stub.add_response("get_query_results", {"results": [], "status": "Failed"})
    with stub, pytest.raises(AgentCoreError, match="not Complete"):
        list(LogsInsightsReader(clients, _config()).read_traces())


def test_an_unparseable_message_is_a_typed_refusal_not_a_dropped_record() -> None:
    # A row whose @message is not valid JSON must surface as the integration's own error, not be
    # silently skipped (which would drop a span) nor let a bare JSONDecodeError escape the read.
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    bad = [[{"field": "@message", "value": "{not json"}, {"field": "@ptr", "value": "p1"}]]
    stub.add_response("start_query", {"queryId": "q"}, _start_params("aws/spans"))
    stub.add_response("get_query_results", _results(bad))
    with stub, pytest.raises(AgentCoreError, match="unparseable"):
        list(LogsInsightsReader(clients, _config()).read_traces())


def test_the_omni_client_is_never_built_on_the_fallback_path() -> None:
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    _stub_insights(stub, "aws/spans", _rows(_span("sp-1")))
    _stub_insights(stub, _CONTENT_GROUP, _rows(_content("sp-1")))
    with stub:
        list(LogsInsightsReader(clients, _config()).read_traces())
    # The fallback path touches CloudWatch Logs only; the Omni client is never constructed.
    assert "cloudwatchomni" not in clients._clients


def test_without_a_trace_id_the_window_query_is_unchanged() -> None:
    # The window readers pass no trace id, so the query they send stays exactly what it was.
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    _stub_insights(stub, "aws/spans", _rows(_span("sp-1")))
    with stub:
        records = run_logs_insights(clients.client("logs"), "aws/spans", _config())
    assert [r["spanId"] for r in records] == ["sp-1"]
    stub.assert_no_pending_responses()


def test_a_trace_id_puts_a_trace_filter_in_the_query() -> None:
    trace_id = "6ae4a4b645731a5a5f27454c968f7694"
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    params = _start_params("aws/spans")
    params["queryString"] = f"fields @message, @ptr | filter traceId = '{trace_id}' | limit 10000"
    stub.add_response("start_query", {"queryId": "q"}, params)
    stub.add_response("get_query_results", _results(_rows(_span("sp-1", trace_id=trace_id))))
    with stub:
        records = run_logs_insights(
            clients.client("logs"), "aws/spans", _config(), trace_id=trace_id
        )
    assert [r["traceId"] for r in records] == [trace_id]
    stub.assert_no_pending_responses()


@pytest.mark.parametrize("bad", ["t-1", "6ae4a4b6' or '1'='1", "6AE4" * 7])
def test_a_malformed_trace_id_is_refused_before_any_query(bad: str) -> None:
    clients = _clients()
    stub = Stubber(clients.client("logs"))  # no responses queued: any call would fail the test
    with stub, pytest.raises(AgentCoreError, match="trace id"):
        run_logs_insights(clients.client("logs"), "aws/spans", _config(), trace_id=bad)


_TRACE = "6ae4a4b645731a5a5f27454c968f7694"


def _filter_params(group: str, **over: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "logGroupName": group,
        "startTime": _epoch(_WINDOW.start) * 1000,
        "endTime": _epoch(_WINDOW.end) * 1000,
        "filterPattern": f'{{ $.traceId = "{_TRACE}" }}',
    }
    params.update(over)
    return params


def _event(event_id: str, record: Any) -> dict[str, Any]:
    return {"eventId": event_id, "message": json.dumps(record)}


def test_a_trace_read_from_the_log_store_pages_to_exhaustion_and_dedups_by_event_id() -> None:
    # An empty first page can carry a token; the read must page on, and an event served on two
    # pages counts once.
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    stub.add_response("filter_log_events", {"events": [], "nextToken": "t1"}, _filter_params("g"))
    stub.add_response(
        "filter_log_events",
        {"events": [_event("e1", _span("sp-1", trace_id=_TRACE))], "nextToken": "t2"},
        _filter_params("g", nextToken="t1"),
    )
    stub.add_response(
        "filter_log_events",
        {"events": [_event("e1", _span("sp-1", trace_id=_TRACE)), _event("e2", _span("sp-2"))]},
        _filter_params("g", nextToken="t2"),
    )
    with stub:
        records = read_trace_events(clients.client("logs"), "g", _config(), _TRACE)
    assert sorted(r["spanId"] for r in records) == ["sp-1", "sp-2"]
    stub.assert_no_pending_responses()


def test_a_log_store_read_that_fails_raises_rather_than_reading_as_empty() -> None:
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    stub.add_client_error("filter_log_events", "AccessDeniedException", "not allowed")
    with stub, pytest.raises(AgentCoreError, match="FilterLogEvents"):
        read_trace_events(clients.client("logs"), "g", _config(), _TRACE)


@pytest.mark.parametrize("message", ["not json", json.dumps(["a", "list"])])
def test_a_log_store_record_that_is_not_a_json_object_is_refused(message: str) -> None:
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    stub.add_response(
        "filter_log_events",
        {"events": [{"eventId": "e1", "message": message}]},
        _filter_params("g"),
    )
    with stub, pytest.raises(AgentCoreError, match="FilterLogEvents message"):
        read_trace_events(clients.client("logs"), "g", _config(), _TRACE)


def test_a_malformed_trace_id_is_refused_before_any_log_store_read() -> None:
    clients = _clients()
    stub = Stubber(clients.client("logs"))  # no responses queued: any call would fail the test
    with stub, pytest.raises(AgentCoreError, match="trace id"):
        read_trace_events(clients.client("logs"), "g", _config(), "6ae4a4b6' or '1'='1")
