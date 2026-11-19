# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Omni reader: query protocol + refusal (via Stubber) and the response decoders."""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("boto3")

import boto3
from botocore.stub import Stubber

from calibrated_agent_evaluation.agentcore.errors import AgentCoreError, AgentCorePartialResult
from calibrated_agent_evaluation.agentcore.readers import (
    LogsInsightsReader,
    OmniReader,
    agentcore_source,
    decode_content_row,
    decode_span_row,
    run_omni_query,
)
from calibrated_agent_evaluation.agentcore.session import AwsClients
from calibrated_agent_evaluation.agentcore.sources import (
    AgentCoreSourceConfig,
    AgentCoreTraceSource,
    TimeWindow,
)

_WINDOW = TimeWindow(start="2026-10-01T00:00:00+00:00", end="2026-10-01T01:00:00+00:00")


def _clients() -> AwsClients:
    session = boto3.Session(
        region_name="us-west-2", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    return AwsClients(session)


def _config(**over: Any) -> AgentCoreSourceConfig:
    base: dict[str, Any] = {"runtime": "nl_sql_agent", "window": _WINDOW, "omni": True}
    base.update(over)
    return AgentCoreSourceConfig(**base)


def _complete(rows: list[dict], partial: bool = False) -> dict:
    return {
        "status": "Complete",
        "rows": rows,
        "statistics": {"partialResults": {"partialResultsDetected": partial}},
    }


def _stub_query(stub: Stubber, rows: list[dict], *, partial: bool = False) -> None:
    """Queue one full Omni query-session round (start session, start query, results, stop)."""
    stub.add_response("start_telemetry_query_session", {"sessionId": "sess"})
    stub.add_response("start_telemetry_query", {"queryId": "q", "sessionId": "sess"})
    stub.add_response("get_telemetry_query_results", _complete(rows, partial))
    stub.add_response("stop_telemetry_query_session", {})


# --- decoders (pure) ----------------------------------------------------------------------


def test_decode_span_row_reads_the_record_and_backfills_ids() -> None:
    row = {
        "traceId": "t-1",
        "spanId": "sp-1",
        "@message": json.dumps({"attributes": {"gen_ai.operation.name": "invoke_agent"}}),
    }
    span = decode_span_row(row)
    assert span["traceId"] == "t-1"
    assert span["spanId"] == "sp-1"
    assert span["attributes"]["gen_ai.operation.name"] == "invoke_agent"


def test_decode_content_row_reads_the_log_record_and_backfills_ids() -> None:
    row = {
        "traceId": "t-1",
        "spanId": "sp-1",
        "@message": json.dumps(
            {"eventName": "gen_ai.user.message", "body": {"content": [{"text": "hi"}]}}
        ),
    }
    record = decode_content_row(row)
    assert record["traceId"] == "t-1"
    assert record["spanId"] == "sp-1"
    assert record["eventName"] == "gen_ai.user.message"
    assert record["body"]["content"] == [{"text": "hi"}]


def test_decoding_an_unparseable_record_is_a_typed_refusal_not_a_raw_json_error() -> None:
    # A malformed @message must surface as the integration's own error, not a bare JSONDecodeError
    # escaping the read: the caller can only catch a family it can name.
    bad_row = {"traceId": "t-1", "spanId": "sp-1", "@message": "{not json"}
    with pytest.raises(AgentCoreError):
        decode_span_row(bad_row)
    with pytest.raises(AgentCoreError):
        decode_content_row(bad_row)


# --- query protocol + refusal (Stubber) ---------------------------------------------------


def test_read_traces_runs_two_queries_and_joins_content() -> None:
    clients = _clients()
    stub = Stubber(clients.client("cloudwatchomni"))
    span_rows = [
        {"traceId": "t-1", "spanId": "sp-1", "session_id": "s-1", "@message": json.dumps({})}
    ]
    content_rows = [
        {
            "traceId": "t-1",
            "spanId": "sp-1",
            "@message": json.dumps(
                {
                    "spanId": "sp-1",
                    "eventName": "gen_ai.user.message",
                    "body": {"content": [{"text": "hi"}]},
                }
            ),
        }
    ]
    _stub_query(stub, span_rows)  # spans query
    _stub_query(stub, content_rows)  # content query
    with stub:
        traces = list(OmniReader(clients, _config()).read_traces())
    assert len(traces) == 1
    assert traces[0].trace_id == "t-1"
    assert traces[0].session_ids == ("s-1",)
    assert traces[0].log_records[0]["eventName"] == "gen_ai.user.message"
    assert traces[0].log_records[0]["body"]["content"] == [{"text": "hi"}]
    stub.assert_no_pending_responses()


def test_a_partial_omni_result_aborts_the_read() -> None:
    clients = _clients()
    stub = Stubber(clients.client("cloudwatchomni"))
    stub.add_response("start_telemetry_query_session", {"sessionId": "sess"})
    stub.add_response("start_telemetry_query", {"queryId": "q", "sessionId": "sess"})
    stub.add_response("get_telemetry_query_results", _complete([], partial=True))
    stub.add_response("stop_telemetry_query_session", {})
    with stub, pytest.raises(AgentCorePartialResult):
        run_omni_query(clients.client("cloudwatchomni"), "SELECT 1")


def test_a_failed_terminal_status_aborts_the_read_rather_than_returning_a_subset() -> None:
    # A query that ends Failed (or Cancelled) is terminal but not success: whatever rows came back
    # are an unknown subset of the match, so the read must refuse rather than report a quietly
    # smaller result as if it were complete.
    clients = _clients()
    stub = Stubber(clients.client("cloudwatchomni"))
    stub.add_response("start_telemetry_query_session", {"sessionId": "sess"})
    stub.add_response("start_telemetry_query", {"queryId": "q", "sessionId": "sess"})
    stub.add_response(
        "get_telemetry_query_results", {"status": "Failed", "rows": [], "statistics": {}}
    )
    stub.add_response("stop_telemetry_query_session", {})
    with stub, pytest.raises(AgentCoreError, match="not Complete"):
        run_omni_query(clients.client("cloudwatchomni"), "SELECT 1")


def test_factory_builds_a_trace_source_when_omni_is_set() -> None:
    source = agentcore_source(_clients(), _config(omni=True))
    assert isinstance(source, AgentCoreTraceSource)


def test_factory_builds_the_logs_insights_reader_without_omni() -> None:
    source = agentcore_source(_clients(), _config(omni=False))
    assert isinstance(source, AgentCoreTraceSource)
    assert isinstance(source._reader, LogsInsightsReader)
