# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete AgentCore readers: the AWS calls + response decode behind the trace source.

This is the vendor seam. It imports boto3 (via ``AwsClients``) and turns AWS responses into
vendor-free :class:`~calibrated_agent_evaluation.agentcore.sources.RawTrace`s and evaluator-result
dicts. The AWS call sequence, pagination and completeness refusals are exercised with a botocore
``Stubber`` over the real service models; the inner decode (the ``@message`` raw record for both
spans and content, the content-record body, the evaluation-result fields) is grounded in the shapes
verified against a live CloudWatch Omni dataset. The faithful raw record is the ``@message`` column;
Omni's ``@record`` projection nulls structured assistant/choice/tool content, so it is never used
for content.

Both read paths ship here: the Omni SQL reader (preferred, when an Omni domain is set) and the
Logs Insights reader (the fallback over CloudWatch's ``aws/spans`` group plus the per-runtime
``otel-rt-logs`` content group, which is what Transaction Search writes).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.agentcore.errors import AgentCoreError, AgentCoreTruncatedResult
from calibrated_agent_evaluation.agentcore.sources import (
    LOGS_INSIGHTS_RETRIEVAL_CAP,
    AgentCoreSourceConfig,
    AgentCoreTraceSource,
    RawTrace,
    build_omni_sql,
    refuse_partial_result,
    refuse_truncated_result,
)

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.session import AwsClients

logger = logging.getLogger(__name__)

_OMNI_SERVICE = "cloudwatchomni"
# Omni reserves session names under /aws/ for service integrations; ours are unique per query.
_SESSION_PREFIX = "calibrated-agent-evaluation-ingest"
_TERMINAL_STATUSES = frozenset({"Complete", "Failed", "Cancelled"})
# How long to wait between polls of a still-running Omni query, and how many polls before giving
# up — so a query that never reaches a terminal status is a bounded refusal, not an unbounded
# busy-loop hammering the service.
_OMNI_POLL_SECONDS = 2
_OMNI_MAX_POLLS = 150


def _unique_session_name() -> str:
    return f"{_SESSION_PREFIX}-{uuid.uuid4().hex[:12]}"


def _refuse_if_omni_poll_exhausted(polls: int) -> None:
    """Raise once a still-running Omni query has been polled past the budget."""
    if polls > _OMNI_MAX_POLLS:
        raise AgentCoreError(
            f"the Omni query did not finish within {_OMNI_POLL_SECONDS * _OMNI_MAX_POLLS}s; it is "
            "still running with no results cursor. Narrow the window and re-run"
        )


def _refuse_if_omni_not_complete(status: str | None) -> None:
    """Raise when a terminal Omni query did not end ``Complete``.

    ``Failed`` and ``Cancelled`` are terminal but not success: the accumulated rows are an unknown
    subset of the match, so the read is refused rather than returned as a quietly smaller result.
    """
    if status is not None and status != "Complete":
        raise AgentCoreError(
            f"the Omni query ended {status!r}, not Complete; its rows would be an unknown subset "
            "of the match, so the read is refused"
        )


def run_omni_query(client: Any, sql: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one Omni SQL query to completion and return its rows and final statistics.

    Walks the three-call session protocol (start session, start query, poll results), paging to
    exhaustion, and refuses a partial result. The session is always stopped. Any botocore failure
    is wrapped into :class:`AgentCoreError`.
    """
    session_name = _unique_session_name()
    try:
        session_id = client.start_telemetry_query_session(sessionName=session_name)["sessionId"]
    except Exception as exc:
        raise AgentCoreError(f"could not start an Omni query session: {exc}") from exc
    rows: list[dict[str, Any]] = []
    statistics: dict[str, Any] = {}
    try:
        query_id = client.start_telemetry_query(sessionId=session_id, queryString=sql)["queryId"]
        next_token: str | None = None
        polls = 0
        while True:
            params = {"queryId": query_id}
            if next_token:
                params["nextToken"] = next_token
            result = client.get_telemetry_query_results(**params)
            rows.extend(result.get("rows", []))
            statistics = result.get("statistics", {})
            refuse_partial_result(statistics)
            next_token = result.get("nextToken")
            if next_token:
                # More pages of a ready result: page on without delay.
                continue
            status = result.get("status")
            if status in _TERMINAL_STATUSES:
                # A terminal status that is not Complete (Failed/Cancelled) is refused, not success.
                _refuse_if_omni_not_complete(status)
                break
            # Still running with no cursor: wait a bounded interval and poll again, refusing once
            # the query has not reached a terminal status within the poll budget.
            polls += 1
            _refuse_if_omni_poll_exhausted(polls)
            time.sleep(_OMNI_POLL_SECONDS)
    except AgentCoreError:
        raise
    except Exception as exc:
        raise AgentCoreError(f"Omni query failed: {exc}") from exc
    finally:
        try:
            client.stop_telemetry_query_session(sessionId=session_id)
        except Exception:  # noqa: BLE001 - a failed stop must not mask the read's own outcome
            logger.warning("could not stop Omni query session %s", session_id)
    return rows, statistics


def decode_span_row(row: dict[str, Any]) -> dict[str, Any]:
    """Decode one Omni span row into an OTLP span dict.

    The ``@message`` column carries the faithful raw span document (a JSON string, or already a
    dict) — unlike ``@record``, Omni's normalized projection, which nulls structured content. The
    explicit ``traceId``/``spanId`` columns backfill those ids when the record omits them.
    """
    record = row.get("@message")
    try:
        span = json.loads(record) if isinstance(record, str) else dict(record or {})
    except json.JSONDecodeError as exc:
        raise AgentCoreError(
            f"an Omni span row carried an unparseable @message record: {exc}"
        ) from exc
    for column, key in (
        ("traceId", "traceId"),
        ("spanId", "spanId"),
        ("parentSpanId", "parentSpanId"),
    ):
        if column in row and not span.get(key):
            span[key] = row[column]
    return span


def decode_content_row(row: dict[str, Any]) -> dict[str, Any]:
    """Decode one Omni content row into a plain ``otel-rt-logs`` log-record dict.

    The ``@message`` column carries the faithful raw OTel log record (a JSON string, or already a
    dict), which holds ``spanId`` / ``eventName`` / ``body`` / ``timeUnixNano`` at top level with
    the message content intact. (The ``@record`` projection nulls assistant/choice/tool bodies, so
    it must not be used for content.) The explicit ``traceId`` / ``spanId`` columns backfill those
    ids when the record omits them. The converter re-homes these onto the spans.
    """
    record_doc = row.get("@message")
    try:
        record = json.loads(record_doc) if isinstance(record_doc, str) else dict(record_doc or {})
    except json.JSONDecodeError as exc:
        raise AgentCoreError(
            f"an Omni content row carried an unparseable @message record: {exc}"
        ) from exc
    for column in ("traceId", "spanId"):
        if column in row and not record.get(column):
            record[column] = row[column]
    return record


class OmniReader:
    """Reads a runtime's traces and evaluator scores through one CloudWatch Omni SQL client."""

    def __init__(self, clients: AwsClients, config: AgentCoreSourceConfig) -> None:
        self._clients = clients
        self._config = config

    def _client(self) -> Any:
        return self._clients.client(_OMNI_SERVICE)

    def read_traces(self) -> Iterator[RawTrace]:
        """Yield each trace in the window, with its spans and its content log records."""
        client = self._client()
        span_rows, _ = run_omni_query(client, build_omni_sql(self._config, session_name="spans"))
        content_rows, _ = run_omni_query(client, _content_sql(self._config))
        logs_by_trace = _group_content(content_rows)
        yield from _group_traces(span_rows, logs_by_trace)

    def read_scores(self) -> Iterator[dict[str, Any]]:
        """Yield each successful evaluator-result record in the window (empty when none).

        Reads the ``gen_ai.evaluation.result`` records (whichever log groups the Omni dataset
        forwards them from) and maps each to an Evaluate-result dict; error-event records (a failed
        evaluation, no score) are skipped, not yielded.
        """
        if not self._config.with_scores:
            return
        rows, _ = run_omni_query(self._client(), _scores_sql(self._config))
        for row in rows:
            result = eval_record_to_result(decode_content_row(row))
            if result is not None:
                yield result


def _content_sql(config: AgentCoreSourceConfig) -> str:
    # Content lives in the `otel-rt-logs` log stream within logs.default, keyed by traceId + spanId;
    # the full log record (eventName/body/spanId/timeUnixNano) is pulled via `@message` — the raw
    # record with content intact — NOT `@record`, whose normalized projection nulls the structured
    # assistant/choice/tool message bodies (verified live). The stream is the dataset's `@logStream`
    # column, not an `attributes` map entry.
    return (
        "SELECT traceId, spanId, `@message` "  # noqa: S608 - inputs validated (see build_omni_sql)
        "FROM logs.default "
        f"WHERE `@timestamp` >= to_timestamp('{config.window.start}') "
        f"AND `@timestamp` < to_timestamp('{config.window.end}') "
        "AND `@logStream` = 'otel-rt-logs'"
    )


def _scores_sql(config: AgentCoreSourceConfig) -> str:
    # `@message` (the raw record), not the normalized `body` column, so the evaluation value/label/
    # explanation survive. The event is identified by its top-level `name` field (verified live) —
    # `gen_ai.evaluation.result` — NOT an `eventName` attribute.
    return (
        "SELECT traceId, spanId, `@message` "  # noqa: S608 - inputs validated (see build_omni_sql)
        "FROM logs.default "
        f"WHERE `@timestamp` >= to_timestamp('{config.window.start}') "
        f"AND `@timestamp` < to_timestamp('{config.window.end}') "
        "AND name = 'gen_ai.evaluation.result'"
    )


def _group_content(content_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in content_rows:
        record = decode_content_row(row)
        trace_id = record.get("traceId") or row.get("traceId")
        if not isinstance(trace_id, str):
            continue
        grouped.setdefault(trace_id, []).append(record)
    return grouped


def _group_traces(
    span_rows: list[dict[str, Any]], logs_by_trace: dict[str, list[dict[str, Any]]]
) -> Iterator[RawTrace]:
    order: list[str] = []
    spans_by_trace: dict[str, list[dict[str, Any]]] = {}
    sessions_by_trace: dict[str, list[str]] = {}
    for row in span_rows:
        span = decode_span_row(row)
        trace_id = span.get("traceId") or row.get("traceId")
        if not isinstance(trace_id, str):
            continue
        if trace_id not in spans_by_trace:
            spans_by_trace[trace_id] = []
            sessions_by_trace[trace_id] = []
            order.append(trace_id)
        spans_by_trace[trace_id].append(span)
        session_id = row.get("session_id")
        if (
            isinstance(session_id, str)
            and session_id
            and session_id not in sessions_by_trace[trace_id]
        ):
            sessions_by_trace[trace_id].append(session_id)
    for trace_id in order:
        # Sort so the collapsed runtime session id is deterministic (neither query sorts rows).
        sessions = sorted(sessions_by_trace[trace_id])
        yield RawTrace(
            trace_id=trace_id,
            span_records=spans_by_trace[trace_id],
            log_records=logs_by_trace.get(trace_id, []),
            session_ids=tuple(sessions),
            runtime_session_id=sessions[0] if sessions else None,
        )


def eval_record_to_result(record: dict[str, Any]) -> dict[str, Any] | None:
    """Map one ``gen_ai.evaluation.result`` log record to an Evaluate-result dict, or ``None``.

    The record (verified against live AgentCore batch/online evaluation output) carries a top-level
    ``name`` == ``gen_ai.evaluation.result`` and a top-level ``traceId``; the verdict rides in its
    ``attributes`` under the OpenTelemetry GenAI-evaluation convention
    (``gen_ai.evaluation.score.value`` / ``.score.label`` / ``.explanation`` / ``.name``) alongside
    ``aws.bedrock_agentcore.evaluator.arn`` and the ``session.id`` / ``gen_ai.response.id`` address.
    An error event (a failed evaluation, ``error.type`` present and no score) is NOT a verdict, so
    it returns ``None`` and the caller skips it rather than storing an empty label.
    """
    if record.get("name") != "gen_ai.evaluation.result":
        return None
    raw_attrs = record.get("attributes")
    attrs: dict[str, Any] = raw_attrs if isinstance(raw_attrs, dict) else {}
    if attrs.get("error.type") or attrs.get("error"):
        return None
    result: dict[str, Any] = {
        "evaluatorArn": attrs.get("aws.bedrock_agentcore.evaluator.arn")
        or attrs.get("gen_ai.evaluation.evaluator.arn"),
        "evaluatorName": attrs.get("gen_ai.evaluation.name"),
        "explanation": attrs.get("gen_ai.evaluation.explanation"),
        "value": attrs.get("gen_ai.evaluation.score.value"),
        "label": attrs.get("gen_ai.evaluation.score.label"),
        "context": {
            "spanContext": {
                "traceId": record.get("traceId") or attrs.get("gen_ai.response.id"),
                "sessionId": attrs.get("session.id"),
                "spanId": attrs.get("gen_ai.evaluation.span.id") or attrs.get("spanId"),
            }
        },
    }
    # A batch result carries its job id (the log stream name) so the label's provenance can record
    # which batch job produced it; an online/console result has none, and the key is left absent.
    job_id = record.get(_LOG_STREAM_KEY)
    if isinstance(job_id, str) and job_id:
        result["jobId"] = job_id
    return result


_LOGS_SERVICE = "logs"
# CloudWatch's Transaction Search writes every runtime's spans into this one shared group.
_SPANS_LOG_GROUP = "aws/spans"
# AgentCore writes evaluation results to dedicated log groups under this prefix (batch:
# .../batch-evaluations/results/*, online: .../results/<config-id>), not the runtime trace group.
_EVALUATIONS_LOG_GROUP_PREFIX = "/aws/bedrock-agentcore/evaluations/"
# A batch-evaluation result group lives under .../evaluations/batch*; its per-record log stream is
# named after the batchEvaluationId (the job id). Online result groups are named by config id, not
# a job id, so only the batch groups carry a job id into the label's provenance (per AWS docs).
_BATCH_EVALUATIONS_PREFIX = "/aws/bedrock-agentcore/evaluations/batch"
# The key the Logs Insights reader stashes a record's source log stream under, so the score reader
# can read it without a wider return type. Private: stripped before the record is mapped to a label.
_LOG_STREAM_KEY = "__agentcore_log_stream__"
# How long to wait for one Logs Insights query to reach a terminal status before giving up.
_INSIGHTS_POLL_SECONDS = 2
_INSIGHTS_MAX_POLLS = 150
_INSIGHTS_TERMINAL = frozenset({"Complete", "Failed", "Cancelled", "Timeout"})
# One Logs Insights query returns at most 10,000 results (AWS hard limit). To read a window that
# matches more, the reader splits the window in time and stitches the pieces (dedup by `@ptr`),
# up to the retrieval cap; a window that still overflows after splitting to one second is refused.
_CWLI_MAX_RESULTS = 10_000
_MAX_SPLIT_DEPTH = 20
# A defensive bound on FilterLogEvents paging in the empty-read cross-check, so a backend that
# keeps returning an empty page with a token can never loop forever.
_FILTER_EVENTS_MAX_PAGES = 1000


def _epoch(iso: str) -> int:
    """The integer Unix second of a tz-aware ISO-8601 bound (validated by ``TimeWindow``)."""
    return int(datetime.fromisoformat(iso).timestamp())


def _row_field(row: list[dict[str, Any]], name: str) -> str | None:
    """The value of a named field (``@message`` / ``@ptr``) in one Logs Insights result row."""
    for field in row:
        if field.get("field") == name:
            return field.get("value")
    return None


def require_otel_trace_id(trace_id: str) -> str:
    """A 32-hex OTel trace id, lowercased, or a refusal, so it is safe to put in a query string.

    The one validator every by-id query uses (the Logs Insights filter and the Omni SQL), so the
    injection guard lives in a single place.
    """
    value = trace_id.strip().lower()
    if len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
        raise AgentCoreError(
            f"trace id {trace_id!r} is not 32 hex characters; refusing to query by it, since it "
            "could match the wrong trace or break the query"
        )
    return value


def _insights_filter(trace_id: str | None) -> str:
    """The query's record filter: one trace's records, or every record that carries a trace id."""
    return f"filter traceId = '{trace_id}'" if trace_id else "filter ispresent(traceId)"


def _one_insights_query(
    client: Any,
    log_group: str,
    start: int,
    end: int,
    *,
    carry_stream: bool = False,
    trace_id: str | None = None,
) -> tuple[list[list[dict[str, Any]]], int]:
    """Run one Logs Insights query over ``[start, end]`` and return its raw rows and match count.

    Selects ``@message`` (the faithful record) and ``@ptr`` (its unique id, used to dedup across
    split windows). ``carry_stream`` additionally selects ``@logStream`` — the log stream a record
    was written to — which for a batch-evaluation result group is the ``batchEvaluationId`` (the
    job id). Polls to a terminal status; a non-``Complete`` status is a typed error.
    """
    # Both groups key their rows by a top-level `traceId` (spans in aws/spans; otel-rt-logs content
    # records); the runtime's plain app-log streams carry none, so this selects telemetry only.
    projection = "@message, @ptr, @logStream" if carry_stream else "@message, @ptr"
    query = f"fields {projection} | {_insights_filter(trace_id)} | limit {_CWLI_MAX_RESULTS}"
    try:
        query_id = client.start_query(
            logGroupName=log_group, startTime=start, endTime=end, queryString=query
        )["queryId"]
    except Exception as exc:
        raise AgentCoreError(
            f"could not start a Logs Insights query on {log_group!r}: {exc}"
        ) from exc
    result: dict[str, Any] = {}
    try:
        for _ in range(_INSIGHTS_MAX_POLLS):
            result = client.get_query_results(queryId=query_id)
            if result.get("status") in _INSIGHTS_TERMINAL:
                break
            time.sleep(_INSIGHTS_POLL_SECONDS)
    except Exception as exc:
        raise AgentCoreError(f"Logs Insights query on {log_group!r} failed: {exc}") from exc
    status = result.get("status")
    if status not in _INSIGHTS_TERMINAL:
        raise AgentCoreError(
            f"Logs Insights query on {log_group!r} did not finish within "
            f"{_INSIGHTS_POLL_SECONDS * _INSIGHTS_MAX_POLLS}s"
        )
    if status != "Complete":
        raise AgentCoreError(f"Logs Insights query on {log_group!r} ended {status!r}, not Complete")
    rows = result.get("results", [])
    matched = int((result.get("statistics") or {}).get("recordsMatched", len(rows)))
    return rows, matched


def run_logs_insights(
    client: Any,
    log_group: str,
    config: AgentCoreSourceConfig,
    *,
    cross_check: bool = False,
    stream: str | None = None,
    carry_stream: bool = False,
    trace_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read every matching record in the window from Logs Insights and decode each ``@message``.

    With ``trace_id`` (a 32-hex OTel trace id), the query itself filters to that one trace's
    records, so a by-id read fetches one run instead of every record in the window; a malformed id
    is refused before any query is sent. Without it the query is the window read the trace source
    has always sent.

    A single query caps at 10,000 results, so a window matching more is split in time and the
    pieces stitched (deduped by ``@ptr``) up to the retrieval cap; a window that still overflows
    after splitting to one second is refused (:class:`AgentCoreTruncatedResult`) rather than
    silently truncated. Each returned dict is the parsed ``@message`` (a flat OTLP span, or an
    ``otel-rt-logs`` content record).

    When ``cross_check`` is set and the read comes back empty, the window is confirmed empty with
    ``FilterLogEvents`` (optionally scoped to ``stream``) before the absence is trusted: the Logs
    Insights index can lag and report zero while the log group already holds events (observed on
    the evaluation group), so an empty Insights result that ``FilterLogEvents`` contradicts is
    raised as a typed error to retry rather than mistaken for "nothing was produced". Only safe for
    a per-runtime / per-group read, never the shared ``aws/spans`` group.
    """
    trace_filter = require_otel_trace_id(trace_id) if trace_id is not None else None
    seen: set[str] = set()
    rows = _read_window(
        client,
        log_group,
        _epoch(config.window.start),
        _epoch(config.window.end),
        0,
        seen,
        carry_stream=carry_stream,
        trace_id=trace_filter,
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        message = _row_field(row, "@message")
        if isinstance(message, str):
            try:
                record = json.loads(message)
            except json.JSONDecodeError as exc:
                # Match the Omni decode: a record we cannot parse is a refusal, not a dropped span.
                raise AgentCoreError(
                    f"a Logs Insights @message on {log_group!r} was unparseable: {exc}"
                ) from exc
            if carry_stream and isinstance(record, dict):
                log_stream = _row_field(row, "@logStream")
                if isinstance(log_stream, str) and log_stream:
                    # The batch-result stream name is the batchEvaluationId (the job id); carry it
                    # on the record so the score reader can stamp it into the label's provenance.
                    record[_LOG_STREAM_KEY] = log_stream
            records.append(record)
    if cross_check and not records and _group_has_events(client, log_group, config, stream):
        raise AgentCoreError(
            f"Logs Insights returned no records for {log_group!r} in the window, but the log group "
            "already holds events there — the Insights index is lagging, not empty. Retry shortly."
        )
    return records


def _group_has_events(
    client: Any, log_group: str, config: AgentCoreSourceConfig, stream: str | None
) -> bool:
    """Whether ``log_group`` (optionally one ``stream``) has any event in the window.

    Uses ``FilterLogEvents`` (epoch milliseconds, unlike the Logs Insights query's seconds), which
    reads the log store directly rather than the eventually-consistent Insights index. It pages to
    exhaustion, because ``FilterLogEvents`` can return an empty page with a ``nextToken`` before a
    page that holds events, so a single empty first page does not mean "no events". A failure to
    run the check **raises** rather than returning ``False``: this check exists to catch an empty
    Insights read that is actually index lag, so failing it open would defeat its whole purpose.
    """
    params: dict[str, Any] = {
        "logGroupName": log_group,
        "startTime": _epoch(config.window.start) * 1000,
        "endTime": _epoch(config.window.end) * 1000,
    }
    if stream is not None:
        params["logStreamNames"] = [stream]
    token: str | None = None
    for _ in range(_FILTER_EVENTS_MAX_PAGES):
        if token:
            params["nextToken"] = token
        try:
            resp = client.filter_log_events(**params)
        except Exception as exc:
            raise AgentCoreError(
                f"could not cross-check {log_group!r} with FilterLogEvents: {exc}"
            ) from exc
        if resp.get("events"):
            return True
        token = resp.get("nextToken")
        if not token:
            return False
    raise AgentCoreError(
        f"FilterLogEvents paged past {_FILTER_EVENTS_MAX_PAGES} empty pages on {log_group!r} "
        "without terminating; refusing the cross-check rather than looping"
    )


def read_trace_events(
    client: Any, log_group: str, config: AgentCoreSourceConfig, trace_id: str
) -> list[dict[str, Any]]:
    """Every record of one trace that ``log_group`` holds in the window, read from the log store.

    Uses ``FilterLogEvents`` with a JSON filter on the top-level ``traceId``, which reads the log
    store directly. The Logs Insights index lags the store, and a run's records reach the index in
    batches minutes apart, so the store is the reference for whether an index read is whole. Pages
    to exhaustion (an empty page can carry a ``nextToken``) and dedups by event id. A call that
    fails, or a record that is not a JSON object, raises rather than reading as fewer records.
    """
    trace_filter = require_otel_trace_id(trace_id)
    params: dict[str, Any] = {
        "logGroupName": log_group,
        "startTime": _epoch(config.window.start) * 1000,
        "endTime": _epoch(config.window.end) * 1000,
        "filterPattern": f'{{ $.traceId = "{trace_filter}" }}',
    }
    records: dict[str, dict[str, Any]] = {}
    token: str | None = None
    for _ in range(_FILTER_EVENTS_MAX_PAGES):
        if token:
            params["nextToken"] = token
        try:
            resp = client.filter_log_events(**params)
        except Exception as exc:
            raise AgentCoreError(
                f"could not read trace {trace_filter!r} from {log_group!r} with FilterLogEvents: "
                f"{exc}"
            ) from exc
        for index, event in enumerate(resp.get("events", [])):
            message = event.get("message")
            try:
                record = json.loads(message) if isinstance(message, str) else None
            except json.JSONDecodeError as exc:
                raise AgentCoreError(
                    f"a FilterLogEvents message on {log_group!r} was unparseable: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise AgentCoreError(
                    f"a FilterLogEvents message on {log_group!r} was not a JSON object; the read "
                    "is refused rather than guessed at"
                )
            records[str(event.get("eventId") or f"{len(records)}-{index}")] = record
        token = resp.get("nextToken")
        if not token:
            return list(records.values())
    raise AgentCoreError(
        f"FilterLogEvents paged past {_FILTER_EVENTS_MAX_PAGES} pages on {log_group!r} without "
        "terminating; refusing the read rather than looping"
    )


def _read_window(
    client: Any,
    log_group: str,
    start: int,
    end: int,
    depth: int,
    seen: set[str],
    *,
    carry_stream: bool = False,
    trace_id: str | None = None,
) -> list[list[dict[str, Any]]]:
    """Read ``[start, end]`` completely, splitting in time when one query can't hold the match."""
    rows, matched = _one_insights_query(
        client, log_group, start, end, carry_stream=carry_stream, trace_id=trace_id
    )
    if matched <= len(rows):
        # Complete for this window: keep the rows not already seen in a sibling window.
        kept: list[list[dict[str, Any]]] = []
        for row in rows:
            ptr = _row_field(row, "@ptr")
            if isinstance(ptr, str):
                if ptr in seen:
                    continue
                seen.add(ptr)
            kept.append(row)
            if len(seen) > LOGS_INSIGHTS_RETRIEVAL_CAP:
                raise AgentCoreTruncatedResult(
                    f"the window matched more than the {LOGS_INSIGHTS_RETRIEVAL_CAP} record "
                    "retrieval cap; narrow --since/--until and re-run"
                )
        return kept
    # More matched than one query returned: split the window and stitch. A window that can no
    # longer be split (one second) but still overflows is refused, never silently truncated.
    if depth >= _MAX_SPLIT_DEPTH or end <= start:
        refuse_truncated_result(matched, len(rows))
    mid = start + (end - start) // 2
    left = _read_window(
        client, log_group, start, mid, depth + 1, seen, carry_stream=carry_stream, trace_id=trace_id
    )
    right = _read_window(
        client,
        log_group,
        mid + 1,
        end,
        depth + 1,
        seen,
        carry_stream=carry_stream,
        trace_id=trace_id,
    )
    return left + right


def _service_name(span: dict[str, Any]) -> str | None:
    return ((span.get("resource") or {}).get("attributes") or {}).get("service.name")


def _content_log_groups(span: dict[str, Any]) -> list[str]:
    # The OTel `aws.log.group.names` convention is a string array, but AgentCore has been observed
    # to emit a single scalar string; accept both so an array-form attribute cannot crash the read
    # (a list is unhashable and is the wrong shape for a scalar logGroupName).
    names = ((span.get("resource") or {}).get("attributes") or {}).get("aws.log.group.names")
    if isinstance(names, str):
        return [names]
    if isinstance(names, list):
        return [n for n in names if isinstance(n, str) and n]
    return []


def _session_id(span: dict[str, Any]) -> str | None:
    return (span.get("attributes") or {}).get("session.id")


class LogsInsightsReader:
    """Reads a runtime's traces (and scores) from CloudWatch Logs Insights — the non-Omni path.

    Spans come from the shared ``aws/spans`` group (what Transaction Search writes), filtered to
    the configured runtime's ``service.name``; content comes from that runtime's own
    ``otel-rt-logs`` group, discovered from the spans' ``aws.log.group.names`` resource attribute.
    The converter re-homes the content onto the spans downstream.
    """

    def __init__(self, clients: AwsClients, config: AgentCoreSourceConfig) -> None:
        self._clients = clients
        self._config = config

    def _client(self) -> Any:
        return self._clients.client(_LOGS_SERVICE)

    def read_traces(self) -> Iterator[RawTrace]:
        """Yield each of the runtime's traces in the window, with its spans and content records."""
        client = self._client()
        spans = [
            span
            for span in run_logs_insights(client, _SPANS_LOG_GROUP, self._config)
            if _service_name(span) == self._config.runtime
        ]
        if not spans:
            return
        content_groups = {g for span in spans for g in _content_log_groups(span)}
        logs_by_trace: dict[str, list[dict[str, Any]]] = {}
        for group in sorted(content_groups):
            # Per-runtime content group: cross-check an empty read against the otel-rt-logs stream,
            # since spans without their content would otherwise persist as empty steps.
            content = run_logs_insights(
                client, group, self._config, cross_check=True, stream="otel-rt-logs"
            )
            for record in content:
                trace_id = record.get("traceId")
                if isinstance(trace_id, str):
                    logs_by_trace.setdefault(trace_id, []).append(record)
        yield from self._group_traces(spans, logs_by_trace)

    def _group_traces(
        self, spans: list[dict[str, Any]], logs_by_trace: dict[str, list[dict[str, Any]]]
    ) -> Iterator[RawTrace]:
        order: list[str] = []
        spans_by_trace: dict[str, list[dict[str, Any]]] = {}
        sessions_by_trace: dict[str, list[str]] = {}
        for span in spans:
            trace_id = span.get("traceId")
            if not isinstance(trace_id, str):
                continue
            if trace_id not in spans_by_trace:
                spans_by_trace[trace_id] = []
                sessions_by_trace[trace_id] = []
                order.append(trace_id)
            spans_by_trace[trace_id].append(span)
            session = _session_id(span)
            if isinstance(session, str) and session and session not in sessions_by_trace[trace_id]:
                sessions_by_trace[trace_id].append(session)
        for trace_id in order:
            yield RawTrace(
                trace_id=trace_id,
                span_records=spans_by_trace[trace_id],
                log_records=logs_by_trace.get(trace_id, []),
                session_ids=tuple(sessions_by_trace[trace_id]),
                runtime_session_id=sessions_by_trace[trace_id][0]
                if sessions_by_trace[trace_id]
                else None,
            )

    def read_scores(self) -> Iterator[dict[str, Any]]:
        """Yield each successful evaluator-result in the window (empty when not requested).

        AgentCore writes evaluation results to DEDICATED log groups under
        ``/aws/bedrock-agentcore/evaluations/`` (batch and online), NOT the runtime's trace group,
        as ``gen_ai.evaluation.result`` records (verified live). This discovers those groups and
        maps each successful record to an Evaluate-result dict; error events are skipped.
        """
        if not self._config.with_scores:
            return
        client = self._client()
        for group in self._evaluation_result_groups(client):
            # A batch result's log stream is its job id; carry it only for the batch groups, since
            # an online group's stream is a config id, not a job id (so it stays unstamped).
            is_batch = group.startswith(_BATCH_EVALUATIONS_PREFIX)
            for record in run_logs_insights(
                client, group, self._config, cross_check=True, carry_stream=is_batch
            ):
                result = eval_record_to_result(record)
                if result is not None:
                    yield result

    def _evaluation_result_groups(self, client: Any) -> list[str]:
        """The CloudWatch log groups AgentCore writes evaluation results to (documented prefix)."""
        groups: list[str] = []
        token: str | None = None
        while True:
            params: dict[str, Any] = {"logGroupNamePrefix": _EVALUATIONS_LOG_GROUP_PREFIX}
            if token:
                params["nextToken"] = token
            try:
                resp = client.describe_log_groups(**params)
            except Exception as exc:
                raise AgentCoreError(
                    f"could not list the evaluation result log groups "
                    f"({_EVALUATIONS_LOG_GROUP_PREFIX}*): {exc}"
                ) from exc
            groups.extend(
                g["logGroupName"] for g in resp.get("logGroups", []) if g.get("logGroupName")
            )
            token = resp.get("nextToken")
            if not token:
                return groups


def agentcore_source(clients: AwsClients, config: AgentCoreSourceConfig) -> AgentCoreTraceSource:
    """Build the trace source for ``config``: the Omni reader, or the Logs Insights fallback.

    Omni is used when an Omni domain is configured; otherwise spans are read from CloudWatch's
    ``aws/spans`` and content from the runtime's ``otel-rt-logs`` group via Logs Insights.
    """
    reader: OmniReader | LogsInsightsReader = (
        OmniReader(clients, config) if config.uses_omni else LogsInsightsReader(clients, config)
    )
    return AgentCoreTraceSource(reader, config)
