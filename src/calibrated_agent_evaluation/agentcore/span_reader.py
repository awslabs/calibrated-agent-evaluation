# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read one AgentCore Runtime run's span tree by id — the live ``RunSpanReader`` the executor uses.

The AgentCore Runtime executor invokes an agent and, before stopping the run's session, asks this
reader for that one run's span tree. Unlike the window ingest (which reads a time window of a
runtime's traces and yields them for the shared ingest path to normalize), this reader looks the
run up by the ids the executor generated for it and returns the normalized ``Span`` tree directly,
or raises a typed :class:`~calibrated_agent_evaluation.agentcore.errors.AgentCoreError` rather than
return a partial or empty result.

It is a by-id sibling of the window readers and reuses them: the window-ingest AWS drivers
(``run_omni_query`` / ``run_logs_insights`` + the row decoders), the ``adot_to_otlp`` content
re-home, and the window ingest's own normalization seam (write the assembled OTLP to a temporary
``.otlp.json`` and run it through ``detect_source`` → ``normalize``, which detects the Strands
grammar and keys every span to the Calibrated Agent Evaluation trace id). The net-new code is the
by-id query, the short settle-poll, and the whole-tree-or-raise orchestration the contract requires.
On the Logs Insights path each poll also reads the run's records straight from the log store with
``FilterLogEvents`` (unbilled), and a read the index has not caught up with is polled again.

Billed calls: only ``cloudwatchomni:StartTelemetryQuery`` is billed, so the Logs Insights path
(the default) spends none of the budget. On the Omni path one read issues two
``StartTelemetryQuery`` calls (spans + content) per poll, so an upper bound for one
``read_spans`` is ``2 * (ceil(settle_deadline_seconds / poll_interval_seconds) + 1)``, which is 62
with the defaults (300s, every 10s); a budget on the injected clients covers the invocations,
retries and this read together. The deadline bounds when a new poll may start, not total wall
time: a single Logs Insights or Omni query can itself take a while to complete.
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation._errors import CalibratedAgentEvaluationDataError
from calibrated_agent_evaluation.agentcore.adot import adot_to_otlp
from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreContentMissing,
    AgentCoreError,
    AgentCoreMixedSession,
    AgentCoreSettlingLag,
)
from calibrated_agent_evaluation.agentcore.readers import (
    decode_content_row,
    decode_span_row,
    read_trace_events,
    require_otel_trace_id,
    run_logs_insights,
    run_omni_query,
)
from calibrated_agent_evaluation.agentcore.sources import AgentCoreSourceConfig, TimeWindow
from calibrated_agent_evaluation.ingest.detect import detect_source
from calibrated_agent_evaluation.normalize import normalize

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.run_spans import RunSpanQuery
    from calibrated_agent_evaluation.agentcore.session import AwsClients
    from calibrated_agent_evaluation.schema.types import Span

_OMNI_SERVICE = "cloudwatchomni"
_LOGS_SERVICE = "logs"
# Transaction Search writes every runtime's spans into this one shared CloudWatch Logs group.
_SPANS_LOG_GROUP = "aws/spans"
# The charset an AgentCore runtime name may use; the (functionally unused) config runtime field is
# sanitized to it so AgentCoreSourceConfig construction does not reject an odd runtime-arn tail.
_RUNTIME_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
# Measured live: a run's spans land within about ten seconds, but its content records can trail
# them in the Logs Insights index for minutes (156s from invoke to a whole tree on the Logs Insights
# path). The default leaves room for that, and polls every ten seconds so a billed Omni read stays
# within a small bound.
_DEFAULT_SETTLE_DEADLINE_SECONDS = 300.0
_DEFAULT_POLL_INTERVAL_SECONDS = 10.0
# Slack added to each side of the invocation window (see read_spans).
_WINDOW_SLACK_SECONDS = 60


def _require_w3c(trace_id: str) -> str:
    """The run's 32-hex W3C trace id, lowercased, or a refusal.

    Validated before it reaches either query (Omni SQL has no bind parameters), through the same
    validator the Logs Insights filter uses.
    """
    return require_otel_trace_id(trace_id)


def _safe_runtime(runtime_arn: str) -> str:
    """A charset-safe runtime name for the config (functionally unused by the by-id read)."""
    tail = runtime_arn.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    safe = "".join(c for c in tail if c in _RUNTIME_NAME_CHARS)
    return safe or "runtime"


def _objects(records: list[Any], w3c: str) -> list[dict[str, Any]]:
    """The decoded records, refusing any that is not a JSON object rather than crashing on it."""
    if any(not isinstance(record, dict) for record in records):
        raise AgentCoreError(
            f"a Logs Insights record for run {w3c!r} was not a JSON object; the read is refused "
            "rather than guessed at"
        )
    return records


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _session_id(span: dict[str, Any]) -> str | None:
    """The ``session.id`` attribute of a decoded span (attributes are a dict in these records)."""
    value = _mapping(span.get("attributes")).get("session.id")
    return value if isinstance(value, str) and value else None


def _content_log_groups(span: dict[str, Any]) -> list[str]:
    """The runtime's own ``otel-rt-logs`` content group(s), from the span's resource attribute."""
    names = _mapping(_mapping(span.get("resource")).get("attributes")).get("aws.log.group.names")
    if isinstance(names, str):
        return [names]
    if isinstance(names, list):
        return [n for n in names if isinstance(n, str) and n]
    return []


class _NoAgentTree(AgentCoreError):
    """A stable read whose spans form no agent span tree (only the runtime's HTTP span, so far)."""


class _IndexBehindStore(AgentCoreSettlingLag):
    """A Logs Insights read missing records the log store already holds for the run."""

    def __init__(self, index: tuple[int, int], store: tuple[int, int]) -> None:
        super().__init__(
            f"the Logs Insights index held {index[0]} of the {store[0]} spans and {index[1]} of "
            f"the {store[1]} content records the log store holds for the run"
        )
        self.index = index
        self.store = store


def _span_ids(spans: list[dict[str, Any]]) -> set[str]:
    return {s["spanId"] for s in spans if isinstance(s.get("spanId"), str)}


def _refuse_half_joined(tree: list[Span], w3c: str) -> None:
    """Refuse a completed agent span whose content came back with its input or output missing.

    ``adot_to_otlp`` refuses an agent span with no content at all, but one that got its input and
    not its output passes it: that is what a read returned live when the agent's final answer was
    not yet in the Logs Insights index. An errored agent can legitimately end without an answer,
    so it is returned for error analysis rather than refused.
    """
    for span in tree:
        if span.status != "completed":
            continue
        missing = [
            half for half, value in (("input", span.input), ("output", span.output)) if not value
        ]
        if missing:
            raise AgentCoreContentMissing(
                f"span {span.span_id!r} of run {w3c!r} completed but came back without its "
                f"{' and '.join(missing)}; its content records were not all readable, so the tree "
                "is refused rather than returned half joined. Retry once the telemetry settles"
            )


def _omni_spans_sql(window: TimeWindow, w3c_trace_id: str) -> str:
    # By-id: filter on the run's trace id, not a runtime's service.name. The id is 32-hex-validated
    # and the window bounds are ISO-8601-validated (TimeWindow), so neither can carry a quote; Omni
    # StartTelemetryQuery takes a query string with no bind-parameter facility.
    return (
        "SELECT traceId, spanId, parentSpanId, name, "  # noqa: S608
        "attributes['session.id'] AS session_id, `@message` "
        "FROM traces.default "
        f"WHERE `@timestamp` >= to_timestamp('{window.start}') "
        f"AND `@timestamp` < to_timestamp('{window.end}') "
        f"AND traceId = '{w3c_trace_id}'"
    )


def _omni_content_sql(window: TimeWindow, w3c_trace_id: str) -> str:
    return (
        "SELECT traceId, spanId, `@message` "  # noqa: S608 - inputs validated (see _omni_spans_sql)
        "FROM logs.default "
        f"WHERE `@timestamp` >= to_timestamp('{window.start}') "
        f"AND `@timestamp` < to_timestamp('{window.end}') "
        "AND `@logStream` = 'otel-rt-logs' "
        f"AND traceId = '{w3c_trace_id}'"
    )


class AgentCoreSpanReader:
    """Reads one AgentCore Runtime run's span tree by id, satisfying the executor's reader contract.

    Built from the same :class:`~calibrated_agent_evaluation.agentcore.session.AwsClients` as the
    executor, so its billed reads count against the shared budget and a guard's refusal (for example
    :class:`~calibrated_agent_evaluation.agentcore.errors.BilledCallBudgetExceeded`) propagates to
    the caller. The read path is the Logs Insights group ``aws/spans`` by default, or CloudWatch
    Omni when ``use_omni=True`` (Omni needs no domain, so the path is an explicit flag, as in the
    window ingest). The Logs Insights path needs ``logs:StartQuery``, ``logs:GetQueryResults`` and
    ``logs:FilterLogEvents`` on ``aws/spans`` and on the runtime's own log group.
    """

    def __init__(
        self,
        clients: AwsClients,
        *,
        use_omni: bool = False,
        settle_deadline_seconds: float = _DEFAULT_SETTLE_DEADLINE_SECONDS,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        # A zero interval would issue billed Omni queries back to back, and a deadline shorter than
        # one interval can never see the two equal polls a whole tree needs.
        if poll_interval_seconds <= 0 or settle_deadline_seconds < poll_interval_seconds:
            raise AgentCoreError(
                f"poll_interval_seconds must be > 0 and settle_deadline_seconds at least that long "
                f"(got {poll_interval_seconds:g}s and {settle_deadline_seconds:g}s)"
            )
        self._clients = clients
        self._use_omni = use_omni
        self._settle_deadline_seconds = settle_deadline_seconds
        self._poll_interval_seconds = poll_interval_seconds

    def read_spans(self, query: RunSpanQuery) -> list[Span]:
        """The run's whole normalized span tree, every span keyed to ``query.trace_id``.

        Polls the by-id read until a stable read (the same span count and content-record count on
        two polls in a row) assembles into a whole tree. On the Logs Insights path a read counts
        only when it holds the same spans and as many content records as the log store already
        holds for the run, since the index serves a run's records in batches minutes apart and
        two equal reads can both be missing one. A stable read that is not yet whole (no
        agent span tree yet, an agent with no content, or a completed agent missing its input or
        output) is polled again; at the deadline the reader raises that refusal
        (:class:`~calibrated_agent_evaluation.agentcore.errors.AgentCoreContentMissing`, or an
        ``AgentCoreError`` naming that no agent tree formed). A run that never appears, or is still
        changing at the deadline, raises :class:`AgentCoreSettlingLag`. A run carrying more than one
        session id raises :class:`AgentCoreMixedSession`. An errored agent span may lack its answer.
        A partial or truncated read raises its typed error (from the reused drivers). Never returns
        a partial or empty list.
        """
        w3c = _require_w3c(query.w3c_trace_id)
        # Pad the invocation window: a record stamped at or just after the client-observed end, or
        # clock skew between client and AWS, would otherwise be missed on every poll. The queries
        # filter on the 32-hex trace id, so a wider window cannot pull in another run.
        slack = timedelta(seconds=_WINDOW_SLACK_SECONDS)
        try:
            started = datetime.fromisoformat(query.started_at)
            ended = datetime.fromisoformat(query.ended_at)
        except (TypeError, ValueError) as exc:
            raise AgentCoreError(
                f"run {query.trace_id!r} has an invocation time that is not ISO-8601 "
                f"(started_at={query.started_at!r}, ended_at={query.ended_at!r})"
            ) from exc
        window = TimeWindow(start=(started - slack).isoformat(), end=(ended + slack).isoformat())
        config = AgentCoreSourceConfig(
            runtime=_safe_runtime(query.runtime_arn), window=window, omni=self._use_omni
        )
        deadline = time.monotonic() + self._settle_deadline_seconds
        previous: tuple[int, int] | None = None
        while True:
            try:
                spans, content, sessions = self._read_once(config, w3c)
            except _IndexBehindStore as behind:
                # Two equal index reads can both miss a batch the index has not served yet, so a
                # read the store contradicts never counts toward the two in a row.
                if time.monotonic() >= deadline:
                    raise AgentCoreSettlingLag(
                        f"run {w3c!r} was still being indexed at the "
                        f"{self._settle_deadline_seconds:g}s deadline: {behind}. Returning it now "
                        "would hand back a partial tree. Refused; retry with a longer settle "
                        "deadline"
                    ) from behind
                previous = None
                time.sleep(self._poll_interval_seconds)
                continue
            # The content records lag the spans in the Logs Insights index (observed live: the spans
            # had settled while an agent's final answer was not yet readable), so the tree has
            # stopped arriving only when both counts hold between two polls.
            signature = (len(spans), len(content))
            not_settled: AgentCoreError | None = None
            if spans and signature == previous:
                if len(sessions) > 1:
                    raise AgentCoreMixedSession(
                        f"run {query.trace_id!r} (w3c {w3c!r}) carries {len(sessions)} session ids "
                        f"({', '.join(sorted(sessions))}); a single invocation carries one "
                        "session, so the tree is ambiguous and is refused rather than fused"
                    )
                try:
                    return self._assemble(spans, content, query.trace_id, w3c)
                except (AgentCoreContentMissing, _NoAgentTree) as exc:
                    # Stable but not yet readable as a whole tree: content still lagging in the
                    # index, or the agent spans not landed yet. Poll again; refuse at the deadline.
                    not_settled = exc
            if time.monotonic() >= deadline:
                if not_settled is not None:
                    raise not_settled
                if not spans:
                    raise AgentCoreSettlingLag(
                        f"no spans for run {w3c!r} within {self._settle_deadline_seconds:g}s. "
                        "Check that the invocation's trace context was sampled: a traceparent "
                        "without the 01 sampled flag makes the runtime record no agent spans, even "
                        "though its content logs still land. Otherwise the telemetry has not "
                        "settled yet. Refused rather than returned empty"
                    )
                raise AgentCoreSettlingLag(
                    f"run {w3c!r} was still changing at the {self._settle_deadline_seconds:g}s "
                    f"deadline ({len(spans)} spans, {len(content)} content records on the last "
                    "poll); returning it now could hand back a partial tree. Refused; retry with a "
                    "longer settle deadline"
                )
            previous = signature
            time.sleep(self._poll_interval_seconds)

    def _assemble(
        self, spans: list[dict[str, Any]], content: list[dict[str, Any]], trace_id: str, w3c: str
    ) -> list[Span]:
        """Join, normalize and check one stable read; raise when it is not yet a whole tree."""
        otlp = adot_to_otlp(spans, content)
        tree = self._normalize(otlp, trace_id)
        if not tree:
            # Raw spans that normalize to no agent span tree (only the runtime's HTTP span so far,
            # or a short-circuited run) degrade to a probed trace; never hand back an empty list.
            raise _NoAgentTree(
                f"run {trace_id!r} (w3c {w3c!r}) normalized to no agent span tree; its spans did "
                "not form an owned trace (agent spans not landed, or a short-circuited run). "
                "Refused rather than returned as an empty list"
            )
        _refuse_half_joined(tree, w3c)
        return tree

    def _read_once(
        self, config: AgentCoreSourceConfig, w3c: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
        """One by-id read of the run's spans + content, through the configured path."""
        if self._use_omni:
            return self._read_omni(config, w3c)
        return self._read_logs_insights(config, w3c)

    def _read_omni(
        self, config: AgentCoreSourceConfig, w3c: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
        client = self._clients.client(_OMNI_SERVICE)
        span_rows, _ = run_omni_query(client, _omni_spans_sql(config.window, w3c))
        content_rows, _ = run_omni_query(client, _omni_content_sql(config.window, w3c))
        sessions = {
            row["session_id"]
            for row in span_rows
            if isinstance(row.get("session_id"), str) and row["session_id"]
        }
        spans = [decode_span_row(row) for row in span_rows]
        content = [decode_content_row(row) for row in content_rows]
        return spans, content, sessions

    def _read_logs_insights(
        self, config: AgentCoreSourceConfig, w3c: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
        client = self._clients.client(_LOGS_SERVICE)
        # Both queries filter to this run inside Logs Insights, so the shared aws/spans group is
        # read for one trace, not every runtime's spans in the window. The content read runs
        # without the window ingest's empty-read cross-check: inside a poll, a content index that
        # has not caught up is a reason to poll again, not an error.
        spans = _objects(run_logs_insights(client, _SPANS_LOG_GROUP, config, trace_id=w3c), w3c)
        sessions = {sid for span in spans if (sid := _session_id(span)) is not None}
        content: list[dict[str, Any]] = []
        for group in sorted({g for span in spans for g in _content_log_groups(span)}):
            content.extend(_objects(run_logs_insights(client, group, config, trace_id=w3c), w3c))
        # The log store has a run's records seconds after it ends; the index serves them in batches
        # minutes apart (observed live: 9 of 25 spans for two polls in a row while the store held
        # all 25), so the index read is only trusted once it matches the store.
        store_spans = read_trace_events(client, _SPANS_LOG_GROUP, config, w3c)
        store_content = sum(
            len(read_trace_events(client, group, config, w3c))
            for group in sorted({g for span in store_spans for g in _content_log_groups(span)})
        )
        if _span_ids(spans) != _span_ids(store_spans) or len(content) != store_content:
            raise _IndexBehindStore((len(spans), len(content)), (len(store_spans), store_content))
        return spans, content, sessions

    def _normalize(self, otlp: dict[str, Any], trace_id: str) -> list[Span]:
        """Normalize the assembled OTLP through the window ingest's seam, keyed to the trace id.

        The temporary file, source detection and normalization can fail outside the AgentCore
        error family (a filesystem error, an ingest or normalization refusal); each is wrapped so
        ``read_spans`` only ever raises an ``AgentCoreError``.
        """
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "run.otlp.json"
                path.write_text(json.dumps(otlp), encoding="utf-8")
                source = detect_source(path, trace_id=trace_id, frame="production-representative")
                trace = normalize(source)
        except (OSError, CalibratedAgentEvaluationDataError) as exc:
            raise AgentCoreError(
                f"could not normalize run {trace_id!r} into a span tree: {type(exc).__name__}"
            ) from exc
        return list(trace.spans)
