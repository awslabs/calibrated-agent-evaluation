# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AgentCore trace source and its configuration.

Read-only: the source pulls AgentCore telemetry from AWS and yields it as canonical OTLP-JSON for
the shared ingest path to normalize. It never writes to AWS. The configuration types are
vendor-free (plain ISO-8601 strings and flags); the source itself (added with the story tasks)
reads through the ``calibrated_agent_evaluation.agentcore`` AWS clients.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from pydantic import ConfigDict, field_validator, model_validator

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.agentcore.adot import adot_to_otlp
from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreContentMissing,
    AgentCoreError,
    AgentCoreIndexingTooLow,
    AgentCoreMixedSession,
    AgentCorePartialResult,
    AgentCoreTruncatedResult,
)
from calibrated_agent_evaluation.backends.base import SourceTrace

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.settle import SettlePolicy

logger = logging.getLogger(__name__)


class TimeWindow(CalibratedAgentEvaluationModel):
    """The read window, as timezone-aware ISO-8601 bounds.

    Timezone-aware is required: a naive bound compared against AWS timestamps silently shifts the
    window by the local offset, so a naive value is refused rather than coerced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = AgentCoreError

    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def _tz_aware_iso(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{value!r} is not ISO-8601") from exc
        if parsed.utcoffset() is None:
            raise ValueError(f"{value!r} has no timezone; window bounds must be tz-aware")
        return value

    @model_validator(mode="after")
    def _start_before_end(self) -> TimeWindow:
        # Swapped --since/--until would otherwise read an empty window and report "0 ingested"
        # with no error, which reads as "nothing happened" rather than "you asked for nothing".
        # Raise the family's typed error directly: a ``mode="after"`` validator's ``ValueError``
        # escapes as a raw pydantic ``ValidationError`` (the base's wrap only catches field-level
        # failures), so the typed error keeps the same surface as the field validators above.
        if datetime.fromisoformat(self.start) >= datetime.fromisoformat(self.end):
            raise AgentCoreError(
                f"window start {self.start!r} is not before end {self.end!r}; "
                "check the order of --since and --until"
            )
        return self


class AgentCoreSourceConfig(CalibratedAgentEvaluationModel):
    """What to read, resolved from the CLI flags.

    ``omni`` selects the CloudWatch Omni SQL path; otherwise spans are read from Logs Insights.
    (Omni queries need no domain — verified live — so this is a plain flag, not a domain name:
    a domain value would only pick the path while never reaching a call, so a typo would read the
    default silently.) ``rekey_sessions`` opts into collapsing a mixed-session trace to the runtime
    session id. ``with_scores`` additionally reads AgentCore evaluator results as advisory labels.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = AgentCoreError

    runtime: str
    window: TimeWindow
    omni: bool = False
    rekey_sessions: bool = False
    with_scores: bool = False

    @field_validator("runtime")
    @classmethod
    def _safe_runtime_name(cls, value: str) -> str:
        # The runtime name is interpolated into the Omni SQL (which has no bind parameters), so it
        # is restricted to the AgentCore runtime-name charset; a name with a quote or space cannot
        # reach the query and is refused here, closing the only free interpolation in the builder.
        if not value or any(c not in _RUNTIME_NAME_CHARS for c in value):
            raise ValueError(
                f"runtime name {value!r} must be non-empty and use only letters, digits, "
                "'.', '_' or '-'"
            )
        return value

    @property
    def uses_omni(self) -> bool:
        """Whether this config reads through the Omni SQL path."""
        return self.omni


# --- read-path helpers: pure, response-shaped, directly testable ---------------------------

_MAX_OMNI_WINDOW_DAYS = 365
# The charset an AgentCore runtime name may use; enforced so it is safe to interpolate into SQL.
_RUNTIME_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
# Logs Insights returns at most this many rows per query; a match above it is a truncated read.
LOGS_INSIGHTS_RETRIEVAL_CAP = 100_000
# X-Ray retrieves traces in batches of at most this many ids.
XRAY_TRACE_BATCH = 100


def build_omni_sql(config: AgentCoreSourceConfig, *, session_name: str) -> str:
    """Build the Omni SQL that reads a runtime's spans in the window.

    Encodes the query rules the verified account needs: a ``@timestamp`` bound written with
    ``to_timestamp`` and shorter than 365 days, and ``attributes['session.id']`` map access (the
    backtick form returns an empty string with no error). ``session_name`` must be unique per
    in-flight query; the caller supplies it (and asserts uniqueness).
    """
    start = datetime.fromisoformat(config.window.start)
    end = datetime.fromisoformat(config.window.end)
    if (end - start).days >= _MAX_OMNI_WINDOW_DAYS:
        raise AgentCoreError(
            f"the Omni window {config.window.start}..{config.window.end} spans "
            f"{(end - start).days} days; it must be shorter than {_MAX_OMNI_WINDOW_DAYS} days"
        )
    # `session_name` is embedded by the caller in StartTelemetryQuerySession, not in the SQL; it is
    # accepted here so the builder can validate it is non-empty alongside the window.
    if not session_name:
        raise AgentCoreError("an Omni query session name is required and must be unique")
    # Interpolation is safe here: the window bounds are ISO-8601-validated (TimeWindow) so cannot
    # carry a quote, and the runtime name is charset-restricted (AgentCoreSourceConfig validator).
    # Omni's StartTelemetryQuery takes a query string with no bind-parameter facility.
    return (
        "SELECT traceId, spanId, parentSpanId, name, "  # noqa: S608
        "attributes['session.id'] AS session_id, `@message` "
        "FROM traces.default "
        f"WHERE `@timestamp` >= to_timestamp('{config.window.start}') "
        f"AND `@timestamp` < to_timestamp('{config.window.end}') "
        f"AND resource['attributes']['service.name'] = '{config.runtime}'"
    )


def refuse_partial_result(statistics: dict[str, Any]) -> None:
    """Raise :class:`AgentCorePartialResult` when an Omni result is partial."""
    partial = (statistics.get("partialResults") or {}).get("partialResultsDetected")
    if partial:
        raise AgentCorePartialResult(
            "the Omni query returned partialResultsDetected=true; the rows are an unknown subset "
            "of the match, so the read is refused rather than returned incomplete"
        )


def refuse_truncated_result(records_matched: int, rows_retrieved: int) -> None:
    """Raise :class:`AgentCoreTruncatedResult` when a Logs Insights match exceeds what was read."""
    if records_matched > rows_retrieved:
        raise AgentCoreTruncatedResult(
            f"Logs Insights matched {records_matched} records but returned only "
            f"{rows_retrieved} (cap {LOGS_INSIGHTS_RETRIEVAL_CAP}); the trace set would be "
            "incomplete, so the read is refused. Narrow the window and re-run"
        )


def refuse_indexing_below_full(indexing_rules: list[dict[str, Any]]) -> None:
    """Raise :class:`AgentCoreIndexingTooLow` when the X-Ray Default rule indexes below 100%."""
    for rule in indexing_rules:
        if rule.get("Name") == "Default":
            pct = (rule.get("Rule", {}).get("Probabilistic", {})).get("DesiredSamplingPercentage")
            if pct is not None and pct < 100:
                raise AgentCoreIndexingTooLow(
                    f"X-Ray indexes {pct}% of spans on the Default rule, not 100%; a read through "
                    "X-Ray or the trace list would see only the indexed fraction, so it is "
                    "refused. Raise the Default rule to 100% for the ingest, then restore it"
                )
            return


def xray_id_batches(trace_ids: list[str]) -> Iterator[list[str]]:
    """Yield ``trace_ids`` in batches of at most :data:`XRAY_TRACE_BATCH`."""
    for start in range(0, len(trace_ids), XRAY_TRACE_BATCH):
        yield trace_ids[start : start + XRAY_TRACE_BATCH]


# --- the pull-only source: orchestration over an injected reader --------------------------


@dataclass(frozen=True)
class RawTrace:
    """One trace as a reader decoded it from AWS, before Calibrated Agent Evaluation normalization.

    ``span_records`` are OTLP span dicts; ``log_records`` are the decoded ``otel-rt-logs`` content
    records (each a plain dict with ``spanId``/``eventName``/``body``) that the converter re-homes
    onto the spans (see :mod:`calibrated_agent_evaluation.agentcore.adot`); ``session_ids`` are the
    distinct ``session.id`` values the trace carried (more than one is a mixed-session trace);
    ``runtime_session_id`` is the runtime's own session id, used when a mixed-session trace is
    re-keyed.
    """

    trace_id: str
    span_records: list[dict[str, Any]]
    log_records: list[dict[str, Any]]
    session_ids: tuple[str, ...] = ()
    runtime_session_id: str | None = None


@runtime_checkable
class AgentCoreReader(Protocol):
    """Reads AgentCore telemetry from AWS and decodes it to :class:`RawTrace`s.

    The vendor seam: each concrete reader (Omni, or Logs Insights + X-Ray) does the AWS calls and
    response decoding. ``read_traces`` must raise ``BackendError`` before yielding anything if the
    account cannot be read, and must raise a whole-read completeness refusal (partial/truncated/
    indexing) before yielding anything, so nothing partial reaches the store.
    """

    def read_traces(self) -> Iterator[RawTrace]:
        """Yield each trace in the window as a :class:`RawTrace`."""
        ...

    def read_scores(self) -> Iterator[dict[str, Any]]:
        """Yield each evaluator result record (empty when there are none)."""
        ...


class AgentCoreTraceSource:
    """A read-only :class:`~calibrated_agent_evaluation.backends.base.TraceSource` for AgentCore.

    Satisfies the ``TraceSource`` protocol (``pull`` only); it is not a store and exposes no write.
    The AWS reads are the injected ``reader``; this class owns the Calibrated Agent Evaluation-side
    orchestration: wait for the window to settle, resolve session identity, convert to OTLP, and
    refuse (per trace) a trace whose content did not come back.
    """

    def __init__(
        self,
        reader: AgentCoreReader,
        config: AgentCoreSourceConfig,
        *,
        settle: SettlePolicy | None = None,
    ) -> None:
        from calibrated_agent_evaluation.agentcore.settle import SettlePolicy as _SettlePolicy

        self._reader = reader
        self._config = config
        self._settle = settle if settle is not None else _SettlePolicy.from_env()
        # Mixed-session traces re-keyed this pull: (original id, runtime-session id). The ingest
        # report reads this to record the re-key for each such trace.
        self.rekeyed: list[tuple[str, str]] = []

    def pull(self) -> Iterator[SourceTrace]:
        """Yield each AgentCore trace as a :class:`SourceTrace` carrying canonical OTLP-JSON."""
        # Wait for the window's writes to settle (bounded); a window too recent to settle in time
        # raises before any trace is yielded, so a partial trace is never ingested.
        self._settle.wait_for_window(self._config.window)
        # The reader raises before its first yield when the account is unreachable or the read is
        # incomplete (partial/truncated/indexing), so those propagate out of pull untouched.
        for raw in self._reader.read_traces():
            yield self._to_source_trace(raw)

    def read_scores(self) -> Iterator[dict[str, Any]]:
        """The reader's evaluator-result records (mapped to labels by the scores module)."""
        return self._reader.read_scores()

    def _to_source_trace(self, raw: RawTrace) -> SourceTrace:
        try:
            trace_id, session_id = self._resolve_session(raw)
            otlp = adot_to_otlp(raw.span_records, raw.log_records)
        except (AgentCoreMixedSession, AgentCoreContentMissing) as exc:
            # One trace that is mixed-session (and not re-keyed) or missing content is reported as
            # its own error and skipped; it never fuses sessions or persists an empty-content span,
            # and the rest of the pull carries on.
            return SourceTrace(source_id=raw.trace_id, error=str(exc))
        return SourceTrace(source_id=trace_id, otlp=otlp, session_id=session_id)

    def _resolve_session(self, raw: RawTrace) -> tuple[str, str | None]:
        """The id to store the trace under, and the session it belongs to (``None`` if none).

        A blank ``session.id`` names no session, so it is dropped before sessions are counted: it
        neither makes a trace mixed-session nor reaches the stored trace, where it is refused.
        """
        distinct = {session for session in raw.session_ids if session.strip()}
        if len(distinct) <= 1:
            return raw.trace_id, next(iter(distinct), None)
        if not self._config.rekey_sessions:
            raise AgentCoreMixedSession(
                f"trace {raw.trace_id!r} carries {len(distinct)} session ids "
                f"({', '.join(sorted(distinct))}); AgentCore Evaluate rejects a mixed-session "
                "trace. Pass --rekey-sessions to collapse them to the runtime session id"
            )
        # A blank runtime session names no session, so it is neither the key nor the session.
        runtime = raw.runtime_session_id if (raw.runtime_session_id or "").strip() else None
        new_id = runtime or raw.trace_id
        # Two different mixed-session traces that collapse to the same runtime session id would be
        # stored under one id, silently overwriting one. Refuse the collision rather than lose a
        # trace; the re-keyed id is deterministic (sessions are sorted) so this is reproducible.
        collided_with = next(
            (old for old, new in self.rekeyed if new == new_id and old != raw.trace_id), None
        )
        if collided_with is not None:
            raise AgentCoreMixedSession(
                f"trace {raw.trace_id!r} re-keys to runtime session {new_id!r}, already used by "
                f"trace {collided_with!r}; collapsing both would overwrite one. Narrow the window "
                "so each runtime session appears once, or ingest them separately"
            )
        self.rekeyed.append((raw.trace_id, new_id))
        logger.info("re-keyed mixed-session trace %s to runtime session %s", raw.trace_id, new_id)
        # The trace id the key falls back to is not a session, so it is never stamped as one.
        return new_id, runtime
