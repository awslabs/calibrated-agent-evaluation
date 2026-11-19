# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Langfuse trace-store adapter.

The both-worlds backend: the *same* normalized methodology plane runs against self-hosted
Langfuse with no change. Langfuse is REUSED unmodified via its public API; no fork,
no schema modification.

**Written against the real Langfuse v3 SDK** (verified against ``langfuse==3.15.0``;
the team rule "never invent a vendor API" applies). The v3
SDK is OpenTelemetry-span based for writes and REST for reads:

  * **write**: deterministic id via :meth:`create_trace_id` (seeded with our ``trace_id`` so
    the same normalized id maps to the same Langfuse trace); spans via ``start_span`` /
    ``start_observation`` bound to a ``TraceContext``; trace-level tags+metadata via the
    span's ``update_trace``; labels via the synchronous REST ``client.api.score.create``, never the
    queued ``create_score``, whose background consumer logs and drops a rejected score.
  * **read**: ``client.api.trace.get`` / ``client.api.trace.list`` / ``client.api.score_v_2.get``.

The three locked adapter conventions:
  * queryable dims ``frame`` / ``ownership`` / ``taxonomy_cell`` are encoded as Langfuse
    **tags** (Langfuse has no arbitrary-metadata filter), so ``query`` is a tag filter;
  * span ``seq`` is persisted **explicitly** in observation metadata (OTel spans carry no
    domain ordering of their own);
  * bespoke ``Label`` fields ride in free-form ``Score.metadata``.

To guarantee the field-for-field round-trip on top of those *query* conventions, the
full normalized artifact also rides in ``metadata``; tags are how you *find* a trace,
metadata is how it is *reconstructed losslessly*. Nothing here leaks a Langfuse type up to
the methodology plane: the surface is ``Trace`` / ``Label`` only.

The Langfuse client is taken by **injection** so unit tests run against a stub that mirrors
the real surface; :meth:`LangfuseBackend.from_credentials` is the production path that builds
the real SDK client for the OSS build with telemetry disabled. The binding to the
live SDK + a real unmodified instance is validated by the opt-in integration test
against the reusable ``docker/langfuse/`` stack.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import uuid
from collections.abc import Callable, Generator, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast

from calibrated_agent_evaluation.backends._bounded import call_bounded
from calibrated_agent_evaluation.backends._export_tap import ExportTap, tapped_tracer_provider
from calibrated_agent_evaluation.backends.base import SkippedTrace, SourceTrace, TraceScan
from calibrated_agent_evaluation.schema.errors import BackendError, SchemaError, TraceNotFoundError
from calibrated_agent_evaluation.schema.serde import (
    dedup_labels,
    label_from_dict,
    label_identity,
    to_dict,
    trace_from_dict,
)
from calibrated_agent_evaluation.schema.types import Frame, Label, Ownership, Span, Trace
from calibrated_agent_evaluation.schema.validate import validate

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _scoped_env(name: str, value: str) -> Generator[None, None, None]:
    """Set ``os.environ[name] = value`` for the block, then restore the prior value."""
    sentinel = object()
    prior: str | object = os.environ.get(name, sentinel)
    os.environ[name] = value
    try:
        yield
    finally:
        if prior is sentinel:
            os.environ.pop(name, None)
        else:
            os.environ[name] = cast("str", prior)


# Langfuse Enterprise-Edition-gated surfaces the adapter must NEVER touch: adopting
# an EE feature would pull in compliance telemetry that cannot be disabled. The
# adapter restricts itself to the MIT/OSS public API; this denylist is asserted in tests.
EE_GATED_METHODS = frozenset(
    {
        "create_annotation_queue",
        "get_annotation_queue",
        "configure_rbac",
        "configure_sso",
        "set_data_retention",
        "get_audit_logs",
    }
)

# Tag encodings for the queryable dimensions. Prefixes keep the three dims
# distinguishable in Langfuse's flat tag list so `query` can filter precisely.
_FRAME_TAG = "frame:"
_OWNERSHIP_TAG = "ownership:"
_CELL_TAG = "cell."

# Page size for the paginated `trace.list` read. The v3 REST list endpoint paginates
# (verified: `trace.list(page=, limit=)` → `.data` + `.meta.total_pages`, langfuse==3.15.0); we
# walk every page so a large result set is never silently truncated ("no silent failure").
_QUERY_PAGE_SIZE = 50

# Metadata envelope keys; where the lossless normalized payload rides (the round-trip).
_TRACE_ENVELOPE = (
    "calibrated_agent_evaluation_trace"  # Trace doc minus spans, on the trace metadata
)
_SPAN_ENVELOPE = "calibrated_agent_evaluation_span"  # full Span doc, on each observation's metadata
_LABEL_ENVELOPE = "calibrated_agent_evaluation_label"  # full Label doc, on the Score metadata
_SCORE_NAME = "calibrated_agent_evaluation_label"
# Retraction tombstones. `create_score` has no update and score deletion
# shares the asynchronous Postgres+ClickHouse delete path that proved unreliable for traces, so a
# retraction is a second, append-only score carrying the retracted Label doc. A separate score name
# keeps tombstones out of Langfuse's own verdict views.
_RETRACTION_ENVELOPE = "calibrated_agent_evaluation_retraction"
_RETRACTION_SCORE_NAME = "calibrated_agent_evaluation_label_retraction"
# UTC write time stamped on label and tombstone scores alike, so a read can order a put against a
# retraction of the same label (last write wins). A score written before this key existed reads as
# the earliest possible write.
_WRITTEN_AT = "calibrated_agent_evaluation_written_at"
# Langfuse keeps every observation a trace id was ever given and merges its tags across updates, so
# a re-put with changed content left the old spans and the old frame/ownership/cell tags in place.
# Each put stamps one generation on the trace metadata and on every observation it writes, plus how
# many spans it wrote; a read keeps only the latest generation's spans and refuses a short set.
_GENERATION = "calibrated_agent_evaluation_generation"
_SPAN_COUNT = "calibrated_agent_evaluation_span_count"


_stamp_lock = threading.Lock()  # guards _last_stamp, the only mutable state here
_last_stamp: datetime | None = None


# Langfuse drops a session id that is not US-ASCII or is 200 characters or longer (its sessions
# docs), so such an id is not sent as the native session; the envelope still keeps it.
_NATIVE_SESSION_LIMIT = 200


def _native_session(trace: Trace) -> str | None:
    """The trace's session as Langfuse can hold it natively.

    Returns ``None`` silently when the trace has no session, and ``None`` with one warning naming
    the trace and the reason when Langfuse cannot hold the session.
    """
    session = trace.session_id
    if session is None:
        return None
    if not session.isascii():
        reason = "is not US-ASCII"
    elif len(session) >= _NATIVE_SESSION_LIMIT:
        reason = (
            f"is {len(session)} characters, and Langfuse's limit is under {_NATIVE_SESSION_LIMIT}"
        )
    else:
        return session
    logger.warning(
        "trace %s: session_id %s, so Langfuse cannot group the trace in its Sessions view. "
        "Nothing is lost: the trace keeps its session_id and reads back with it. Use a shorter "
        "US-ASCII session id to group it in Langfuse",
        trace.trace_id,
        reason,
    )
    return None


def _written_at() -> str:
    """A UTC write stamp, strictly increasing within this process.

    Fixed-width ISO-8601, so string order is time order. Strictly increasing because a coarse
    clock (Windows ticks in milliseconds) could give a retraction and the re-put right after it
    the same stamp, and the tie would hide a label that was written last.
    """
    global _last_stamp
    with _stamp_lock:
        now = datetime.now(tz=UTC)
        if _last_stamp is not None and now <= _last_stamp:
            now = _last_stamp + timedelta(microseconds=1)
        _last_stamp = now
    return now.isoformat(timespec="microseconds")


class LangfuseSpanLike(Protocol):
    """The span/observation handle returned by ``start_span`` / ``start_observation``.

    Mirrors the real ``langfuse._client.span.LangfuseSpan`` public methods the adapter uses.
    """

    id: str

    def update_trace(
        self,
        *,
        tags: list[str] | None = ...,
        metadata: Any | None = ...,
        input: Any | None = ...,
        output: Any | None = ...,
        session_id: str | None = ...,
    ) -> Any:
        """Mirror of ``LangfuseSpan.update_trace``: set trace-level tags, metadata, I/O, session."""

    def end(self) -> Any:
        """Mirror of ``LangfuseSpan.end``: close the span."""


class LangfuseApiLike(Protocol):
    """The public REST read surface (``client.api``); mirrors the real ``FernLangfuse``."""

    trace: Any
    score: Any
    score_v_2: Any
    projects: Any


class LangfuseLike(Protocol):
    """The minimal **public** Langfuse v3 client surface the adapter depends on.

    Every member below exists on the real ``langfuse.Langfuse`` client (verified against
    ``langfuse==3.15.0``); the adapter never reaches a private/underscored attribute.
    """

    api: LangfuseApiLike

    def create_trace_id(self, *, seed: str | None = ...) -> str:
        """Mirror of ``Langfuse.create_trace_id``: a deterministic trace id from ``seed``."""
        ...

    def start_span(
        self,
        *,
        trace_context: Any,
        name: str,
        input: Any | None = ...,
        output: Any | None = ...,
        metadata: Any | None = ...,
    ) -> LangfuseSpanLike:
        """Mirror of ``Langfuse.start_span``: open the root span of a trace."""
        ...

    def start_observation(
        self,
        *,
        trace_context: Any,
        name: str,
        as_type: str = ...,
        input: Any | None = ...,
        output: Any | None = ...,
        metadata: Any | None = ...,
    ) -> LangfuseSpanLike:
        """Mirror of ``Langfuse.start_observation``: open a child observation."""
        ...

    def create_score(
        self,
        *,
        trace_id: str,
        observation_id: str | None = ...,
        name: str,
        value: float | str,
        data_type: str | None = ...,
        comment: str | None = ...,
        metadata: Any | None = ...,
    ) -> None:
        """Mirror of ``Langfuse.create_score``: queue one score on a trace or observation."""

    def flush(self) -> None:
        """Mirror of ``Langfuse.flush``: block until queued writes are sent."""

    def shutdown(self) -> None:
        """Mirror of ``Langfuse.shutdown``: flush and stop the client's background workers."""

    def auth_check(self) -> bool:
        """Mirror of ``Langfuse.auth_check``: whether the credentials are accepted."""
        ...


def _matches(
    trace: Trace,
    frame: Frame | None,
    cell: dict[str, str] | None,
    ownership: Ownership | None,
) -> bool:
    """Whether a reconstructed trace itself satisfies the filters the tag listing applied.

    The tags only find candidates. Langfuse merges a trace's tags across puts, so a trace reframed
    from discovery to production-representative still carries ``frame:discovery``; and the cell
    tag ``cell.{k}:{v}`` reads the same for ``{"a:b": "c"}`` and ``{"a": "b:c"}``. Checking the
    trace's own fields is what keeps one frame's data out of another frame's rates.
    """
    if frame is not None and trace.frame != frame:
        return False
    if ownership is not None and trace.ownership != ownership:
        return False
    return all(trace.taxonomy_cell.get(k) == v for k, v in (cell or {}).items())


def _refuse_unresolved_parents(trace_id: str, write_order: list[Span]) -> None:
    """Refuse a span whose parent cannot be resolved within the trace before any write happens.

    ``_parents_first`` leaves such a span at the end of the order rather than dropping it, so a
    cycle or a dangling parent surfaces here. Writing it anyway would silently re-parent it to the
    trace root -- a changed topology, and the topology is what crew attribution reads.
    """
    emitted: set[str] = set()
    for span in write_order:
        if span.parent is not None and span.parent not in emitted:
            raise BackendError(
                f"span {span.span_id!r} names parent {span.parent!r}, which is not in trace "
                f"{trace_id!r} (or forms a cycle); refusing to re-parent it to the trace root, "
                "which would silently change the topology"
            )
        emitted.add(span.span_id)


def _parents_first(spans: list[Span]) -> list[Span]:
    """Order spans so every span follows its parent, seq-ordered within a level.

    Was ``sorted(spans, key=seq)``, which assumes a parent always has a lower ``seq`` than its
    children. Nothing enforces that -- ``seq`` is an execution-order field, not a depth -- so an
    out-of-order ``calibrated_agent_evaluation.span.seq`` left the parent's Langfuse id unknown when
    the child was written, and the child was silently re-parented to the trace root instead of
    raising.

    Emits any span whose parent is already emitted (or absent), repeating until nothing more can be;
    a span in a cycle or naming a missing parent is left out, and the caller refuses it explicitly.
    """
    remaining = sorted(spans, key=lambda s: s.seq)
    emitted: set[str] = set()
    ordered: list[Span] = []
    while remaining:
        ready = [s for s in remaining if s.parent is None or s.parent in emitted]
        if not ready:
            break  # a cycle or a dangling parent: the caller names it
        for span in ready:
            emitted.add(span.span_id)
            ordered.append(span)
        remaining = [s for s in remaining if s.span_id not in emitted]
    return ordered + remaining


class LangfuseBackend:
    """A ``TraceStoreBackend`` over an unmodified Langfuse v3 deployment.

    Writes block until the Langfuse client confirms its queue drained, so a returned write is
    durable. Build one with :meth:`from_credentials`; the constructor takes an injected client.

    Args:
        client: A Langfuse v3 client, or a stand-in with the same surface.
        telemetry_disabled: Recorded for provenance; the client never sends usage telemetry.
        flush_timeout_seconds: How long a write waits for the client's queue to drain before it
            fails with ``BackendError``.
        export_tap: Counts the client's failed span exports, so a trace write whose export
            failed raises instead of returning. :meth:`from_credentials` wires one in.
    """

    def __init__(
        self,
        client: LangfuseLike,
        *,
        telemetry_disabled: bool = True,
        flush_timeout_seconds: float = 20.0,
        export_tap: ExportTap | None = None,
    ) -> None:
        self._client = client
        self._export_tap = export_tap
        self.telemetry_disabled = telemetry_disabled
        # Bounds the blocking flush() (see _flush_bounded). Defaults higher than the per-request
        # timeout because flush waits on the whole queued batch to drain, not one request.
        self._flush_timeout_seconds = flush_timeout_seconds

    @classmethod
    def from_credentials(
        cls,
        public_key: str,
        secret_key: str,
        host: str,
        *,
        timeout_seconds: int = 5,
        flush_timeout_seconds: float = 20.0,
        client_factory: Callable[..., Any] | None = None,
    ) -> LangfuseBackend:
        """Construct against the OSS build with telemetry disabled.

        Telemetry-off is enforced in two real places, NOT by the client alone: (a) the OSS
        *server* reads ``TELEMETRY_ENABLED`` (set in ``docker/langfuse/docker-compose.yml``), and
        (b) this adapter restricts itself to the MIT/OSS public surface; the ``EE_GATED_METHODS``
        denylist (asserted in tests) is the client-side guarantee that no enterprise-licensed
        feature is invoked. The v3 Python client does NOT read ``TELEMETRY_ENABLED`` (it only
        reads ``LANGFUSE_TRACING_ENABLED`` / the ``tracing_enabled`` kwarg, which gate our own
        span *writes*; flipping those would break ``put_trace``, so we must not). We still set the
        server-recognized ``TELEMETRY_ENABLED=false`` in-process while the client is built; a
        harmless belt-and-suspenders for a co-located server that reads the process env; and
        restore the prior value after, so constructing a backend has no lasting global side effect.
        ``client_factory`` is an injection seam for tests; in production it imports the real MIT
        SDK. The real ``langfuse.Langfuse`` client structurally satisfies :class:`LangfuseLike`
        at the call sites the adapter uses (verified surface).

        ``timeout_seconds`` bounds every API request; including the synchronous reachability
        probe (``auth_check``); so an unreachable backend fails fast instead of hanging. The
        real SDK accepts ``timeout`` (seconds; verified langfuse==3.15.0) and defaults it to 5;
        we surface it as an explicit, tunable parameter rather than rely on that default.
        ``flush_timeout_seconds`` separately bounds the blocking ``flush()`` after a write; the
        SDK's ``flush`` does an unbounded ``queue.join()`` (verified langfuse==3.15.0) that hangs
        forever if the ingestion queue wedges, so :meth:`_flush_bounded` caps it and raises a typed
        ``BackendError`` on timeout rather than blocking the caller indefinitely.
        """
        factory: Callable[..., Any]
        if client_factory is None:  # pragma: no cover - real SDK path, integration-tested
            from langfuse import Langfuse  # opt-in [langfuse] extra; MIT build

            factory = Langfuse
        else:
            factory = client_factory
        # The client gets a provider of its own (the SDK's documented `tracer_provider` argument)
        # rather than taking over the process's global one, so it exports only the spans this
        # backend writes, and its failed exports are counted (see
        # `calibrated_agent_evaluation.backends._export_tap`).
        tap = ExportTap()
        provider = tapped_tracer_provider(tap)
        # cast: the real langfuse.Langfuse client structurally provides every member of
        # LangfuseLike at the call sites used here (surface verified against langfuse==3.15.0);
        # pyright cannot prove the wider real signatures match the protocol.
        # Server-side telemetry env (the v3 client itself ignores it; see docstring); scoped to
        # construction and restored after so there is no lasting global os.environ mutation.
        with _scoped_env("TELEMETRY_ENABLED", "false"):
            raw_client = factory(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                timeout=timeout_seconds,
                tracer_provider=provider,
            )
        return cls(
            cast("LangfuseLike", raw_client),
            telemetry_disabled=True,
            flush_timeout_seconds=flush_timeout_seconds,
            export_tap=tap,
        )

    def __enter__(self) -> LangfuseBackend:
        return self

    def __exit__(self, *_exc: object) -> None:
        # On context exit we FLUSH (the durability boundary) but do NOT call the SDK's shutdown().
        #
        # The langfuse v3 client is a thin wrapper over a PROCESS-GLOBAL singleton
        # (``LangfuseResourceManager``, keyed by public_key) that owns the OTel tracer provider, the
        # ingestion queues, and 4 shared daemon threads (verified langfuse==3.15.0 source +
        # docs: "The Langfuse client is a singleton"). ``shutdown()`` permanently pause()+join()s
        # those SHARED consumer threads and unregisters the SDK's atexit hook, yet leaves the dead
        # manager cached in the registry; so a LATER backend for the same key reuses a manager
        # whose consumers are dead, and its next flush()'s unbounded ``queue.join()`` hangs forever.
        # Calling shutdown() per context-exit therefore poisons the singleton for every subsequent
        # caller (this is what wedged the end-to-end demo). The SDK's design is one long-lived
        # client per process, cleaned up ONCE by its own ``atexit.register(shutdown)`` hook; the
        # workers are daemon threads, so flush-and-leave-resident leaks nothing at process exit.
        # We still flush here so a caller's writes are confirmed durable before the ``with`` closes.
        # __exit__ must never raise (it can run while an exception is already propagating), so the
        # bounded flush's BackendError is suppressed here; each put_trace/put_score already ran its
        # own bounded flush and raised at that call site if the write did not confirm, so this is a
        # best-effort final drain, not the durability gate.
        with contextlib.suppress(BackendError):
            self._flush_bounded("__exit__")

    def _trace_context(self, trace_id: str, parent_span_id: str | None = None) -> Any:
        # Map our normalized trace_id deterministically onto a Langfuse OTel trace id, so the
        # same normalized id always addresses the same Langfuse trace (idempotent put/get).
        lf_trace_id = self._client.create_trace_id(seed=trace_id)
        ctx: dict[str, str] = {"trace_id": lf_trace_id}
        if parent_span_id is not None:
            ctx["parent_span_id"] = parent_span_id
        return ctx

    def check_reachable(self) -> None:
        """Check that Langfuse can be reached and accepts these credentials, writing nothing.

        It is the synchronous probe every write runs first: the client's ``auth_check``, bounded
        by its request timeout. Run it before costly work whose results go here, so an unreachable
        host or refused keys stop the work before it starts.

        Raises:
            BackendError: Langfuse can't be reached, or it refuses the credentials.
        """
        self._require_reachable("reachability check")

    def _require_reachable(self, op: str) -> None:
        # The v3 SDK writes via a fire-and-forget async OTel exporter; start_span /
        # flush DO NOT raise when the backend is unreachable (they log and drop in the
        # background). So a write to a dead endpoint would silently no-op, violating
        # "unreachable → raise BackendError, never silently degrade worlds".
        # We therefore probe synchronously via the documented public auth_check() and surface a
        # typed BackendError before attempting the write. (Discovered by the both-worlds
        # acceptance gate against a real instance; the unit stub raised synchronously and
        # hid this.)
        try:
            ok = self._client.auth_check()
        except Exception as exc:  # connection/auth failure → typed BackendError
            raise self._as_backend_error(exc, op) from exc
        if not ok:
            raise BackendError(f"Langfuse {op} failed: backend unreachable or auth rejected")

    def _flush_bounded(self, op: str) -> None:
        # Bounded so a wedged ingestion queue fails as a typed BackendError instead of hanging
        # (see calibrated_agent_evaluation.backends._bounded).
        outcome = call_bounded(
            self._client.flush, timeout_seconds=self._flush_timeout_seconds, name="langfuse-flush"
        )
        if not outcome.finished:
            raise BackendError(
                f"Langfuse {op} failed: flush did not complete within "
                f"{self._flush_timeout_seconds:g}s (ingestion queue wedged); "
                "the write was not confirmed durable"
            )
        if outcome.error is not None:
            err = outcome.error
            raise self._as_backend_error(
                err if isinstance(err, Exception) else BackendError(str(err)), op
            )

    def _refuse_failed_export(self, op: str, failures_before: int) -> None:
        # The flush drained the span processor, so any export of this write has run by now. A
        # failure is counted by the tap rather than raised, because the SDK only logs it.
        if self._export_tap is None:
            return
        failed = self._export_tap.failures - failures_before
        if failed > 0:
            raise BackendError(
                f"Langfuse {op} failed: {failed} span export(s) to Langfuse did not succeed "
                f"({self._export_tap.last_error}); the write was not confirmed"
            )

    # --- write path -----------------------------------------------------------------------

    def put_trace(self, trace: Trace) -> None:
        """Write ``trace`` and its span tree to Langfuse, parents before children.

        Args:
            trace: The normalized trace to store.

        Raises:
            SchemaError: The trace fails validation, or a span names a parent that isn't in the
                trace; nothing is written.
            BackendError: Langfuse is unreachable, rejects the write, or doesn't confirm it within
                ``flush_timeout_seconds``.
        """
        from calibrated_agent_evaluation.schema.validate import validate

        validate(trace)
        self._require_reachable(f"put_trace({trace.trace_id!r})")
        failures_before = self._export_tap.failures if self._export_tap is not None else 0
        doc = to_dict(trace)
        tags = self._dimension_tags(trace)
        trace_envelope = {k: v for k, v in doc.items() if k != "spans"}
        generation = uuid.uuid4().hex
        trace_metadata = {
            _TRACE_ENVELOPE: trace_envelope,
            _GENERATION: generation,
            _SPAN_COUNT: len(trace.spans),
        }
        # Resolve the write order BEFORE the try: this refusal is about the trace, not the SDK, and
        # must not be caught and re-wrapped as a transport failure.
        write_order = _parents_first(trace.spans)
        _refuse_unresolved_parents(trace.trace_id, write_order)
        try:
            root = self._client.start_span(
                trace_context=self._trace_context(trace.trace_id),
                name=trace.trace_id,
                input=trace.input,
                output=trace.output,
                metadata={_TRACE_ENVELOPE: trace_envelope},
            )
            # Queryable dims as trace tags; full normalized trace (minus spans) in metadata.
            root.update_trace(
                tags=tags,
                metadata=trace_metadata,
                input=trace.input,
                output=trace.output,
                # Langfuse's own session too, so its Sessions view groups a session's traces.
                session_id=_native_session(trace),
            )
            lf_parent_ids: dict[str, str] = {}
            for span in write_order:
                parent_lf_id = lf_parent_ids.get(span.parent) if span.parent else None
                obs = self._client.start_observation(
                    trace_context=self._trace_context(trace.trace_id, parent_lf_id),
                    name=span.agent,
                    as_type="span",
                    input=span.input,
                    output=span.output,
                    # seq persisted explicitly; full span (incl. tool_calls) in metadata.
                    metadata={
                        "seq": span.seq,
                        _SPAN_ENVELOPE: to_dict(span),
                        _GENERATION: generation,
                    },
                )
                lf_parent_ids[span.span_id] = obs.id
                obs.end()
            root.end()
        except Exception as exc:
            raise self._as_backend_error(exc, f"put_trace({trace.trace_id!r})") from exc
        # Flush outside the try: _flush_bounded already raises a typed BackendError on timeout,
        # so wrapping it again would double-wrap. It is the durability confirmation for this write.
        self._flush_bounded(f"put_trace({trace.trace_id!r})")
        self._refuse_failed_export(f"put_trace({trace.trace_id!r})", failures_before)

    def put_score(self, label: Label) -> None:
        """Write ``label`` to Langfuse as a score on its trace (and span, if it names one).

        Args:
            label: The label to store.

        Raises:
            SchemaError: The label fails validation; nothing is written.
            BackendError: Langfuse is unreachable, rejects the write, or doesn't confirm it.
        """
        from calibrated_agent_evaluation.schema.validate import validate

        validate(label)
        self._create_score(
            f"put_score({label.trace_id!r})",
            trace_id=label.trace_id,
            name=_SCORE_NAME,
            value=label.outcome,
            comment=label.critique,
            # Bespoke Label fields (attribution, expected, …) ride in Score.metadata.
            metadata={_LABEL_ENVELOPE: to_dict(label), _WRITTEN_AT: _written_at()},
        )

    def retract_score(self, label: Label) -> None:
        """Retract ``label`` by appending a tombstone score (see ``TraceStoreBackend``).

        ``get_scores`` then hides every put of this label stamped before the tombstone, including
        one still in Langfuse's ingestion queue; a later put of the same label is visible again.
        There is deliberately no read-first existence check: score ingestion is asynchronous
        (verified live, a score takes seconds to become readable), so an undo pressed right after
        a verdict would find nothing and refuse to retract a label that is about to appear.
        """
        self._create_score(
            f"retract_score({label.trace_id!r})",
            trace_id=label.trace_id,
            name=_RETRACTION_SCORE_NAME,
            value="RETRACTED",
            comment=f"retracts a {label.outcome} label",
            metadata={_RETRACTION_ENVELOPE: to_dict(label), _WRITTEN_AT: _written_at()},
        )

    def _create_score(
        self,
        op: str,
        *,
        trace_id: str,
        name: str,
        value: str,
        comment: str,
        metadata: dict[str, Any],
    ) -> None:
        # The SDK's create_score only queues: its consumer logs a 4xx or a serialization failure
        # and drops the score, and flush() returns normally (langfuse==3.15.0, confirmed live
        # with a body over the server's 1 MB limit). The REST create answers synchronously and
        # raises on any non-2xx, so a returned call is a score the server accepted.
        from langfuse.api import CreateScoreRequest, ScoreDataType

        self._require_reachable(op)
        try:
            response = self._client.api.score.create(
                request=CreateScoreRequest(
                    traceId=self._client.create_trace_id(seed=trace_id),
                    name=name,
                    value=value,
                    dataType=ScoreDataType.CATEGORICAL,
                    comment=comment,
                    metadata=metadata,
                )
            )
        except Exception as exc:
            raise self._as_backend_error(exc, op) from exc
        if not getattr(response, "id", None):
            raise BackendError(
                f"Langfuse {op} failed: the server returned no score id, so the write is "
                "not confirmed"
            )

    # --- read path ------------------------------------------------------------------------

    def get_trace(self, trace_id: str) -> Trace:
        """Read the trace Calibrated Agent Evaluation wrote under ``trace_id`` back from Langfuse.

        Args:
            trace_id: The Calibrated Agent Evaluation trace id (not the Langfuse id derived from
                it).

        Returns:
            The trace, rebuilt from the envelope stored with it.

        Raises:
            TraceNotFoundError: Langfuse has no trace for that id.
            BackendError: Langfuse can't be read, or the stored trace can't be rebuilt.
        """
        lf_trace_id = self._client.create_trace_id(seed=trace_id)
        try:
            raw = self._client.api.trace.get(lf_trace_id)
        except Exception as exc:
            # The SDK raises an ApiError with status_code 404 (NotFoundError, or a bare ApiError
            # when the 404 body is not JSON); anything else is a failed read, not an absence.
            if getattr(exc, "status_code", None) == 404:
                raise TraceNotFoundError(
                    f"no trace with trace_id {trace_id!r} in Langfuse"
                ) from exc
            raise self._as_backend_error(exc, f"get_trace({trace_id!r})") from exc
        if raw is None:
            raise TraceNotFoundError(f"no trace with trace_id {trace_id!r} in Langfuse")
        # Reconstruction is wrapped separately: a partial/corrupt envelope makes trace_from_dict
        # raise a bare KeyError that would otherwise escape the CalibratedAgentEvaluationDataError
        # family the data plane contracts on (query() wraps its reconstruction the same way).
        try:
            return self._reconstruct_trace(raw)
        except BackendError:
            raise
        except Exception as exc:
            raise self._as_backend_error(exc, f"get_trace({trace_id!r})") from exc

    def query(
        self,
        frame: Frame | None = None,
        cell: dict[str, str] | None = None,
        ownership: Ownership | None = None,
    ) -> list[Trace]:
        """Return every stored trace matching all given filters, reading every result page.

        Args:
            frame: Keep only traces in this sampling frame.
            cell: Keep only traces whose taxonomy cell has all of these key/value pairs.
            ownership: Keep only owned or only probed traces.

        Returns:
            The matching traces.

        Raises:
            BackendError: Langfuse can't be read, or a listed trace can't be rebuilt. Use
                :meth:`scan` to skip traces Calibrated Agent Evaluation didn't write instead of
                failing.
        """
        tags = self._tags(frame, cell, ownership)
        try:
            # Walk every page; the v3 list endpoint paginates, so reading only the first page
            # would silently truncate a large result set (violates "no silent failure").
            items = list(self._list_all_traces(tags))
            # list items carry observation *ids* only; re-get each for the full detail we read.
            traces = [
                self._reconstruct_trace(self._client.api.trace.get(item.id)) for item in items
            ]
        except Exception as exc:
            raise self._as_backend_error(exc, "query") from exc
        return [t for t in traces if _matches(t, frame, cell, ownership)]

    def scan(
        self,
        frame: Frame | None = None,
        cell: dict[str, str] | None = None,
        ownership: Ownership | None = None,
    ) -> TraceScan:
        """``query``, with every trace Calibrated Agent Evaluation did not write skipped and named.

        A project shared with another client holds traces without the
        ``calibrated_agent_evaluation_trace`` envelope, and ``query`` raises on the first one. A
        pipeline stage reads with ``scan`` instead: the library's traces come back as ``query``
        returns them, and each foreign one is a :class:`SkippedTrace` whose reason says how to bring
        it in (``calibrated-agent-evaluation ingest --from-langfuse``, which maps it through the
        path and its gates) or that it already was, under the same id. Skipped, not mapped on the
        fly: a stage would otherwise score a trace that has passed none of the ingest gates, under
        span ids no stored label can name. A Calibrated Agent Evaluation trace that fails to
        reconstruct still raises, since that is a corrupt copy of our own data, not someone else's.
        """
        tags = self._tags(frame, cell, ownership)
        try:
            items = list(self._list_all_traces(tags))
        except Exception as exc:
            raise self._as_backend_error(exc, "scan") from exc
        traces: list[Trace] = []
        foreign: list[str] = []
        for item in items:
            try:
                raw = self._client.api.trace.get(item.id)
                if self._read_metadata(raw).get(_TRACE_ENVELOPE):
                    trace = self._reconstruct_trace(raw)
                    if _matches(trace, frame, cell, ownership):
                        traces.append(trace)
                else:
                    foreign.append(str(item.id))
            except Exception as exc:
                raise self._as_backend_error(exc, f"scan({item.id!r})") from exc
        ingested = {t.trace_id for t in traces}
        skipped = tuple(
            SkippedTrace(
                trace_id=tid,
                reason=(
                    "written by another client; already ingested as the Calibrated Agent "
                    "Evaluation trace of the "
                    "same id, which is read instead"
                    if tid in ingested
                    else "written by another client "
                    "(no calibrated_agent_evaluation_trace envelope); "
                    "`calibrated-agent-evaluation ingest --from-langfuse` brings such traces in"
                ),
            )
            for tid in foreign
        )
        return TraceScan(traces=traces, skipped=skipped)

    def pull(self) -> Iterator[SourceTrace]:
        """Yield every trace in the project, including ones this library did not write.

        ``query`` reads only the library's own traces: it reconstructs from the
        ``calibrated_agent_evaluation_trace`` envelope and raises on a trace without one, so a
        single foreign trace aborted the whole read. ``pull`` returns each trace in whichever form
        it can: a Calibrated Agent Evaluation-authored trace reconstructed losslessly as before, a
        foreign one mapped to OTLP-JSON
        (:mod:`~calibrated_agent_evaluation.backends.langfuse_foreign`) for the ingest path to
        normalize, and a trace that fails either way as a per-trace ``error``.

        The listing is read to exhaustion before anything is yielded, so an unreachable or
        rejecting server raises ``BackendError`` before the caller persists a single trace.
        Re-putting a trace leaves every earlier put's observations server-side; the
        Calibrated Agent Evaluation-authored path keeps only the latest put's spans, as
        ``get_trace`` does.
        """
        from calibrated_agent_evaluation.backends.langfuse_foreign import foreign_trace_to_otlp

        try:
            items = list(self._list_all_traces([]))
        except Exception as exc:
            raise self._as_backend_error(exc, "pull") from exc
        for item in items:
            trace_id = str(item.id)
            try:
                raw = self._client.api.trace.get(trace_id)
                if self._read_metadata(raw).get(_TRACE_ENVELOPE):
                    yield SourceTrace(source_id=trace_id, trace=self._reconstruct_trace(raw))
                else:
                    yield SourceTrace(source_id=trace_id, otlp=foreign_trace_to_otlp(raw))
            # One unreadable trace is reported as its own error, never fatal to the pull; the
            # error is the message the caller shows, so nothing is swallowed.
            except Exception as exc:  # noqa: BLE001
                error = self._as_backend_error(exc, f"pull({trace_id!r})")
                yield SourceTrace(source_id=trace_id, error=str(error))

    def project(self) -> tuple[str, str]:
        """The ``(id, name)`` of the project these credentials address.

        Langfuse API keys are project-scoped and the public API has no project selector: the
        project is whichever one the key pair belongs to (verified: ``api.projects.get()`` returns
        exactly that project for a project key, langfuse==3.15.0 against server 3.202.1). So the
        caller can check that it is reading the project it meant, but cannot choose another.
        """
        try:
            projects = list(self._client.api.projects.get().data)
        except Exception as exc:
            raise self._as_backend_error(exc, "project lookup") from exc
        if len(projects) != 1:
            raise BackendError(
                f"Langfuse credentials resolve to {len(projects)} projects; use a project-scoped "
                "key pair, which addresses exactly one"
            )
        return str(projects[0].id), str(projects[0].name)

    def _list_all_traces(self, tags: list[str]) -> Iterator[Any]:
        page = 1
        while True:
            listing = self._client.api.trace.list(tags=tags, page=page, limit=_QUERY_PAGE_SIZE)
            yield from listing.data
            # Stop once we've read the last page. `meta.total_pages` is the authoritative bound
            # (verified shape, langfuse==3.15.0); fall back to "short page = last page" if a
            # backend omits meta, so the loop always terminates.
            meta = getattr(listing, "meta", None)
            total_pages = getattr(meta, "total_pages", None)
            if total_pages is not None:
                if page >= total_pages:
                    return
            elif len(listing.data) < _QUERY_PAGE_SIZE:
                return
            page += 1

    def get_scores(self, trace_id: str, span_id: str | None = None) -> list[Label]:
        """Return the labels stored for a trace, oldest first, retracted and duplicates removed.

        Args:
            trace_id: The trace whose labels to read.
            span_id: Narrow the result to labels on this span.

        Returns:
            The labels; an empty list when the trace has none.

        Raises:
            BackendError: Langfuse can't be read, or a stored score isn't a Calibrated Agent
                Evaluation label.
        """
        lf_trace_id = self._client.create_trace_id(seed=trace_id)
        try:
            # Walk every page; score_v_2.get paginates (data + meta.total_pages, verified shape
            # langfuse==3.15.0), so reading only page 1 would silently drop a heavily-labeled
            # trace's later labels (violates "no silent truncation"), mirroring query()/_list_all.
            # Reconstruction stays INSIDE the try so a partial envelope surfaces as a typed
            # BackendError, not a bare KeyError escaping the CalibratedAgentEvaluationDataError
            # family.
            puts: list[tuple[Label, str]] = []
            retracted_at: dict[str, str] = {}
            for score in self._list_all_scores(lf_trace_id):
                meta = self._read_metadata(score)
                written = str(meta.get(_WRITTEN_AT) or "")
                if meta.get(_LABEL_ENVELOPE) is not None:
                    puts.append((self._reconstruct_label(score), written))
                elif meta.get(_RETRACTION_ENVELOPE) is not None:
                    key = label_identity(label_from_dict(meta[_RETRACTION_ENVELOPE]))
                    retracted_at[key] = max(retracted_at.get(key, ""), written)
        except Exception as exc:
            raise self._as_backend_error(exc, f"get_scores({trace_id!r})") from exc
        # Last write wins per label: a put survives only if it is newer than every tombstone for
        # the same identity. Scores carry no server order we read, so the order is the stamped time,
        # and the result is returned oldest first like LocalBackend's, which `dedup_labels` and
        # `latest_verdicts` rely on to keep the newest write. The sort is stable for equal stamps.
        puts.sort(key=lambda put: put[1])
        labels = [
            lbl
            for lbl, written in puts
            if (key := label_identity(lbl)) not in retracted_at or written > retracted_at[key]
        ]
        if span_id is not None:
            labels = [lbl for lbl in labels if lbl.span_id == span_id]
        # Same append-only root cause as `_reconstruct_trace`'s span dedup: `create_score` has no
        # update, so a re-`put_score` leaves N scores server-side for one label. Keep the last.
        return dedup_labels(labels)

    def _list_all_scores(self, lf_trace_id: str) -> Iterator[Any]:
        page = 1
        while True:
            resp = self._client.api.score_v_2.get(
                trace_id=lf_trace_id, page=page, limit=_QUERY_PAGE_SIZE
            )
            yield from resp.data
            meta = getattr(resp, "meta", None)
            total_pages = getattr(meta, "total_pages", None)
            if total_pages is not None:
                if page >= total_pages:
                    return
            elif len(resp.data) < _QUERY_PAGE_SIZE:
                return
            page += 1

    # --- mapping helpers ------------------------------------------------------------------

    @staticmethod
    def _tags(
        frame: Frame | None, cell: dict[str, str] | None, ownership: Ownership | None
    ) -> list[str]:
        tags: list[str] = []
        if frame is not None:
            tags.append(f"{_FRAME_TAG}{frame}")
        if ownership is not None:
            tags.append(f"{_OWNERSHIP_TAG}{ownership}")
        if cell:
            tags.extend(f"{_CELL_TAG}{k}:{v}" for k, v in cell.items())
        return tags

    @staticmethod
    def _dimension_tags(trace: Trace) -> list[str]:
        # The queryable dimensions land as Langfuse tags.
        tags = [f"{_FRAME_TAG}{trace.frame}", f"{_OWNERSHIP_TAG}{trace.ownership}"]
        tags.extend(f"{_CELL_TAG}{k}:{v}" for k, v in trace.taxonomy_cell.items())
        return tags

    @staticmethod
    def _read_metadata(raw: Any) -> dict[str, Any]:
        meta = getattr(raw, "metadata", None)
        if meta is None and isinstance(raw, dict):
            meta = raw.get("metadata")
        if isinstance(meta, str):
            # Corrupt JSON must surface as a typed BackendError, not a raw JSONDecodeError that
            # escapes the CalibratedAgentEvaluationDataError family the data plane contracts on.
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError as exc:
                raise BackendError(f"Langfuse metadata is not valid JSON: {exc}") from exc
        return meta or {}

    def _reconstruct_trace(self, raw: Any) -> Trace:
        meta = self._read_metadata(raw)
        trace_doc = dict(meta.get(_TRACE_ENVELOPE, {}))
        if not trace_doc:
            raise BackendError(
                "Langfuse trace is missing the calibrated_agent_evaluation_trace metadata envelope"
            )
        observations = getattr(raw, "observations", None)
        if observations is None and isinstance(raw, dict):
            observations = raw.get("observations", [])
        # Keep only the latest put's observations, deduplicated by span_id. The v3 SDK's OTel write
        # path is append-only: re-putting a trace_id mints FRESH observation ids, so every put's
        # spans stay server-side, including spans the newer put no longer has; and api.trace.delete
        # is asynchronous across Postgres+ClickHouse (verified: even a 25s wait does not reliably
        # purge), so a delete-then-rewrite cannot fix it. Filtering on read by the generation the
        # trace metadata names restores parity with LocalBackend's replace. A trace written before
        # generations existed has none and keeps every enveloped observation, as before.
        generation = meta.get(_GENERATION)
        by_span_id: dict[str, dict[str, Any]] = {}
        unenveloped = 0
        for obs in observations or []:
            obs_meta = self._read_metadata(obs)
            span_doc = obs_meta.get(_SPAN_ENVELOPE)
            if span_doc is None:
                unenveloped += 1
                continue
            if generation is not None and str(obs_meta.get(_GENERATION)) != str(generation):
                continue  # an earlier put's span
            by_span_id[span_doc["span_id"]] = span_doc
        skipped = (
            f"; {unenveloped} observation(s) carried no {_SPAN_ENVELOPE} envelope"
            if unenveloped
            else ""
        )
        expected = meta.get(_SPAN_COUNT)
        if expected is not None and len(by_span_id) != int(expected):
            raise BackendError(
                f"Langfuse trace {trace_doc.get('trace_id')!r} holds {len(by_span_id)} of the "
                f"{expected} span(s) its latest write recorded; the write is still being ingested "
                f"or did not complete. Retry the read, or re-put the trace{skipped}"
            )
        spans = sorted(by_span_id.values(), key=lambda s: s["seq"])
        trace_doc["spans"] = spans
        trace = trace_from_dict(trace_doc)
        # Validate on the way OUT, mirroring `put_trace`'s discipline on the way in. Without it the
        # read path was the one place a Trace entered the methodology plane unvalidated: an `owned`
        # envelope whose observations all lacked the span envelope reconstructed as
        # `ownership="owned"` with `spans == []`, which `validate` forbids outright,
        # and every downstream consumer then read a zero-step trace as a complete one.
        try:
            validate(trace)
        except SchemaError as exc:
            raise BackendError(
                f"Langfuse trace {trace_doc.get('trace_id')!r} reconstructed into an invalid "
                f"Trace: {exc}{skipped}"
            ) from exc
        return trace

    def _reconstruct_label(self, raw: Any) -> Label:
        label_doc = self._read_metadata(raw).get(_LABEL_ENVELOPE)
        if label_doc is None:
            raise BackendError(
                "Langfuse score is missing the calibrated_agent_evaluation_label metadata envelope"
            )
        return label_from_dict(label_doc)

    @staticmethod
    def _as_backend_error(exc: Exception, op: str) -> BackendError:
        # Any SDK failure surfaces as a typed BackendError. Never a LocalBackend
        # fallback; failure must not silently degrade worlds.
        if isinstance(exc, BackendError):
            return exc
        return BackendError(f"Langfuse {op} failed: {type(exc).__name__}: {exc}")
