# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dashboard sinks for the numeric trend/drift series.

The run stage computes a vendor-neutral numeric series
(:func:`calibrated_agent_evaluation.run.report.numeric_series`). This module publishes it:

  * :class:`LocalDashboardSink` -- writes the series to a CSV+JSON file and reads it back
    field-for-field. **Authoritative in both worlds**: the local artifact is the round-trip
    of record, offline and dependency-free.
  * :class:`LangfuseDashboardSink` -- ADDITIONALLY maps each numeric point to a Langfuse-native
    numeric score so the reused Langfuse dashboards render the trend. Scores go through the
    synchronous REST ``client.api.score.create``, which raises on any non-2xx, never the queued
    ``create_score``, whose background consumer logs and drops a rejected score. Reachability is
    probed (``auth_check``) first, so an unreachable backend fails before anything is written.

This is the ONLY dashboard module that touches a vendor SDK; it lives in ``backends/`` (the
adapter package) so no vendor type leaks into the ``run/`` methodology plane (a static test guards
that). The Langfuse client is taken by **injection** (a ``LangfuseScoreClient`` stub mirrors the
verified surface in unit tests); the real MIT SDK client binds via :meth:`from_credentials`, proven
against the live stack by the opt-in integration test.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from calibrated_agent_evaluation._recordstore import atomic_write_text
from calibrated_agent_evaluation.backends._bounded import call_bounded
from calibrated_agent_evaluation.run.errors import RunError, RunStoreError
from calibrated_agent_evaluation.run.report import NumericPoint, RunReport, numeric_series
from calibrated_agent_evaluation.schema.errors import BackendError

_CSV_NAME = "dashboard_series.csv"
_JSON_NAME = "dashboard_series.json"

# Bounds the blocking flush() after a write. Mirrors LangfuseBackend.flush_timeout_seconds: the v3
# SDK's flush() does an unbounded queue.join() (verified langfuse==3.15.0) that hangs forever if the
# ingestion queue wedges, so we cap it and raise a typed BackendError on timeout rather than block
# the caller indefinitely (the same incident that wedged the end-to-end demo; see LangfuseBackend).
_DEFAULT_FLUSH_TIMEOUT_SECONDS = 20.0


@runtime_checkable
class DashboardSink(Protocol):
    """Publishes a run report's numeric series to a dashboard target."""

    def publish(self, report: RunReport) -> None:
        """Publish ``report``'s numeric series; raise on a target failure (never silently no-op)."""
        ...


class LocalDashboardSink:
    """The authoritative file sink: numeric series -> CSV+JSON, round-trippable.

    Offline, dependency-free, identical in both worlds. Writes both files under ``directory``;
    :meth:`load` reads the JSON back field-for-field. Any filesystem/parse failure is wrapped into
    :class:`RunStoreError` (no raw ``OSError``/``JSONDecodeError`` escapes -- "no silent failure").
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)

    def publish(self, report: RunReport) -> None:
        """Write the numeric series to ``dashboard_series.{csv,json}`` under the directory."""
        points = numeric_series(report)
        rows = [p.model_dump(mode="json") for p in points]
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self._dir / _JSON_NAME, json.dumps(rows, indent=2, sort_keys=True))
            fieldnames = [
                "name",
                "value",
                "kind",
                "run_id",
                "lo",
                "hi",
                "interval_method",
                "p_value",
                "refusal_reason",
            ]
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            atomic_write_text(self._dir / _CSV_NAME, buffer.getvalue())
        except OSError as exc:
            raise RunStoreError(f"failed to write dashboard series to {self._dir}: {exc}") from exc

    def load(self) -> list[NumericPoint]:
        """Read the numeric series back field-for-field (the round-trip)."""
        file = self._dir / _JSON_NAME
        if not file.exists():
            return []
        try:
            rows = json.loads(file.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RunStoreError(f"failed to read dashboard series from {self._dir}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RunStoreError(
                f"dashboard series in {self._dir} is not valid JSON: {exc}"
            ) from exc
        try:
            return [NumericPoint.model_validate(r) for r in rows]
        except RunError as exc:
            raise RunStoreError(f"malformed dashboard series in {self._dir} ({exc})") from exc


class RunTraceSpan(Protocol):
    """The span handle ``start_span`` returns (mirrors ``langfuse._client.span.LangfuseSpan``)."""

    def update_trace(self, *, name: str | None = ..., metadata: Any | None = ...) -> Any:
        """Mirror of ``LangfuseSpan.update_trace``: name the trace and attach metadata."""

    def end(self) -> Any:
        """Mirror of ``LangfuseSpan.end``: close the span."""


@runtime_checkable
class LangfuseScoreClient(Protocol):
    """The minimal Langfuse v3 surface the dashboard sink uses (verified langfuse==3.15.0).

    A structural subset of the client
    :class:`calibrated_agent_evaluation.backends.langfuse.LangfuseBackend` uses:
    ``api.score.create`` (the synchronous REST score write), ``create_trace_id`` (deterministic id
    from our run id), ``start_span`` (opens the run's trace, so the scores have a trace to attach
    to), ``auth_check`` (the synchronous reachability probe), and ``flush`` (sends the span). The
    real ``langfuse.Langfuse`` client satisfies it at these call sites.
    """

    api: Any

    def create_trace_id(self, *, seed: str) -> str:
        """Mirror of ``Langfuse.create_trace_id``: a deterministic trace id from ``seed``."""
        ...

    def start_span(
        self, *, trace_context: Any, name: str, metadata: Any | None = ...
    ) -> RunTraceSpan:
        """Mirror of ``Langfuse.start_span``: open the root span of a trace."""
        ...

    def auth_check(self) -> bool:
        """Mirror of ``Langfuse.auth_check``: whether the credentials are accepted."""
        ...

    def flush(self) -> None:
        """Mirror of ``Langfuse.flush``: block until queued writes are sent."""


def _comment(point: NumericPoint) -> str:
    """The score comment: what the point is and its confidence treatment."""
    base = f"{point.kind} for run {point.run_id}"
    if point.lo is not None and point.hi is not None:
        return f"{base}; 95% CI [{point.lo:.4f}, {point.hi:.4f}] ({point.interval_method})"
    if point.p_value is not None:
        return f"{base}; permutation p = {point.p_value:.4g}"
    return base


def _dedup_points(points: Sequence[NumericPoint]) -> list[NumericPoint]:
    """Collapse points sharing ``(name, kind)``, keeping the last, preserving first-seen order.

    A trend series has one value per named metric per run, so two points with the same name and kind
    within one report are the same point reported twice.
    """
    by_key: dict[tuple[str, str], NumericPoint] = {}
    for point in points:
        by_key[(point.name, point.kind)] = point
    return list(by_key.values())


class LangfuseDashboardSink:
    """Publishes the numeric series to Langfuse as numeric scores for dashboard rendering.

    Mirrors :class:`calibrated_agent_evaluation.backends.langfuse.LangfuseBackend`: :meth:`publish`
    calls ``auth_check`` first and raises :class:`BackendError` on an unreachable or rejecting
    backend before writing, and every score is a synchronous REST create, so a score the server
    refuses raises instead of vanishing. The ``local`` sink (if provided) is written first and is
    authoritative; the Langfuse leg is additive. Client injected for tests; :meth:`from_credentials`
    builds the real MIT SDK client.
    """

    def __init__(
        self,
        client: LangfuseScoreClient,
        *,
        local: LocalDashboardSink | None = None,
        flush_timeout_seconds: float = _DEFAULT_FLUSH_TIMEOUT_SECONDS,
    ) -> None:
        self._client = client
        self._local = local
        self._flush_timeout_seconds = flush_timeout_seconds

    def publish(self, report: RunReport) -> None:
        """Publish the numeric series to Langfuse (and the local sink first, if configured)."""
        if self._local is not None:
            self._local.publish(report)  # authoritative round-trip artifact, written first

        # The async OTel write path does NOT raise on a dead backend, so probe first.
        try:
            reachable = self._client.auth_check()
        except Exception as exc:  # connection/auth failure -> typed BackendError (never silent)
            raise BackendError(
                f"Langfuse dashboard publish failed: backend unreachable ({exc})"
            ) from exc
        if not reachable:
            raise BackendError(
                "Langfuse dashboard publish failed: backend unreachable or auth rejected"
            )

        # Deduplicate by (name, kind) before writing. A score create is append-only (the same root
        # cause as the label-retraction duplicates) and the trace id is seeded on run_id, so
        # re-publishing a run appended a second copy of every trend point and the chart showed each
        # value twice. Dedup here rather than on read, because a dashboard sink has no read path.
        points = _dedup_points(numeric_series(report))
        trace_id = self._client.create_trace_id(
            seed=f"calibrated-agent-evaluation-run-{report.run_id}"
        )
        try:
            # A score on a trace id Langfuse has never seen is accepted and then never readable
            # (verified against Langfuse 3.202), so the run gets a trace of its own first.
            span = self._client.start_span(
                trace_context={"trace_id": trace_id},
                name="calibrated-agent-evaluation-run",
                metadata={"run_id": report.run_id, "mode": report.mode},
            )
            span.update_trace(
                name=f"calibrated-agent-evaluation run {report.run_id}",
                metadata={"run_id": report.run_id},
            )
            span.end()
        except Exception as exc:  # any SDK failure -> typed BackendError (no raw error escapes)
            raise BackendError(
                f"Langfuse dashboard publish failed while opening the run trace: {exc}"
            ) from exc
        # The span travels on the queued OTel path; send it before the scores that hang off it.
        self._flush_bounded()
        for point in points:
            if point.value is None:
                # A refused correction has no number to chart; publish the refusal itself as a
                # categorical score so the dashboard shows it rather than a gap.
                self._create_score(
                    trace_id,
                    point.name,
                    "refused",
                    "CATEGORICAL",
                    f"{point.kind} for run {point.run_id}: {point.refusal_reason}",
                )
            else:
                self._create_score(trace_id, point.name, point.value, "NUMERIC", _comment(point))

    def _create_score(
        self, trace_id: str, name: str, value: float | str, data_type: str, comment: str
    ) -> None:
        # The SDK's create_score only queues, and its consumer drops a score the server rejects
        # while flush() returns normally (langfuse==3.15.0). The REST create raises on non-2xx.
        from langfuse.api import CreateScoreRequest, ScoreDataType

        try:
            response = self._client.api.score.create(
                request=CreateScoreRequest(
                    traceId=trace_id,
                    name=name,
                    value=value,
                    dataType=ScoreDataType(data_type),
                    comment=comment,
                )
            )
        except Exception as exc:  # any SDK write failure -> typed BackendError
            raise BackendError(
                f"Langfuse dashboard publish failed while writing score {name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not getattr(response, "id", None):
            raise BackendError(
                f"Langfuse dashboard publish failed: the server returned no id for score "
                f"{name!r}, so the write is not confirmed"
            )

    def _flush_bounded(self) -> None:
        # Bounded so a wedged ingestion queue fails as a typed BackendError instead of hanging
        # (see calibrated_agent_evaluation.backends._bounded).
        outcome = call_bounded(
            self._client.flush,
            timeout_seconds=self._flush_timeout_seconds,
            name="langfuse-dashboard-flush",
        )
        if not outcome.finished:
            raise BackendError(
                f"Langfuse dashboard flush did not confirm within {self._flush_timeout_seconds:g}s "
                "(ingestion queue wedged); the run trace the scores attach to may not exist"
            )
        if outcome.error is not None:
            err = outcome.error
            raise BackendError(
                f"Langfuse dashboard flush failed: {type(err).__name__}: {err}"
            ) from err

    @classmethod
    def from_credentials(
        cls,
        public_key: str,
        secret_key: str,
        host: str,
        *,
        timeout_seconds: int = 5,
        local: LocalDashboardSink | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> LangfuseDashboardSink:
        """Build against the real MIT SDK client (or an injected factory in tests).

        ``timeout_seconds`` bounds every request including the ``auth_check`` probe (the SDK accepts
        ``timeout`` in seconds, verified langfuse==3.15.0), so an unreachable backend fails fast.
        """
        factory: Callable[..., Any]
        if (
            client_factory is None
        ):  # pragma: no cover - real SDK path, exercised by the integration test
            from langfuse import Langfuse  # opt-in [langfuse] extra; MIT build

            factory = Langfuse
        else:
            factory = client_factory
        raw = factory(
            public_key=public_key, secret_key=secret_key, host=host, timeout=timeout_seconds
        )
        return cls(cast("LangfuseScoreClient", raw), local=local)
