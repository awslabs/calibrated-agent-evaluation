# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Count span exports that failed, so a write can refuse to report success after one.

The Langfuse v3 SDK writes spans through an OpenTelemetry ``BatchSpanProcessor`` whose exporter
runs in the background. A failed export is logged by the processor and never raised, and the
SDK's ``flush()`` ignores ``force_flush``'s result (verified langfuse==3.15.0,
opentelemetry-sdk 1.39), so a rejected write looked exactly like a delivered one.

The backend gives the client a tracer provider of its own (the SDK's documented ``tracer_provider``
argument). That provider wraps the ``export`` of every span exporter added to it, through the
processor's public ``span_exporter`` property, and counts the results that were not ``SUCCESS``.
A write compares the count before and after its flush. Owning the provider also keeps the client
off the process's global provider, so the client never exports spans other code emits.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

__all__ = ["ExportTap", "tapped_tracer_provider"]


class ExportTap:
    """A thread-safe count of span exports that did not succeed, with the last reason."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failures = 0
        self._last_error: str | None = None

    @property
    def failures(self) -> int:
        """How many exports have failed so far."""
        with self._lock:
            return self._failures

    @property
    def last_error(self) -> str | None:
        """Why the most recent failed export failed, as far as the exporter said."""
        with self._lock:
            return self._last_error

    def record_failure(self, reason: str) -> None:
        """Count one failed export."""
        with self._lock:
            self._failures += 1
            self._last_error = reason

    def watch(self, exporter: Any) -> None:
        """Wrap ``exporter.export`` so each result that is not ``SUCCESS`` is counted."""
        if getattr(exporter, "_calibrated_agent_evaluation_tapped", False):
            return
        from opentelemetry.sdk.trace.export import SpanExportResult

        inner: Callable[[Sequence[Any]], Any] = exporter.export

        def export(spans: Sequence[Any]) -> Any:
            try:
                result = inner(spans)
            except Exception as exc:
                self.record_failure(f"{type(exc).__name__}: {exc}")
                raise
            if result is not SpanExportResult.SUCCESS:
                self.record_failure(f"the exporter returned {result.name} for {len(spans)} span(s)")
            return result

        exporter.export = export
        exporter._calibrated_agent_evaluation_tapped = True


def tapped_tracer_provider(tap: ExportTap) -> Any:
    """An SDK ``TracerProvider`` that taps the exporter of every span processor added to it."""
    from opentelemetry.sdk.trace import SpanProcessor, TracerProvider

    class _TappedTracerProvider(TracerProvider):
        def add_span_processor(self, span_processor: SpanProcessor) -> None:
            super().add_span_processor(span_processor)
            exporter = getattr(span_processor, "span_exporter", None)
            if exporter is not None:
                tap.watch(exporter)

    return _TappedTracerProvider()
