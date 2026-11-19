# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Langfuse trace write whose span export fails raises instead of reporting success.

The SDK exports spans from a background batch processor that only logs a failed export, and its
``flush()`` ignores the result, so a rejected write used to return normally. These tests run the
real OpenTelemetry batch processor and, in the last case, the real Langfuse client against a port
nothing listens on.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, cast

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

from _helpers import make_span, make_trace
from calibrated_agent_evaluation.backends._export_tap import ExportTap, tapped_tracer_provider
from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend, LangfuseLike
from calibrated_agent_evaluation.schema.errors import BackendError
from test_backends_langfuse import StubLangfuseClient


class _RejectingExporter(SpanExporter):
    def export(self, spans: Sequence[Any]) -> SpanExportResult:
        return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        return None


def test_the_tap_counts_an_export_the_batch_processor_only_logged() -> None:
    tap = ExportTap()
    provider = tapped_tracer_provider(tap)
    provider.add_span_processor(BatchSpanProcessor(_RejectingExporter()))
    with provider.get_tracer("t").start_as_current_span("write"):
        pass
    provider.force_flush()
    assert tap.failures == 1
    assert "FAILURE" in (tap.last_error or "")


class _ClientWhoseExportFails(StubLangfuseClient):
    def __init__(self, tap: ExportTap) -> None:
        super().__init__()
        self._tap = tap

    def flush(self) -> None:
        super().flush()
        self._tap.record_failure("HTTP 401 from the OTLP endpoint")


def test_put_trace_raises_when_its_export_failed() -> None:
    tap = ExportTap()
    backend = LangfuseBackend(cast("LangfuseLike", _ClientWhoseExportFails(tap)), export_tap=tap)
    trace = make_trace(trace_id="X-1", spans=[make_span(trace_id="X-1", span_id="X-1/r")])
    with pytest.raises(BackendError, match="did not succeed"):
        backend.put_trace(trace)


def test_put_trace_returns_when_every_export_succeeded() -> None:
    tap = ExportTap()
    backend = LangfuseBackend(cast("LangfuseLike", StubLangfuseClient()), export_tap=tap)
    trace = make_trace(trace_id="X-2", spans=[make_span(trace_id="X-2", span_id="X-2/r")])
    backend.put_trace(trace)
    assert tap.failures == 0


def test_the_real_client_reports_a_failed_export_and_leaves_the_global_provider_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("langfuse")
    global_before = trace_api.get_tracer_provider()
    backend = LangfuseBackend.from_credentials(
        f"pk-lf-{uuid.uuid4().hex}",
        "sk-lf-test",
        "http://127.0.0.1:9",
        timeout_seconds=1,
        flush_timeout_seconds=30,
    )
    client = cast("Any", backend._client)
    monkeypatch.setattr(client, "auth_check", lambda: True)
    trace = make_trace(trace_id="X-3", spans=[make_span(trace_id="X-3", span_id="X-3/r")])
    with pytest.raises(BackendError, match="did not succeed"):
        backend.put_trace(trace)
    assert trace_api.get_tracer_provider() is global_before
