# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture OpenTelemetry SDK spans and serialize them as canonical OTLP-JSON.

Promoted from ``corpus/generate/otel_capture.py`` into the shipped package so an installed user
(who has no ``examples/`` tree) can run ``StrandsExecutor``: it installs an in-memory OTel exporter
as the global ``TracerProvider`` so a framework crew's genuine instrumentation drives capture, then
encodes the finished spans into the real OTLP/HTTP wire format (``resourceSpans -> scopeSpans ->
spans``) the normalizer reads; *not* the SDK's debug ``ReadableSpan.to_json()`` shape.

The path is the standard one: SDK ``ReadableSpan``s -> ``encode_spans`` (the official proto encoder)
-> OTLP ``TracesData`` protobuf -> ``MessageToDict`` (canonical JSON). Nothing about span content is
synthesized here. This module imports ``opentelemetry`` (the ``[strands]`` extra brings it in); the
executor lazy-imports it so ``import calibrated_agent_evaluation.execute`` stays cheap.
"""

from __future__ import annotations

import contextvars
import json
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from google.protobuf.json_format import MessageToDict
from opentelemetry import trace as trace_api
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace import Span as SdkSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from calibrated_agent_evaluation.execute.errors import ExecutionError


class CaptureProviderError(ExecutionError):
    """The process's global tracer provider cannot carry spans to a capture.

    Raised when something other than the OpenTelemetry SDK's ``TracerProvider`` is already the
    global provider (a vendor distribution, or the API's no-op provider), so capture has no way to
    attach a span processor to it.
    """


class SpanCapture:
    """Owns a ``TracerProvider`` + in-memory exporter; yields a tracer and the captured spans.

    Use as a context manager so the provider is flushed and shut down deterministically
    (I/O resources are context managers). After the ``with`` block,
    ``finished_spans`` holds every span the agent emitted, in completion order.

    Args:
        service_name: The OTel service name the captured spans are recorded under.
    """

    def __init__(self, service_name: str) -> None:
        self._exporter = InMemorySpanExporter()
        self._provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        # SimpleSpanProcessor exports on span end; synchronous, deterministic, no batching delay.
        self._provider.add_span_processor(SimpleSpanProcessor(self._exporter))
        self._attached = False
        self._active_token: contextvars.Token[SpanCapture | None] | None = None

    @property
    def provider(self) -> TracerProvider:
        """The real provider: hand this to ``StrandsTelemetry(provider)`` or set it as global."""
        return self._provider

    def register_global(self) -> None:
        """Receive the spans the agent run inside this capture emits through global tracers.

        Framework-instrumented code (Strands, or a module-level ``trace.get_tracer(...)``) emits
        through the OTel global provider, and OTel lets that be set only once per process. A
        ``ProxyTracer`` handed out before then also binds to the first real provider for good. So
        capture never replaces the global: it installs one long-lived provider the first time (or
        joins an SDK provider the application already installed) and hangs a router off it.

        The router gives each capture only its own spans. A span belongs to the capture whose
        context it started in, or to the capture that owns its parent, so the agent's spans on a
        worker thread still arrive and two captures running at once never see each other's spans.
        Host telemetry that started outside any capture (a web request, a database call) is not
        captured. Strands' tracer is pointed at a private provider while any capture is open, so
        its spans reach the capture without passing through an exporter the global provider
        carries (a Langfuse client's, for example). Self-built emitters (OrderBot) can use
        :meth:`tracer`.

        Raises :class:`CaptureProviderError` if the global provider is not an SDK provider.
        """
        _router().attach(self)
        self._attached = True
        self._active_token = _ACTIVE_CAPTURE.set(self)

    def _receive(self, span: ReadableSpan) -> None:
        self._exporter.export((_with_resource(span, self._provider.resource),))

    def tracer(self, name: str) -> Tracer:
        """A tracer whose spans this capture collects."""
        return self._provider.get_tracer(name)

    @property
    def finished_spans(self) -> tuple[ReadableSpan, ...]:
        """The spans collected so far, in the order they finished."""
        return tuple(self._exporter.get_finished_spans())

    def __enter__(self) -> SpanCapture:
        return self

    def __exit__(self, *_exc: object) -> None:
        try:
            if self._attached:
                # Drain any batching processor on the global provider before detaching, so a span
                # that ends during the flush still reaches this capture. Bounded, because the
                # global provider's exporters belong to the application and may be slow or down.
                global_provider = trace_api.get_tracer_provider()
                if isinstance(global_provider, TracerProvider):
                    global_provider.force_flush(timeout_millis=_EXIT_FLUSH_MILLIS)
                _router().detach(self)
                self._attached = False
        finally:
            if self._active_token is not None:
                try:
                    _ACTIVE_CAPTURE.reset(self._active_token)
                except ValueError:
                    # Exited in a different context than it was entered in; clear it instead.
                    _ACTIVE_CAPTURE.set(None)
                self._active_token = None
            self._provider.force_flush(timeout_millis=_EXIT_FLUSH_MILLIS)
            self._provider.shutdown()


class _CaptureRouter(SpanProcessor):
    """Forwards each ended span to the capture that owns it, and drops the rest.

    Ownership is decided when a span starts: the capture that owns its parent, else the capture
    active in the context it started in. Children inherit from parents, so a span emitted on a
    thread that did not copy the context still reaches the capture its parent belongs to.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._captures: list[SpanCapture] = []
        self._owners: dict[tuple[int, int], SpanCapture] = {}
        self._strands_previous: tuple[Any, Any] | None = None

    def attach(self, capture: SpanCapture) -> None:
        with self._lock:
            if capture not in self._captures:
                self._captures.append(capture)
            if len(self._captures) == 1:
                self._strands_previous = _bind_strands_tracer(_strands_provider())

    def detach(self, capture: SpanCapture) -> None:
        with self._lock:
            if capture in self._captures:
                self._captures.remove(capture)
            self._owners = {k: v for k, v in self._owners.items() if v is not capture}
            if not self._captures and self._strands_previous is not None:
                _restore_strands_tracer(self._strands_previous)
                self._strands_previous = None

    def on_start(
        self,
        span: SdkSpan,
        parent_context: Context | None = None,  # noqa: ARG002 (SDK signature)
    ) -> None:
        context = span.get_span_context()
        parent = span.parent
        with self._lock:
            owner = self._owners.get((parent.trace_id, parent.span_id)) if parent else None
            if owner is None:
                active = _ACTIVE_CAPTURE.get()
                owner = active if active in self._captures else None
            if owner is not None:
                self._owners[(context.trace_id, context.span_id)] = owner

    def on_end(self, span: ReadableSpan) -> None:
        context = span.get_span_context()
        if context is None:
            return
        with self._lock:
            owner = self._owners.pop((context.trace_id, context.span_id), None)
            if owner is not None and owner not in self._captures:
                owner = None
        if owner is not None:
            owner._receive(span)

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: ARG002 (SDK signature)
        return True


# The capture whose block the current code runs in; set by `register_global`.
_ACTIVE_CAPTURE: contextvars.ContextVar[SpanCapture | None] = contextvars.ContextVar(
    "calibrated_agent_evaluation_active_capture", default=None
)
_EXIT_FLUSH_MILLIS = 5000

# One router per process, because the OTel global provider it hangs off is one per process.
_ROUTER = _CaptureRouter()
_ROUTER_LOCK = threading.Lock()
_router_host: object | None = None
_STRANDS_PROVIDER: TracerProvider | None = None


def _strands_provider() -> TracerProvider:
    """The private provider Strands' tracer emits through while a capture is open."""
    global _STRANDS_PROVIDER
    if _STRANDS_PROVIDER is None:
        _STRANDS_PROVIDER = TracerProvider()
        _STRANDS_PROVIDER.add_span_processor(_ROUTER)
    return _STRANDS_PROVIDER


def _router() -> _CaptureRouter:
    """Return the router, first hanging it off the current global provider if it is not there."""
    global _router_host
    with _ROUTER_LOCK:
        current = trace_api.get_tracer_provider()
        if current is _router_host:
            return _ROUTER
        if isinstance(current, trace_api.ProxyTracerProvider):
            provider = TracerProvider()
            provider.add_span_processor(_ROUTER)
            trace_api.set_tracer_provider(provider)
            current = trace_api.get_tracer_provider()
            if current is not provider:
                raise CaptureProviderError(
                    "another tracer provider was installed while capture was installing its own; "
                    "capture cannot see that provider's spans"
                )
        elif isinstance(current, TracerProvider):
            current.add_span_processor(_ROUTER)
        else:
            raise CaptureProviderError(
                "the process's global OpenTelemetry tracer provider is a "
                f"{type(current).__module__}.{type(current).__qualname__}, which is not the "
                "OpenTelemetry SDK's TracerProvider, so capture cannot attach to it. Run the agent "
                "in a process that does not install its own provider, or export OTLP-JSON from "
                "that provider and ingest the file"
            )
        _router_host = current
        return _ROUTER


def _with_resource(span: ReadableSpan, resource: Resource) -> ReadableSpan:
    """Copy ``span`` under ``resource``, keeping bounded collections so dropped counts survive."""
    # The private fields are read because the public properties return copies that lose the
    # dropped-attribute, dropped-event and dropped-link counts the OTLP encoder writes.
    return ReadableSpan(
        name=span.name,
        context=span.get_span_context(),
        parent=span.parent,
        resource=resource,
        attributes=getattr(span, "_attributes", span.attributes),
        events=getattr(span, "_events", span.events),
        links=getattr(span, "_links", span.links),
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def _bind_strands_tracer(provider: TracerProvider) -> tuple[Any, Any] | None:
    """Point Strands' process-wide tracer at ``provider``; return what it pointed at before.

    Strands' ``telemetry.tracer.get_tracer()`` memoizes one ``Tracer`` per process (``tracer.py``:
    ``_tracer_instance``), and every ``Agent``, ``Graph`` and ``Swarm`` keeps a reference to it from
    construction. Its ``tracer`` attribute is rebound in place, so agents built before the capture
    opened emit into it too. Returns ``None`` when Strands is not imported, so this module stays
    importable without the strands dependency.
    """
    if "strands" not in sys.modules:
        return None
    import strands.telemetry.tracer as strands_tracer

    instance = strands_tracer.get_tracer()
    previous = (instance.tracer, instance.tracer_provider)
    instance.tracer_provider = provider
    instance.tracer = provider.get_tracer(instance.service_name)
    return previous


def _restore_strands_tracer(previous: tuple[Any, Any]) -> None:
    """Point Strands' tracer back where it was before the first open capture rebound it."""
    import strands.telemetry.tracer as strands_tracer

    instance = strands_tracer.get_tracer()
    instance.tracer, instance.tracer_provider = previous


def spans_to_otlp_json(spans: tuple[ReadableSpan, ...]) -> dict[str, Any]:
    """Encode finished SDK spans into a canonical OTLP-JSON dict (``{"resourceSpans": [...]}``)."""
    traces_data = encode_spans(spans)
    # preserving_proto_field_name=False -> camelCase wire keys (resourceSpans/scopeSpans/...),
    # which is the JSON form the OTLP/HTTP exporter and collectors emit and the normalizer reads.
    return MessageToDict(traces_data, preserving_proto_field_name=False)


def write_otlp_json(spans: tuple[ReadableSpan, ...], path: Path) -> dict[str, Any]:
    """Freeze ``spans`` to ``path`` as pretty OTLP-JSON; returns the written dict."""
    payload = spans_to_otlp_json(spans)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


@contextmanager
def capture_spans(service_name: str) -> Generator[SpanCapture]:
    """Collect the spans emitted inside the block under ``service_name``.

    Use it as ``with capture_spans("orderbot") as cap:``, then read ``cap.finished_spans``.
    """
    cap = SpanCapture(service_name)
    with cap:
        yield cap
