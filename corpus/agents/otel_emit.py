# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render executed agent records into real OpenTelemetry spans (gen_ai semantic conventions).

This is the shared OTel emitter for the OrderBot builds. It produces a genuine span tree via the
OTel SDK — one ``invoke_agent`` span per agent, one ``execute_tool {name}`` child span per tool —
carrying prompt/completion/tool content. Two content placements are supported so the
normalizer's OTel adapter is exercised on both:

  * ``events``     — content in span **events** (``gen_ai.user.message`` /
    ``gen_ai.assistant.message`` / ``gen_ai.tool.message``), the OTel-idiomatic placement.
  * ``attributes`` — the *same* content flattened onto span **attributes**, the placement a naive
    attribute-only normalizer would read (and the variant proving both are handled).

These are the literal ``gen_ai.*`` names the OpenTelemetry GenAI semantic conventions define and
that Strands emits (verified against the installed strands tracer) — the corpus speaks the real
vocabulary so the normalizer is coded against real names, not invented ones.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from opentelemetry.trace import SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.span import Span

from corpus.agents.common import ExecutedToolCall, ExecutionRecord

# gen_ai.system value for the hand-coded/Anthropic OrderBot builds (NOT "strands-agents").
GEN_AI_SYSTEM = "calibrated-agent-evaluation-orderbot"


class ContentLocation(StrEnum):
    """Where message/tool content is written on the emitted spans."""

    EVENTS = "events"
    ATTRIBUTES = "attributes"


def emit_trace(
    tracer: Tracer,
    records: list[ExecutionRecord],
    *,
    trace_input: dict[str, Any],
    content_location: ContentLocation = ContentLocation.EVENTS,
) -> None:
    """Emit a full span tree for ``records`` (one agent for OrderBot, many for a crew).

    Spans are emitted in ``seq`` order with real parent/child context so the OTLP export carries the
    genuine tree (``parentSpanId`` set by the SDK). The root span's input mirrors ``trace_input``
    (the terminal request); every span's output is the agent's **real** emitted output (the model's
    answer) — nothing is rewritten. Returns nothing — the caller captures the spans via
    :class:`~corpus.generate.otel_capture.SpanCapture`.
    """
    spans: dict[str, Span] = {}
    for record in sorted(records, key=lambda r: r.seq):
        parent_ctx = None
        if record.parent and record.parent in spans:
            parent_ctx = set_span_in_context(spans[record.parent])
        agent_input = trace_input if record.parent is None else record.input
        span = _emit_agent_span(
            tracer,
            record,
            agent_input=agent_input,
            agent_output=record.output,
            parent_ctx=parent_ctx,
            content_location=content_location,
        )
        spans[record.agent] = span
    # Close spans in reverse seq so children end before parents (valid OTel timing).
    for record in sorted(records, key=lambda r: r.seq, reverse=True):
        spans[record.agent].end()


def emit_degraded_trace(
    tracer: Tracer,
    record: ExecutionRecord,
    *,
    trace_input: dict[str, Any],
) -> None:
    """Emit a single under-instrumented span: overall input/output only, no per-step detail.

    This models a real but minimally-instrumented agent: the app wrapped its whole run in one
    span carrying the terminal request and final answer, but emitted **no per-step tool spans and no
    message events**. The normalizer must recognize there is no recoverable per-step content and
    degrade the trace to outcome-only (``ownership="probed"``, ``spans=[]``) rather than present an
    ``owned`` tree with empty steps. The agent genuinely ran (the same buggy OrderBot); only its
    instrumentation is thin — a common real-world gap, not a hand-shaped fake.
    """
    span = tracer.start_span(
        f"invoke_agent {record.agent}",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.system": GEN_AI_SYSTEM,
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": record.agent,
            # Only top-level I/O as attributes — no per-step events, no tool child spans.
            "gen_ai.input.content": _ser(trace_input),
            "gen_ai.output.content": _ser(record.output),
        },
    )
    span.end()


def _emit_agent_span(
    tracer: Tracer,
    record: ExecutionRecord,
    *,
    agent_input: dict[str, Any],
    agent_output: dict[str, Any],
    parent_ctx: Any,
    content_location: ContentLocation,
) -> Span:
    """Start (but do not end) an ``invoke_agent`` span and emit its tool-call child spans."""
    span = tracer.start_span(
        f"invoke_agent {record.agent}",
        context=parent_ctx,
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.system": GEN_AI_SYSTEM,
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": record.agent,
            "calibrated_agent_evaluation.agent.role": record.role,
            "calibrated_agent_evaluation.span.seq": record.seq,
        },
    )
    _write_content(span, "input", agent_input, content_location)
    agent_ctx = set_span_in_context(span)
    for tool in record.tool_calls:
        _emit_tool_span(tracer, tool, parent_ctx=agent_ctx, content_location=content_location)
    _write_content(span, "output", agent_output, content_location)
    return span


def _emit_tool_span(
    tracer: Tracer,
    tool: ExecutedToolCall,
    *,
    parent_ctx: Any,
    content_location: ContentLocation,
) -> None:
    """Emit one ``execute_tool {name}`` child span carrying the tool args + result."""
    span = tracer.start_span(
        f"execute_tool {tool.name}",
        context=parent_ctx,
        kind=SpanKind.INTERNAL,
        attributes={
            "gen_ai.system": GEN_AI_SYSTEM,
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool.name,
        },
    )
    if content_location is ContentLocation.EVENTS:
        span.add_event("gen_ai.tool.message", {"gen_ai.tool.input": _ser(tool.args)})
        span.add_event("gen_ai.tool.message", {"gen_ai.tool.output": _ser(tool.result)})
    else:
        span.set_attribute("gen_ai.tool.input", _ser(tool.args))
        span.set_attribute("gen_ai.tool.output", _ser(tool.result))
    span.end()


def _write_content(
    span: Span, kind: str, payload: dict[str, Any], content_location: ContentLocation
) -> None:
    """Write an agent input/output as either a span event or a span attribute."""
    if content_location is ContentLocation.EVENTS:
        event_name = "gen_ai.user.message" if kind == "input" else "gen_ai.assistant.message"
        span.add_event(event_name, {"content": _ser(payload)})
    else:
        span.set_attribute(f"gen_ai.{kind}.content", _ser(payload))


def _ser(value: Any) -> str:
    """Serialize a payload to a JSON string (how gen_ai content rides in events/attributes)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
