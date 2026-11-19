# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialize a normalized ``Trace`` into the AgentCore on-demand Evaluate input payload.

Source-agnostic by design: this reads the normalized ``Trace`` schema alone, so a trace from any
agent serializes to the same ``sessionSpans`` shape the on-demand Evaluate call accepts.

Tool calls are sent as their own ``execute_tool`` child spans (name, call id, arguments, result),
because a trace-level judge that cannot see what the tools returned cannot tell a grounded answer
from a made-up one. No source stores tool-call times, so those spans always get synthetic times.

Synthetic times are used without asking, because Evaluate requires a start and end time on every
span and tool calls never have them. They are kept honest instead: every span document whose times
are synthetic is marked with :data:`SYNTHETIC_TIMING_ATTRIBUTE`, the runner marks each verdict on
such a trace and warns once per run, and the stored trace is never changed.

- A trace with no span timing gets nested windows: a depth-first walk visits the roots, and each
  span's children, in ``seq`` order (a span's tool calls first, as leaves), and each span opens one
  step after the previous open or close and closes one step after its last child. Every parent
  encloses its children, siblings do not overlap, and the last root ends last, so the root is the
  span read as the answer. The base and the step are constants, so a trace yields the same request.
- Under a timed span with window [start, end], its k-th of n tool calls is a zero-length instant at
  start + k * (end - start) / (n + 1), so the calls keep their order and fall inside the window.

Refused, on purpose: a trace with no serializable content (an outcome-only / probed trace); a trace
with timing on some spans or fields but not all, since real and synthetic agent-span times on one
axis can order spans in a way that never happened; and a trace with no timing whose span names a
parent outside it, or cannot be reached from a root, rather than leave that span out.

The payload uses the **unified** telemetry shape: message content rides on the span's own
attributes (the gen_ai convention), with no separate log-event records, under a **generic**
instrumentation scope so Evaluate reads it through its documented generic path rather than holding
it to a specific framework's exact layout.

An on-demand call scores one trace as its own session, so the session/trace/span ids are derived
from the trace's own id (conformed to the exact length the Evaluate target requires); the runner
maps each result back to the original id.

Vendor-free surface: the payload is a plain dict (the boto3 ``evaluationInput`` wire shape) handed
to the Evaluate call in :mod:`calibrated_agent_evaluation.agentcore.evaluate`; nothing here imports
boto3.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreBlankAssertion,
    AgentCoreContentUnserializable,
    AgentCoreEvaluateLimitExceeded,
    AgentCoreNotAConversation,
)

if TYPE_CHECKING:
    from calibrated_agent_evaluation.schema.types import Span, Trace

# The boto3 ``evaluationInput`` wire shape (``sessionSpans`` and friends). A plain dict confined to
# this package and handed straight to the Evaluate call, so it is typed structurally rather than as
# a methodology-plane model.
EvaluationInput = dict[str, Any]

_ONE_SECOND_NANOS = 1_000_000_000

#: Set on every span document whose times were made up, so a reader of the request can tell.
SYNTHETIC_TIMING_ATTRIBUTE = "calibrated_agent_evaluation.timestamps.synthetic"

# Synthetic windows start from a fixed instant and advance one millisecond per open or close, so a
# trace always yields the same request. The values carry no meaning beyond their order.
_SYNTHETIC_BASE_NANOS = int(datetime(2000, 1, 1, tzinfo=UTC).timestamp()) * _ONE_SECOND_NANOS
_SYNTHETIC_STEP_NANOS = 1_000_000

# A generic OpenTelemetry instrumentation scope. Evaluate reads any
# ``opentelemetry.instrumentation.*`` scope through its documented generic path (following the
# standard gen_ai conventions), whereas the Strands scope is held to Strands' exact layout. We emit
# standard gen_ai attributes, so we declare the generic scope rather than impersonate a framework.
_SUPPORTED_SCOPE = "opentelemetry.instrumentation.calibrated_agent_evaluation"

# The Evaluate target requires ids of an exact length (a trace id is 32 chars, a span id is 16),
# but a Calibrated Agent Evaluation trace id is arbitrary (e.g. "ORD-1001"). So both the inline span
# documents and the evaluation target carry a conforming id derived deterministically from ours, and
# the runner maps each result back to the original id. Derivation is a pure function of our id, so
# the serializer and the runner independently produce the same conforming id without sharing state.
_AWS_TRACE_ID_LEN = 32
_AWS_SPAN_ID_LEN = 16


def aws_trace_id(trace_id: str) -> str:
    """A 32-char hex id for the Evaluate target, derived deterministically from ``trace_id``."""
    return hashlib.sha256(trace_id.encode("utf-8")).hexdigest()[:_AWS_TRACE_ID_LEN]


def aws_span_id(span_id: str) -> str:
    """A 16-char hex id for the Evaluate target, derived deterministically from ``span_id``."""
    return hashlib.sha256(span_id.encode("utf-8")).hexdigest()[:_AWS_SPAN_ID_LEN]


def _unix_nano(span: Span, field: Literal["started_at", "ended_at"]) -> int:
    """The integer Unix-nanosecond of a span's stored time, which must carry a UTC offset.

    Raises :class:`AgentCoreContentUnserializable` naming the trace and span when the time does
    not parse, or has no offset: read in the host's local time zone, the same trace would build a
    different request on another machine.
    """
    iso = getattr(span, field)
    try:
        moment = datetime.fromisoformat(iso)
    except (TypeError, ValueError) as exc:
        raise AgentCoreContentUnserializable(
            f"trace {span.trace_id!r}: span {span.span_id!r} has {field} {iso!r}, which is not an "
            "ISO-8601 time; the run is refused before any billed call"
        ) from exc
    if moment.utcoffset() is None:
        raise AgentCoreContentUnserializable(
            f"trace {span.trace_id!r}: span {span.span_id!r} has {field} {iso!r}, which has no UTC "
            "offset (Z or +HH:MM); the run is refused before any billed call rather than read the "
            "time in this machine's time zone"
        )
    return int(moment.timestamp() * _ONE_SECOND_NANOS)


def _as_text(span: Span, field: Literal["input", "output"]) -> str:
    """A plain-text rendering of a span's input/output payload for a content attribute.

    A single-field payload (the common ``{"message": "..."}`` / ``{"answer": "..."}`` shape) is
    unwrapped to its string value so the evaluator reads the real text, not a JSON envelope; any
    other shape is serialized stably.
    """
    payload: dict[str, Any] = getattr(span, field)
    if len(payload) == 1:
        (value,) = payload.values()
        if isinstance(value, str):
            return value
    return _json(payload, span=span, field=field)


def _json(value: Any, *, span: Span, field: str) -> str:
    """``value`` as stable JSON, or a typed refusal naming the trace, span and field."""
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise AgentCoreContentUnserializable(
            f"trace {span.trace_id!r}: span {span.span_id!r} has a {field} that cannot be written "
            f"as JSON ({exc}); the run is refused before any billed call"
        ) from exc


def _tool_documents(
    span: Span, *, trace_id: str, windows: list[tuple[int, int]]
) -> list[dict[str, Any]]:
    """One ``execute_tool`` child span document per tool call on ``span``, in call order.

    Tool calls have no stored times, so every window here is synthetic and every document is marked.
    """
    documents = []
    for index, (call, (start, end)) in enumerate(zip(span.tool_calls, windows, strict=True)):
        documents.append(
            {
                "traceId": trace_id,
                "spanId": aws_span_id(_tool_key(span.span_id, index)),
                "parentSpanId": aws_span_id(span.span_id),
                "name": f"execute_tool {call.name}",
                "startTimeUnixNano": start,
                "endTimeUnixNano": end,
                "scope": {"name": _SUPPORTED_SCOPE},
                "attributes": {
                    "gen_ai.operation.name": "execute_tool",
                    "session.id": trace_id,
                    "gen_ai.tool.name": call.name,
                    "gen_ai.tool.call.id": f"{span.span_id}#{index}",
                    "gen_ai.tool.call.arguments": _json(
                        call.args, span=span, field=f"tool call {call.name!r} argument"
                    ),
                    "gen_ai.tool.call.result": _json(
                        call.result, span=span, field=f"tool call {call.name!r} result"
                    ),
                    SYNTHETIC_TIMING_ATTRIBUTE: True,
                },
            }
        )
    return documents


def _tool_key(span_id: str, index: int) -> str:
    """The Calibrated Agent Evaluation-side key a tool call's span id is derived from."""
    return f"{span_id}#tool-{index}"


def _instants_inside(start: int, end: int, count: int) -> list[tuple[int, int]]:
    """``count`` zero-length instants spaced evenly inside [start, end], in order."""
    return [
        (at, at) for at in (start + k * (end - start) // (count + 1) for k in range(1, count + 1))
    ]


def _span_document(
    span: Span,
    *,
    trace_id: str,
    span_id: str,
    window: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """One unified span document: content on the span's own gen_ai attributes, no log events.

    ``window`` is a synthetic (start, end) in Unix nanoseconds; given, it replaces the span's own
    (absent) times and the document is marked as synthetically timed.
    """
    operation = "execute_tool" if span.role == "tool" else "invoke_agent"
    user_prompt = _as_text(span, "input")
    agent_response = _as_text(span, "output")
    attributes: dict[str, Any] = {
        "gen_ai.operation.name": operation,
        "gen_ai.agent.name": span.agent,
        # Each trace is sent as its own session, so its session id is its trace id.
        "session.id": trace_id,
        # The top-level turn I/O, in every documented placement so the evaluator reads it wherever
        # it looks: the primary task attributes, the structured message lists (content as text
        # parts, matching the captured record shape), and the read-from-any-scope fallback.
        "gen_ai.task.input": user_prompt,
        "gen_ai.task.output": agent_response,
        "gen_ai.input.messages": [{"role": "user", "content": [{"text": user_prompt}]}],
        "gen_ai.output.messages": [{"role": "assistant", "content": [{"text": agent_response}]}],
        "agentcore.invocation.user_prompt": user_prompt,
        "agentcore.invocation.agent_response": agent_response,
    }
    if window is None:
        start = _unix_nano(span, "started_at")
        end = _unix_nano(span, "ended_at")
    else:
        start, end = window
        attributes[SYNTHETIC_TIMING_ATTRIBUTE] = True
    document: dict[str, Any] = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": span.agent,
        "startTimeUnixNano": start,
        "endTimeUnixNano": end,
        "scope": {"name": _SUPPORTED_SCOPE},
        "attributes": attributes,
    }
    if span.parent is not None:
        document["parentSpanId"] = aws_span_id(span.parent)
    return document


def _has_serializable_content(trace: Trace) -> bool:
    """Whether any span carries input, output or a tool call to serialize."""
    return any(span.input or span.output or span.tool_calls for span in trace.spans)


def _timing(trace: Trace) -> Literal["full", "none", "partial"]:
    """``"full"`` if every span has both times, ``"none"`` if none has either, else partial."""
    fields = [field for span in trace.spans for field in (span.started_at, span.ended_at)]
    if all(field is not None for field in fields):
        return "full"
    if all(field is None for field in fields):
        return "none"
    return "partial"


def _synthetic_windows(trace: Trace) -> dict[str, tuple[int, int]]:
    """Nested synthetic (start, end) windows, from a depth-first walk in ``seq`` order.

    Keyed by span id, and by :func:`_tool_key` for each tool call, which is a leaf visited before
    the span's child spans.

    Raises :class:`AgentCoreContentUnserializable` if a span names a parent outside the trace, or
    is otherwise unreachable from a root, rather than leave it out of the request.
    """
    ids = {span.span_id for span in trace.spans}
    for span in trace.spans:
        if span.parent is not None and span.parent not in ids:
            raise AgentCoreContentUnserializable(
                f"trace {trace.trace_id!r}: span {span.span_id!r} names parent {span.parent!r}, "
                "which is not a span of the trace, so synthetic timing has nowhere to place it; "
                "the run is refused rather than sent without that span"
            )
    children: dict[str | None, list[Span]] = {}
    for span in sorted(trace.spans, key=lambda span: span.seq):
        children.setdefault(span.parent, []).append(span)
    windows: dict[str, tuple[int, int]] = {}
    opened: dict[str, int] = {}
    clock = _SYNTHETIC_BASE_NANOS
    # A node is a span, or a tool call (as its key) visited as a leaf before the span's children.
    stack: list[tuple[Span | str, bool]] = [
        (root, False) for root in reversed(children.get(None, []))
    ]
    while stack:
        node, closing = stack.pop()
        clock += _SYNTHETIC_STEP_NANOS
        key = node if isinstance(node, str) else node.span_id
        if closing:
            windows[key] = (opened[key], clock)
            continue
        opened[key] = clock
        stack.append((node, True))
        if isinstance(node, str):
            continue
        visits: list[Span | str] = [
            _tool_key(node.span_id, index) for index in range(len(node.tool_calls))
        ]
        visits.extend(children.get(node.span_id, []))
        stack.extend((visit, False) for visit in reversed(visits))
    if any(span.span_id not in windows for span in trace.spans):
        raise AgentCoreContentUnserializable(
            f"trace {trace.trace_id!r}: some spans cannot be reached from a root span, so "
            "synthetic timing has nowhere to place them; the run is refused rather than sent "
            "without them"
        )
    return windows


def trace_to_evaluation_input(trace: Trace) -> EvaluationInput:
    """Serialize ``trace`` to the unified Evaluate ``sessionSpans`` payload.

    Each span becomes one span document carrying its turn I/O on gen_ai attributes, followed by one
    ``execute_tool`` document per tool call; ids are derived from the trace's id to meet the
    Evaluate target's length constraints. A trace with no span timing gets nested synthetic
    windows, tool calls always get synthetic times, and every document with a synthetic time is
    marked with :data:`SYNTHETIC_TIMING_ATTRIBUTE`. ``trace`` itself is never changed.

    Raises :class:`AgentCoreContentUnserializable` for a trace with no serializable content
    (checked first), one with partial timing, and one with no timing whose span names a parent
    outside it or cannot be reached from a root.
    """
    if trace.ownership == "probed" or not trace.spans or not _has_serializable_content(trace):
        raise AgentCoreContentUnserializable(
            f"trace {trace.trace_id!r} carries no message content to serialize into an evaluation "
            "input (an outcome-only or probed trace has no spans/content to score); it is refused "
            "rather than sent as an empty payload"
        )
    timing = _timing(trace)
    if timing == "partial":
        raise AgentCoreContentUnserializable(
            f"trace {trace.trace_id!r} has partial span timing (started_at/ended_at set on some "
            "spans but not all); Calibrated Agent Evaluation will not mix real and synthetic "
            "times, so the trace is "
            "refused"
        )
    windows = _synthetic_windows(trace) if timing == "none" else {}
    trace_id = aws_trace_id(trace.trace_id)
    documents: list[dict[str, Any]] = []
    for span in trace.spans:
        window = windows.get(span.span_id)
        documents.append(
            _span_document(
                span,
                trace_id=trace_id,
                span_id=aws_span_id(span.span_id),
                window=window,
            )
        )
        if window is None:
            parent = documents[-1]
            tool_windows = _instants_inside(
                parent["startTimeUnixNano"], parent["endTimeUnixNano"], len(span.tool_calls)
            )
        else:
            tool_windows = [
                windows[_tool_key(span.span_id, index)] for index in range(len(span.tool_calls))
            ]
        documents.extend(_tool_documents(span, trace_id=trace_id, windows=tool_windows))
    return {"sessionSpans": documents}


#: The most span documents one ``Evaluate`` call takes (``sessionSpans``, botocore service model).
MAX_SESSION_SPANS = 20_000
#: The most assertions one reference input carries (``assertions``, botocore service model).
MAX_ASSERTIONS = 100

_TURN_PREFIX = re.compile(r"t([1-9][0-9]*):")


def is_conversation_trace(trace: Trace) -> bool:
    """Whether ``trace`` is a simulated conversation: its input carries the transcript."""
    return isinstance(trace.input.get("conversation"), list)


def conversation_session_id(trace: Trace) -> str:
    """The session id a conversation is sent under: the same id its documents' ``session.id`` is."""
    return aws_trace_id(trace.trace_id)


def conversation_to_evaluation_input(
    trace: Trace,
) -> tuple[EvaluationInput, list[dict[str, Any]]]:
    """Serialize a simulated conversation as one ``Evaluate`` session, with its assertions.

    The span documents are :func:`trace_to_evaluation_input`'s, with every document's
    ``session.id`` the conversation's, and each turn's documents (its spans, re-keyed ``t<n>:``,
    and their tool calls) under that turn's own trace id, so AgentCore reads one session of
    request-response turns. The reference input names that session and carries the trace's
    assertions in order; there is none when the trace has no assertions.

    Returns:
        The ``evaluationInput`` and the ``evaluationReferenceInputs`` (possibly empty).

    Raises:
        AgentCoreNotAConversation: ``trace`` is not a simulated conversation.
        AgentCoreBlankAssertion: An assertion is empty or only whitespace.
        AgentCoreEvaluateLimitExceeded: More span documents than one call takes, or more
            assertions than one reference input carries.
        AgentCoreContentUnserializable: As :func:`trace_to_evaluation_input`.
    """
    if not is_conversation_trace(trace):
        raise AgentCoreNotAConversation(
            f"trace {trace.trace_id!r} is not a simulated conversation (its input has no "
            "transcript), so a session-level evaluator would score it as a one-turn session; "
            "it is refused before any billed call"
        )
    assertions = trace.input.get("assertions") or []
    if not isinstance(assertions, list) or not all(isinstance(a, str) for a in assertions):
        raise AgentCoreNotAConversation(
            f"conversation trace {trace.trace_id!r} has assertions that are not a list of strings"
        )
    blank = [index for index, text in enumerate(assertions, start=1) if not text.strip()]
    if blank:
        raise AgentCoreBlankAssertion(
            f"conversation trace {trace.trace_id!r} has a blank assertion (number "
            f"{', '.join(map(str, blank))}), which asserts nothing; give it text or remove it from "
            "the scenario. Refused before any billed call"
        )
    if len(assertions) > MAX_ASSERTIONS:
        raise AgentCoreEvaluateLimitExceeded(
            f"conversation trace {trace.trace_id!r} has {len(assertions)} assertions, and one "
            f"Evaluate reference input takes at most {MAX_ASSERTIONS}; refused before any billed "
            "call"
        )
    documents = trace_to_evaluation_input(trace)["sessionSpans"]
    if len(documents) > MAX_SESSION_SPANS:
        raise AgentCoreEvaluateLimitExceeded(
            f"conversation trace {trace.trace_id!r} serializes to {len(documents)} span "
            f"documents, and one Evaluate call takes at most {MAX_SESSION_SPANS}; refused before "
            "any billed call"
        )
    turn_trace = {
        aws_span_id(span.span_id): aws_trace_id(f"{trace.trace_id}#t{_turn(span.span_id)}")
        for span in trace.spans
    }
    for document in documents:
        own = turn_trace.get(document["spanId"]) or turn_trace.get(document.get("parentSpanId", ""))
        if own is None:  # pragma: no cover - every document is built from one of the trace's spans
            raise AgentCoreContentUnserializable(
                f"conversation trace {trace.trace_id!r} produced a span document that belongs to "
                "none of its turns, so it cannot be sent as part of a turn"
            )
        document["traceId"] = own
    session = conversation_session_id(trace)
    references: list[dict[str, Any]] = []
    if assertions:
        references.append(
            {
                "context": {"spanContext": {"sessionId": session}},
                "assertions": [{"text": text} for text in assertions],
            }
        )
    return {"sessionSpans": documents}, references


def conversation_turn_trace_ids(trace: Trace) -> frozenset[str]:
    """The trace id each turn of a conversation is sent under, one per turn its spans name."""
    return frozenset(
        aws_trace_id(f"{trace.trace_id}#t{n}") for n in {_turn(s.span_id) for s in trace.spans}
    )


def _turn(span_id: str) -> int:
    """The turn a conversation span belongs to, from its ``t<n>:`` prefix (1 when it has none)."""
    match = _TURN_PREFIX.match(span_id)
    return int(match.group(1)) if match else 1
