# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a normalized trace into the ``strands_evals`` session the failure detector reads.

Each Calibrated Agent Evaluation span, in ``seq`` order, becomes an agent-invocation span, followed
right after it by one tool-execution span per tool call, with id ``<span_id>#tool<i>`` so a
detection can point at a single call. The facts the vendor span types have no field for (the span's
agent, role, occurrence and status) and what the normalized trace does not record (the agent's full
toolbox, its system prompt) are written into each converted span's ``metadata``.

The serialized session is the prompt the detector sends and the key its cassette is recorded
under, so nothing in it may depend on the process or the backend: tool names are listed in
first-call order and every JSON payload is rebuilt with sorted keys.

A trace the converter cannot represent faithfully is refused with
:class:`~calibrated_agent_evaluation.agentcore.SessionConversionError`, before any model call: a
probed or span-less trace, a span time that is unparseable, naive or reversed, a tool span id that
another span already has, and a span payload (input, output, tool arguments or result) that is not
JSON, which a trace built in code rather than read from a store can hold.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, Literal

from pydantic import ValidationError
from strands_evals.types.trace import (
    AgentInvocationSpan,
    Session,
    SpanInfo,
    SpanUnion,
    ToolCall,
    ToolConfig,
    ToolExecutionSpan,
    ToolResult,
)
from strands_evals.types.trace import Trace as StrandsTrace

from calibrated_agent_evaluation.agentcore.errors import SessionConversionError
from calibrated_agent_evaluation.schema.types import Span, Trace

__all__ = ["ConvertedSession", "canonical_json", "session_json", "trace_to_session"]

TimeSource = Literal["recorded", "placeholder"]
Interval = tuple[datetime, datetime]

# Placeholder times encode only the seq order: span k runs from 2k to 2k+1 seconds after this.
_PLACEHOLDER_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_TOOLS_NOTE = "only the tools this span called"
_SYSTEM_PROMPT_NOTE = "not recorded"
_NEEDS_SPANS = "the detector needs an owned trace with spans"


def canonical_json(value: Any) -> str:
    """Serialize ``value`` as compact JSON with sorted keys and non-ASCII text kept as written.

    Args:
        value: A JSON-serializable value.

    Returns:
        The same text for equal values, whatever order their keys were built in.
    """
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class ConvertedSession:
    """A trace converted for the failure detector, with the map back to its normalized spans.

    Attributes:
        session: The vendor session holding one vendor trace whose id is the Calibrated Agent
            Evaluation trace id.
        times: ``"recorded"`` when the span times are the trace's own, ``"placeholder"`` when
            they encode only the ``seq`` order.
        locations: Each vendor span id, in converted-session order, mapped to the Calibrated Agent
            Evaluation span it came from and the tool call's index (``None`` for the agent span
            itself).
        tool_names: Each tool span id (``<span_id>#tool<i>``) mapped to the name of its tool.
    """

    session: Session
    times: TimeSource
    locations: Mapping[str, tuple[str, int | None]]
    tool_names: Mapping[str, str]


def trace_to_session(trace: Trace) -> ConvertedSession:
    """Convert a normalized trace into the vendor session the failure detector reads.

    The session id, the one vendor trace's id and every span's trace and session id are the
    trace id. Spans are taken in ``seq`` order. Each becomes an agent-invocation span with the
    same id and parent, its canonical-JSON input and output as prompt and response, and the
    distinct names of the tools it called, in first-call order, as its available tools. Each
    tool call follows as a tool-execution span under it, with its arguments rebuilt with sorted
    keys and its canonical-JSON result.

    When every span records both ``started_at`` and ``ended_at``, those times are used, and each
    tool span takes its agent span's times. Otherwise every span gets placeholder times that
    encode only the ``seq`` order: agent span ``k`` (counted from 0) starts ``2k`` seconds and
    ends ``2k+1`` seconds after ``1970-01-01T00:00:00Z``, and each tool span takes its agent
    span's interval.

    The trace is checked before anything is built, so a refused trace never reaches a model.

    Args:
        trace: The normalized trace.

    Returns:
        The converted session, which time source it used, and the map from each vendor span id
        back to its Calibrated Agent Evaluation span and tool call.

    Raises:
        SessionConversionError: The trace is probed or has no spans; a span's time does not
            parse as ISO-8601 or has no timezone (checked even when another span's missing time
            would send the trace to placeholders); a span ends before it starts; a tool span
            id the converter would build is already the id of another span; or a span's input,
            output, tool arguments or tool result is not JSON, or the detector's span types
            refuse it.
    """
    _require_owned_spans(trace)
    spans = sorted(trace.spans, key=lambda span: span.seq)
    _require_distinct_tool_span_ids(trace.trace_id, spans)
    times, intervals = _span_times(trace.trace_id, spans)
    vendor_spans: list[SpanUnion] = []
    locations: dict[str, tuple[str, int | None]] = {}
    tool_names: dict[str, str] = {}
    for span, interval in zip(spans, intervals, strict=True):
        vendor_spans.append(_agent_span(trace.trace_id, span, interval, times))
        locations[span.span_id] = (span.span_id, None)
        for index in range(len(span.tool_calls)):
            tool_span = _tool_span(trace.trace_id, span, index, interval, times)
            tool_span_id = _tool_span_id(span.span_id, index)
            vendor_spans.append(tool_span)
            locations[tool_span_id] = (span.span_id, index)
            tool_names[tool_span_id] = tool_span.tool_call.name
    vendor_trace = StrandsTrace(
        spans=vendor_spans, trace_id=trace.trace_id, session_id=trace.trace_id
    )
    return ConvertedSession(
        session=Session(traces=[vendor_trace], session_id=trace.trace_id),
        times=times,
        locations=MappingProxyType(locations),
        tool_names=MappingProxyType(tool_names),
    )


def session_json(converted: ConvertedSession) -> str:
    """Serialize a converted session the way the vendor serializes a session for its prompt.

    Args:
        converted: The converted session.

    Returns:
        The JSON array of the session's traces, indented by two spaces, exactly as the detector
        puts it into its prompt.
    """
    return json.dumps(
        [trace.model_dump() for trace in converted.session.traces], indent=2, default=str
    )


def _require_owned_spans(trace: Trace) -> None:
    """Refuse a trace that holds no span for the detector to read."""
    if trace.ownership == "probed":
        raise SessionConversionError(f"trace {trace.trace_id} is probed: {_NEEDS_SPANS}")
    if not trace.spans:
        raise SessionConversionError(f"trace {trace.trace_id} has no spans: {_NEEDS_SPANS}")


def _tool_span_id(span_id: str, index: int) -> str:
    return f"{span_id}#tool{index}"


def _require_distinct_tool_span_ids(trace_id: str, spans: Sequence[Span]) -> None:
    """Refuse a tool span id that a span of the trace already has.

    A detection names its location by vendor span id, so two vendor spans with one id would map
    a failure back to the wrong span or tool call.
    """
    span_ids = {span.span_id for span in spans}
    for span in spans:
        for index in range(len(span.tool_calls)):
            tool_span_id = _tool_span_id(span.span_id, index)
            if tool_span_id in span_ids:
                raise SessionConversionError(
                    f"trace {trace_id}: tool call {index} of span {span.span_id!r} would get the "
                    f"id {tool_span_id!r}, which another span of the trace already has; give the "
                    f"span {tool_span_id!r} another id"
                )


def _span_times(trace_id: str, spans: Sequence[Span]) -> tuple[TimeSource, list[Interval]]:
    """Each span's interval: its own times when every span has both, else placeholders.

    Every time a span records is checked first, so a bad one is refused rather than dropped
    unread when another span's missing time sends the trace to placeholders.
    """
    checked = [_checked_times(trace_id, span) for span in spans]
    recorded = [(start, end) for start, end in checked if start is not None and end is not None]
    if len(recorded) == len(spans):
        return "recorded", recorded
    return "placeholder", [
        (
            _PLACEHOLDER_EPOCH + timedelta(seconds=2 * k),
            _PLACEHOLDER_EPOCH + timedelta(seconds=2 * k + 1),
        )
        for k in range(len(spans))
    ]


def _checked_times(trace_id: str, span: Span) -> tuple[datetime | None, datetime | None]:
    """The span's start and end as aware datetimes, ``None`` where the span records none."""
    start = _checked_time(trace_id, span.span_id, "started_at", span.started_at)
    end = _checked_time(trace_id, span.span_id, "ended_at", span.ended_at)
    if start is not None and end is not None and end < start:
        raise SessionConversionError(
            f"trace {trace_id}: span {span.span_id!r} ends at {span.ended_at}, before it starts "
            f"at {span.started_at}; fix its started_at or ended_at"
        )
    return start, end


def _checked_time(trace_id: str, span_id: str, field: str, value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SessionConversionError(
            f"trace {trace_id}: span {span_id!r} has {field} {value!r}, which is not an "
            "ISO-8601 time; write it as one with a UTC offset, such as '2026-10-16T07:00:00+00:00'"
        ) from exc
    # A naive time names no moment (its offset is unknown) and cannot be ordered against another.
    if parsed.utcoffset() is None:
        raise SessionConversionError(
            f"trace {trace_id}: span {span_id!r} has {field} {value!r} with no timezone; the "
            "detector needs times with a UTC offset"
        )
    return parsed


def _notes(span: Span, times: TimeSource) -> dict[str, Any]:
    """The metadata block every vendor span built from ``span`` carries."""
    return {
        "calibrated_agent_evaluation": {
            "agent": span.agent,
            "role": span.role,
            "occurrence": span.occurrence,
            "status": span.status,
            "times": times,
            "available_tools": _TOOLS_NOTE,
            "system_prompt": _SYSTEM_PROMPT_NOTE,
        }
    }


def _span_info(trace_id: str, span_id: str, parent: str | None, interval: Interval) -> SpanInfo:
    start, end = interval
    return SpanInfo(
        trace_id=trace_id,
        span_id=span_id,
        session_id=trace_id,
        parent_span_id=parent,
        start_time=start,
        end_time=end,
    )


def _agent_span(
    trace_id: str, span: Span, interval: Interval, times: TimeSource
) -> AgentInvocationSpan:
    # dict.fromkeys keeps first-call order; a set would follow the process's string hashing.
    called = dict.fromkeys(call.name for call in span.tool_calls)
    user_prompt = _payload_json(trace_id, span.span_id, "input", span.input)
    agent_response = _payload_json(trace_id, span.span_id, "output", span.output)
    with _refused_by_the_span_types(trace_id, span.span_id):
        return AgentInvocationSpan(
            span_info=_span_info(trace_id, span.span_id, span.parent, interval),
            metadata=_notes(span, times),
            user_prompt=user_prompt,
            agent_response=agent_response,
            available_tools=[ToolConfig(name=name) for name in called],
            system_prompt=None,
        )


def _tool_span(
    trace_id: str, span: Span, index: int, interval: Interval, times: TimeSource
) -> ToolExecutionSpan:
    call = span.tool_calls[index]
    where = f"tool call {index}"
    # The vendor serializes without sort_keys, so the arguments are rebuilt in sorted order.
    arguments = json.loads(_payload_json(trace_id, span.span_id, f"{where} arguments", call.args))
    result = _payload_json(trace_id, span.span_id, f"{where} result", call.result)
    with _refused_by_the_span_types(trace_id, span.span_id):
        return ToolExecutionSpan(
            span_info=_span_info(
                trace_id, _tool_span_id(span.span_id, index), span.span_id, interval
            ),
            metadata=_notes(span, times),
            tool_call=ToolCall(name=call.name, arguments=arguments),
            tool_result=ToolResult(content=result),
            agent_span_id=span.span_id,
        )


def _payload_json(trace_id: str, span_id: str, field: str, value: Any) -> str:
    """``value`` as canonical JSON; refused, naming the span and the field, when it is not JSON."""
    try:
        return canonical_json(value)
    except (TypeError, ValueError, RecursionError) as exc:
        # json's messages name the offending type, never the payload's content.
        raise SessionConversionError(
            f"trace {trace_id}: span {span_id!r} cannot be converted: its {field} is not JSON "
            f"({type(exc).__name__}: {exc}); the detector reads each payload as JSON text"
        ) from exc


@contextmanager
def _refused_by_the_span_types(trace_id: str, span_id: str) -> Iterator[None]:
    """Turn the detector span types' refusal of a span into a refusal naming it."""
    try:
        yield
    except ValidationError as exc:
        # Only the type name: a validation message can quote the payload it refused.
        raise SessionConversionError(
            f"trace {trace_id}: span {span_id!r} cannot be converted: the detector's span types "
            f"refuse it ({type(exc).__name__} on {_error_fields(exc)}); check those fields of "
            "the span. If they are valid, the installed strands-agents-evals checks spans "
            "differently from the version Calibrated Agent Evaluation was checked against: "
            "install that version"
        ) from exc


def _error_fields(exc: ValidationError) -> str:
    fields = dict.fromkeys(
        ".".join(str(part) for part in error["loc"]) or "the span"
        for error in exc.errors(include_url=False, include_context=False, include_input=False)
    )
    return ", ".join(fields)
