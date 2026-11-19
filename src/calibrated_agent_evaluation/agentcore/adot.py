# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""ADOT-to-OTLP-JSON assembly and content re-home.

AgentCore's ``aws/spans`` export carries the span tree but, once CloudWatch Transaction Search is
on, strips the ``gen_ai.*`` content events off the spans; the message content rides instead in
separate ``otel-rt-logs`` log records keyed by ``traceId`` + ``spanId``. Those content records key
to the per-LLM-call ``chat`` spans, while the normalizer's Strands convention reads an agent's
handoff I/O off its ``invoke_agent`` span and a tool's args/result off its ``execute_tool`` span.

This module bridges that gap: it assembles the span records into the canonical OTLP-JSON document
the normalizer reads and **re-homes** the content from the ``chat`` records onto the spans the
convention reads, in the legacy ``gen_ai.user.message`` / ``gen_ai.choice`` /
``gen_ai.tool.message`` events the adapter already understands:

  * an ``invoke_agent`` span gets the first user message of its subtree's earliest ``chat`` as its
    input, and the last non-empty choice of its latest ``chat`` as its output (never crossing a
    nested ``invoke_agent`` boundary, so a sub-agent's turns stay with the sub-agent);
  * an ``execute_tool`` span gets its arguments and result matched by ``gen_ai.tool.call.id`` to the
    assistant tool-call request and the tool-result record.

A dangling ``parentSpanId`` (a root whose parent is outside the pulled window) is cleared so the
tree re-roots cleanly rather than being rejected as broken. A content-bearing span that recovers
no content from either inline events or the log records is refused
(:class:`AgentCoreContentMissing`) rather than persisted as an empty step.

Vendor-free by contract: the raw AWS-response decoding lives in
:mod:`calibrated_agent_evaluation.agentcore.readers`; this module takes already-decoded OTLP span
dicts and plain log-record dicts, so it holds no boto3 and no AWS-response-envelope assumptions.
"""

from __future__ import annotations

import json
import string
from collections import defaultdict
from typing import Any

from calibrated_agent_evaluation.agentcore.errors import AgentCoreContentMissing, AgentCoreError
from calibrated_agent_evaluation.normalize import otlp_json
from calibrated_agent_evaluation.normalize.otlp_json import (
    ATTR_INPUT_CONTENT,
    ATTR_INPUT_MESSAGES,
    ATTR_OP,
    ATTR_OUTPUT_CONTENT,
    ATTR_OUTPUT_MESSAGES,
    EVENT_ASSISTANT_MESSAGE,
    EVENT_CHOICE,
    EVENT_TOOL_MESSAGE,
    EVENT_USER_MESSAGE,
)

# AgentCore stamps each execute_tool span with the id of the tool-use it ran; the content records
# carry the same id, so it is the join key between a tool span and its arguments/result.
_ATTR_TOOL_CALL_ID = "gen_ai.tool.call.id"


def xray_trace_id_to_otel(raw: str) -> str:
    """Normalize an X-Ray ``1-<8 hex>-<24 hex>`` trace id to the 32-hex OTel id.

    An already-32-hex id is returned lowercased unchanged. A value in neither form raises rather
    than returning a corrupt id, so a malformed id never silently points at a nonexistent trace.
    """
    value = raw.strip()
    if _is_hex(value, 32):
        return value.lower()
    parts = value.split("-")
    if len(parts) == 3 and parts[0] == "1" and _is_hex(parts[1], 8) and _is_hex(parts[2], 24):
        return (parts[1] + parts[2]).lower()
    raise AgentCoreError(
        f"trace id {raw!r} is neither a 32-hex OTel id nor an X-Ray '1-<8hex>-<24hex>' id"
    )


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(c in string.hexdigits for c in value)


def adot_to_otlp(
    span_records: list[dict[str, Any]],
    log_records: list[dict[str, Any]],
    *,
    resource_attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble OTLP span dicts into one canonical OTLP-JSON document, content re-homed.

    ``span_records`` are already-decoded OTLP span dicts (``spanId``/``traceId``/``parentSpanId``/
    ``attributes``/...). ``log_records`` are the decoded ``otel-rt-logs`` content records, each a
    plain dict with ``spanId``, ``eventName`` and ``body`` (and an optional timestamp used only for
    ordering). Content is re-homed from the ``chat`` records onto each ``invoke_agent`` and
    ``execute_tool`` span; a content-bearing span that recovers nothing raises
    :class:`AgentCoreContentMissing` naming the trace and span.
    """
    spans = [dict(span) for span in span_records]
    by_id = {s.get("spanId"): s for s in spans if isinstance(s.get("spanId"), str)}
    agent_ids = {sid for sid, s in by_id.items() if _op(s) == "invoke_agent"}

    logs_by_span: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in log_records:
        span_id = record.get("spanId")
        if isinstance(span_id, str):
            logs_by_span[span_id].append(record)
    tool_args, tool_results = _index_tool_payloads(log_records)

    for span_id in agent_ids:
        _rehome_agent_content(by_id[span_id], span_id, spans, by_id, agent_ids, logs_by_span)
    for span in spans:
        if _op(span) == "execute_tool":
            _rehome_tool_content(span, tool_args, tool_results)

    # The content refusal gates on agent handoff I/O only. That is where wholesale content
    # stripping shows up (an invoke_agent with no recoverable input or output means its whole
    # conversation was dropped). A leaf execute_tool whose args/result were never emitted as a
    # separate content record is left best-effort (empty args/result) rather than failing the whole
    # trace: the tool call is still structurally recovered, and the normalizer's content gate counts
    # it as contentless, so the gap is reported, not silent. Verified against a live trace whose
    # search_kb tool logged no gen_ai content record of its own.
    for span in spans:
        if _op(span) == "invoke_agent":
            _refuse_if_agent_content_missing(span)

    present = set(by_id)
    for span in spans:
        parent = span.get("parentSpanId")
        if isinstance(parent, str) and parent and parent not in present:
            # A root whose parent is outside the pulled window; clear it so the tree re-roots here
            # rather than being rejected as a dangling edge.
            span["parentSpanId"] = ""

    return {
        "resourceSpans": [
            {
                "resource": {"attributes": dict(resource_attributes or {})},
                "scopeSpans": [{"scope": {"name": "strands.telemetry.tracer"}, "spans": spans}],
            }
        ]
    }


def _op(span: dict[str, Any]) -> str | None:
    return otlp_json.attributes(span).get(ATTR_OP)


def _rehome_agent_content(
    agent_span: dict[str, Any],
    agent_id: Any,
    spans: list[dict[str, Any]],
    by_id: dict[Any, dict[str, Any]],
    agent_ids: set[Any],
    logs_by_span: dict[str, list[dict[str, Any]]],
) -> None:
    """Re-home an agent's handoff I/O from its subtree's ``chat`` records onto its span.

    Input is the first ``gen_ai.user.message`` of the earliest ``chat`` whose nearest enclosing
    ``invoke_agent`` is this one; output is the last non-empty ``gen_ai.choice`` of the latest such
    ``chat`` (the final answer, past the intermediate tool-use turns whose choice has no text).
    """
    chats = sorted(
        (s for s in spans if _op(s) == "chat" and _nearest_agent(s, by_id, agent_ids) == agent_id),
        key=_start_time,
    )
    first_input: Any = None
    last_output: str | None = None
    for chat in chats:
        chat_id = chat.get("spanId")
        if not isinstance(chat_id, str):
            continue
        for record in sorted(logs_by_span.get(chat_id, []), key=_record_time):
            event = record.get("eventName")
            body = _body(record)
            if event == EVENT_USER_MESSAGE and first_input is None:
                first_input = body.get("content")
            elif event == EVENT_CHOICE:
                text = _text_of((body.get("message") or {}).get("content"))
                if text.strip():
                    last_output = text
    events: list[dict[str, Any]] = []
    if first_input is not None and not _has_event(agent_span, EVENT_USER_MESSAGE):
        events.append(_event(EVENT_USER_MESSAGE, "content", json.dumps(first_input)))
    if last_output is not None and not _has_event(agent_span, EVENT_CHOICE):
        events.append(_event(EVENT_CHOICE, "message", last_output))
    if events:
        agent_span.setdefault("events", []).extend(events)


def _rehome_tool_content(
    tool_span: dict[str, Any],
    tool_args: dict[str, Any],
    tool_results: dict[str, Any],
) -> None:
    """Re-home a tool's arguments and result, matched to its span by ``gen_ai.tool.call.id``."""
    call_id = otlp_json.attributes(tool_span).get(_ATTR_TOOL_CALL_ID)
    if not isinstance(call_id, str):
        return
    events: list[dict[str, Any]] = []
    if call_id in tool_args and not _has_event(tool_span, EVENT_TOOL_MESSAGE):
        events.append(_event(EVENT_TOOL_MESSAGE, "content", json.dumps(tool_args[call_id])))
    if call_id in tool_results and not _has_event(tool_span, EVENT_CHOICE):
        events.append(_event(EVENT_CHOICE, "message", json.dumps(tool_results[call_id])))
    if events:
        tool_span.setdefault("events", []).extend(events)


def _index_tool_payloads(
    log_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Index tool arguments (from assistant tool-call requests) and results by tool-use id.

    Arguments come from a ``gen_ai.assistant.message``'s ``tool_calls``/``toolUse`` blocks; the
    result comes from the ``gen_ai.tool.message`` that carries the same id. Both are keyed by the
    tool-use id that an ``execute_tool`` span repeats in ``gen_ai.tool.call.id``.
    """
    args: dict[str, Any] = {}
    results: dict[str, Any] = {}
    for record in log_records:
        event = record.get("eventName")
        body = _body(record)
        if event == EVENT_ASSISTANT_MESSAGE:
            for call in body.get("tool_calls") or []:
                call_id = call.get("id")
                if isinstance(call_id, str):
                    args.setdefault(call_id, (call.get("function") or {}).get("arguments"))
            for part in body.get("content") or []:
                use = part.get("toolUse") if isinstance(part, dict) else None
                if isinstance(use, dict) and isinstance(use.get("toolUseId"), str):
                    args.setdefault(use["toolUseId"], use.get("input"))
        elif event == EVENT_TOOL_MESSAGE:
            call_id = body.get("id")
            if isinstance(call_id, str):
                results.setdefault(call_id, body.get("content"))
    return args, results


def _refuse_if_agent_content_missing(span: dict[str, Any]) -> None:
    """Refuse an ``invoke_agent`` span that recovered no handoff content, inline or re-homed.

    This is the wholesale-stripping guard: an agent whose subtree yielded neither input nor output
    means its conversation content was dropped, so the step would persist empty. A leaf tool with
    no logged payload is NOT gated here (it degrades to empty args/result); only the agent's own
    handoff I/O gates, because that is what makes a trace analyzable step by step.
    """
    if _has_agent_content(span):
        return
    raise AgentCoreContentMissing(
        f"span {span.get('spanId')!r} of trace {span.get('traceId')!r} carries operation "
        "'invoke_agent' but no handoff content was recovered from its span events or the "
        "otel-rt-logs records; its conversation was stripped and not returned. Confirm the content "
        "log group (otel-rt-logs) was read and AWS_GENAI_CONTENT_EXTRACTION_OPT_OUT is set"
    )


def _has_agent_content(span: dict[str, Any]) -> bool:
    # Recognise content in every placement the normalizer recovers from, so the gate is never
    # narrower than the adapter (a narrower gate falsely refuses a trace the pipeline could read):
    # latest-convention messages on the span OR the inference-details event (latest_content tries
    # both), the legacy single-blob content attributes, and the legacy user/choice events.
    if otlp_json.latest_content(span, ATTR_INPUT_MESSAGES) or otlp_json.latest_content(
        span, ATTR_OUTPUT_MESSAGES
    ):
        return True
    attrs = otlp_json.attributes(span)
    if attrs.get(ATTR_INPUT_CONTENT) or attrs.get(ATTR_OUTPUT_CONTENT):
        return True
    return _has_event(span, EVENT_USER_MESSAGE) or _has_event(span, EVENT_CHOICE)


def _has_event(span: dict[str, Any], name: str) -> bool:
    return any(e.get("name") == name for e in span.get("events") or [])


def _nearest_agent(
    span: dict[str, Any], by_id: dict[Any, dict[str, Any]], agent_ids: set[Any]
) -> Any:
    """The nearest enclosing ``invoke_agent`` span id via ``parentSpanId`` (``None`` at a root)."""
    pid = span.get("parentSpanId")
    seen: set[Any] = set()
    while pid and pid not in seen:
        seen.add(pid)
        if pid in agent_ids:
            return pid
        parent = by_id.get(pid)
        pid = parent.get("parentSpanId") if parent else None
    return None


def _start_time(span: dict[str, Any]) -> int:
    try:
        return int(span.get("startTimeUnixNano") or 0)
    except (TypeError, ValueError):
        return 0


def _record_time(record: dict[str, Any]) -> int:
    for key in ("timeUnixNano", "observedTimeUnixNano"):
        try:
            value = int(record.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value:
            return value
    return 0


def _body(record: dict[str, Any]) -> dict[str, Any]:
    body = record.get("body")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return {}
    return body if isinstance(body, dict) else {}


def _text_of(content: Any) -> str:
    """Join the ``text`` parts of a ``[{"text": ...}]`` content list; ``""`` when there are none."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]
        )
    return ""


def _event(name: str, key: str, value: str) -> dict[str, Any]:
    """One OTLP span event carrying a single string attribute, in KeyValue-list form."""
    return {"name": name, "attributes": [{"key": key, "value": {"stringValue": value}}]}
