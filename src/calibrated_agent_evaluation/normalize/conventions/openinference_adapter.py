# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenInference (Arize/Phoenix) convention adapter.

OpenInference flattens messages onto **indexed** span attributes (verified against the Arize
openinference spec + a real published fixture):
``llm.input_messages.{i}.message.role`` / ``.content`` (or the nested
``.contents.{k}.message_content.text``), ``llm.output_messages.{i}.message.*``, tool calls under
``llm.*_messages.{i}.message.tool_calls.{j}.tool_call.{id, function.name, function.arguments}``,
and a tool result as a ``message.role="tool"`` entry carrying ``message.tool_call_id`` +
``message.content``. Terminal I/O also appears as ``input.value`` / ``output.value`` JSON blobs.

Collapse-first (see :mod:`messages_adapter`): this convention has no agent/tool boundary spans, so
the session is a single agent; a one-span ``owned`` Trace holding the reconstructed ``tool_calls``
(paired by id via :mod:`pairing`). A span carrying only ``input.value``/``output.value`` and no
per-message attributes still recovers terminal I/O.

Reconstruction is **per span** and then concatenated, never over one flattened attribute map: the
indexed keys restart on every span, so merging them lost every span but the last
(``). Collapse-first is about the *tree*, not about the content.
"""

from __future__ import annotations

import re
from typing import Any

from calibrated_agent_evaluation.normalize import content as content_mod
from calibrated_agent_evaluation.normalize import otlp_json
from calibrated_agent_evaluation.normalize.conventions import registry
from calibrated_agent_evaluation.normalize.conventions.pairing import (
    RawToolCall,
    keyed_span_calls,
    pair,
    trace_id_from,
)
from calibrated_agent_evaluation.normalize.sources import SessionSpans
from calibrated_agent_evaluation.normalize.tree import AgentNode, Intermediate

_INPUT_VALUE = "input.value"
_OUTPUT_VALUE = "output.value"
_AGENT_NAME = "agent.name"
_USAGE_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("input_tokens", ("llm.token_count.prompt",)),
    ("output_tokens", ("llm.token_count.completion",)),
    ("total_tokens", ("llm.token_count.total",)),
)

# Indexed-attribute patterns (zero-based). Verified key shapes.
_TOOL_CALL_KEY = re.compile(
    r"^(llm\.(input|output)_messages\.(\d+)\.message\.tool_calls\.(\d+)\.tool_call)"
    r"\.(?:id|function\.name|function\.arguments)$"
)
_MSG_ROLE = re.compile(r"^llm\.(input|output)_messages\.(\d+)\.message\.role$")


def to_intermediate(session: SessionSpans) -> Intermediate:
    """Reconstruct an OpenInference session into the vendor-free intermediate.

    Reconstructed **per span**, then concatenated; see
    `openllmetry_adapter.to_intermediate` for the shared reasoning. The indexed grammar restarts its
    numbering on every span, so flattening the session with ``dict.update`` made every span
    overwrite the last: a 2-agent, 2-span trace normalized to 1 span and 1 tool call, exit 0, with
    agent 1's `web_search` gone and the trace input replaced by agent 2's.

    **Still collapse-first**: one node, because this convention carries no agent-boundary spans.
    Crew attribution stays unavailable here; the loss is fixed, the tree is not added.
    """
    spans = _span_order(session.spans)

    calls: list[RawToolCall] = []
    results_by_id: dict[str, Any] = {}
    seen_ids: set[str] = set()
    trace_input: dict[str, Any] = {}
    trace_output: dict[str, Any] = {}
    agent_name: str | None = None
    for span in spans:
        attrs = otlp_json.attributes(span)
        span_calls, span_results = _tool_calls_and_results(attrs)
        for call in span_calls:
            if call.id in seen_ids:
                continue  # the same call re-reported on a later span (history resend)
            seen_ids.add(call.id or "")
            calls.append(call)
        results_by_id.update(span_results)
        if not trace_input:
            trace_input = _terminal(attrs, _INPUT_VALUE, role="user")  # the FIRST turn's request
        span_output = _terminal(attrs, _OUTPUT_VALUE, role="assistant")
        if span_output:
            trace_output = span_output  # the LAST-ended span's reply
        if agent_name is None and attrs.get(_AGENT_NAME):
            agent_name = str(attrs[_AGENT_NAME])

    trace_id = session.trace_id or trace_id_from(session.spans) or "openinference-trace"

    tool_calls = pair(calls, results_by_id)
    if not tool_calls and not trace_input and not trace_output:
        return Intermediate(
            trace_id=trace_id,
            frame=session.frame,
            taxonomy_cell=dict(session.taxonomy_cell),
            trace_input=trace_input,
            trace_output=trace_output,
            nodes=[],
            degrade_reason="no OpenInference message/tool content recovered",
        )

    started, ended = otlp_json.session_window(spans)
    node = AgentNode(
        agent=agent_name or "agent",
        role="specialist",
        parent=None,
        input=trace_input,
        output=trace_output,
        tool_calls=tool_calls,
        order=0,
        started_at=started,
        ended_at=ended,
        usage=otlp_json.session_usage(spans, _USAGE_KEYS),
    )
    return Intermediate(
        trace_id=trace_id,
        frame=session.frame,
        taxonomy_cell=dict(session.taxonomy_cell),
        trace_input=trace_input,
        trace_output=trace_output,
        nodes=[node],
    )


def _span_order(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Spans in end-time order, falling back to start time then wire order.

    End time, not wire order: OTLP carries no ordering guarantee, and a concurrent or re-exported
    session arrives in whatever order the exporter flushed. Terminal-output selection
    draws the same distinction.
    """

    def key(indexed: tuple[int, dict[str, Any]]) -> tuple[int, int, int]:
        i, span = indexed
        end = _nanos(span.get("endTimeUnixNano"))
        start = _nanos(span.get("startTimeUnixNano"))
        return (end or start, start, i)

    return [span for _, span in sorted(enumerate(spans), key=key)]


def _nanos(raw: Any) -> int:
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _tool_calls_and_results(
    attrs: dict[str, Any],
) -> tuple[list[RawToolCall], dict[str, Any]]:
    """Recover tool calls (indexed) + tool results (tool-role messages), keyed by tool_call id.

    A call is found by any of its ``id``/``function.name``/``function.arguments`` keys, since the
    id is optional in the convention and a provider or proxy that mints none would otherwise lose
    the call. Calls come back in message then call index order, compared as integers.
    """
    prefixes: dict[tuple[int, int, int], str] = {}
    for key in attrs:
        m = _TOOL_CALL_KEY.match(key)
        if m:
            side = 0 if m.group(2) == "input" else 1
            prefixes[(side, int(m.group(3)), int(m.group(4)))] = m.group(1)
    entries: list[tuple[str | None, str, Any]] = []
    seen: set[str] = set()
    for index in sorted(prefixes):
        base = prefixes[index]
        cid = attrs.get(f"{base}.id")
        cid_str = str(cid) if cid not in (None, "") else None
        if cid_str is not None:
            if cid_str in seen:
                continue
            seen.add(cid_str)
        entries.append(
            (
                cid_str,
                str(attrs.get(f"{base}.function.name") or "tool"),
                attrs.get(f"{base}.function.arguments"),
            )
        )
    return keyed_span_calls(entries), _tool_results_by_id(attrs)


def _tool_results_by_id(attrs: dict[str, Any]) -> dict[str, Any]:
    """A tool result is a message with role=='tool' carrying a tool_call_id + content."""
    results: dict[str, Any] = {}
    for key, value in attrs.items():
        m = _MSG_ROLE.match(key)
        if not m or value != "tool":
            continue
        base = key[: -len(".role")]
        cid = attrs.get(f"{base}.tool_call_id")
        if cid is not None:
            results[str(cid)] = attrs.get(f"{base}.content")
    return results


def _terminal(attrs: dict[str, Any], value_key: str, *, role: str) -> dict[str, Any]:
    """Recover terminal input/output: prefer the value blob, else the first message of ``role``."""
    if value_key in attrs:
        return content_mod.as_object(attrs[value_key])
    # Fall back to the first message text of the wanted role.
    for key, val in attrs.items():
        m = _MSG_ROLE.match(key)
        if m and val == role:
            base = key[: -len(".role")]
            text = attrs.get(f"{base}.content") or attrs.get(
                f"{base}.contents.0.message_content.text"
            )
            if text:
                return content_mod.as_object(text)
    return {}


registry.register_builtin("openinference", to_intermediate)
