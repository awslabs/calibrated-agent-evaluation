# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenLLMetry / Traceloop convention adapter.

OpenLLMetry has two eras (both verified against real traces):

* **legacy dotted-index** (Traceloop ≤ ~0.33): messages flatten onto
  ``gen_ai.prompt.{i}.{role, content, tool_call_id}`` and
  ``gen_ai.completion.{i}.{role, finish_reason, tool_calls.{j}.{id, name, arguments}}``; and
* **latest** (`main`): migrated to the same ``gen_ai.input.messages`` / ``gen_ai.output.messages``
  JSON that :mod:`messages_adapter` handles (tool result under the ``response`` key).

This adapter reconstructs the legacy dotted-index form directly, and **falls through** to the
messages adapter when the span instead carries the latest JSON shape; so one ``openllmetry`` kind
spans both wire shapes. Tool calls (completion side) pair to their results (a matching
``gen_ai.prompt`` entry with the same ``tool_call_id``) by id via :mod:`pairing`. Collapse-first: a
single-agent one-span ``owned`` Trace (no boundary spans in this convention).
"""

from __future__ import annotations

import re
from typing import Any

from calibrated_agent_evaluation.normalize import agent_boundaries, content_placements, otlp_json
from calibrated_agent_evaluation.normalize import content as content_mod
from calibrated_agent_evaluation.normalize.conventions import messages_adapter, registry
from calibrated_agent_evaluation.normalize.conventions.pairing import (
    RawToolCall,
    keyed_span_calls,
    pair,
    trace_id_from,
)
from calibrated_agent_evaluation.normalize.sources import SessionSpans
from calibrated_agent_evaluation.normalize.tree import AgentNode, Intermediate

_PROMPT = "gen_ai.prompt"
_COMPLETION = "gen_ai.completion"
_LATEST_MESSAGES = ("gen_ai.input.messages", "gen_ai.output.messages")
# The legacy era spelled the total `llm.usage.total_tokens`; its prompt and completion counts
# already use the `gen_ai.usage.*` names the shared reader knows.
_USAGE_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    *otlp_json.USAGE_KEYS[:2],
    ("total_tokens", ("gen_ai.usage.total_tokens", "llm.usage.total_tokens")),
)

_COMPLETION_TOOL_KEY = re.compile(
    r"^(gen_ai\.completion\.(\d+)\.tool_calls\.(\d+))\.(?:id|name|arguments)$"
)
_PROMPT_ROLE = re.compile(r"^gen_ai\.prompt\.(\d+)\.role$")


def to_intermediate(session: SessionSpans) -> Intermediate:
    """Reconstruct an OpenLLMetry session (legacy dotted-index, or latest-JSON fall-through).

    Reconstructed **per span**, then concatenated. The dotted-index grammar restarts its numbering
    on every span (each span has its own ``gen_ai.completion.0.*``), so flattening the session into
    one attribute map with ``dict.update`` made every span overwrite the last: a 3-span session with
    one tool call each recovered exactly ONE call, the final span's, and reported ``owned`` with
    ``status="ingested"``. The trace input was whichever span merged last, not the first turn's.

    This matters more than one convention: OpenLLMetry and OpenInference are the only on-ramp
    documented for LangGraph, LangChain, CrewAI, AutoGen, the OpenAI Agents SDK and Pydantic AI, so
    every multi-turn session ingested the documented way lost all but its last span.

    **Still collapse-first.** The recovered calls land on ONE node, because this convention carries
    no agent-boundary spans to build a tree from. So crew attribution remains unavailable on the
    convention path: this fixes the loss, it does not add a tree. This is said explicitly to
    stop a reader assuming otherwise.
    """
    ordered = _span_order(session.spans)

    # Fall-through: if the source uses the latest JSON messages shape, defer to the shared
    # messages adapter ("openllmetry" spans both wire eras through one kind).
    if any(k in otlp_json.attributes(span) for span in ordered for k in _LATEST_MESSAGES):
        return messages_adapter.to_intermediate(session)

    calls: list[RawToolCall] = []
    results_by_id: dict[str, Any] = {}
    seen_ids: set[str] = set()
    trace_input: dict[str, Any] = {}
    trace_output: dict[str, Any] = {}
    agent_name: str | None = None
    for span in ordered:
        attrs = otlp_json.attributes(span)
        span_calls, span_results = _tool_calls_and_results(attrs)
        for call in span_calls:
            if call.id in seen_ids:
                continue  # the same call re-reported on a later span (history resend)
            seen_ids.add(call.id or "")
            calls.append(call)
        results_by_id.update(span_results)
        if not trace_input:
            trace_input = _prompt_text(attrs, "user")  # the FIRST turn's request
        span_output = _completion_text(attrs)
        if span_output:
            trace_output = span_output  # the LAST-ended span's reply
        if agent_name is None and attrs.get("gen_ai.system"):
            agent_name = str(attrs["gen_ai.system"])

    trace_id = session.trace_id or trace_id_from(session.spans) or "openllmetry-trace"

    tool_calls = pair(calls, results_by_id)
    if not tool_calls and not trace_input and not trace_output:
        return Intermediate(
            trace_id=trace_id,
            frame=session.frame,
            taxonomy_cell=dict(session.taxonomy_cell),
            trace_input=trace_input,
            trace_output=trace_output,
            nodes=[],
            degrade_reason="no OpenLLMetry prompt/completion content recovered",
        )

    started, ended = otlp_json.session_window(ordered)
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
        usage=otlp_json.session_usage(ordered, _USAGE_KEYS),
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
    session arrives in whatever order the exporter flushed. Terminal-output selection draws the
    same distinction.
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


def _tool_calls_and_results(attrs: dict[str, Any]) -> tuple[list[RawToolCall], dict[str, Any]]:
    """Completion-side tool calls, keyed to results via a prompt entry sharing the tool_call_id.

    A call is found by any of its ``id``/``name``/``arguments`` keys, since the id is optional and
    a provider or proxy that mints none would otherwise lose the call. Calls come back in
    completion then call index order, compared as integers.
    """
    prefixes: dict[tuple[int, int], str] = {}
    for key in attrs:
        m = _COMPLETION_TOOL_KEY.match(key)
        if m:
            prefixes[(int(m.group(2)), int(m.group(3)))] = m.group(1)
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
            (cid_str, str(attrs.get(f"{base}.name") or "tool"), attrs.get(f"{base}.arguments"))
        )
    return keyed_span_calls(entries), _tool_results_by_id(attrs)


def _tool_results_by_id(attrs: dict[str, Any]) -> dict[str, Any]:
    """A tool result rides a prompt entry carrying a tool_call_id + content (the tool-role turn)."""
    results: dict[str, Any] = {}
    for key, value in attrs.items():
        if key.endswith(".tool_call_id") and key.startswith(_PROMPT + "."):
            base = key[: -len(".tool_call_id")]
            results[str(value)] = attrs.get(f"{base}.content")
    return results


def _prompt_text(attrs: dict[str, Any], role: str) -> dict[str, Any]:
    for key, val in attrs.items():
        m = _PROMPT_ROLE.match(key)
        if m and val == role:
            base = key[: -len(".role")]
            text = attrs.get(f"{base}.content")
            if text:
                return content_mod.as_object(text)
    return {}


def _completion_text(attrs: dict[str, Any]) -> dict[str, Any]:
    content = attrs.get(f"{_COMPLETION}.0.content")
    return content_mod.as_object(content) if content else {}


registry.register_builtin("openllmetry", to_intermediate)

# Traceloop's own content keys, contributed to the shared placement registry so the
# convention-neutral OTel reader can recover them WITHOUT any vendor name appearing in
# `otel_adapter`. They live here because this is the module that owns the Traceloop
# dialect. Verified against `opentelemetry-instrumentation-langchain` 0.62.3, which is what
# gpt-researcher emits: its content sits on these keys alone, so before this registration the
# generic reader built agents with `{}` on every one of them.
content_placements.register_builtin(
    "traceloop",
    {
        "agent_input": ("traceloop.entity.input", "gen_ai.task.input"),
        "agent_output": ("traceloop.entity.output", "gen_ai.task.output"),
        "tool_args": ("traceloop.entity.input",),
        "tool_result": ("traceloop.entity.output",),
    },
)

# Traceloop names each compiled LangGraph run `invoke_agent LangGraph` and emits every graph NODE as
# an `execute_task` span. The nodes are the crew's roles (gpt-researcher's browser, planner,
# researcher, writer, reviewer...), and they are where the content is; read by the standard
# boundary alone, a seven-role crew normalized to three contentless agents all named `LangGraph`.
# Measured on a live gpt-researcher 0.15.1 capture under instrumentation-langchain 0.62.3.
#
# A node span is identified by LangGraph's own metadata, which Traceloop copies onto the span as an
# association property: `langgraph_node` names the node, and a node span's own entity name equals
# it. Every runnable *inside* a node inherits the same `langgraph_node` property but has a
# different entity name (a subgraph wrapper is `LangGraph`, an inner lambda `unknown`), so the
# equality is what separates the node from its internals.
_LANGGRAPH_NODE = "traceloop.association.properties.langgraph_node"
_ENTITY_NAME = "traceloop.entity.name"


def _langgraph_node(span: dict[str, Any]) -> str | None:
    attrs = otlp_json.attributes(span)
    if attrs.get("gen_ai.operation.name") != "execute_task":
        return None
    node = attrs.get(_LANGGRAPH_NODE)
    if isinstance(node, str) and node and node == attrs.get(_ENTITY_NAME):
        return node
    return None


agent_boundaries.register_agent_boundary("traceloop-langgraph", _langgraph_node)
