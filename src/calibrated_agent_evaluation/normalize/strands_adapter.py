# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strands session adapter: multi-agent-pattern-agnostic.

Strands is OTel-native (``gen_ai.system="strands-agents"``): every multi-agent topology (Graph,
Swarm, Agent-as-Tools) is emitted from ONE shared span-name grammar; ``invoke_agent {name}`` /
``execute_event_loop_cycle`` / ``chat`` / ``execute_tool {name}`` under a multiagent root
``invoke_{instance}`` (``invoke_graph`` / ``invoke_swarm``); verified against
``strands-agents==1.45.0`` ``telemetry/tracer.py`` and the captured corpus. So this adapter
recovers **agents from ``invoke_agent`` spans** and **tools from ``execute_tool`` spans**, builds
the tree from OTel ``parentSpanId`` edges, and needs no per-pattern branch:

  * a leaf ``execute_tool`` (no nested ``invoke_agent``) → a ``ToolCall`` on its enclosing agent;
  * an ``execute_tool {child}`` span that **nests an ``invoke_agent {child}``** is an agent
    boundary → the sub-agent is a child ``Span`` (the Agent-as-Tools discriminator);
  * a **Swarm**'s ``handoff_to_agent`` is a registered *tool* → an ``execute_tool
    handoff_to_agent`` span → a ``ToolCall`` (never a phantom agent span);
  * a **Graph**'s ``invoke_agent`` nodes are siblings under ``invoke_graph`` → the entry agent
    becomes the tree root and the others its children (the supervisor-rooted crew tree).

It reads content through the shared stdlib OTLP-JSON core; no ``strands`` runtime import; it
rides the ``[otel]``/stdlib read path; from **both** placements Strands emits:

  * **legacy** (the default): span *events* ``gen_ai.user.message`` / ``gen_ai.choice`` /
    ``gen_ai.tool.message``;
  * **latest-experimental** (``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental``): the
    structured ``gen_ai.input.messages`` / ``gen_ai.output.messages`` and
    ``gen_ai.tool.call.arguments`` / ``gen_ai.tool.call.result`` payloads, which ride on span
    *attributes* when the emitter is also told to write attributes only, and on one details
    *event* otherwise.

``StrandsSession.mode`` orders which placement is tried first and is never a gate; a source is
free to mix them, and the reason this adapter has to try all of them is that some pipelines keep
span attributes and **drop span events entirely** (AWS CloudWatch Transaction Search does), so a
mode-gated reader would recover a correctly-shaped tree with every step's content empty and no
error. That is the silent drop the content gate exists to prevent, and it is invisible to a
tool-call *count* check because the tool spans still name their tools.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.normalize import content as content_mod
from calibrated_agent_evaluation.normalize import guidance, otlp_json
from calibrated_agent_evaluation.normalize.errors import NormalizationError
from calibrated_agent_evaluation.normalize.otlp_json import (
    ATTR_AGENT_NAME,
    ATTR_INPUT_MESSAGES,
    ATTR_OP,
    ATTR_OUTPUT_MESSAGES,
    ATTR_TOOL_CALL_ARGUMENTS,
    ATTR_TOOL_CALL_RESULT,
    ATTR_TOOL_NAME,
    EVENT_CHOICE,
    EVENT_TOOL_MESSAGE,
    EVENT_USER_MESSAGE,
)
from calibrated_agent_evaluation.normalize.sources import StrandsSession
from calibrated_agent_evaluation.normalize.tree import (
    AgentNode,
    Intermediate,
    ToolInvocation,
    reroot_siblings,
)

# Same logger name `ownership` uses, so header synthesis and degradation read as one stream.
logger = logging.getLogger("calibrated_agent_evaluation.normalize")

_H_TRACE_ID = guidance.ATTR_TRACE_ID
_H_FRAME = guidance.ATTR_FRAME
_H_TAX = guidance.ATTR_TAXONOMY_PREFIX
# Sentinel returned when a span's parent chain references a span id absent from the file; a
# dangling edge the tree must surface (validate rejects it), never silently re-root.
_DANGLING = "\x00dangling\x00"


def _read_otlp(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise NormalizationError(f"could not read Strands source {str(path)!r}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise NormalizationError(
            f"Strands source {str(path)!r} is not valid OTLP-JSON: {exc}"
        ) from exc
    if not isinstance(data, dict) or "resourceSpans" not in data:
        raise NormalizationError(
            f"Strands source {str(path)!r} is not canonical OTLP-JSON (missing 'resourceSpans')"
        )
    return data


def to_intermediate(source: StrandsSession) -> Intermediate:
    """Map a :class:`StrandsSession` to the vendor-free intermediate."""
    otlp = _read_otlp(Path(source.path))
    spans = otlp_json.iter_spans(otlp)
    by_id = {s.get("spanId"): s for s in spans}

    agent_spans = [s for s in spans if _op(s) == "invoke_agent"]
    tool_spans = [s for s in spans if _op(s) == "execute_tool"]
    agent_ids = {s.get("spanId") for s in agent_spans}

    trace_id, frame, taxonomy = _resolve_headers(source, spans, otlp)

    if not agent_spans:
        return Intermediate(
            trace_id=trace_id,
            frame=frame,
            taxonomy_cell=taxonomy,
            trace_input={},
            trace_output={},
            nodes=[],
            degrade_reason="no strands invoke_agent spans in the session",
        )

    # An execute_tool that nests an invoke_agent is an agent boundary, not a leaf tool.
    boundary_tool_ids = {
        t.get("spanId") for t in tool_spans if _has_descendant_agent(t, by_id, agent_ids)
    }
    leaf_tools = [t for t in tool_spans if t.get("spanId") not in boundary_tool_ids]

    nodes = _build_agent_nodes(agent_spans, leaf_tools, by_id, agent_ids, source.mode)
    trace_input, trace_output = _trace_io(nodes)
    return Intermediate(
        trace_id=trace_id,
        frame=frame,
        taxonomy_cell=taxonomy,
        trace_input=trace_input,
        trace_output=trace_output,
        nodes=nodes,
    )


def _op(span: dict[str, Any]) -> str | None:
    return otlp_json.attributes(span).get(ATTR_OP)


def _start_time(span: dict[str, Any]) -> int:
    try:
        return int(span.get("startTimeUnixNano", 0))
    except (TypeError, ValueError):
        return 0


def _build_agent_nodes(
    agent_spans: list[dict[str, Any]],
    leaf_tools: list[dict[str, Any]],
    by_id: dict[Any, dict[str, Any]],
    agent_ids: set[Any],
    mode: str,
) -> list[AgentNode]:
    """Build one AgentNode per invoke_agent span, re-rooting siblings under the entry."""
    ordered = sorted(agent_spans, key=_start_time)
    names = [_agent_name(span) for span in ordered]
    # A dangling parent reference is carried through so validate() rejects the broken tree; never
    # silently repointed to a root; so it is resolved before the re-rooting policy, which only
    # ever moves a genuinely parentless agent.
    raw_parents = [_enclosing_agent_name(span, by_id, agent_ids) for span in ordered]
    dangling = {index for index, parent in enumerate(raw_parents) if parent == _DANGLING}
    resolvable = [index for index in range(len(ordered)) if index not in dangling]
    rerooted = reroot_siblings([raw_parents[i] for i in resolvable], [names[i] for i in resolvable])
    parents: list[str | None] = [_DANGLING] * len(ordered)
    for index, parent in zip(resolvable, rerooted, strict=True):
        parents[index] = parent
    entry_name = next(
        (names[i] for i, parent in zip(resolvable, rerooted, strict=True) if parent is None), None
    )
    # `supervisor` is a claim about a fan-out, so it needs another agent beneath the entry; an agent
    # that merely ran twice is not one.
    crew = len({names[i] for i in resolvable}) > 1

    nodes: list[AgentNode] = []
    for order, (name, span) in enumerate(zip(names, ordered, strict=True)):
        nodes.append(
            AgentNode(
                agent=name,
                role="supervisor" if crew and name == entry_name else "specialist",
                parent=parents[order],
                input=_agent_content(span, EVENT_USER_MESSAGE, mode),
                output=_agent_content(span, EVENT_CHOICE, mode),
                tool_calls=_tools_for(span, leaf_tools, by_id, agent_ids, mode),
                order=order,
                status="errored" if otlp_json.status_code(span) == "ERROR" else "completed",
                started_at=otlp_json.started_at(span),
                ended_at=otlp_json.ended_at(span),
                usage=otlp_json.usage(span),
            )
        )
    return nodes


def _agent_name(span: dict[str, Any]) -> str:
    return str(otlp_json.attributes(span).get(ATTR_AGENT_NAME) or "agent")


def _enclosing_agent_name(
    span: dict[str, Any], by_id: dict[Any, dict[str, Any]], agent_ids: set[Any]
) -> str | None:
    """Walk parentSpanId to the nearest ancestor invoke_agent; ``None`` at a root/multiagent root.

    Returns :data:`_DANGLING` when the chain references a span id absent from the file (a broken
    tree the caller surfaces via validate); never silently treats it as a root.
    """
    pid = span.get("parentSpanId")
    seen: set[Any] = set()
    while pid:
        if pid in seen:  # defensive cycle guard
            return None
        seen.add(pid)
        parent = by_id.get(pid)
        if parent is None:
            return _DANGLING
        if pid in agent_ids:
            return _agent_name(parent)
        pid = parent.get("parentSpanId")
    return None


def _has_descendant_agent(
    tool_span: dict[str, Any], by_id: dict[Any, dict[str, Any]], agent_ids: set[Any]
) -> bool:
    """True iff ``tool_span`` has a descendant invoke_agent span (the Agent-as-Tools boundary)."""
    tool_id = tool_span.get("spanId")
    for candidate in agent_ids:
        span = by_id.get(candidate)
        pid = span.get("parentSpanId") if span else None
        seen: set[Any] = set()
        while pid:
            if pid in seen:
                break
            seen.add(pid)
            if pid == tool_id:
                return True
            parent = by_id.get(pid)
            pid = parent.get("parentSpanId") if parent else None
    return False


def _tools_for(
    agent_span: dict[str, Any],
    leaf_tools: list[dict[str, Any]],
    by_id: dict[Any, dict[str, Any]],
    agent_ids: set[Any],
    mode: str = "legacy",
) -> list[ToolInvocation]:
    """Leaf tools whose nearest ancestor invoke_agent is this agent (in wire order)."""
    agent_id = agent_span.get("spanId")
    mine = [t for t in leaf_tools if _nearest_agent_id(t, by_id, agent_ids) == agent_id]
    mine.sort(key=_start_time)
    return [_tool_invocation(t, mode) for t in mine]


def _nearest_agent_id(
    span: dict[str, Any], by_id: dict[Any, dict[str, Any]], agent_ids: set[Any]
) -> Any:
    pid = span.get("parentSpanId")
    seen: set[Any] = set()
    while pid:
        if pid in seen:
            return None
        seen.add(pid)
        if pid in agent_ids:
            return pid
        parent = by_id.get(pid)
        pid = parent.get("parentSpanId") if parent else None
    return None


def _tool_invocation(tool_span: dict[str, Any], mode: str = "legacy") -> ToolInvocation:
    """Map an execute_tool span to a ToolCall, from whichever placement carries its I/O."""
    name = str(otlp_json.attributes(tool_span).get(ATTR_TOOL_NAME) or "tool")
    args = _first_present(
        mode,
        latest=lambda: otlp_json.latest_content(tool_span, ATTR_TOOL_CALL_ARGUMENTS),
        legacy=lambda: _legacy_tool_args(tool_span),
    )
    result = _first_present(
        mode,
        latest=lambda: otlp_json.latest_content(tool_span, ATTR_TOOL_CALL_RESULT),
        legacy=lambda: _choice_message(tool_span),
    )
    return ToolInvocation(
        name=name,
        args=content_mod.as_object(args),
        result=content_mod.as_object(result),
    )


def _legacy_tool_args(tool_span: dict[str, Any]) -> Any:
    """A legacy tool span's arguments: the first ``gen_ai.tool.message`` event's content."""
    for ev in otlp_json.event_attributes(tool_span, EVENT_TOOL_MESSAGE):
        if "content" in ev:
            return ev["content"]
    return None


def _agent_content(span: dict[str, Any], event_name: str, mode: str = "legacy") -> dict[str, Any]:
    """Recover an agent span's handoff input or output from whichever placement carries it."""
    latest_key = ATTR_INPUT_MESSAGES if event_name == EVENT_USER_MESSAGE else ATTR_OUTPUT_MESSAGES
    value = _first_present(
        mode,
        latest=lambda: _messages_text(span, latest_key),
        legacy=lambda: _legacy_agent_content(span, event_name),
    )
    return content_mod.as_object(value)


def _messages_text(span: dict[str, Any], key: str) -> Any:
    """A latest-convention messages payload as prose; a malformed one is a typed error."""
    raw = otlp_json.latest_content(span, key)
    try:
        return content_mod.messages_text(raw) or None
    except ValueError as exc:
        raise NormalizationError(f"strands {key} content is not valid JSON: {exc}") from exc


def _legacy_agent_content(span: dict[str, Any], event_name: str) -> Any:
    """A legacy agent span's content: every ``user.message``, or the last ``choice``.

    An agent run over a conversation carries one ``gen_ai.user.message`` event per user turn, and
    this returned the first one only, so a multi-turn run's input read as its opening line with
    the rest of what the user said dropped. Every turn is kept, in wire order.
    """
    if event_name == EVENT_USER_MESSAGE:
        contents = [
            ev["content"]
            for ev in otlp_json.event_attributes(span, EVENT_USER_MESSAGE)
            if "content" in ev
        ]
        if not contents:
            return None
        if len(contents) == 1:
            return contents[0]
        return "\n".join(text for text in (_as_prose(c) for c in contents) if text)
    return _choice_message(span)


def _as_prose(value: Any) -> str:
    """One message event's content as prose, so several of them can be read in order."""
    decoded = content_mod.as_object(value)
    answer = decoded.get("answer")
    if isinstance(answer, str):
        return answer
    return json.dumps(decoded, sort_keys=True) if decoded else ""


def _first_present(mode: str, *, latest: Any, legacy: Any) -> Any:
    """Try the placement ``mode`` names first, then the other; ``None`` when neither carries it.

    Ordering rather than gating, matching ``otel_adapter``'s ``content_location`` hint: a declared
    mode says where to look *first*, never where content is allowed to be, because a real trace can
    mix placements and a gate would turn that into a silent drop.
    """
    readers = (latest, legacy) if mode == "latest" else (legacy, latest)
    for reader in readers:
        value = reader()
        if value not in (None, "", {}, []):
            return value
    return None


def _choice_message(span: dict[str, Any]) -> Any:
    """Return the ``message`` of a span's last ``gen_ai.choice`` event (its emitted output)."""
    choices = otlp_json.event_attributes(span, EVENT_CHOICE)
    for ev in reversed(choices):
        if "message" in ev:
            return ev["message"]
    return None


def _trace_io(nodes: list[AgentNode]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Terminal trace I/O: the entry (root) agent's input, the final answer.

    Output prefers the ROOT agent's own output and falls back to the last agent to *end*. See
    `otel_adapter._trace_io` for the reasoning; both adapters had the identical defect, which was
    reading `ordered[-1]` where `order` is assigned by START time -- so in a fan-out the reported
    trace output was whichever agent happened to be invoked last, not the one that finished last.
    """
    if not nodes:
        return {}, {}
    ordered = sorted(nodes, key=lambda n: n.order)
    root = next((n for n in ordered if n.parent is None), ordered[0])
    if root.output:
        return dict(root.input), dict(root.output)
    # `ended_at` is an ISO-8601 UTC string, so lexical order is chronological order; a node the
    # source left unfinished sorts first rather than winning the tie-break by accident.
    last_ended = max(nodes, key=lambda n: (n.ended_at or "", n.order))
    return dict(root.input), dict(last_ended.output)


def _resolve_headers(
    source: StrandsSession, spans: list[dict[str, Any]], otlp: dict[str, Any]
) -> tuple[str, str, dict[str, str]]:
    """Resolve trace_id/frame/taxonomy_cell in the layered header order."""
    convention = {**otlp_json.resource_attributes(otlp), **_root_attributes(spans)}
    trace_id = (
        convention.get(_H_TRACE_ID)
        or source.trace_id
        or _otel_trace_hex(spans, fallback="strands-trace")
    )
    frame = convention.get(_H_FRAME) or source.frame
    taxonomy = {
        k[len(_H_TAX) :]: str(v) for k, v in convention.items() if k.startswith(_H_TAX)
    } or dict(source.taxonomy_cell)
    return str(trace_id), str(frame), taxonomy


def _root_attributes(spans: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {s.get("spanId") for s in spans}
    roots = [s for s in spans if s.get("parentSpanId") not in ids]
    if not roots:
        return {}
    return otlp_json.attributes(sorted(roots, key=_start_time)[0])


def _otel_trace_hex(spans: list[dict[str, Any]], *, fallback: str) -> str:
    """Fallback trace_id: the OTel traceId as stable hex (always a valid non-empty id).

    Announces itself. This is the layered header resolution's LAST layer -- no
    `calibrated_agent_evaluation.trace_id` on the wire and no descriptor override -- so the id the
    caller ends up with is one Calibrated Agent Evaluation made up from the transport, and
    `guidance.synthesized_header_hint` names the attribute that would have given a stable one
    instead. That hint existed and was called from nowhere, so a synthesized header was silent even
    though the module was written to explain it.
    """
    for span in spans:
        raw = span.get("traceId")
        if isinstance(raw, str) and raw:
            resolved = otlp_json.trace_id_hex(raw)
            logger.info("%s", guidance.synthesized_header_hint("trace_id", resolved))
            return resolved
    logger.info("%s", guidance.synthesized_header_hint("trace_id", fallback))
    return fallback
