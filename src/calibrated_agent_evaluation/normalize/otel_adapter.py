# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic OpenTelemetry adapter.

Reads a frozen OTLP-JSON file with the stdlib (``[otel]`` only needed for binary ``.otlp``),
walks **every** span container, and branches on where content lives; span **events**
or span **attributes**; recovering per-step ``input``/``output`` and
``tool_calls{args,result}`` from wherever it is (never dropping event-located content). One
``invoke_agent {name}`` span becomes an :class:`AgentNode`; each ``execute_tool {name}`` child
becomes a leaf :class:`ToolInvocation` on its enclosing agent. Trace-level headers
resolve in a layered order (``calibrated_agent_evaluation.*`` convention attribute → descriptor
override → OTel-``traceId`` hex / defaults). A source carrying only top-level I/O (no per-step
spans) degrades to ``probed`` via the ownership gate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.normalize import (
    agent_boundaries,
    content_placements,
    guidance,
    otlp_json,
)
from calibrated_agent_evaluation.normalize import content as content_mod
from calibrated_agent_evaluation.normalize.content_placements import ContentRole
from calibrated_agent_evaluation.normalize.errors import NormalizationError
from calibrated_agent_evaluation.normalize.otlp_json import (
    ATTR_AGENT_NAME,
    ATTR_OP,
    ATTR_TOOL_NAME,
    EVENT_ASSISTANT_MESSAGE,
    EVENT_TOOL_MESSAGE,
    EVENT_USER_MESSAGE,
)
from calibrated_agent_evaluation.normalize.sources import OTelSpans
from calibrated_agent_evaluation.normalize.tree import (
    AgentNode,
    Intermediate,
    ToolInvocation,
    reroot_siblings,
)

# The calibrated-agent-evaluation header convention keys (first layer of header resolution).
# Same logger name `ownership` uses, so header synthesis and degradation read as one stream.
logger = logging.getLogger("calibrated_agent_evaluation.normalize")

_H_TRACE_ID = guidance.ATTR_TRACE_ID
_H_FRAME = guidance.ATTR_FRAME
_H_TAX = guidance.ATTR_TAXONOMY_PREFIX
_H_ROLE = "calibrated_agent_evaluation.agent.role"
_H_SEQ = "calibrated_agent_evaluation.span.seq"


def read_otlp(path: Path) -> dict[str, Any]:
    """Read a ``.otlp`` binary or OTLP-JSON file to its canonical OTLP-JSON dict.

    Public for the ingest gates, which measure the source before normalizing it and need the
    binary form decoded exactly as this adapter decodes it. Raises :class:`NormalizationError`.
    """
    return _read_otlp(path)


def _read_otlp(path: Path) -> dict[str, Any]:
    """Read an OTLP source to a canonical OTLP-JSON dict, wrapping any read error typed.

    A ``.otlp`` file is binary protobuf → decoded via ``opentelemetry.proto`` (the ``[otel]``
    extra; a missing extra surfaces as a named error). Any other suffix is frozen
    OTLP-JSON read with the stdlib ``json`` module (zero dependency; the default path).
    """
    if path.suffix == ".otlp":
        return _decode_binary_otlp(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise NormalizationError(f"could not read OTLP source {str(path)!r}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise NormalizationError(
            f"OTLP source {str(path)!r} is not valid OTLP-JSON: {exc}"
        ) from exc
    if not isinstance(data, dict) or "resourceSpans" not in data:
        raise NormalizationError(
            f"OTLP source {str(path)!r} is not canonical OTLP-JSON (missing 'resourceSpans')"
        )
    return data


def _decode_binary_otlp(path: Path) -> dict[str, Any]:
    """Decode a binary ``.otlp`` protobuf into canonical OTLP-JSON via the ``[otel]`` extra."""
    try:
        from google.protobuf.json_format import MessageToDict
        from opentelemetry.proto.trace.v1.trace_pb2 import TracesData
    except ImportError as exc:
        raise NormalizationError(
            "decoding a binary .otlp protobuf requires the optional extra; install it with "
            f"`{install_hint('otel')}`"
        ) from exc
    try:
        raw = path.read_bytes()
        message = TracesData()
        message.ParseFromString(raw)
    except OSError as exc:
        raise NormalizationError(f"could not read OTLP source {str(path)!r}: {exc}") from exc
    except Exception as exc:
        raise NormalizationError(
            f"OTLP source {str(path)!r} is not a valid binary OTLP protobuf: {exc}"
        ) from exc
    return MessageToDict(message)


# Semantic conventions this adapter implements. The gen_ai generations share the
# invoke_agent/execute_tool grammar + gen_ai.* content this reader recovers; any other declared
# value is refused rather than silently mis-read as gen_ai.
_SUPPORTED_SEMCONV = frozenset({"gen_ai/legacy", "gen_ai/latest"})


# Conventions the shared convention layer (not this grammar reader) reconstructs.
# "auto" runs detection. When an OTelSpans file declares one of these (or "auto") this adapter
# reads the file and delegates to the convention layer (one convention → one adapter,
# whether the spans arrive from an OTLP file or a dataset row).
_CONVENTION_SEMCONV = frozenset({"auto", "gen_ai/messages", "openllmetry", "openinference"})


def to_intermediate(source: OTelSpans) -> Intermediate:
    """Map an :class:`OTelSpans` source to the vendor-free intermediate.

    Convention resolution: ``"auto"`` detects from the spans; a convention-layer convention
    (``gen_ai/messages``/``openllmetry``/``openinference``) is reconstructed by the shared
    convention layer; the two built-in grammars (``gen_ai/legacy``/``gen_ai/latest``) use this
    reader.
    """
    otlp = _read_otlp(Path(source.path))
    spans = otlp_json.iter_spans(otlp)

    resolved = source.semconv
    if resolved == "auto":
        from calibrated_agent_evaluation.normalize.conventions.detect import detect

        resolved = detect(spans)  # raises typed if undetectable/ambiguous

    if resolved in _CONVENTION_SEMCONV - {"auto"}:
        # Reconstruct via the shared convention layer, reusing the exact adapters the
        # dataset path uses. Pass the already-read spans (no second file read).
        return _via_convention_layer(source, spans, resolved)
    if resolved not in _SUPPORTED_SEMCONV:
        # An unsupported semconv is refused loudly; never mis-read under a wrong grammar.
        raise NormalizationError(
            f"unsupported OTel semantic convention {resolved!r}; "
            f"supported: {sorted(_SUPPORTED_SEMCONV | _CONVENTION_SEMCONV)}"
        )
    res_attrs = otlp_json.resource_attributes(otlp)

    agent_spans = [s for s in spans if _is_agent_span(s)]
    tool_spans = [s for s in spans if _op(s) == "execute_tool"]

    trace_id, frame, taxonomy = _resolve_headers(source, spans, res_attrs)
    trace_input, trace_output = _trace_io(agent_spans)

    degrade_reason = _degrade_reason(agent_spans, tool_spans)
    if degrade_reason is not None:
        # Only a terminal-I/O envelope (no per-step tool spans, no per-step message events); an
        # under-instrumented run. Preserve the terminal input/output; the gate builds probed.
        return Intermediate(
            trace_id=trace_id,
            frame=frame,
            taxonomy_cell=taxonomy,
            trace_input=trace_input,
            trace_output=trace_output,
            nodes=[],
            degrade_reason=degrade_reason,
            convention=resolved,
        )

    nodes = _build_nodes(agent_spans, tool_spans, spans, source.content_location)
    return Intermediate(
        trace_id=trace_id,
        frame=frame,
        taxonomy_cell=taxonomy,
        trace_input=trace_input,
        trace_output=trace_output,
        nodes=nodes,
        convention=resolved,
    )


def _via_convention_layer(
    source: OTelSpans, spans: list[dict[str, Any]], resolved: str
) -> Intermediate:
    """Dispatch already-read OTLP spans to the shared convention layer.

    Reuses the same convention adapters the dataset path uses, so a single OTLP file in the
    ``gen_ai/messages`` / ``openllmetry`` / ``openinference`` convention (or ``auto``-detected to
    one) normalizes identically to a dataset row of the same convention. ``resolved`` is the
    concrete convention (never ``"auto"``); header resolution stays the descriptor's.
    """
    # dispatch imports the built-in adapters (self-registration) and runs the resolved one; passing
    # the concrete `resolved` (never "auto") means it uses that adapter without re-detecting.
    from calibrated_agent_evaluation.normalize.conventions.dispatch import (
        to_intermediate as convention_dispatch,
    )
    from calibrated_agent_evaluation.normalize.sources import SessionSpans

    session = SessionSpans(
        spans=spans,
        semconv=resolved,  # type: ignore[arg-type]  # a concrete Convention (caller narrowed)
        trace_id=source.trace_id,
        frame=source.frame,
        taxonomy_cell=dict(source.taxonomy_cell),
    )
    return convention_dispatch(session)


def _op(span: dict[str, Any]) -> str | None:
    return otlp_json.attributes(span).get(ATTR_OP)


def _is_agent_span(span: dict[str, Any]) -> bool:
    """An agent step: the standard ``invoke_agent`` boundary, or one a vendor family registered.

    The registry only adds boundaries; see
    :mod:`~calibrated_agent_evaluation.normalize.agent_boundaries` for why the standard one is not
    enough on real instrumentor output.
    """
    return _op(span) == "invoke_agent" or agent_boundaries.boundary_name(span) is not None


def _agent_name(span: dict[str, Any]) -> str:
    named = otlp_json.attributes(span).get(ATTR_AGENT_NAME)
    if _op(span) == "invoke_agent" and named:
        return str(named)
    return agent_boundaries.boundary_name(span) or str(named or "agent")


# The per-step message events that signal genuine step-level instrumentation (vs a terminal-only
# envelope). A run with these (even with zero tool calls) is `owned`; one with only
# top-level gen_ai.input/output.content attributes and no tool spans degrades to `probed`.
_STEP_EVENTS = frozenset({EVENT_USER_MESSAGE, EVENT_ASSISTANT_MESSAGE, EVENT_TOOL_MESSAGE})


def _degrade_reason(
    agent_spans: list[dict[str, Any]], tool_spans: list[dict[str, Any]]
) -> str | None:
    """Return a degradation reason when the source carries no recoverable per-step content.

    ``probed`` iff there are no ``execute_tool`` spans AND no per-step message events on any agent
    span; i.e. only a terminal input/output envelope was instrumented. A
    single agent that emitted message events but no tools stays ``owned``.
    """
    if not agent_spans:
        return "no gen_ai invoke_agent spans in the OTLP source"
    if tool_spans:
        return None
    has_step_event = any(
        (event.get("name") in _STEP_EVENTS)
        for span in agent_spans
        for event in span.get("events", []) or []
    )
    if has_step_event:
        return None
    # Content on the attributes of more than one agent span is per-step by construction: a
    # terminal envelope is one input and one output, not one per agent. Without this, a crew whose
    # every agent carried its own content, with no tool spans, was degraded as "only a terminal
    # envelope", which is the opposite of what the trace holds.
    carrying = sum(
        1
        for span in agent_spans
        if _agent_from_attributes(span, "input") or _agent_from_attributes(span, "output")
    )
    if carrying > 1:
        return None
    return (
        "only a terminal input/output envelope was instrumented "
        "(no per-step tool spans and no per-step gen_ai message events)"
    )


def _build_nodes(
    agent_spans: list[dict[str, Any]],
    tool_spans: list[dict[str, Any]],
    all_spans: list[dict[str, Any]],
    prefer: str,
) -> list[AgentNode]:
    """Build one AgentNode per agent span, attaching the execute_tool spans it encloses.

    Parentage is read off the span tree: a node's parent is its nearest *enclosing agent span*,
    walked through whatever non-agent spans sit between (a runtime's message-processing spans, a
    LangGraph subgraph wrapper, a Strands cycle). It used to be the immediate parent only, which
    made every agent under an intermediate span parentless. The parent is pinned to that exact
    execution by span identity (``parent_key``), because a name does not identify one:
    gpt-researcher's section subgraph has its own ``researcher`` node inside the outer
    ``researcher`` node, and resolving the subgraph's reviewer by name attached it to the wrong
    one.

    Agents with no enclosing agent (the Graph/Swarm fan-out under a non-agent root) are re-rooted
    by the shared :func:`~calibrated_agent_evaluation.normalize.tree.reroot_siblings` policy, the
    same one the Strands adapter applies. Without it a fan-out arriving as generic OTLP became N
    parentless roots, which ``validate()`` refuses on duplicate ``seq``.
    """
    by_id = {s.get("spanId"): s for s in all_spans}
    ordered = sorted(agent_spans, key=_start_time)
    index_of = {s.get("spanId"): i for i, s in enumerate(ordered)}
    names = [_agent_name(s) for s in ordered]
    _refuse_repeated_seq(ordered, names)
    orders = [_declared_seq(s, i) for i, s in enumerate(ordered)]
    keys = [_span_key(s, i) for i, s in enumerate(ordered)]
    enclosing = [_enclosing_index(s, by_id, index_of) for s in ordered]
    parents = reroot_siblings([names[i] if i is not None else None for i in enclosing], names)
    entry_index = next((i for i, parent in enumerate(parents) if parent is None), None)
    entry = names[entry_index] if entry_index is not None else None
    # `supervisor` is a claim about a fan-out, so it needs another agent beneath the entry; an agent
    # that merely ran twice is not one.
    crew = len(set(names)) > 1

    tools_of: dict[int, list[ToolInvocation]] = {}
    for tool in sorted(tool_spans, key=_start_time):
        owner = _tool_owner_index(tool, by_id, index_of)
        if owner is not None:
            tools_of.setdefault(owner, []).append(_tool_invocation(tool, prefer))

    nodes: list[AgentNode] = []
    for index, (agent, span) in enumerate(zip(names, ordered, strict=True)):
        attrs = otlp_json.attributes(span)
        # A declared `calibrated_agent_evaluation.agent.role` still wins; absent one, a re-rooted
        # entry with agents beneath it is a supervisor, which is what the fan-out shape means.
        role = attrs.get(_H_ROLE) or ("supervisor" if crew and agent == entry else "specialist")
        parent_index = enclosing[index]
        if parent_index is None and parents[index] is not None:
            parent_index = entry_index  # re-rooted under the entry execution
        nodes.append(
            AgentNode(
                agent=agent,
                role=str(role),  # type: ignore[arg-type]  # validated by schema.validate
                parent=parents[index],
                key=keys[index],
                parent_key=keys[parent_index] if parent_index is not None else None,
                input=_agent_content(span, "input", prefer),
                output=_agent_content(span, "output", prefer),
                tool_calls=tools_of.get(index, []),
                order=orders[index],
                status="errored" if otlp_json.status_code(span) == "ERROR" else "completed",
                started_at=otlp_json.started_at(span),
                ended_at=otlp_json.ended_at(span),
                usage=otlp_json.usage(span),
            )
        )
    return nodes


def _declared_seq(span: dict[str, Any], fallback: int) -> int:
    seq_attr = otlp_json.attributes(span).get(_H_SEQ)
    return int(seq_attr) if isinstance(seq_attr, int) else fallback


def _span_key(span: dict[str, Any], index: int) -> str:
    span_id = span.get("spanId")
    return span_id if isinstance(span_id, str) and span_id else f"#{index}"


def _refuse_repeated_seq(ordered: list[dict[str, Any]], names: list[str]) -> None:
    """Refuse agent spans that declare the same ``calibrated_agent_evaluation.span.seq``.

    The attribute states a span's place in the execution order, so two spans claiming one place
    leave the order undefined.
    """
    claimed: dict[int, list[str]] = {}
    for span, name in zip(ordered, names, strict=True):
        seq = otlp_json.attributes(span).get(_H_SEQ)
        if isinstance(seq, int) and not isinstance(seq, bool):
            claimed.setdefault(seq, []).append(name)
    repeated = {seq: who for seq, who in claimed.items() if len(who) > 1}
    if repeated:
        detail = "; ".join(
            f"seq {seq} on " + ", ".join(repr(n) for n in who)
            for seq, who in sorted(repeated.items())
        )
        raise NormalizationError(
            f"agent spans declare the same {_H_SEQ}, so their execution order is undefined: "
            f"{detail}. Give each span its own seq, or drop the attribute to order by start time."
        )


def _enclosing_index(
    span: dict[str, Any], by_id: dict[Any, dict[str, Any]], index_of: dict[Any, int]
) -> int | None:
    """The execution index of the nearest agent span enclosing ``span``, or ``None`` at a root.

    A parent id absent from the file ends the walk as a root, matching how the reader treated a
    dangling parent before; a cycle in the parent links ends it too rather than looping.
    """
    seen: set[Any] = set()
    parent_id = span.get("parentSpanId")
    while parent_id and parent_id not in seen:
        seen.add(parent_id)
        if parent_id in index_of:
            return index_of[parent_id]
        parent = by_id.get(parent_id)
        if parent is None:
            return None
        parent_id = parent.get("parentSpanId")
    return None


_STRANDS_CYCLE_OP = "execute_event_loop_cycle"


def _tool_owner_index(
    tool: dict[str, Any], by_id: dict[Any, dict[str, Any]], index_of: dict[Any, int]
) -> int | None:
    """The agent a tool span belongs to: its nearest enclosing agent, through non-agent spans.

    A tool under an intermediate span (``invoke_agent`` > ``chat`` > ``execute_tool``) used to be
    dropped because only the direct parent was looked at. The walk stops at a Strands
    event-loop cycle span: those tools carry their arguments in Strands' own placement, which
    this reader does not read, so linking them would ingest every argument as ``{}``; the Strands
    adapter owns them, and leaving them unlinked keeps the tool-loss gate firing here.
    """
    seen: set[Any] = set()
    parent_id = tool.get("parentSpanId")
    while parent_id and parent_id not in seen:
        seen.add(parent_id)
        if parent_id in index_of:
            return index_of[parent_id]
        parent = by_id.get(parent_id)
        if parent is None or _op(parent) == _STRANDS_CYCLE_OP:
            return None
        parent_id = parent.get("parentSpanId")
    return None


def _tool_invocation(tool_span: dict[str, Any], prefer: str = "events") -> ToolInvocation:
    """Map an execute_tool span to a leaf ToolInvocation.

    Recovers args/result from BOTH placements; the ``content_location`` hint only orders which is
    tried first (real traces mix placements, so the reader always falls back to the other rather
    than silently dropping content). ``prefer`` is a best-effort preference, not a
    hard gate.
    """
    name = otlp_json.attributes(tool_span).get(ATTR_TOOL_NAME) or "tool"
    args = _tool_field(tool_span, "tool_args", prefer)
    result = _tool_field(tool_span, "tool_result", prefer)
    return ToolInvocation(
        name=str(name),
        args=content_mod.as_object(args),
        result=content_mod.as_object(result),
    )


def _tool_field(tool_span: dict[str, Any], role: ContentRole, prefer: str) -> Any:
    """Read a tool field (args/result) from the preferred location first, then fall back."""
    readers = (_tool_from_attributes, _tool_from_events)
    if prefer == "events":
        readers = (_tool_from_events, _tool_from_attributes)
    for reader in readers:
        value = reader(tool_span, role)
        if value is not None:
            return value
    return None


def _tool_from_attributes(tool_span: dict[str, Any], role: ContentRole) -> Any:
    """Read tool content from EVERY registered attribute placement for ``role``.

    Previously read one hardcoded key, so an instrumentor using the latest convention's
    ``gen_ai.tool.call.arguments`` (or Traceloop's own names) yielded ``{}`` args on a tool
    call that plainly had them.
    """
    return content_placements.read(tool_span, role)


def _tool_from_events(tool_span: dict[str, Any], role: ContentRole) -> Any:
    key = content_placements.standard_key(role)
    for ev in otlp_json.event_attributes(tool_span, EVENT_TOOL_MESSAGE):
        if key in ev:
            return ev[key]
    return None


def _agent_content(span: dict[str, Any], kind: str, prefer: str = "events") -> dict[str, Any]:
    """Recover an agent span's input/output from attributes AND events.

    ``prefer`` only orders which placement is tried first; the reader always falls back to the
    other so event-located content is never dropped (the silent-drop anti-pattern this guards).
    """
    readers = (_agent_from_attributes, _agent_from_events)
    if prefer == "events":
        readers = (_agent_from_events, _agent_from_attributes)
    for reader in readers:
        value = reader(span, kind)
        if value:
            return value
    return {}


def _agent_from_attributes(span: dict[str, Any], kind: str) -> dict[str, Any]:
    """Read agent content from EVERY registered attribute placement for ``kind``.

    Previously read only ``gen_ai.input``/``output.content``, so a trace carrying the latest
    convention's ``gen_ai.*.messages`` (or a vendor family's keys) normalized to a correct tree
    with empty content on every node and still passed as ``owned``.
    """
    role: ContentRole = "agent_input" if kind == "input" else "agent_output"
    value = content_placements.read(span, role)
    if value is None:
        return {}
    return content_mod.as_object(value)


def _agent_from_events(span: dict[str, Any], kind: str) -> dict[str, Any]:
    event_name = EVENT_USER_MESSAGE if kind == "input" else EVENT_ASSISTANT_MESSAGE
    for ev in otlp_json.event_attributes(span, event_name):
        if "content" in ev:
            return content_mod.as_object(ev["content"])
    return {}


def _start_time(span: dict[str, Any]) -> int:
    try:
        return int(span.get("startTimeUnixNano", 0))
    except (TypeError, ValueError):
        return 0


def _end_time(span: dict[str, Any]) -> int:
    """The span's end time, falling back to its start so an unfinished span still orders."""
    try:
        end = int(span.get("endTimeUnixNano", 0))
    except (TypeError, ValueError):
        end = 0
    return end or _start_time(span)


def _trace_io(agent_spans: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive the terminal trace input/output: the entry agent's input, the final answer.

    Input is the earliest-started agent span's recovered input.

    Output prefers the ROOT agent's own output, and falls back to the last agent span to *end*.
    Both halves of that matter:

    * Root-first is what a supervisor topology needs. In an agent-as-tools crew the supervisor
      delegates, reads what came back, and composes the answer -- so its output IS the trace's
      output, and a sub-agent's intermediate result is not.
    * Last-*ended* rather than last-*started* is the fix for a genuine handoff-terminal topology
      (no single root composing an answer). Ordering by start time returned whichever agent was
      invoked last, which under any concurrency or fan-out is not the one that finished last --
      so `trace.output` was an arbitrary intermediate result presented as the final answer, with
      nothing anywhere saying so.
    """
    if not agent_spans:
        return {}, {}
    by_start = sorted(agent_spans, key=_start_time)
    # The first agent that recorded an input, not merely the first to start: a framework's own
    # outer span (Traceloop's `invoke_agent LangGraph` around the graph) can open first and carry
    # nothing, and taking it made the trace's input `{}` while the request sat one span down.
    trace_input = next((c for s in by_start if (c := _agent_content(s, "input"))), {})

    ids = {s.get("spanId") for s in agent_spans}
    roots = [s for s in agent_spans if s.get("parentSpanId") not in ids]
    if roots:
        root_output = _agent_content(sorted(roots, key=_start_time)[0], "output")
        if root_output:
            return trace_input, root_output
    by_end = sorted(agent_spans, key=_end_time, reverse=True)
    return trace_input, next((c for s in by_end if (c := _agent_content(s, "output"))), {})


def _resolve_headers(
    source: OTelSpans,
    spans: list[dict[str, Any]],
    res_attrs: dict[str, Any],
) -> tuple[str, str, dict[str, str]]:
    """Resolve trace_id/frame/taxonomy_cell in the layered header order."""
    root_attrs = _root_span_attributes(spans)
    convention = {**res_attrs, **root_attrs}

    trace_id = (
        convention.get(_H_TRACE_ID)
        or source.trace_id
        or _otel_trace_hex(spans, fallback="otel-trace")
    )
    frame = convention.get(_H_FRAME) or source.frame
    taxonomy = _taxonomy(convention) or dict(source.taxonomy_cell)
    return str(trace_id), str(frame), taxonomy


def _root_span_attributes(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Attributes of the root-most span (no resolvable parent within this trace)."""
    ids = {s.get("spanId") for s in spans}
    roots = [s for s in spans if s.get("parentSpanId") not in ids]
    if not roots:
        return {}
    return otlp_json.attributes(sorted(roots, key=_start_time)[0])


def _taxonomy(attrs: dict[str, Any]) -> dict[str, str]:
    return {
        key[len(_H_TAX) :]: str(value) for key, value in attrs.items() if key.startswith(_H_TAX)
    }


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
