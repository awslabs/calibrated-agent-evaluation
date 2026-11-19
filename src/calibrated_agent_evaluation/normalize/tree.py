# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared, vendor-free tree builder + handoff synthesis.

Every adapter (Inspect / OTel / Strands) reduces its vendor-shaped input to the same
vendor-free **intermediate**: an ordered list of :class:`AgentNode` (one per recovered agent,
each carrying its recovered ``input``/``output`` content and its leaf :class:`ToolInvocation`
list) plus the trace-level headers. This module turns that intermediate into the span
tree; assigning explicit ``seq`` by node order, setting ``parent`` edges, flattening tool
invocations onto ``Span.tool_calls``, synthesizing each child span's handoff
``input``/``output`` where the source has no native field, and collapsing a
single-agent run to a one-span tree whose root mirrors the trace I/O.

The intermediate is the seam that keeps the policy (here) separate from the parsing (the
adapters): ``tree.py`` never sees a vendor type, so single-agent collapse and tool-vs-span
placement are source-independent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from calibrated_agent_evaluation.schema.types import (
    Role,
    Span,
    SpanStatus,
    TokenUsage,
    ToolCall,
    Trace,
)

# The seq the root span always carries.
_ROOT_SEQ = 0


@dataclass(frozen=True)
class ToolInvocation:
    """One recovered *leaf* tool call (no nested agent): becomes a ``Span.tool_calls`` entry.

    An agent/subagent boundary is NOT a ``ToolInvocation``; the adapter promotes it to its own
    :class:`AgentNode` (the tool-vs-span discriminator). ``args``/``result`` are already
    decoded to dicts by the adapter's content reader.
    """

    name: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass
class AgentNode:
    """One recovered agent in the run (vendor-free intermediate).

    ``agent`` is the human agent name; ``role`` its normalized role; ``parent`` the *agent
    name* of its parent (``None`` for the entry node). ``input``/``output`` are the recovered
    per-step content (empty dicts when the source carried none; the ownership gate reads that).
    ``span_id`` lets an adapter pin the ``{trace_id}/{agent}`` id; when ``None`` the
    builder derives it. ``order`` is the adapter's execution order hint used to assign ``seq``.
    ``status`` is the recovered per-step outcome: an adapter that can read the source's error
    signal sets ``"errored"``, and one that cannot leaves the ``"completed"`` default; which is
    a claim about the *adapter*, so it must not be read as a claim that the step succeeded.
    """

    agent: str
    role: Role
    parent: str | None
    input: dict[str, Any] = field(default_factory=dict)
    #: The source identity of this execution (its span id), unique within the intermediate.
    key: str | None = None
    #: The ``key`` of the parent node's exact execution, when the adapter read it off the span
    #: tree. Takes precedence over resolving ``parent`` by name, which cannot tell two executions
    #: of one agent apart (a subgraph that reuses a node name inside that node). Matching by
    #: ``order`` instead attached a child to whichever node shared its parent's order value.
    parent_key: str | None = None
    output: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    span_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    order: int = 0
    status: SpanStatus = "completed"
    usage: TokenUsage | None = None


@dataclass
class Intermediate:
    """The vendor-free hand-off from an adapter to the tree builder + ownership gate.

    ``nodes`` are the recovered agents (already ordered); ``trace_input``/``trace_output`` are
    the terminal request/decision; the headers (``trace_id``/``frame``/``taxonomy_cell``) are
    resolved by the adapter in the layered header order. ``degrade_reason`` is set by an adapter
    that recovered no per-step content and wants the ownership gate to build a ``probed`` trace.
    """

    trace_id: str
    frame: str
    taxonomy_cell: dict[str, str]
    trace_input: dict[str, Any]
    trace_output: dict[str, Any]
    nodes: list[AgentNode] = field(default_factory=list)
    captured_at: str | None = None
    degrade_reason: str | None = None
    # The convention that read the source (`gen_ai/messages`, `openinference`, ...), so a report
    # can say which grammar interpreted a trace, not only which descriptor it arrived in.
    convention: str | None = None


def reroot_siblings(parents: Sequence[str | None], names: Sequence[str]) -> list[str | None]:
    """Re-root a fan-out of parentless agents under the earliest of them (the crew-tree policy).

    A Graph or Swarm emits its agent spans as *siblings* under a multiagent root that is not
    itself an agent span, so every agent resolves to no agent parent. Left alone that is N roots,
    which :func:`~calibrated_agent_evaluation.schema.validate.validate` refuses; every root takes
    ``seq`` 0, so the trace dies on ``duplicate seq values`` and the topology is unrepresentable
    rather than merely unattributed. The entry agent (earliest in execution order) becomes the root
    and the rest its children, which is the supervisor-rooted crew tree the methodology plane reads.

    Positional, one entry per **execution**: ``names[i]`` is the agent of the i-th agent span in
    execution order and ``parents[i]`` its resolved parent agent name (``None`` for parentless).
    It was keyed by agent name, which held only while every agent ran once. The commonest real crew
    shape runs the entry agent again (an AutoGen ``SelectorGroupChat`` goes planner, worker,
    planner): the name-keyed map collapsed both planner executions into one entry, so the second
    stayed a root, both took ``seq`` 0, and the trace was refused. An agent that ran twice under
    *different* parents was worse, because it was silently given whichever parent the map saw last.

    A re-execution of the entry agent becomes a child of the entry name, which the tree builder
    resolves to the nearest preceding execution. A single parentless execution is already a valid
    root and is left untouched, and a nested (Agent-as-Tools) parent is never repointed.

    Policy, not parsing, so it lives here and every adapter shares it: a fan-out topology has to
    normalize the same way whether it arrived as a Strands session or as generic OTLP.
    """
    if len(parents) != len(names):
        raise ValueError(f"reroot_siblings: {len(parents)} parents for {len(names)} names")
    parentless = [index for index, parent in enumerate(parents) if parent is None]
    if len(parentless) < 2:
        return list(parents)
    entry_index = parentless[0]
    entry = names[entry_index]
    return [
        entry if parent is None and index != entry_index else parent
        for index, parent in enumerate(parents)
    ]


def _span_id_for(node: AgentNode, trace_id: str, occurrence: int = 1) -> str:
    """The normalized span id: per **execution**, not per agent.

    It was `f"{trace_id}/{agent}"`, which is unique only while every agent runs at most once. A
    cyclic topology breaks that by design; a repair or retry edge re-runs one node, emitting
    sibling spans that are perfectly legitimate OTel; and the collision then surfaced as
    ``validate``'s ``duplicate span_id values within trace``, i.e. the normalizer refusing
    a correct trace. The producer cannot avoid it either: the only escape would be encoding loop
    bookkeeping into ``gen_ai.agent.name`` (``generator#2``), which breaks counting agents by name.

    The **first** execution keeps the historical id, so every previously normalized artifact is
    byte-identical and only a re-execution introduces a new one.
    """
    if node.span_id:
        return node.span_id
    suffix = "" if occurrence <= 1 else f"#{occurrence}"
    return f"{trace_id}/{node.agent}{suffix}"


def build_spans(inter: Intermediate) -> list[Span]:
    """Build the span tree from the intermediate.

    Single-agent runs collapse to a one-span tree whose root ``input``/``output`` mirror the
    trace I/O; multi-agent runs emit one span per node with a ``parent`` edge and a
    synthesized handoff ``input``/``output``. ``seq`` is assigned by node ``order``
    (stable, contiguous from 0 with the entry node at 0).

    An agent may appear more than once (that is what a cyclic Graph does) so both the id and the
    ``seq`` are computed **per node**, never keyed by agent name.
    """
    ordered = sorted(inter.nodes, key=lambda n: n.order)
    single = len(ordered) == 1

    # Occurrence index per agent, in execution order: 1 for a node that runs once, 2 for the
    # second execution of the same agent, and so on.
    counts: dict[str, int] = {}
    occurrences: list[int] = []
    for node in ordered:
        counts[node.agent] = counts.get(node.agent, 0) + 1
        occurrences.append(counts[node.agent])

    span_ids = [
        _span_id_for(n, inter.trace_id, occ) for n, occ in zip(ordered, occurrences, strict=True)
    ]

    # The entry node (parent None) always carries seq 0; the remaining nodes follow in their order,
    # contiguous from 1. Ranking every node by start-time order and then force-overriding the root
    # to 0 would COLLIDE the root's 0 with whatever node sorted to position 0 whenever the root is
    # not the earliest-starting span (clock skew / near-simultaneous supervisor-fanout starts;
    # exactly the crew topology this exercises), refusing an otherwise-valid trace. Ranking only
    # the non-root nodes keeps the root's 0 unique and the sequence unique + contiguous.
    #
    # Assigned positionally rather than into a dict keyed by agent, which had the same collapse the
    # span id did: a re-executed agent overwrote its own earlier entry, so two nodes came out
    # sharing one seq and the trace died on `duplicate seq` instead.
    seqs: list[int] = []
    rank = 0
    for node in ordered:
        if node.parent is None:
            seqs.append(_ROOT_SEQ)
            continue
        rank += 1
        seqs.append(rank)

    spans: list[Span] = []
    for index, node in enumerate(ordered):
        is_root = node.parent is None
        # A single-agent root mirrors the trace I/O; a multi-agent child carries its
        # synthesized handoff payload (its own recovered input/output), the root the trace input.
        span_input = inter.trace_input if (single or is_root) else node.input
        span_output = inter.trace_output if single else node.output
        spans.append(
            Span(
                span_id=span_ids[index],
                trace_id=inter.trace_id,
                agent=node.agent,
                role=node.role,
                parent=(None if is_root else _parent_span_id(node, ordered, span_ids, index)),
                seq=seqs[index],
                status=node.status,
                input=dict(span_input),
                tool_calls=[
                    ToolCall(name=t.name, args=t.args, result=t.result) for t in node.tool_calls
                ],
                output=dict(span_output),
                started_at=node.started_at,
                ended_at=node.ended_at,
                usage=node.usage,
                occurrence=occurrences[index],
            )
        )
    return spans


def _parent_span_id(
    node: AgentNode, ordered: list[AgentNode], span_ids: list[str], index: int
) -> str | None:
    """Resolve a node's ``parent`` agent-name to a span_id (kept as-is if it doesn't resolve).

    A ``parent`` naming an agent not in ``ordered`` is left as the raw reference so the schema
    ``validate()`` catches the dangling edge as a typed error; the builder never silently drops
    or repoints it.

    ``parent`` names an *agent*, and an agent may have executed several times, so the reference is
    resolved to the **nearest preceding** execution of it. For a repair loop that is exactly right;
    the second validator follows the second generator, not the first; and where the parent ran only
    once it is the same answer the previous agent-keyed lookup gave. A parent that appears only
    *after* this node (out-of-order input) falls back to its first execution rather than dangling.
    """
    if node.parent is None:
        return None
    if node.parent_key is not None:
        pinned = [i for i, other in enumerate(ordered) if other.key == node.parent_key]
        if pinned:
            return span_ids[pinned[0]]
    preceding = [i for i in range(index) if ordered[i].agent == node.parent]
    if preceding:
        return span_ids[preceding[-1]]
    later = [i for i in range(len(ordered)) if ordered[i].agent == node.parent]
    if later:
        return span_ids[later[0]]
    return node.parent


def build_trace(inter: Intermediate, spans: list[Span], ownership: str) -> Trace:
    """Assemble the final ``Trace`` from the resolved spans + ownership (caller validates)."""
    return Trace(
        trace_id=inter.trace_id,
        frame=inter.frame,  # type: ignore[arg-type]  # validated downstream by schema.validate
        ownership=ownership,  # type: ignore[arg-type]
        input=dict(inter.trace_input),
        output=dict(inter.trace_output),
        taxonomy_cell=dict(inter.taxonomy_cell),
        spans=spans,
        captured_at=inter.captured_at,
    )
