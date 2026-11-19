# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default trace renderer: the tool-call tree.

The fallback renderer used for any trace no registered plugin renderer matches. It
shows the trace's terminal input/output and, for an ``owned`` trace, the span tree with each
span's received handoff ``input``, ``tool_calls`` (name/args/result), and emitted ``output``;
the "look at the data" primary view. A ``probed`` trace (no spans) shows the
terminal input/output only. Produces FastHTML components; imported only when the ``[ui]`` extra
is installed.
"""

from __future__ import annotations

import json
from typing import Any

from fasthtml.common import H3, H4, Details, Div, Li, Pre, Section, Span, Strong, Summary, Ul

from calibrated_agent_evaluation.schema import Trace
from calibrated_agent_evaluation.schema.types import Span as TraceSpan

# A JSON block longer than this many lines renders collapsed, so one large tool result does not
# bury the rest of the span tree. Nothing is truncated: expanding shows the whole value.
FOLD_LINES = 30


def _pretty(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def _block(value: Any, cls: str) -> Any:
    text = _pretty(value)
    lines = text.count("\n") + 1
    if lines <= FOLD_LINES:
        return Pre(text, cls=cls)
    return Details(Summary(f"{lines} lines (expand)"), Pre(text, cls=cls), cls="hd-fold")


def _tool_calls(span: TraceSpan) -> Any:
    if not span.tool_calls:
        return Span("(no tool calls)", cls="hd-muted")
    items = [
        Li(
            Strong(tc.name),
            _block({"args": tc.args, "result": tc.result}, "hd-toolcall"),
            cls="hd-tool",
        )
        for tc in span.tool_calls
    ]
    return Ul(*items, cls="hd-tools")


def _span_node(span: TraceSpan, children: list[Any]) -> Any:
    return Li(
        Div(
            Strong(f"{span.agent} "),
            Span(f"[{span.role}] seq={span.seq} status={span.status}", cls="hd-muted"),
            cls="hd-span-head",
        ),
        Div(Strong("input: "), _block(span.input, "hd-io"), cls="hd-span-input"),
        Div(Strong("tool_calls:"), _tool_calls(span), cls="hd-span-tools"),
        Div(Strong("output: "), _block(span.output, "hd-io"), cls="hd-span-output"),
        Ul(*children, cls="hd-children") if children else "",
        cls="hd-span",
        data_span_id=span.span_id,
    )


def _span_tree(trace: Trace) -> Any:
    by_parent: dict[str | None, list[TraceSpan]] = {}
    for span in sorted(trace.spans, key=lambda s: s.seq):
        by_parent.setdefault(span.parent, []).append(span)

    def build(parent_id: str | None) -> list[Any]:
        return [_span_node(sp, build(sp.span_id)) for sp in by_parent.get(parent_id, [])]

    roots = build(None)
    if not roots:  # defensive: an owned trace with orphaned parents still shows every span flat
        roots = [_span_node(sp, []) for sp in sorted(trace.spans, key=lambda s: s.seq)]
    return Ul(*roots, cls="hd-span-tree")


def tool_call_tree_renderer(trace: Trace) -> Any:
    """Render ``trace`` as its terminal I/O plus (for owned traces) the tool-call span tree."""
    header = Div(
        H3(f"Trace {trace.trace_id}"),
        Span(
            f"frame={trace.frame} · ownership={trace.ownership} · "
            f"cell={_pretty(trace.taxonomy_cell)}",
            cls="hd-muted",
        ),
        cls="hd-trace-header",
    )
    terminal = Section(
        Div(Strong("input: "), _block(trace.input, "hd-io")),
        Div(Strong("output: "), _block(trace.output, "hd-io")),
        cls="hd-terminal",
    )
    if trace.spans:
        body = Section(H4("Span tree"), _span_tree(trace), cls="hd-spans")
    else:
        body = Section(
            Span("probed trace: no spans (outcome-only)", cls="hd-muted"), cls="hd-spans"
        )
    return Div(header, terminal, body, cls="hd-trace", data_trace_id=trace.trace_id)
