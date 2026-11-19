# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a stored trace, or one span of it, into the text a judge reads.

The runtime is deliberately decoupled from how a trace becomes a prompt (``JudgeTarget.content``
is caller-rendered). These are the renderings the shipped surfaces share, so a verdict recorded
through one of them replays through another.

  * A **workflow** judge reads the whole trace: the request, every tool call in span order, and the
    final answer.
  * A **span** judge reads one agent step together with the path that led to it: each ancestor
    from the root down, with what it received and the tools it called, then the judged span in
    full, its output included. That is the span tree the review surface shows, cut to the judged
    span's ancestry. No descendant and no sibling is shown, so the verdict is about
    the judged span's own step and is keyed on it, the culprit side of the crew walk-back. An
    ancestor's output is withheld: an orchestrator writes it after its children
    ran, so it can summarize the judged span's result and leak the answer to the judge. Tool calls
    carry no timestamps, so an ancestor's tool calls are all shown, including any it made after
    the judged span ran.

Numbers render by value. Langfuse returns a whole-number float such as
``149.0`` as ``149``; the two traces compare equal, and a judge must read the same text for both,
or a live judge sees a different prompt by world and a cassette recorded against one misses on the
other. A whole-number float renders as the integer it equals. A float past 2**53 keeps its float
form: it is not an exact integer, and Langfuse returns it as a float as well.
"""

from __future__ import annotations

from typing import Any

from calibrated_agent_evaluation.judges.errors import JudgeLevelError
from calibrated_agent_evaluation.judges.runtime import JudgeTarget
from calibrated_agent_evaluation.schema.types import Span, Trace

__all__ = ["ancestor_path", "render_span", "render_trace", "span_target", "workflow_target"]


# The largest magnitude at which every whole number is an exact float.
_EXACT_INT = 2**53


def _by_value(value: Any) -> Any:
    """``value`` with each whole-number float replaced by the int it equals, at any depth."""
    if isinstance(value, float):
        if value.is_integer() and abs(value) <= _EXACT_INT:
            return int(value)
        return value
    if isinstance(value, dict):
        return {k: _by_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_by_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_by_value(v) for v in value)
    return value


def _show(value: Any) -> str:
    """How a judge prompt shows ``value``: its ``str``, with numbers by value."""
    return str(_by_value(value))


def _tool_line(indent: str, name: str, args: Any, result: Any) -> str:
    return f"{indent}tool {name}(args={_show(args)}) -> {_show(result)}"


def render_trace(trace: Trace) -> str:
    """The judge-facing text for the whole of ``trace``.

    Args:
        trace: The trace to render.

    Returns:
        The trace's input, each span's agent, tool calls and output, and the final output, as text.
    """
    tool_lines = [
        _tool_line("  ", tc.name, tc.args, tc.result)
        for span in trace.spans
        for tc in span.tool_calls
    ]
    tools = "\n".join(tool_lines) if tool_lines else "  (none)"
    return (
        f"Customer request: {_show(trace.input)}\n"
        f"Agent tool calls:\n{tools}\n"
        f"Agent final answer: {_show(trace.output)}\n"
    )


def workflow_target(trace: Trace) -> JudgeTarget:
    """A whole-trace :class:`JudgeTarget` for ``trace``."""
    return JudgeTarget(trace_id=trace.trace_id, span_id=None, content=render_trace(trace))


def ancestor_path(trace: Trace, span: Span) -> list[Span]:
    """The spans from the root down to ``span``'s parent, in that order.

    Raises :class:`JudgeLevelError` when ``span`` is not in ``trace``, or a parent link names a
    span the trace does not hold or loops back on itself: the path would be partial, and a span
    judged on part of its context is a different judgment.
    """
    by_id = {s.span_id: s for s in trace.spans}
    if by_id.get(span.span_id) != span:
        raise JudgeLevelError(f"span {span.span_id!r} is not in trace {trace.trace_id!r}")
    path: list[Span] = []
    seen = {span.span_id}
    parent = span.parent
    while parent is not None:
        node = by_id.get(parent)
        if node is None:
            raise JudgeLevelError(
                f"span {span.span_id!r} in trace {trace.trace_id!r} has an ancestor {parent!r} the "
                "trace does not hold; refusing to judge it on a partial path"
            )
        if node.span_id in seen:
            raise JudgeLevelError(
                f"span {span.span_id!r} in trace {trace.trace_id!r} has a parent cycle at "
                f"{node.span_id!r}"
            )
        seen.add(node.span_id)
        path.append(node)
        parent = node.parent
    return list(reversed(path))


def _step_lines(span: Span, indent: str, *, with_output: bool = True) -> list[str]:
    tools = [_tool_line(f"{indent}  ", tc.name, tc.args, tc.result) for tc in span.tool_calls]
    lines = [
        f"{indent}received: {_show(span.input)}",
        f"{indent}tool calls:",
        *(tools or [f"{indent}  (none)"]),
    ]
    if with_output:
        lines.append(f"{indent}emitted: {_show(span.output)}")
    return lines


def render_span(trace: Trace, span: Span) -> str:
    """The judge-facing text for ``span``: the request, its ancestor path, then the span itself.

    Each ancestor shows what it received and the tools it called, never what it emitted:
    an orchestrator's output is written after its children ran and can state the
    judged span's result, which would hand the judge the answer.
    """
    lines = [f"Customer request: {_show(trace.input)}"]
    path = ancestor_path(trace, span)
    if path:
        lines.append("Steps leading to the step being judged, from the root (outputs withheld):")
        for depth, node in enumerate(path, start=1):
            lines.append(f"  [{depth}] agent {node.agent} ({node.role})")
            lines.extend(_step_lines(node, "      ", with_output=False))
    else:
        lines.append("The step being judged is the root of the run.")
    lines.append(f"Step being judged: agent {span.agent} ({span.role})")
    lines.extend(_step_lines(span, "  "))
    return "\n".join(lines) + "\n"


def span_target(trace: Trace, span: Span) -> JudgeTarget:
    """A span-level :class:`JudgeTarget` for ``span``, keyed on ``(trace_id, span_id)``."""
    return JudgeTarget(
        trace_id=trace.trace_id, span_id=span.span_id, content=render_span(trace, span)
    )
