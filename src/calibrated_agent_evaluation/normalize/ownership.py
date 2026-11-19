# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ownership gate + honest degradation.

The normalizer is the *single* point that sets ``ownership``: a trace is ``owned``
only when the adapter recovered a span tree with real per-step content (an agent whose
``input``/``output`` or ``tool_calls`` carry data); otherwise it degrades to an outcome-only
``probed`` trace (``spans == []``, terminal ``input``/``output`` preserved) and surfaces a
``WARNING`` naming what could not be recovered. It is recoverability, not agent
architecture, that decides; so a source that carried only top-level I/O is never dressed up as
a complete ``owned`` trace with silently-empty steps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from calibrated_agent_evaluation.normalize.tree import Intermediate, build_spans, build_trace
from calibrated_agent_evaluation.schema.types import Span, Trace

logger = logging.getLogger("calibrated_agent_evaluation.normalize")


def _has_recovered_content(spans: list[Span]) -> bool:
    """True iff at least one span carries recovered per-step content (input, output, or a tool).

    A span whose ``input``/``output`` are empty and that made no tool call carries no recoverable
    per-step signal; a tree of only such spans is not analyzable step-by-step and must degrade.

    A tool call counts only when it carries ``args`` or ``result``. Testing a
    ``tool_calls`` entry's mere *presence* was the gap: a placement mismatch recovers the tree and
    every tool's name while emptying every payload, which passed this check and shipped as
    ``owned``. A tool name with no arguments and no result records that something happened, not
    what; and "the agent called `search`" is not per-step content a judge or labeler can read.
    """
    return any(
        span.input or span.output or any(call.args or call.result for call in span.tool_calls)
        for span in spans
    )


def _empty_content_reason(spans: list[Span]) -> str:
    """The contentless degrade reason, naming empty content and how much of the tree was empty.

    Two distinct sources reach this line and a reader needs to tell them apart. No tree at all
    means the source carried only a terminal envelope. A tree whose every node is empty means the
    structure was recovered and the content was not, which is a placement or instrumentation gap;
    the node count is what says which, so it goes in the message.
    """
    if not spans:
        return (
            "no per-step content or tool-call attribution recoverable from the source "
            "(only terminal input/output present)"
        )
    tools = sum(len(span.tool_calls) for span in spans)
    named = f" and {tools} named tool call(s) with empty args and results" if tools else ""
    return (
        f"every one of the {len(spans)} recovered node(s) carries empty content{named} — the tree "
        "was recovered but nothing a judge or labeler could read was; the content is absent from "
        "the source or sits in a placement no registered convention reads"
    )


@dataclass(frozen=True)
class Recovery:
    """What the adapter recovered, stated *before* the ownership gate collapsed it.

    The gate's output alone cannot answer "did normalization work?", because a ``probed`` trace has
    ``spans == []`` whether the adapter recovered nothing or recovered a full tree whose every node
    turned out to be empty. Those are a defect and a declared degradation respectively, and the
    ingest gates have to tell them apart: the loss gate refuses tool-execution spans the adapter
    failed to link, while the degrade rule degrades an all-contentless tree and explicitly must
    *not* refuse it.

    So the pre-collapse counts travel beside the trace rather than being re-derived downstream.
    ``reason`` is ``None`` for an ``owned`` trace and otherwise the message that was logged.
    """

    ownership: str
    reason: str | None
    recovered_spans: int
    recovered_tool_calls: int
    convention: str | None = None  # the convention that read the source, when the adapter knows


def resolve(inter: Intermediate) -> tuple[Trace, str]:
    """Return the ``(Trace, ownership)`` for an intermediate, applying the degradation gate.

    ``owned`` when a span tree with recovered per-step content exists; otherwise ``probed`` with
    ``spans == []`` and a logged reason. The returned trace is not yet
    ``validate()``-d; the caller (``normalize``) validates and wraps any ``SchemaError``.

    Kept as the two-value form for every caller that only needs the verdict; see
    :func:`resolve_with_recovery` for the pre-collapse detail the ingest gates need.
    """
    trace, recovery = resolve_with_recovery(inter)
    return trace, recovery.ownership


def resolve_with_recovery(inter: Intermediate) -> tuple[Trace, Recovery]:
    """:func:`resolve`, additionally reporting what was recovered before any collapse."""
    reason = inter.degrade_reason
    spans = [] if reason else build_spans(inter)
    recovered_tool_calls = sum(len(span.tool_calls) for span in spans)

    if reason is None and not _has_recovered_content(spans):
        reason = _empty_content_reason(spans)

    if reason is not None:
        # Surface the degradation with a reason naming what could not be recovered.
        logger.warning("trace %s degraded to ownership='probed': %s", inter.trace_id, reason)
        recovery = Recovery(
            "probed", reason, len(spans), recovered_tool_calls, convention=inter.convention
        )
        return build_trace(inter, [], "probed"), recovery

    return (
        build_trace(inter, spans, "owned"),
        Recovery("owned", None, len(spans), recovered_tool_calls, convention=inter.convention),
    )
