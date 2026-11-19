# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which agent a label is about: the shared rule ``run`` and ``improve`` count by.

A crew label names its ``culprit_span`` (the walk-back's origin), and a judge or scorer verdict on
a span is about the span it judged, so the label's agent is that span's agent. Pure: reads
normalized ``Trace`` and ``Label`` only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from calibrated_agent_evaluation.schema import Label, Trace

__all__ = ["SpanAgents", "attributed_agent", "span_agents"]

#: ``(trace_id, span_id) -> agent`` over the traces in scope.
SpanAgents = Mapping[tuple[str, str], str]

_SCORED_KINDS = ("llm-judge", "code")


def span_agents(traces: Iterable[Trace]) -> dict[tuple[str, str], str]:
    """``(trace_id, span_id) -> agent`` for every span of ``traces``."""
    return {(t.trace_id, s.span_id): s.agent for t in traces for s in t.spans}


def attributed_agent(label: Label, agents: SpanAgents | None = None) -> str | None:
    """The agent ``label`` attributes its verdict to, or ``None`` for a trace-level label.

    The source span is the human's ``culprit_span``, or for a judge or scorer verdict the span it
    judged. Its agent is read from ``agents`` when the span is in scope, since a span id need not
    spell its agent; otherwise from the id convention ``"CLM-4471/intake"`` (the segment after the
    last ``/``). A re-executed node's occurrence suffix (``"CLM-4471/intake#2"``) is stripped:
    both executions are the same agent, and keying them apart would split one agent's rate.
    """
    source = label.culprit_span
    if source is None and label.kind in _SCORED_KINDS:
        source = label.span_id
    if not source:
        return None
    if agents and (label.trace_id, source) in agents:
        return agents[(label.trace_id, source)]
    if "/" in source:
        return source.rsplit("/", 1)[-1].split("#", 1)[0]
    return None
