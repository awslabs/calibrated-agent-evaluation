# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The counterfactual walk-back.

Two pieces, both honoring the human-gold boundary:

  * :func:`validate_walkback` -- a pure, deterministic validator. Given the labeler's recorded
    ``culprit_spans`` / ``manifestation_span`` / ``handoff_implicated`` and the real ``Trace``, it
    confirms every named span exists, that each culprit is at or before the manifestation in causal
    (``seq``/ancestry) order (a failure cannot originate after it surfaced), and that an implicated
    handoff is a declared seam. It supports multiple / ``disputed`` culprits
    and the outcome-pass / span-fail case -- a span defect recorded independently of the
    trace outcome. Structural violations raise a typed :class:`AttributionError`.

  * :func:`ratify_suggestion` -- an LLM MAY *propose* a culprit, but the proposal is an
    unratified :class:`AttributionSuggestion` until a human ratifies it; an unratified suggestion is
    refused. This layer ships no proposer and carries no LLM dependency.

"Causal order" is the crew's **declared pipeline ordering** (``CrewDefinition``'s
``ExpectedProcess.expected_agents`` -- supervisor -> intake -> ... -> payout), so a culprit that
originated a failure must run at or before the agent where the harm manifested. It is NOT the span
``seq``: the normalizer assigns ``seq`` in emission order, which does not encode a flat
sibling pipeline (all specialists are children of the supervisor), so seq is only a fallback when an
agent is absent from the declared ordering. A parent (ancestor) is at-or-before its descendant.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from calibrated_agent_evaluation.attribution.errors import AttributionError
from calibrated_agent_evaluation.attribution.types import (
    Attribution,
    AttributionSuggestion,
    CrewDefinition,
)
from calibrated_agent_evaluation.review.errors import RatificationError
from calibrated_agent_evaluation.schema.types import Span, Trace


def validate_walkback(
    trace: Trace,
    crew: CrewDefinition,
    *,
    culprit_spans: Sequence[str],
    manifestation_span: str | None,
    handoff_implicated: str | None = None,
    span_defect: bool = False,
) -> Attribution:
    """Validate a recorded counterfactual walk-back against the real span tree.

    ``culprit_spans`` must be non-empty and every id must exist in ``trace``; ``manifestation_span``
    (if given) must exist; each culprit must be at or before the manifestation in causal order; and
    ``handoff_implicated`` (if given) must be a seam declared in ``crew.handoffs``. More than one
    culprit yields ``status="disputed"`` -- the culprits are kept, never collapsed to one.

    Raises :class:`AttributionError` (recording no attribution) when a named span is absent, a
    culprit lies strictly after the manifestation, or the handoff is undeclared.
    """
    culprits = tuple(culprit_spans)
    if not culprits:
        raise AttributionError("A walk-back must record at least one culprit span; got none.")

    by_id = _span_index(trace.spans)
    agent_order = _agent_order(trace, crew)
    for span_id in culprits:
        _require_span(span_id, by_id, trace, role="culprit")
    if manifestation_span is not None:
        _require_span(manifestation_span, by_id, trace, role="manifestation")
        for span_id in culprits:
            _require_causal_order(span_id, manifestation_span, by_id, agent_order)

    if handoff_implicated is not None and handoff_implicated not in crew.handoffs:
        raise AttributionError(
            f"handoff_implicated {handoff_implicated!r} is not a declared seam in the crew "
            f"handoff axis {list(crew.handoffs)}."
        )

    status = "disputed" if len(culprits) > 1 else "single"
    return Attribution(
        trace_id=trace.trace_id,
        culprit_spans=culprits,
        manifestation_span=manifestation_span,
        handoff_implicated=handoff_implicated,
        status=status,
        span_defect=span_defect,
    )


def ratify_suggestion(
    suggestion: AttributionSuggestion,
    *,
    ratified_by: str | None,
) -> AttributionSuggestion:
    """Return ``suggestion`` iff a human ratified it; otherwise refuse.

    An LLM-proposed attribution may draft the culprit, but it is not a Label until a human accepts
    or overrides it. Raises :class:`RatificationError` when ``ratified_by`` is empty -- an
    unratified LLM proposal never reaches the gold path.
    """
    if not ratified_by:
        raise RatificationError(
            "This attribution was proposed by an LLM and has not been ratified by a human. "
            "A human must accept or override the suggested culprit before it is recorded."
        )
    return suggestion


def _span_index(spans: Iterable[Span]) -> dict[str, Span]:
    return {s.span_id: s for s in spans}


def _require_span(span_id: str, by_id: dict[str, Span], trace: Trace, *, role: str) -> None:
    if span_id not in by_id:
        raise AttributionError(
            f"{role} span {span_id!r} is not present in trace {trace.trace_id!r} "
            f"(known spans: {sorted(by_id)})."
        )


def _agent_order(trace: Trace, crew: CrewDefinition) -> dict[str, int]:
    """Map each agent to its rank in the crew's declared valid ordering.

    Causal order for sibling crew spans is the *pipeline* order the labeler reasons about, which the
    crew Define artifact declares as ``ExpectedProcess.expected_agents`` (supervisor -> intake ->
    ... -> payout) -- NOT the normalizer's ``seq``, which is span emission order and does not encode
    a flat sibling pipeline (the normalizer assigns ``seq`` per emission). The rank comes from the
    scenario matching ``trace.taxonomy_cell["scenario"]`` if declared, else from the union of every
    scenario's ordering, so a single expected route needs no scenario lookup.
    """
    scenario = trace.taxonomy_cell.get("scenario", "")
    proc = crew.scenarios.get(scenario)
    if proc is not None:
        return {agent: i for i, agent in enumerate(proc.expected_agents)}
    order: dict[str, int] = {}
    for expected in crew.scenarios.values():
        for i, agent in enumerate(expected.expected_agents):
            order.setdefault(agent, i)
    return order


def _require_causal_order(
    culprit: str, manifestation: str, by_id: dict[str, Span], agent_order: dict[str, int]
) -> None:
    """Refuse a culprit that lies strictly after the manifestation in causal order.

    Causal order is the crew's declared pipeline ordering (``agent_order``); an ancestor of the
    manifestation is always at-or-before it (a parent runs before its child) regardless of ordering.
    When either agent is absent from the declared ordering, fall back to the ``seq`` timeline so a
    trace whose crew route is undeclared is still guarded.
    """
    if culprit == manifestation:
        return
    if _is_ancestor(culprit, manifestation, by_id):
        return
    culprit_agent = by_id[culprit].agent
    manifestation_agent = by_id[manifestation].agent
    culprit_rank = agent_order.get(culprit_agent)
    manifestation_rank = agent_order.get(manifestation_agent)
    if culprit_rank is not None and manifestation_rank is not None:
        if culprit_rank > manifestation_rank:
            raise AttributionError(
                f"culprit span {culprit!r} (agent {culprit_agent!r}) runs strictly after the "
                f"manifestation span {manifestation!r} (agent {manifestation_agent!r}) in the "
                f"crew's declared pipeline order; a failure cannot originate after it has already "
                f"surfaced."
            )
        return
    culprit_seq = by_id[culprit].seq
    manifestation_seq = by_id[manifestation].seq
    if culprit_seq > manifestation_seq:
        raise AttributionError(
            f"culprit span {culprit!r} (seq={culprit_seq}) runs strictly after the manifestation "
            f"span {manifestation!r} (seq={manifestation_seq}); a failure cannot originate after "
            f"it has already surfaced."
        )


def _is_ancestor(ancestor: str, descendant: str, by_id: dict[str, Span]) -> bool:
    """Return whether ``ancestor`` is on the parent chain above ``descendant``."""
    seen: set[str] = set()
    current = by_id[descendant].parent
    while current is not None and current not in seen:
        if current == ancestor:
            return True
        seen.add(current)
        parent_span = by_id.get(current)
        current = parent_span.parent if parent_span is not None else None
    return False
