# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-axis inter-coder attribution agreement (two-axis dual-coding).

The ~20% dual-coding check yields agreement on **two** axes for a crew: the outcome (PASS/FAIL) and
the culprit (which agent). A high outcome-axis agreement with a *low* culprit-axis agreement is the
most useful multiagent signal -- a fuzzy agent/handoff responsibility boundary (a real product bug),
not labeler noise to suppress. This reuses
:func:`calibrated_agent_evaluation.stats.kappa.cohen_kappa_interval` over the subset both coders
labeled; when there is no dual-coded overlap, no agreement is fabricated -- the function returns
``None`` (capability-absent).
"""

from __future__ import annotations

from collections.abc import Sequence

from calibrated_agent_evaluation.attribution.errors import AttributionError
from calibrated_agent_evaluation.attribution.types import AgreementReport
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.schema.types import Label
from calibrated_agent_evaluation.stats.agreement import UndefinedAgreementError
from calibrated_agent_evaluation.stats.kappa import cohen_kappa_interval


def _kappa(a: Sequence[object], b: Sequence[object]) -> tuple[Interval | None, str | None]:
    try:
        return cohen_kappa_interval(a, b), None
    except UndefinedAgreementError as exc:
        return None, str(exc)


def _by_trace(labels: Sequence[Label], coder: str) -> dict[str, Label]:
    """One coder's trace-level labels keyed by trace, refusing span-level or repeated labels."""
    span_level = sorted({lb.trace_id for lb in labels if lb.span_id is not None})
    if span_level:
        raise AttributionError(
            f"coder {coder} has span-level label(s) on trace(s) {span_level[:5]}; attribution "
            "agreement compares trace-level labels (outcome and culprit per trace). Pass only the "
            "labels with span_id=None"
        )
    by_trace: dict[str, Label] = {}
    repeated: list[str] = []
    for lb in labels:
        if lb.trace_id in by_trace:
            repeated.append(lb.trace_id)
        by_trace[lb.trace_id] = lb
    if repeated:
        raise AttributionError(
            f"coder {coder} labeled trace(s) {sorted(set(repeated))[:5]} more than once; keep one "
            "label per trace per coder so each trace is counted once"
        )
    return by_trace


def attribution_agreement(
    coder_a: Sequence[Label],
    coder_b: Sequence[Label],
) -> AgreementReport | None:
    """Return per-axis (outcome, culprit) inter-coder agreement, or ``None`` if none dual-coded.

    Labels are trace-level and matched on ``trace_id``. The outcome axis compares ``outcome`` over
    every trace both coders labeled. The culprit axis compares ``culprit_span`` over the traces
    both coders called FAIL: a PASS has no culprit, and counting the PASS traces where both
    culprits are ``None`` would carry the outcome agreement into the culprit axis (80 shared PASS
    traces and 20 FAILs with culprits picked at random read as culprit kappa 0.54 instead of about
    0). Returns:

      * an :class:`AgreementReport` (``outcome_kappa`` / ``culprit_kappa`` / ``n_dual_coded`` /
        ``n_culprit_coded``) when at least one trace was co-labeled, each kappa with its 95%
        interval or ``None`` plus a reason when it is 0/0 or has no traces on that axis;
      * ``None`` when the coders share no trace -- there is nothing to agree on, so no figure is
        fabricated from a single coder (the capability-absent path).

    Raises:
        AttributionError: A coder passed a span-level label or labeled a trace twice.
    """
    a_by_trace = _by_trace(coder_a, "A")
    b_by_trace = _by_trace(coder_b, "B")
    shared = sorted(a_by_trace.keys() & b_by_trace.keys())
    if not shared:
        return None

    outcome_a = [a_by_trace[t].outcome for t in shared]
    outcome_b = [b_by_trace[t].outcome for t in shared]
    co_fail = [t for t in shared if a_by_trace[t].outcome == b_by_trace[t].outcome == "FAIL"]

    outcome, outcome_reason = _kappa(outcome_a, outcome_b)
    if co_fail:
        culprit, culprit_reason = _kappa(
            [a_by_trace[t].culprit_span for t in co_fail],
            [b_by_trace[t].culprit_span for t in co_fail],
        )
    else:
        culprit, culprit_reason = (
            None,
            "no trace was called FAIL by both coders, so there is no culprit to agree on",
        )
    return AgreementReport(
        outcome_kappa=outcome,
        culprit_kappa=culprit,
        n_dual_coded=len(shared),
        outcome_undefined_reason=outcome_reason,
        culprit_undefined_reason=culprit_reason,
        n_culprit_coded=len(co_fail),
    )
