# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The failure-relocation primitive.

When a crew is improved, the gate metric is the **workflow outcome rate**, not the touched agent's
rate. "Fix" intake so its drop-rate falls, but if policy-checker still won't apply the exclusion,
the failure has merely *relocated* -- the crew outcome rate is flat, which is not a fix. This is the
standalone primitive the improvement-guard panel later composes (which must include
the *other* agents and handoffs); this layer supplies only the workflow-outcome accounting + the
flag, not the McNemar/pre-registration machinery. Each verdict reads an interval on the
after-minus-before change in rate, never a raw point difference or a raw count.
"""

from __future__ import annotations

from calibrated_agent_evaluation.attribution.errors import AttributionError
from calibrated_agent_evaluation.attribution.types import CrewOutcomes, RelocationResult
from calibrated_agent_evaluation.calibrate.errors import StatisticsInputError
from calibrated_agent_evaluation.calibrate.record import METHOD_NEWCOMB_DIFF, Interval

# Two-sided level of the after-minus-before intervals the verdict reads.
RELOCATION_LEVEL = 0.95


def relocation_check(
    before: CrewOutcomes,
    after: CrewOutcomes,
    *,
    touched_agent: str,
) -> RelocationResult:
    """Classify a crew before/after improvement, gating on the workflow outcome rate.

    Both slices are read as rates over their own traces, so slices of different sizes compare
    fairly, and each comparison is the 95% Newcombe interval for the after-minus-before
    difference of two independent proportions. The verdict is:

      * ``regressed`` -- the workflow failure rate's interval lies above 0 (a relocation flag never
        masks this);
      * ``improved`` -- the workflow failure rate's interval lies below 0;
      * ``relocation`` -- the workflow rate's interval straddles 0 while the touched agent's
        failure rate's interval lies below 0: a per-agent gain with no shown workflow gain is
        failure relocation, not a fix;
      * ``flat`` -- neither interval excludes 0.

    ``relocated_from`` names the touched agent when (and only when) the verdict is ``relocation``.
    The point rates are reported beside the intervals and decide nothing.

    Raises :class:`AttributionError` when either slice holds zero traces. A rate over no traces
    is not a rate, and the ladder below reads two of them as equal -- so an empty after-slice used
    to come back ``flat``, or ``improved`` against any non-zero before-rate: a verdict about an
    improvement that was never measured. Same refusal ``run/gate.py:run_ci_gate`` makes over zero
    labels rather than reporting a vacuous pass. A failure count above its slice's trace count is
    refused too.
    """
    _refuse_empty_slice(before, "before")
    _refuse_empty_slice(after, "after")
    agent_before = before.per_agent_failures.get(touched_agent, 0)
    agent_after = after.per_agent_failures.get(touched_agent, 0)
    workflow = _delta(
        after.workflow_failures,
        after.workflow_total,
        before.workflow_failures,
        before.workflow_total,
        "workflow failures",
    )
    agent = _delta(
        agent_after, after.workflow_total, agent_before, before.workflow_total, touched_agent
    )

    relocated_from = None
    if workflow.lo > 0.0:
        verdict = "regressed"
    elif workflow.hi < 0.0:
        verdict = "improved"
    elif agent.hi < 0.0:
        verdict = "relocation"
        relocated_from = touched_agent
    else:
        verdict = "flat"

    return RelocationResult(
        workflow_outcome_rate_before=_rate(before.workflow_failures, before.workflow_total),
        workflow_outcome_rate_after=_rate(after.workflow_failures, after.workflow_total),
        per_agent_failures_before=dict(before.per_agent_failures),
        per_agent_failures_after=dict(after.per_agent_failures),
        verdict=verdict,
        relocated_from=relocated_from,
        workflow_rate_change=workflow,
        touched_agent_rate_change=agent,
    )


def _delta(
    count_after: int, total_after: int, count_before: int, total_before: int, what: str
) -> Interval:
    """The after-minus-before rate change with its 95% Newcombe interval."""
    from calibrated_agent_evaluation.stats.drift import delta_proportion_ci

    try:
        point, lo, hi = delta_proportion_ci(
            count_after, total_after, count_before, total_before, alpha=1.0 - RELOCATION_LEVEL
        )
    except StatisticsInputError as exc:
        raise AttributionError(
            f"relocation_check: the {what} counts ({count_before} of {total_before} before, "
            f"{count_after} of {total_after} after) are not counts over their slices; each "
            f"failure count must lie between 0 and its slice's trace count ({exc})"
        ) from exc
    return Interval(
        point=point,
        lo=lo,
        hi=hi,
        method=METHOD_NEWCOMB_DIFF,
        level=RELOCATION_LEVEL,
        sided="two-sided",
        n=total_after + total_before,
    )


def _refuse_empty_slice(outcomes: CrewOutcomes, side: str) -> None:
    """Refuse a slice with no traces, naming which side was empty."""
    if outcomes.workflow_total <= 0:
        raise AttributionError(
            f"relocation_check: the {side} slice holds {outcomes.workflow_total} workflow "
            f"traces, so its outcome rate is undefined; refusing to compare a rate that was never "
            f"measured (check the {side} slice's query -- a frame or taxonomy_cell filter matching "
            f"nothing produces exactly this)"
        )


def _rate(failures: int, total: int) -> float:
    """Return failures/total. ``total`` is guaranteed positive by :func:`_refuse_empty_slice`.

    The ``or 0.0`` guard here is now a divide-by-zero backstop only; it is unreachable through
    ``relocation_check``, and it was the bug: 0.0 is a *rate*, so reporting one for a slice with no
    traces made "no data" indistinguishable from "no failures".
    """
    return failures / total if total else 0.0
