# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Holm guard-slice panel -- reject any collateral regression.

A fix that improves the target while quietly breaking something else is not a fix. The guard panel
watches a set of non-targeted slices for regression and rejects the change if *any* of them
regresses -- with the per-slice p-values corrected for multiplicity so watching many slices does not
hand back a pile of false alarms:

  * :func:`guard_panel` -- corrects the per-slice regression p-values with **Holm** at a
    family-wise alpha of 0.05 via the :func:`~calibrated_agent_evaluation.run.apply_multiplicity`
    wrapper (which reuses ``multiplicity`` -- no correction is re-implemented), states ``m``, and
    names the slices whose regression survived correction. One survivor rejects the fix, so the
    error rate that matters is the chance of any false rejection: Holm bounds it at 0.05 under any
    dependence between slices (they share traces) and whichever slices truly regressed. An FDR
    method bounds only the expected share of false rejections among the rejections, which is the
    wrong guarantee for a hard reject.
  * :func:`crew_panel` -- for a crew improvement, derives a guard slice per
    *other* agent (from ``CrewDefinition.agents``) and per *declared handoff* (from
    ``CrewDefinition.handoffs``, matched against each label's ``handoff_implicated``), and folds in
    :func:`~calibrated_agent_evaluation.attribution.relocation_check` so the workflow-outcome-rate
    gate + the failure-relocation flag are authoritative. The attribution layer exposes per-*agent*
    counts and a single-``touched_agent`` relocation primitive but no per-*handoff* slice, so the
    per-handoff aggregation is built here.
"""

from __future__ import annotations

from collections.abc import Sequence

from calibrated_agent_evaluation.attribution import CrewDefinition, CrewOutcomes, relocation_check
from calibrated_agent_evaluation.calibrate.errors import StatisticsInputError
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.improve.errors import ImproveError
from calibrated_agent_evaluation.improve.report import GuardPanelReport, GuardSlice
from calibrated_agent_evaluation.run import apply_multiplicity

# The family-wise error rate of the guard panel. Any surviving slice rejects the fix, so the panel
# is a hard gate and is held to the project's hard-gate alpha (amended 2026-10-14).
GUARD_FWER_ALPHA = 0.05


def guard_panel(
    slices: Sequence[GuardSlice], *, alpha: float = GUARD_FWER_ALPHA
) -> GuardPanelReport:
    """Correct the guard-slice regression p-values with Holm and name any survivor.

    ``slices`` each carry a ``regression_p`` (the p-value that the slice worsened). The panel
    corrects them with ``holm`` at family-wise level ``alpha`` (via
    :func:`calibrated_agent_evaluation.run.apply_multiplicity` -> ``multiplicity``;
    ``fdr_bh``/unknown methods are refused by the primitive as :class:`StatisticsInputError`),
    states ``m`` = the slice count, and lists in
    ``regressed`` the slices whose regression survived correction. Raises :class:`ImproveError` on
    an empty panel -- a declared-but-empty guard panel is a caller bug, not a vacuous "no
    regression".
    """
    if not slices:
        raise ImproveError(
            "guard_panel: the guard-slice panel is empty; refusing to return a vacuous "
            "'no regression' for a declared-but-empty panel"
        )
    result = apply_multiplicity(
        [s.regression_p for s in slices], family="improve_guard", method="holm", alpha=alpha
    )
    regressed = tuple(s.name for s, rej in zip(slices, result.reject, strict=True) if rej)
    return GuardPanelReport(
        slices=tuple(slices),
        method="holm",
        m=result.m,
        regressed=regressed,
        relocation=None,
    )


def crew_panel(
    crew: CrewDefinition,
    before: CrewOutcomes,
    after: CrewOutcomes,
    *,
    touched_agent: str,
    slices: Sequence[GuardSlice],
) -> GuardPanelReport:
    """Build the crew guard panel: agent+handoff slices + the relocation fold-in.

    ``slices`` are the caller-derived per-agent and per-handoff guard slices (each ``kind="agent"``
    or ``"handoff"``, its ``name`` an agent in ``crew.agents`` or a handoff in ``crew.handoffs`` --
    see :func:`build_crew_slices`). This runs the Holm panel over them AND folds in
    :func:`relocation_check` over ``before``/``after`` at ``touched_agent`` so the
    workflow-outcome-rate verdict + the failure-relocation flag are attached to the report. Refuses
    a ``touched_agent`` absent from ``crew.agents``.
    """
    if touched_agent not in crew.agents:
        raise ImproveError(
            f"crew_panel: touched_agent {touched_agent!r} is not a declared crew agent "
            f"{crew.agents}; cannot gate a relocation check on an unknown agent"
        )
    base = guard_panel(slices)
    relocation = relocation_check(before, after, touched_agent=touched_agent)
    return GuardPanelReport(
        slices=base.slices,
        method=base.method,
        m=base.m,
        regressed=base.regressed,
        relocation=relocation,
    )


def build_crew_slices(
    crew: CrewDefinition,
    *,
    touched_agent: str,
    agent_regression_p: dict[str, float],
    agent_rates: dict[str, tuple[Interval, Interval]],
    handoff_regression_p: dict[str, float],
    handoff_rates: dict[str, tuple[Interval, Interval]],
) -> tuple[GuardSlice, ...]:
    """Derive the per-other-agent + per-declared-handoff guard slices for a crew.

    Includes a slice for every crew agent EXCEPT ``touched_agent`` (the guard panel watches the
    agents the fix did NOT target), and a slice for every declared handoff in ``crew.handoffs``.
    Each slice carries its before/after :class:`Interval` rates and the p-value that it regressed. A
    ``handoff_regression_p`` key naming a seam not in ``crew.handoffs``, or an
    ``agent_regression_p`` key naming an undeclared agent, is refused (the crew seam guard).
    """
    for agent in agent_regression_p:
        if agent not in crew.agents:
            raise ImproveError(
                f"build_crew_slices: agent {agent!r} is not a declared crew agent {crew.agents} "
            )
    for seam in handoff_regression_p:
        if seam not in crew.handoffs:
            raise ImproveError(
                f"build_crew_slices: handoff {seam!r} is not a declared crew handoff "
                f"{crew.handoffs}; refusing to guard an undeclared seam"
            )
    # Completeness (the reverse direction): every non-touched agent + every declared handoff must
    # be supplied in BOTH its regression-p and its rates map. A missing key would otherwise surface
    # as a raw KeyError off this public surface -- refuse with a typed error instead (no silent
    # failure; the caller cannot catch a family it can't name).
    expected_agents = [a for a in crew.agents if a != touched_agent]
    for agent in expected_agents:
        missing = [
            n
            for n, d in (("agent_regression_p", agent_regression_p), ("agent_rates", agent_rates))
            if agent not in d
        ]
        if missing:
            raise ImproveError(
                f"build_crew_slices: guarded agent {agent!r} is missing from {missing} "
                "(every non-touched crew agent needs a regression p-value and before/after rates)"
            )
    for seam in crew.handoffs:
        missing = [
            n
            for n, d in (
                ("handoff_regression_p", handoff_regression_p),
                ("handoff_rates", handoff_rates),
            )
            if seam not in d
        ]
        if missing:
            raise ImproveError(
                f"build_crew_slices: declared handoff {seam!r} is missing from {missing} "
                "(every declared handoff needs a regression p-value and before/after rates)"
            )
    out: list[GuardSlice] = []
    for agent in crew.agents:
        if agent == touched_agent:
            continue
        rate_before, rate_after = agent_rates[agent]
        out.append(
            GuardSlice(
                name=agent,
                kind="agent",
                regression_p=agent_regression_p[agent],
                rate_before=rate_before,
                rate_after=rate_after,
            )
        )
    for seam in crew.handoffs:
        rate_before, rate_after = handoff_rates[seam]
        out.append(
            GuardSlice(
                name=seam,
                kind="handoff",
                regression_p=handoff_regression_p[seam],
                rate_before=rate_before,
                rate_after=rate_after,
            )
        )
    return tuple(out)


# Re-export so the guard/orchestrator can surface a forbidden-method refusal by its shared type.
__all__ = ["StatisticsInputError", "build_crew_slices", "crew_panel", "guard_panel"]
