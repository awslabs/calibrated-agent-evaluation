# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The composed improve guard -- evaluate_improvement.

Orchestrates the four guards into one :class:`ImproveVerdict` + a persisted :class:`ImproveRecord`:

  1. :func:`~calibrated_agent_evaluation.improve.require_prereg` -- the headline metric must be
     pre-registered, un-mutated, consistent with every entry an earlier evaluation recorded, and
     frozen before the after run it gates.
  2. :func:`~calibrated_agent_evaluation.improve.assert_held_out` -- the confirmation slice must be
     disjoint from the development set.
  3. :func:`~calibrated_agent_evaluation.improve.require_adjudication` -- judge-labeled discordant
     pairs must carry a human verdict on the target, which
     :func:`~calibrated_agent_evaluation.improve.adjudicate_pairs` puts into the pair before the
     test.
  4. :func:`~calibrated_agent_evaluation.improve.paired_mcnemar` -- the paired significance.
  5. :func:`~calibrated_agent_evaluation.improve.guard_panel` /
     :func:`~calibrated_agent_evaluation.improve.crew_panel` -- the Holm guard panel, crew-aware.

A change is ``rejected`` when the paired test does not show a significant improvement, OR the
paired effect interval does not clear the pre-registered ``success_threshold`` (a minimum absolute
reduction in fail rate: ``effect.hi <= -success_threshold``), OR any guard slice regressed, OR a
crew improvement relocated its failure; ``accepted`` only when all guards pass. Every reason is
recorded on the verdict + record. Pure -- no vendor SDK, no I/O beyond the optional record dump the
caller drives.
"""

from __future__ import annotations

from calibrated_agent_evaluation._provenance import capture_provenance
from calibrated_agent_evaluation.improve.confirm import assert_held_out
from calibrated_agent_evaluation.improve.errors import ImproveError
from calibrated_agent_evaluation.improve.paired import (
    adjudicate_pairs,
    discordant_pairs,
    paired_mcnemar,
    require_adjudication,
)
from calibrated_agent_evaluation.improve.panel import GUARD_FWER_ALPHA, crew_panel, guard_panel
from calibrated_agent_evaluation.improve.prereg import (
    require_frozen_before,
    require_prereg,
    verify_anchors,
)
from calibrated_agent_evaluation.improve.report import (
    GuardPanelReport,
    ImproveRecord,
    ImproveRequest,
    ImproveVerdict,
    McNemarResult,
    PreRegEntry,
)

# A paired result is "significant" below this two-sided alpha (the gate level). Requiring the
# effect to point the right way as well makes the decision a one-sided test at _ALPHA / 2.
_ALPHA = 0.05
ALPHA_ONE_SIDED = _ALPHA / 2


def _improved(paired: McNemarResult) -> bool:
    """True iff the paired test shows a *significant improvement* (fewer failures after).

    Requires a defined test (``p_value`` is not None -- an undefined zero-discordant table is never
    an improvement), a significant p-value, and an effect whose point is negative (the
    fail-rate fell: after - before < 0). A two-sided p below 0.05 plus the sign check is a
    one-sided test at 0.025 (:data:`ALPHA_ONE_SIDED`), recorded on the improve record.
    """
    if paired.p_value is None or paired.effect.point is None:
        return False
    return paired.p_value < _ALPHA and paired.effect.point < 0.0


def _clears_threshold(paired: McNemarResult, success_threshold: float) -> bool:
    """True iff the whole effect interval shows a fail-rate reduction of at least the threshold.

    ``success_threshold`` is the pre-registered minimum absolute reduction in fail rate, and the
    effect is after minus before, so a reduction is negative: the upper bound of the 95% interval
    must satisfy ``effect.hi <= -success_threshold``. Reading the bound, not the point, keeps a fix
    whose estimate looks big enough but whose interval reaches a smaller reduction from passing.
    """
    return paired.effect.hi <= -success_threshold


def _success_threshold(request: ImproveRequest, prereg: PreRegEntry) -> float:
    """Return the ledger's ``success_threshold``; refuse one outside (0, 1) or a differing caller.

    The ledger entry is the only source. A caller that also states a threshold must state the same
    number, so a threshold chosen after seeing the result cannot stand in for the committed one.
    """
    threshold = prereg.success_threshold
    if not 0.0 < threshold < 1.0:
        raise ImproveError(
            f"evaluate_improvement: pre-registered success_threshold {threshold!r} for "
            f"{prereg.target_metric!r} is not in (0, 1); it is a minimum absolute reduction in "
            "fail rate"
        )
    if request.success_threshold is not None and request.success_threshold != threshold:
        raise ImproveError(
            f"evaluate_improvement: caller success_threshold {request.success_threshold!r} differs "
            f"from the pre-registered {threshold!r} for {prereg.target_metric!r}; the gate reads "
            "only the threshold frozen before the fix"
        )
    return threshold


def _verdict(
    paired: McNemarResult, panel: GuardPanelReport, success_threshold: float
) -> tuple[str, tuple[str, ...]]:
    """Compose the accept/reject verdict + its reasons from the paired test and the guard panel."""
    reasons: list[str] = []
    if not _improved(paired):
        if paired.method == "undefined":
            reasons.append("paired test undefined (no discordant pairs) -- no improvement shown")
        else:
            reasons.append("paired before/after not a significant improvement")
    if not _clears_threshold(paired, success_threshold):
        reasons.append(
            f"paired effect upper bound {paired.effect.hi:+.4f} does not clear the pre-registered "
            f"reduction in fail rate of {success_threshold} (needs <= {-success_threshold})"
        )
    if panel.regressed:
        reasons.append(f"guard slices regressed: {', '.join(panel.regressed)}")
    if panel.relocation is not None and panel.relocation.verdict == "relocation":
        reasons.append(
            "failure relocation: the touched agent's failures fell but the workflow outcome rate "
            f"did not improve (the failures moved from {panel.relocation.relocated_from!r})"
        )
    if panel.relocation is not None and panel.relocation.verdict == "regressed":
        reasons.append("workflow outcome rate regressed")
    verdict = "rejected" if reasons else "accepted"
    return verdict, tuple(reasons)


def evaluate_improvement(request: ImproveRequest) -> ImproveVerdict:
    """Evaluate one proposed improvement against all four causal guards.

    Runs the guards in order (pre-registration -> held-out -> adjudication -> McNemar -> panel) and
    returns the composed :class:`ImproveVerdict`. Raises before returning a verdict when a guard
    refuses: :class:`~calibrated_agent_evaluation.improve.ImproveError` (unregistered/mutated
    prereg, a ledger that disagrees with the entries in ``anchors``, a pre-registration frozen after
    a time in ``after_run_at``, a trace paired twice, a ``success_threshold`` outside (0, 1) or
    differing from the caller's, empty input),
    :class:`~calibrated_agent_evaluation.improve.ConfirmationContaminationError`,
    :class:`~calibrated_agent_evaluation.improve.AdjudicationRequiredError`, or
    :class:`~calibrated_agent_evaluation.calibrate.errors.StatisticsInputError` (a forbidden
    multiplicity method). For a crew request
    (``crew``/``crew_before``/``crew_after``/``touched_agent`` set), the panel is the crew panel
    with the relocation fold-in.

    Args:
        request: The pre-registration, the paired before/after observations and the guard slices.

    Returns:
        Whether the improvement is accepted, with each guard's result.
    """
    verify_anchors(request.ledger, request.anchors)
    prereg = require_prereg(request.ledger, request.target_metric)
    require_frozen_before(prereg, request.after_run_started_at)
    success_threshold = _success_threshold(request, prereg)
    assert_held_out(request.confirmation_ids, request.development_ids)

    judge_produced = any(p.labels_are_judge_produced for p in request.pairs)
    discordant = discordant_pairs(request.pairs)
    require_adjudication(
        discordant,
        request.adjudications,
        labels_are_judge_produced=judge_produced,
        target_metric=request.target_metric,
    )
    pairs = adjudicate_pairs(
        request.pairs, request.adjudications, target_metric=request.target_metric
    )

    paired = paired_mcnemar(pairs)

    if request.crew is not None:
        if (
            request.crew_before is None
            or request.crew_after is None
            or request.touched_agent is None
        ):
            raise _crew_incomplete(prereg)
        panel = crew_panel(
            request.crew,
            request.crew_before,
            request.crew_after,
            touched_agent=request.touched_agent,
            slices=request.slices,
        )
    else:
        panel = guard_panel(request.slices)

    verdict, reasons = _verdict(paired, panel, success_threshold)
    return ImproveVerdict(
        verdict=verdict,  # type: ignore[arg-type]  # "accepted"|"rejected" from _verdict
        prereg=prereg,
        paired=paired,
        panel=panel,
        reasons=reasons,
    )


def _crew_incomplete(prereg: PreRegEntry) -> Exception:
    return ImproveError(
        f"evaluate_improvement: crew improvement for {prereg.target_metric!r} requires "
        "crew_before, crew_after, and touched_agent to run the relocation check"
    )


def record_of(
    verdict: ImproveVerdict,
    *,
    improve_id: str,
    confirmation_slice_id: str,
    development_set_id: str,
    inputs: dict[str, str],
) -> ImproveRecord:
    """Assemble the persisted :class:`ImproveRecord` from a verdict.

    A thin constructor so the caller (or the CLI) can persist the audit artifact via
    :func:`calibrated_agent_evaluation.improve.dump_improve_records`. ``inputs`` must be non-empty
    (the record refuses otherwise).
    """
    thresholds = {
        "alpha": _ALPHA,
        "alpha_one_sided": ALPHA_ONE_SIDED,
        # The key names the method: the guard panel is Holm at this family-wise alpha.
        "guard_holm_alpha": GUARD_FWER_ALPHA,
        "success_threshold": verdict.prereg.success_threshold,
    }
    return ImproveRecord(
        improve_id=improve_id,
        verdict=verdict.verdict,
        prereg=verdict.prereg,
        paired=verdict.paired,
        panel=verdict.panel,
        confirmation_slice_id=confirmation_slice_id,
        development_set_id=development_set_id,
        inputs=inputs,
        reasons=verdict.reasons,
        # McNemar's exact test and the Holm panel draw no random numbers: no seed to record.
        provenance=capture_provenance(thresholds=thresholds),
    )
