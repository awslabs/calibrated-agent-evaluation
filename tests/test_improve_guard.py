# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the composed improve guard.

Behavioural: evaluate_improvement composes the four guards into an ImproveVerdict -- accepting a
genuine, guard-clean improvement and rejecting one that regresses a guard slice, relocates a crew
failure, or shows no significant paired improvement. Every effect/rate is carried as an Interval
, and the orchestrator raises the guard refusals (unregistered prereg, contamination,
unadjudicated judge labels) before returning a verdict. The fix must also clear the
pre-registered minimum reduction in fail rate on the upper bound of its effect interval,
read from the ledger and never from a differing caller value.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from _helpers import TEST_INTERVAL_KIND
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.attribution import CrewDefinition, CrewOutcomes
from calibrated_agent_evaluation.attribution.types import ExpectedProcess
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.improve import (
    AdjudicationRequiredError,
    ConfirmationContaminationError,
    GuardSlice,
    ImproveError,
    ImproveRequest,
    LedgerIntegrityError,
    PairedObs,
    PreRegEntry,
    PreRegLedger,
    evaluate_improvement,
    record_of,
    register_prereg,
)
from calibrated_agent_evaluation.schema import Label


def _iv(point: float) -> Interval:
    return Interval(
        point=point, lo=point - 0.05, hi=point + 0.05, refusal_reason=None, **TEST_INTERVAL_KIND
    )


def _pairs(
    before_fail: int, after_fail: int, n: int, *, judge: bool = False
) -> tuple[PairedObs, ...]:
    return tuple(
        PairedObs(
            f"t{i}",
            before_fail=(i < before_fail),
            after_fail=(i < after_fail),
            labels_are_judge_produced=judge,
        )
        for i in range(n)
    )


# A ten-point minimum reduction in fail rate: 18/40 -> 4/40 (effect hi -0.186) clears it.
def _ledger(metric: str = "mode_a", *, threshold: float = 0.10, crew: bool = False) -> PreRegLedger:
    return register_prereg(
        PreRegLedger(),
        PreRegEntry(
            prereg_id="p1", target_metric=metric, success_threshold=threshold, is_crew=crew
        ),
    )


def _slice(name: str, p: float) -> GuardSlice:
    return GuardSlice(
        name=name, kind="other", regression_p=p, rate_before=_iv(0.1), rate_after=_iv(0.1)
    )


def test_accepts_a_clean_significant_improvement() -> None:
    """A significant paired improvement with no guard regression is accepted."""
    req = ImproveRequest(
        ledger=_ledger(),
        target_metric="mode_a",
        pairs=_pairs(before_fail=18, after_fail=4, n=40),  # big drop, many discordant
        confirmation_ids=tuple(f"t{i}" for i in range(40)),
        development_ids=("d1", "d2"),
        slices=(_slice("other", 0.8),),
    )
    verdict = evaluate_improvement(req)
    assert verdict.verdict == "accepted"
    assert verdict.paired.effect.point is not None  # an Interval, not a bare point
    assert verdict.reasons == ()


def test_rejects_when_a_guard_slice_regresses() -> None:
    """A fix that regresses a guard slice (surviving Holm) is rejected, naming it."""
    req = ImproveRequest(
        ledger=_ledger(),
        target_metric="mode_a",
        pairs=_pairs(before_fail=18, after_fail=4, n=40),
        confirmation_ids=tuple(f"t{i}" for i in range(40)),
        development_ids=("d1",),
        slices=(_slice("collateral", 0.0005), _slice("ok", 0.9)),
    )
    verdict = evaluate_improvement(req)
    assert verdict.verdict == "rejected"
    assert "collateral" in " ".join(verdict.reasons)


def test_rejects_when_paired_shows_no_improvement() -> None:
    """No significant paired improvement (a wash) is rejected even with clean guards."""
    req = ImproveRequest(
        ledger=_ledger(),
        target_metric="mode_a",
        pairs=_pairs(before_fail=10, after_fail=9, n=40),  # a single-trace change, not significant
        confirmation_ids=tuple(f"t{i}" for i in range(40)),
        development_ids=("d1",),
        slices=(_slice("other", 0.8),),
    )
    verdict = evaluate_improvement(req)
    assert verdict.verdict == "rejected"


def test_unregistered_metric_raises_before_verdict() -> None:
    """Evaluating a metric absent from the ledger raises rather than returning a verdict."""
    req = ImproveRequest(
        ledger=_ledger("mode_a"),
        target_metric="mode_never",
        pairs=_pairs(before_fail=8, after_fail=3, n=20),
        confirmation_ids=tuple(f"t{i}" for i in range(20)),
        development_ids=("d1",),
        slices=(_slice("other", 0.8),),
    )
    with pytest.raises(ImproveError):
        evaluate_improvement(req)


def test_contaminated_slice_raises_before_verdict() -> None:
    """A confirmation slice overlapping the dev set raises before any measurement."""
    req = ImproveRequest(
        ledger=_ledger(),
        target_metric="mode_a",
        pairs=_pairs(before_fail=8, after_fail=3, n=20),
        confirmation_ids=("t0", "t1", "shared"),
        development_ids=("shared",),
        slices=(_slice("other", 0.8),),
    )
    with pytest.raises(ConfirmationContaminationError):
        evaluate_improvement(req)


def test_unadjudicated_judge_labels_raise() -> None:
    """Judge-produced labels with unadjudicated discordant pairs raise before McNemar."""
    req = ImproveRequest(
        ledger=_ledger(),
        target_metric="mode_a",
        pairs=_pairs(before_fail=8, after_fail=3, n=20, judge=True),
        confirmation_ids=tuple(f"t{i}" for i in range(20)),
        development_ids=("d1",),
        slices=(_slice("other", 0.8),),
    )
    with pytest.raises(AdjudicationRequiredError):
        evaluate_improvement(req)


def _crew() -> CrewDefinition:
    return CrewDefinition(
        agents=("intake", "policy-checker"),
        handoffs=("intake->policy-checker",),
        scenarios={
            "s1": ExpectedProcess(
                scenario="s1",
                expected_agents=("intake", "policy-checker"),
                terminal_decision="approve",
                payout_allowed=True,
                escalation_required=False,
            )
        },
        rules=(),
    )


def test_rejects_crew_failure_relocation() -> None:
    """A crew fix where the touched agent improved but the workflow rate stayed flat is
    rejected as failure relocation (via relocation_check)."""
    before = CrewOutcomes(
        workflow_failures=60, workflow_total=200, per_agent_failures={"intake": 60}
    )
    after = CrewOutcomes(
        workflow_failures=60, workflow_total=200, per_agent_failures={"intake": 20}
    )
    req = ImproveRequest(
        ledger=_ledger("workflow_outcome_rate", crew=True),
        target_metric="workflow_outcome_rate",
        pairs=_pairs(before_fail=18, after_fail=4, n=40),  # touched agent looks improved
        confirmation_ids=tuple(f"t{i}" for i in range(40)),
        development_ids=("d1",),
        slices=(_slice("policy-checker", 0.8),),
        crew=_crew(),
        crew_before=before,
        crew_after=after,
        touched_agent="intake",
    )
    verdict = evaluate_improvement(req)
    assert verdict.verdict == "rejected"
    assert any("relocation" in r for r in verdict.reasons)


def test_accepts_genuine_crew_improvement() -> None:
    """A crew fix where the workflow outcome rate genuinely improved and no guard regressed
    is accepted."""
    before = CrewOutcomes(workflow_failures=8, workflow_total=20, per_agent_failures={"intake": 8})
    after = CrewOutcomes(workflow_failures=2, workflow_total=20, per_agent_failures={"intake": 2})
    req = ImproveRequest(
        ledger=_ledger("workflow_outcome_rate", crew=True),
        target_metric="workflow_outcome_rate",
        pairs=_pairs(before_fail=18, after_fail=4, n=40),
        confirmation_ids=tuple(f"t{i}" for i in range(40)),
        development_ids=("d1",),
        slices=(_slice("policy-checker", 0.8),),
        crew=_crew(),
        crew_before=before,
        crew_after=after,
        touched_agent="intake",
    )
    verdict = evaluate_improvement(req)
    assert verdict.verdict == "accepted"
    assert verdict.panel.relocation is not None
    assert verdict.panel.relocation.verdict == "improved"


def test_record_of_carries_intervals_and_reasons() -> None:
    """The persisted record carries the verdict, the Interval effect, and reasons."""
    req = ImproveRequest(
        ledger=_ledger(),
        target_metric="mode_a",
        pairs=_pairs(before_fail=18, after_fail=4, n=40),
        confirmation_ids=tuple(f"t{i}" for i in range(40)),
        development_ids=("d1",),
        slices=(_slice("collateral", 0.0005),),
    )
    verdict = evaluate_improvement(req)
    record = record_of(
        verdict,
        improve_id="imp-1",
        confirmation_slice_id="conf-v1",
        development_set_id="dev-v1",
        inputs={"target_metric": "mode_a"},
    )
    assert record.verdict == verdict.verdict
    assert record.paired.effect.point is not None
    assert record.reasons == verdict.reasons
    # The sign check on a two-sided p makes the decision one-sided at 0.025; the record says so.
    assert record.provenance.thresholds["alpha"] == 0.05
    assert record.provenance.thresholds["alpha_one_sided"] == 0.025
    # The guard panel's method and family-wise alpha.
    assert record.provenance.thresholds["guard_holm_alpha"] == 0.05
    assert "guard_fdr_q" not in record.provenance.thresholds


def _threshold_request(
    before_fail: int, after_fail: int, n: int, threshold: float, **kw: object
) -> ImproveRequest:
    return ImproveRequest(
        ledger=_ledger(threshold=threshold),
        target_metric="mode_a",
        pairs=_pairs(before_fail=before_fail, after_fail=after_fail, n=n),
        confirmation_ids=tuple(f"t{i}" for i in range(n)),
        development_ids=("d1",),
        slices=(_slice("other", 0.8),),
        **kw,  # type: ignore[arg-type]
    )


def test_fix_whose_interval_clears_the_reduction_is_accepted() -> None:
    """18/40 -> 4/40 has Newcombe interval [-0.4964, -0.1862]; -0.1862 <= -0.15 passes."""
    verdict = evaluate_improvement(_threshold_request(18, 4, 40, 0.15))
    assert verdict.paired.effect.hi == pytest.approx(-0.186161, abs=1e-6)
    assert verdict.verdict == "accepted"
    assert verdict.reasons == ()


def test_upper_bound_just_short_of_the_reduction_is_rejected() -> None:
    """The same fix against a 0.19 reduction: -0.1862 > -0.19, so it is rejected."""
    verdict = evaluate_improvement(_threshold_request(18, 4, 40, 0.19))
    assert verdict.verdict == "rejected"
    assert len(verdict.reasons) == 1
    assert "does not clear the pre-registered reduction in fail rate of 0.19" in verdict.reasons[0]


def test_point_clears_but_upper_bound_does_not_is_rejected() -> None:
    """8/20 -> 3/20 is significant (mid-p 0.03125) with point -0.25, which
    clears a 0.10 reduction, but the interval's upper bound is -0.0292, so the fix is rejected."""
    verdict = evaluate_improvement(_threshold_request(8, 3, 20, 0.10))
    assert verdict.paired.p_value == pytest.approx(0.03125, abs=1e-12)
    assert verdict.paired.effect.point == pytest.approx(-0.25)
    assert verdict.paired.effect.hi == pytest.approx(-0.029154, abs=1e-6)
    assert verdict.verdict == "rejected"
    assert len(verdict.reasons) == 1
    assert "upper bound -0.0292" in verdict.reasons[0]


def test_fix_that_changes_nothing_is_rejected() -> None:
    """Identical before/after (10/40 both) has an undefined test and an effect
    interval [-0.0641, 0.0641] that shows no reduction, so both reasons are recorded."""
    verdict = evaluate_improvement(_threshold_request(10, 10, 40, 0.05))
    assert verdict.verdict == "rejected"
    assert verdict.paired.method == "undefined"
    assert len(verdict.reasons) == 2
    assert "does not clear" in verdict.reasons[1]


def test_caller_threshold_matching_the_ledger_proceeds() -> None:
    """A caller who restates the pre-registered threshold gets the normal verdict."""
    verdict = evaluate_improvement(_threshold_request(18, 4, 40, 0.10, success_threshold=0.10))
    assert verdict.verdict == "accepted"


def test_caller_threshold_differing_from_the_ledger_is_refused() -> None:
    """A caller threshold that differs from the frozen one is refused."""
    with pytest.raises(ImproveError, match="differs from the pre-registered 0.1"):
        evaluate_improvement(_threshold_request(8, 3, 20, 0.10, success_threshold=0.02))


def test_hand_built_ledger_with_an_out_of_range_threshold_is_refused() -> None:
    """An entry re-hashed around a threshold of 1.0, bypassing freeze, is still
    refused by the gate rather than rejecting every fix silently."""

    good = _ledger().entries[0]
    bad = replace(good, success_threshold=1.0)
    bad = replace(bad, content_hash=bad.compute_hash())
    request = replace(_threshold_request(18, 4, 40, 0.10), ledger=PreRegLedger(entries=(bad,)))
    with pytest.raises(ImproveError, match=r"not in \(0, 1\)"):
        evaluate_improvement(request)


def _adjudicated_request(outcome: str, mode: str | None) -> ImproveRequest:
    """100 judge pairs, 30 judged fixed; a human labels each of the 30 after-run outputs."""
    pairs = tuple(
        PairedObs(f"t{i}", before_fail=i < 30, after_fail=False, labels_are_judge_produced=True)
        for i in range(100)
    )
    adjudications = {
        f"t{i}": Label(
            trace_id=f"t{i}",
            span_id=None,
            kind="human",
            outcome=outcome,  # type: ignore[arg-type]
            critique="checked the after run",
            failure_mode=mode,
            rater_id="r",
        )
        for i in range(30)
    }
    return ImproveRequest(
        ledger=_ledger(threshold=0.05),
        target_metric="mode_a",
        pairs=pairs,
        confirmation_ids=tuple(p.trace_id for p in pairs),
        development_ids=("dev",),
        slices=(_slice("other", 0.9),),
        adjudications=adjudications,
    )


def test_humans_who_say_the_judged_fixes_still_fail_reject_the_fix() -> None:
    """The human verdict replaces the judge's, so flips the humans overturn are not counted."""
    verdict = evaluate_improvement(_adjudicated_request("FAIL", "mode_a"))
    assert verdict.verdict == "rejected"
    assert verdict.paired.n_discordant == 0
    assert verdict.paired.effect.point == 0.0


def test_humans_who_confirm_the_judged_fixes_accept_it() -> None:
    verdict = evaluate_improvement(_adjudicated_request("PASS", None))
    assert verdict.verdict == "accepted"
    assert verdict.paired.effect.point == pytest.approx(-0.30)


def test_a_human_label_about_another_failure_mode_does_not_adjudicate() -> None:
    with pytest.raises(AdjudicationRequiredError, match="another failure mode"):
        evaluate_improvement(_adjudicated_request("FAIL", "other_mode"))
    with pytest.raises(AdjudicationRequiredError, match="another failure mode"):
        evaluate_improvement(_adjudicated_request("PASS", "other_mode"))


def test_a_span_level_human_label_does_not_adjudicate() -> None:
    req = _adjudicated_request("PASS", None)
    span_level = {t: replace(lb, span_id="s-1") for t, lb in req.adjudications.items()}
    with pytest.raises(AdjudicationRequiredError, match="span-level"):
        evaluate_improvement(replace(req, adjudications=span_level))


def test_a_trace_paired_twice_is_refused() -> None:
    """Listing every pair twice would double each McNemar cell and shrink the p-value."""
    once = _pairs(before_fail=6, after_fail=0, n=20)
    req = ImproveRequest(
        ledger=_ledger(threshold=0.05),
        target_metric="mode_a",
        pairs=once + once,
        confirmation_ids=tuple(p.trace_id for p in once),
        development_ids=("dev",),
        slices=(_slice("other", 0.9),),
    )
    with pytest.raises(ImproveError, match="more than once"):
        evaluate_improvement(req)


def test_a_ledger_missing_an_entry_an_evaluation_recorded_is_refused() -> None:
    recorded = _ledger(threshold=0.30).entries[0]
    req = replace(_adjudicated_request("PASS", None), anchors=(recorded,))
    with pytest.raises(LedgerIntegrityError, match="edited, replaced or deleted"):
        evaluate_improvement(req)


def test_a_target_frozen_after_the_after_run_began_is_refused() -> None:
    req = _adjudicated_request("PASS", None)
    frozen = datetime.fromisoformat(req.ledger.entries[0].frozen_at)
    with pytest.raises(LedgerIntegrityError, match="after the after run began"):
        evaluate_improvement(replace(req, after_run_started_at=frozen - timedelta(hours=1)))
    later = evaluate_improvement(replace(req, after_run_started_at=frozen + timedelta(hours=1)))
    assert later.verdict == "accepted"
