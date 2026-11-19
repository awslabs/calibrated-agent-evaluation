# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the calibration gate ( engine): certify, corrected_rate, disciplines.

Covers the lower-bound gate, prevalence correction + its interval
, the J-guard and identifiability, the two-samples discipline
, the human-gold/few-shot/locked-test disciplines, version pinning
, the uncalibrated refusal, per-judge certification with no family, and
the no-bare-number output contract. Every test embeds its FR; tests assert behavior
(results, raised typed errors, recovered values), never mock calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from _helpers import TEST_STAMP, make_label, rate_ids
from calibrated_agent_evaluation.calibrate import InMemoryLockedTestLedger
from calibrated_agent_evaluation.calibrate.engine import (
    certifies,
    certify,
    corrected_rate,
)
from calibrated_agent_evaluation.calibrate.errors import (
    FewShotEvaluationError,
    LockedTestReuseError,
    NonIdentifiableRateError,
    SampleMisuseError,
    UncalibratedJudgeError,
    UnratifiedGoldError,
    WeakJudgeError,
)
from calibrated_agent_evaluation.calibrate.record import (
    METHOD_RG_JEFFREYS,
    METHOD_RG_JEFFREYS_CLUSTERED,
    Bound,
    CalibrationRecord,
    CalibrationSet,
    Interval,
    LabelPair,
)


@dataclass(frozen=True)
class _Judge:
    """A minimal judge identity (the duck type certify/certifies pin to)."""

    judge_id: str = "intake-extraction"
    model: str = "bedrock/claude-sonnet-4"
    prompt_version: str = "sha1"


def _pairs(n_pos: int, tp: int, n_neg: int, tn: int) -> tuple[LabelPair, ...]:
    """Build locked-test pairs realizing exactly the given confusion counts (FAIL = positive).

    ``tp`` of ``n_pos`` true-FAIL items are caught (verdict FAIL); ``tn`` of ``n_neg`` true-PASS
    items are correctly cleared (verdict PASS). Gold is human.
    """
    pairs: list[LabelPair] = []
    idx = 0
    for i in range(n_pos):
        idx += 1
        v = "FAIL" if i < tp else "PASS"
        pairs.append(
            LabelPair(
                gold=make_label(trace_id=f"p{idx}", kind="human", outcome="FAIL"),
                verdict=make_label(trace_id=f"p{idx}", kind="llm-judge", outcome=v),
            )
        )
    for i in range(n_neg):
        idx += 1
        v = "PASS" if i < tn else "FAIL"
        pairs.append(
            LabelPair(
                gold=make_label(trace_id=f"n{idx}", kind="human", outcome="PASS"),
                verdict=make_label(trace_id=f"n{idx}", kind="llm-judge", outcome=v),
            )
        )
    return tuple(pairs)


def _locked(pairs: tuple[LabelPair, ...], set_id: str = "locked-1") -> CalibrationSet:
    return CalibrationSet(pairs=pairs, split="locked_test", set_id=set_id)


# --- the lower-bound gate --------------------------------


def test_well_sized_high_accuracy_set_certifies() -> None:
    # 80/class at 90% accuracy: both lower bounds clear 0.80 → CERTIFIED.
    rec = certify(_Judge(), _locked(_pairs(80, 72, 80, 72)), ledger=InMemoryLockedTestLedger())
    assert rec.result == "CERTIFIED"
    assert rec.tpr.lb95 >= 0.80
    assert rec.tnr.lb95 >= 0.80


def test_high_point_but_short_lower_bound_is_refused() -> None:
    # ~20/class at 90% point accuracy: the POINT estimates are 0.90, but the 95% lower bound sits
    # below 0.80 (small n) → REFUSED. This is the core "gate on the bound, not the point" behavior.
    rec = certify(_Judge(), _locked(_pairs(20, 18, 20, 18)), ledger=InMemoryLockedTestLedger())
    assert rec.tpr.point == pytest.approx(0.90)
    assert rec.tpr.lb95 < 0.80
    assert rec.result == "REFUSED"


def test_point_exactly_at_threshold_does_not_certify() -> None:
    # Boundary: a TPR point of exactly 0.80 has a lower bound below 0.80 → not certified.
    rec = certify(
        _Judge(), _locked(_pairs(50, 40, 80, 76)), ledger=InMemoryLockedTestLedger()
    )  # TPR point = 0.80
    assert rec.tpr.point == pytest.approx(0.80)
    assert rec.tpr.lb95 < 0.80
    assert rec.result == "REFUSED"


# --- CERTIFIED only from the locked test --------------------------------


def test_dev_set_never_certifies() -> None:
    # A dev-set source (even with great numbers) never yields CERTIFIED — certification derives only
    # from the locked test. Refused as a dev-set evaluation.
    dev = CalibrationSet(pairs=_pairs(80, 72, 80, 72), split="dev", set_id="dev-1")
    with pytest.raises(FewShotEvaluationError):
        certify(_Judge(), dev, ledger=InMemoryLockedTestLedger())


# --- few-shot pool never evaluated ---------------------------------------


def test_few_shot_pool_is_refused() -> None:
    few = CalibrationSet(pairs=_pairs(10, 9, 10, 9), split="few_shot", set_id="fs-1")
    with pytest.raises(FewShotEvaluationError):
        certify(_Judge(), few, ledger=InMemoryLockedTestLedger())


# --- locked test single-use ----------------------------------------------


def test_locked_test_is_single_use() -> None:
    locked = _locked(_pairs(80, 72, 80, 72), set_id="locked-once")
    ledger = InMemoryLockedTestLedger()
    first = certify(_Judge(), locked, ledger=ledger)  # first use succeeds
    assert first.result == "CERTIFIED"
    with pytest.raises(LockedTestReuseError):  # second use refused
        certify(_Judge(), locked, ledger=ledger)


def test_a_ledger_seeded_with_a_consumed_id_refuses_it() -> None:
    """A ledger rebuilt from stored records refuses the set they consumed, in a new process."""
    locked = _locked(_pairs(80, 72, 80, 72), set_id="stored-earlier")
    with pytest.raises(LockedTestReuseError, match="stored-earlier"):
        certify(_Judge(), locked, ledger=InMemoryLockedTestLedger(["stored-earlier"]))


def test_a_set_that_fails_pairing_is_not_consumed() -> None:
    """Only an evaluated locked test is spent; one refused before evaluation can be fixed."""
    ledger = InMemoryLockedTestLedger()
    judge_gold = LabelPair(
        gold=make_label(trace_id="t1", kind="llm-judge", outcome="FAIL"),
        verdict=make_label(trace_id="t1", kind="llm-judge", outcome="FAIL"),
    )
    with pytest.raises(UnratifiedGoldError):
        certify(_Judge(), _locked((judge_gold,), set_id="judge-gold"), ledger=ledger)
    assert not ledger.is_consumed("judge-gold")


# --- corrected rate, not raw count ---------------------------------------


def _certified_9090() -> CalibrationRecord:
    """A CERTIFIED record for a 90/90 judge (80/class), ready for corrected_rate."""
    return certify(
        _Judge(),
        _locked(_pairs(80, 72, 80, 72), set_id="hunted-9090"),
        ledger=InMemoryLockedTestLedger(),
    )


def test_reports_corrected_rate_not_raw_fraction() -> None:
    rec = _certified_9090()
    # A separate production-representative sample measuring ~0.148 raw → corrected ~0.06.
    eval_verdicts = [1] * 148 + [0] * 852  # measured = 0.148
    out = corrected_rate(
        rec, eval_verdicts, rate_set_id="prod-sample-1", rate_trace_ids=rate_ids(eval_verdicts)
    )
    assert out.corrected_rate is not None
    # Rogan-Gladen is exact algebra here: (0.148 + 0.90 - 1) / (0.90 + 0.90 - 1) = 0.06.
    assert out.corrected_rate.require_point() == pytest.approx(0.06, abs=1e-9)
    # The raw fraction (0.148) is NOT what the record surfaces.
    assert out.corrected_rate.require_point() < 0.10


# --- the interval carries eval-sample variance; an eval-fixed one does not ------


def test_interval_wider_than_eval_fixed_and_shrinks_with_the_eval_sample() -> None:
    # The interval propagates eval-sample variance that the judgy-style (eval-rate-fixed)
    # bootstrap omits. Two consequences, both checked on one 90/90 judge at 80
    # per class: at a prevalence away from the [0, 1] edge it is wider than the eval-fixed foil, and
    # it narrows toward the foil as the eval sample grows, which the foil cannot do by construction.
    # (Near the edge both intervals clip at 0 and their widths stop measuring this.)
    import numpy as np
    from scipy.stats import bootstrap

    from calibrated_agent_evaluation.stats.rogan_gladen import rg_interval

    cal_pos = [1] * 72 + [0] * 8  # TPR 0.90
    cal_neg = [1] * 8 + [0] * 72  # 1-TNR 0.10
    eval_preds = [1] * 136 + [0] * 264  # measured 0.34: true rate 0.30
    got = rg_interval(cal_pos, cal_neg, eval_preds, n_resamples=2000, seed=7)
    ours = got.hi - got.lo
    big = rg_interval(cal_pos, cal_neg, [1] * 13_600 + [0] * 26_400, n_resamples=2000, seed=7)

    # Eval-rate-fixed foil: resample only the two calibration classes; hold measured constant.
    measured = sum(eval_preds) / len(eval_preds)

    def _fixed_eval(p: np.ndarray, n: np.ndarray, axis: int = -1) -> np.ndarray:
        tpr = np.mean(p, axis=axis)
        fpr = np.mean(n, axis=axis)
        denom = tpr - fpr
        with np.errstate(divide="ignore", invalid="ignore"):
            corrected = np.where(denom > 0, (measured - fpr) / denom, measured)
        return np.clip(corrected, 0.0, 1.0)

    res = bootstrap(
        (np.asarray(cal_pos, float), np.asarray(cal_neg, float)),
        _fixed_eval,
        method="percentile",
        n_resamples=2000,
        vectorized=True,
        rng=np.random.default_rng(7),
    )
    judgy_like = float(res.confidence_interval.high) - float(res.confidence_interval.low)
    assert ours == pytest.approx(0.1688, abs=5e-4)
    assert judgy_like == pytest.approx(0.1305, abs=5e-4)
    assert ours > 1.25 * judgy_like
    # 100x the eval sample: the eval share of the width is nearly gone.
    assert big.hi - big.lo == pytest.approx(0.1324, abs=5e-4)
    assert big.hi - big.lo < ours


def test_corrected_rate_refuses_the_interval_when_too_many_draws_are_not_identifiable() -> None:
    """Refuse the interval as well as the point, and say why.

    A J=0.20 judge cannot certify, so the record is a certified 90/90 one with its bounds replaced;
    corrected_rate only reads the points and class sizes. The reported bound is [0, 1], the
    interval that claims nothing, never one built from a mixture of two estimands.
    """
    from calibrated_agent_evaluation._model import replace

    rec = _certified_9090()
    weak = replace(
        rec,
        n_pos=35,
        n_neg=35,
        tpr=Bound(point=0.60, lb95=0.45, n=35),
        tnr=Bound(point=0.60, lb95=0.45, n=35),
        youden_j=0.20,
    )
    out = corrected_rate(
        weak,
        [1] * 80 + [0] * 120,
        rate_set_id="prod-weak",
        rate_trace_ids=rate_ids([1] * 80 + [0] * 120),
    )
    iv = out.corrected_rate
    assert iv is not None
    assert (iv.point, iv.lo, iv.hi) == (None, 0.0, 1.0)
    assert iv.refusal_reason is not None
    assert iv.refusal_reason.startswith("non-identifiable interval:")
    with pytest.raises(NonIdentifiableRateError, match="non-identifiable interval"):
        iv.require_point()


def test_corrected_rate_takes_clustered_rate_verdicts_at_their_effective_size() -> None:
    """Span-level verdicts from one trace are not independent draws.

    Traces of 5 verdicts that agree within the trace: the clustered interval is wider than the
    one that treats the 1000 verdicts as independent, and says which it is.
    """
    rec = _certified_9090()
    verdicts = ([1] * 5) * 30 + ([0] * 5) * 170  # 150 FAIL of 1000, whole traces at a time
    ids = [f"t{i // 5}" for i in range(1000)]
    plain = corrected_rate(
        rec, verdicts, rate_set_id="prod-spans", rate_trace_ids=rate_ids(verdicts)
    )
    grouped = corrected_rate(
        rec,
        verdicts,
        rate_set_id="prod-spans",
        rate_cluster_ids=ids,
        rate_trace_ids=rate_ids(verdicts),
    )
    assert plain.corrected_rate is not None
    assert grouped.corrected_rate is not None
    assert plain.corrected_rate.method == METHOD_RG_JEFFREYS
    assert grouped.corrected_rate.method == METHOD_RG_JEFFREYS_CLUSTERED
    assert grouped.corrected_rate.point == plain.corrected_rate.point
    plain_w = plain.corrected_rate.hi - plain.corrected_rate.lo
    assert grouped.corrected_rate.hi - grouped.corrected_rate.lo > plain_w


def test_degenerate_single_value_eval_still_finite() -> None:
    rec = _certified_9090()
    out = corrected_rate(
        rec, [1] * 150, rate_set_id="prod-allpos", rate_trace_ids=rate_ids([1] * 150)
    )  # measured = 1.0
    assert out.corrected_rate is not None
    lo, hi = out.corrected_rate.lo, out.corrected_rate.hi
    assert 0.0 <= lo <= hi <= 1.0


# --- the true rate is never read off the hunted set ----------------------


def test_hunted_set_as_rate_sample_is_refused() -> None:
    rec = _certified_9090()  # hunted_set_id == "hunted-9090"
    with pytest.raises(SampleMisuseError):
        corrected_rate(
            rec,
            [1] * 148 + [0] * 852,
            rate_set_id="hunted-9090",
            rate_trace_ids=rate_ids([1] * 148 + [0] * 852),
        )


# --- Youden J guard ------------------------------------------------------


def test_weak_judge_cannot_certify_because_j_caps_the_bounds() -> None:
    # A weak judge (TPR=TNR=0.72 → J=0.44) can never clear the 0.80 lower-bound gate — so the
    # "CERTIFIED but weak" state is unreachable through certify(). The J-guard's point-refusal is
    # therefore tested against a hand-built CERTIFIED-but-weak record below (test_FR015_j_below).
    rec = certify(
        _Judge(),
        _locked(_pairs(400, 288, 400, 288), set_id="weak"),
        ledger=InMemoryLockedTestLedger(),
    )  # 0.72/0.72
    assert rec.youden_j < 0.50
    assert rec.result == "REFUSED"


def _record_with(
    tpr_pt: float, tnr_pt: float, *, result: str = "CERTIFIED", hunted: str = "h"
) -> CalibrationRecord:
    """A hand-built record for isolating corrected_rate's J-guard / identifiability behavior."""
    return CalibrationRecord(
        provenance=TEST_STAMP,
        calib_id="c",
        judge_id="j",
        model="m",
        prompt_version="v",
        hunted_set_id=hunted,
        n_pos=80,
        n_neg=80,
        tpr=Bound(point=tpr_pt, lb95=tpr_pt - 0.02, n=80),
        tnr=Bound(point=tnr_pt, lb95=tnr_pt - 0.02, n=80),
        youden_j=tpr_pt + tnr_pt - 1.0,
        result=result,  # type: ignore[arg-type]
    )


def test_j_below_half_refuses_point_but_gives_interval() -> None:
    weak = _record_with(0.72, 0.72)  # J = 0.44 < 0.5
    out = corrected_rate(
        weak,
        [1] * 60 + [0] * 140,
        rate_set_id="prod",
        rate_trace_ids=rate_ids([1] * 60 + [0] * 140),
    )
    assert out.corrected_rate is not None
    # The point is refused (require_point raises); the interval is the honest signal.
    with pytest.raises(WeakJudgeError):
        out.corrected_rate.require_point()
    assert 0.0 <= out.corrected_rate.lo <= out.corrected_rate.hi <= 1.0


def test_j_above_half_produces_a_point() -> None:
    strong = _record_with(0.90, 0.90)  # J = 0.80
    out = corrected_rate(
        strong,
        [1] * 148 + [0] * 852,
        rate_set_id="prod",
        rate_trace_ids=rate_ids([1] * 148 + [0] * 852),
    )
    assert out.corrected_rate is not None
    # Rogan-Gladen is exact algebra here: (0.148 + 0.90 - 1) / (0.90 + 0.90 - 1) = 0.06.
    assert out.corrected_rate.require_point() == pytest.approx(0.06, abs=1e-9)


# --- non-identifiable rate -----------------------------------------------


def test_measured_below_false_positive_floor_refuses_point() -> None:
    strong = _record_with(0.90, 0.90)  # false-positive floor = 1-TNR = 0.10
    out = corrected_rate(
        strong,
        [1] * 10 + [0] * 190,
        rate_set_id="prod",
        rate_trace_ids=rate_ids([1] * 10 + [0] * 190),
    )  # measured 0.05 < 0.10
    assert out.corrected_rate is not None
    assert out.corrected_rate.point is None  # refused, not silently clamped to 0
    with pytest.raises(NonIdentifiableRateError):
        out.corrected_rate.require_point()


# --- uncalibrated judge refused ------------------------------------------


def test_corrected_rate_refuses_uncertified_judge() -> None:
    refused = _record_with(0.90, 0.90, result="REFUSED")
    with pytest.raises(UncalibratedJudgeError):
        corrected_rate(
            refused,
            [1] * 20 + [0] * 180,
            rate_set_id="prod",
            rate_trace_ids=rate_ids([1] * 20 + [0] * 180),
        )


def test_certified_record_proceeds() -> None:
    ok = _record_with(0.90, 0.90, result="CERTIFIED")
    out = corrected_rate(
        ok,
        [1] * 148 + [0] * 852,
        rate_set_id="prod",
        rate_trace_ids=rate_ids([1] * 148 + [0] * 852),
    )
    assert out.corrected_rate is not None


# --- version pinning -----------------------------------------------------


def test_record_certifies_only_the_pinned_version() -> None:
    rec = certify(
        _Judge(model="X", prompt_version="sha1"),
        _locked(_pairs(80, 72, 80, 72)),
        ledger=InMemoryLockedTestLedger(),
    )
    assert certifies(rec, _Judge(model="X", prompt_version="sha1")) is True
    # A prompt bump forces re-calibration...
    assert certifies(rec, _Judge(model="X", prompt_version="sha2")) is False
    # ...as does a model bump.
    assert certifies(rec, _Judge(model="Y", prompt_version="sha1")) is False


def test_refused_record_never_certifies() -> None:
    rec = certify(
        _Judge(model="X", prompt_version="sha1"),
        _locked(_pairs(20, 18, 20, 18)),
        ledger=InMemoryLockedTestLedger(),
    )
    assert rec.result == "REFUSED"
    assert certifies(rec, _Judge(model="X", prompt_version="sha1")) is False


# --- no bare number: the output contract ---------------------------------


def test_every_surfaced_figure_carries_its_confidence_treatment() -> None:
    rec = _certified_9090()
    out = corrected_rate(
        rec,
        [1] * 148 + [0] * 852,
        rate_set_id="prod",
        rate_trace_ids=rate_ids([1] * 148 + [0] * 852),
    )
    # TPR/TNR are Bounds (point + lb95), never bare floats; the corrected rate is an Interval.
    assert isinstance(rec.tpr, Bound)
    assert isinstance(rec.tnr, Bound)
    assert isinstance(out.corrected_rate, Interval)
    # The gate reads lb95; a bare float would have no lower bound to read.
    assert hasattr(rec.tpr, "lb95")
    assert hasattr(rec.tnr, "lb95")


# --- each judge's certification stands alone (amended 2026-10-14) -------------


def test_each_judge_is_certified_on_its_own_bounds() -> None:
    """Certifying two judges in one run applies no correction across them.

    Each judge at 72/80 per class gets the plain one-sided 95% Wilson bound 0.831099 on both TPR
    and TNR, the value a judge certified alone gets, and the record carries no multiplicity
    family.
    """

    r1 = certify(
        _Judge(judge_id="j1"),
        _locked(_pairs(80, 72, 80, 72), set_id="lk1"),
        ledger=InMemoryLockedTestLedger(),
    )
    r2 = certify(
        _Judge(judge_id="j2"),
        _locked(_pairs(80, 72, 80, 72), set_id="lk2"),
        ledger=InMemoryLockedTestLedger(),
    )
    for rec in (r1, r2):
        assert rec.tpr.lb95 == pytest.approx(0.831099146, abs=1e-9)
        assert rec.tnr.lb95 == pytest.approx(0.831099146, abs=1e-9)
        assert rec.result == "CERTIFIED"
    names = set(type(r1).model_fields)
    assert not names & {"m_comparisons", "gate_family"}


def test_certify_takes_no_judge_count() -> None:
    """`certify` has no `m`, so no caller can imply a family."""
    locked = _locked(_pairs(80, 72, 80, 72), set_id="lk3")
    with pytest.raises(TypeError):
        certify(_Judge(judge_id="j3"), locked, m=2, ledger=InMemoryLockedTestLedger())  # type: ignore[call-arg]


def test_holm_family_is_distinct_from_exploratory_fdr_by() -> None:
    # The exploratory dependence-aware family (fdr_by) is available AND produces a materially
    # different verdict from the Holm hard-gate family on the same p-values — proving the gate's
    # choice of family is load-bearing, not cosmetic. On [0.001, 0.013, 0.021, 0.030] Holm (FWER)
    # rejects all four while BY (dependence-aware FDR, the conservative Yekutieli variant with its
    # extra log-factor penalty) rejects only the strongest — verified against statsmodels.
    from calibrated_agent_evaluation.stats.multiplicity import multiplicity

    pvals = [0.001, 0.013, 0.021, 0.030]
    holm_reject, m = multiplicity(pvals, "holm")
    by_reject, _ = multiplicity(pvals, "fdr_by")
    assert m == 4
    assert holm_reject == [True, True, True, True]
    assert by_reject == [True, False, False, False]


def test_tpr_tnr_bound_pair_is_not_a_joint_95_region() -> None:
    """The simultaneous coverage of the two one-sided bounds.

    Computed exactly from the binomial: P(wilson_lower(X, n) <= p) summed over X ~ Bin(n, p), per
    class, and squared for two independent classes. Pinned so a change to GATE_THRESHOLD, the
    Wilson alpha, or the class size that moves it is caught. At the gate threshold each bound
    covers 0.9435 (Wilson's coverage oscillates below the nominal 0.95 there) and the pair 0.890.
    """
    from scipy.stats import binom

    from calibrated_agent_evaluation.calibrate.engine import GATE_THRESHOLD
    from calibrated_agent_evaluation.stats import wilson_lower

    def coverage(n: int, p: float) -> float:
        return float(sum(binom.pmf(x, n, p) for x in range(n + 1) if wilson_lower(x, n) <= p))

    assert GATE_THRESHOLD == 0.80
    single = coverage(80, GATE_THRESHOLD)
    assert single == pytest.approx(0.943539, abs=1e-6)
    assert single * single == pytest.approx(0.890266, abs=1e-6)
    assert coverage(80, 0.90) ** 2 == pytest.approx(0.930634, abs=1e-6)
