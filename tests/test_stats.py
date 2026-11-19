# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the statistics contracts (stats/).

Known-answer tests against values re-derivable by hand / cross-checked against the verified vendor
surfaces, plus hypothesis properties. Every test name embeds its FR. The
oracles (sklearn for Cohen's κ, closed-form Wilson/RG values) keep the reimplementations honest.
"""

from __future__ import annotations

import math
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from calibrated_agent_evaluation.calibrate.errors import (
    ErrorRatesDoNotTransferError,
    NonIdentifiableRateError,
    StatisticsInputError,
)
from calibrated_agent_evaluation.stats.agreement import (
    cohen_kappa,
    minority_fraction,
    should_hide_raw_agreement,
)
from calibrated_agent_evaluation.stats.multiplicity import multiplicity
from calibrated_agent_evaluation.stats.proportions import wilson_lower
from calibrated_agent_evaluation.stats.rogan_gladen import rg_interval, rogan_gladen, youden_j

# --- wilson_lower ---------------------------------------------------------


def test_wilson_lower_known_answer() -> None:
    # The one-sided 95% Wilson lower bound at k=72, n=80, from the closed form
    # (p + z^2/2n - z sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n) with z = z_0.95 = 1.6448536:
    # 0.831099146 -- the number the gate and the planner both key on.
    z, n, p = 1.6448536269514722, 80, 72 / 80
    closed = (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (
        1 + z * z / n
    )
    assert closed == pytest.approx(0.831099146, abs=1e-9)
    assert wilson_lower(72, 80) == pytest.approx(closed, abs=1e-9)


def test_wilson_lower_boundaries() -> None:
    # A perfect sample's lower bound is still < 1 (uncertainty never vanishes at finite n)...
    assert wilson_lower(80, 80) < 1.0
    # ...and an all-failure sample's lower bound is exactly 0 (clamped, never negative).
    assert wilson_lower(0, 80) == 0.0


def test_wilson_lower_rejects_out_of_domain_input() -> None:
    # Negative path: a proportion from a non-count is a typed error, never a NaN leak.
    with pytest.raises(StatisticsInputError):
        wilson_lower(5, 0)
    with pytest.raises(StatisticsInputError):
        wilson_lower(81, 80)  # k > n
    with pytest.raises(StatisticsInputError):
        wilson_lower(-1, 80)


@given(n=st.integers(min_value=1, max_value=500), frac=st.floats(min_value=0.0, max_value=1.0))
def test_wilson_lower_never_exceeds_point_estimate(n: int, frac: float) -> None:
    # Property: the lower bound is always <= the point estimate k/n and stays in [0, 1].
    k = round(frac * n)
    lb = wilson_lower(k, n)
    assert 0.0 <= lb <= 1.0
    assert lb <= k / n + 1e-9


# --- rogan_gladen -------------------------------------------------------


def test_rogan_gladen_recovers_true_prevalence() -> None:
    # The HLD §6 worked case: a 90/90 judge at a true 6% prevalence reports ~0.148 raw; inverting
    # recovers ~0.06. This is the whole reason the correction exists.
    assert rogan_gladen(0.148, 0.90, 0.90) == pytest.approx(0.06, abs=1e-3)


def test_rogan_gladen_refuses_below_false_positive_floor() -> None:
    # Negative path: a measured rate below the judge's false-positive floor (1-TNR=0.10)
    # would invert to a negative true rate — refused, not silently clamped to 0.
    with pytest.raises(NonIdentifiableRateError):
        rogan_gladen(0.05, 0.90, 0.90)


def test_rogan_gladen_refuses_above_true_positive_ceiling() -> None:
    # Negative path, the symmetric mirror of the false-positive floor: a measured rate
    # ABOVE the judge's true-positive ceiling (TPR=0.90) would invert to a true rate > 1 — refused,
    # not silently clamped to 1.0 (a fabricated 100% is exactly the hazard). The boundary
    # measured == TPR is identifiable (inclusive region [1-TNR, TPR]).
    with pytest.raises(NonIdentifiableRateError):
        rogan_gladen(0.95, 0.90, 0.90)
    assert rogan_gladen(0.90, 0.90, 0.90) == pytest.approx(1.0)  # boundary is still identifiable


def test_rogan_gladen_refuses_when_judge_no_better_than_chance() -> None:
    # Negative path: denominator TPR-(1-TNR) <= 0 (J<=0) → not identifiable.
    with pytest.raises(NonIdentifiableRateError):
        rogan_gladen(0.30, 0.50, 0.50)


# --- rg_interval ------------------------------------------------


def _class_array(rate: float, n: int) -> list[int]:
    """A 1/0 array of length n whose mean is ~rate (for building calibration/eval samples)."""
    k = round(rate * n)
    return [1] * k + [0] * (n - k)


def test_rg_interval_brackets_the_true_rate() -> None:
    # A 90/90 judge (TPR=0.90 → cal_pos mean 0.90; TNR=0.90 → cal_neg mean 0.10) on an eval sample
    # measuring ~0.148 should yield an interval that brackets the true 0.06, entirely within [0,1].
    cal_pos = _class_array(0.90, 80)  # judge caught 90% of true fails
    cal_neg = _class_array(0.10, 80)  # judge false-alarmed on 10% of true passes
    eval_preds = _class_array(0.148, 400)
    got = rg_interval(cal_pos, cal_neg, eval_preds, n_resamples=999, seed=1)
    assert 0.0 <= got.lo <= got.hi <= 1.0
    assert got.lo <= 0.06 <= got.hi


def test_rg_interval_stays_finite_at_the_boundary() -> None:
    # Edge (the BCa hazard the percentile interval avoids): a measured rate at the judge's
    # true-positive ceiling pushes the corrected rate to the [0,1] boundary. The interval must
    # still be a finite, in-range interval, never NaN or a bound outside [0,1].
    cal_pos = _class_array(0.90, 80)
    cal_neg = _class_array(0.10, 80)
    eval_preds = [1] * 90 + [0] * 10  # measured == TPR, corrected point exactly 1
    got = rg_interval(cal_pos, cal_neg, eval_preds, n_resamples=999, seed=2)
    assert not math.isnan(got.lo)
    assert not math.isnan(got.hi)
    assert 0.0 <= got.lo <= got.hi <= 1.0


def test_rg_interval_refuses_a_sample_beyond_the_judge_ceiling() -> None:
    # 100 of 100 FAIL against a judge that catches 90% of real failures: almost every draw clips
    # to 1, and a zero-width [1, 1] would claim every trace fails. The sample contradicts the
    # judge's calibrated error rates instead.
    with pytest.raises(ErrorRatesDoNotTransferError, match="true-positive ceiling"):
        rg_interval(_class_array(0.90, 80), _class_array(0.10, 80), [1] * 100, n_resamples=999)


def test_rg_interval_rejects_empty_sample() -> None:
    # Negative: an empty sample can't be resampled — typed error, not a scipy crash.
    with pytest.raises(StatisticsInputError):
        rg_interval([1, 0], [1, 0], [])


def test_rg_interval_covers_the_truth_when_the_judge_made_no_false_positives() -> None:
    """The planner's p_hat=0.95 pick, 20 per class.

    A 95/95 judge shows no false positive in 20 true passes about 36% of the time (0.95**20). A
    resample of an all-zero class is all zeros, so the bootstrap took 1 - TNR as exactly 0, and with
    a large eval sample the interval sat on (0.068 / TPR) entirely above the true 0.02. The
    interval has to admit that a false-positive rate of 0/20 is not a rate of 0.
    """
    cal_pos = [1] * 19 + [0]
    cal_neg = [0] * 20
    # measured = 0.02 * 0.95 + 0.98 * 0.05 = 0.068 at a true prevalence of 0.02
    eval_preds = [1] * 340 + [0] * 4660
    got = rg_interval(cal_pos, cal_neg, eval_preds, n_resamples=9999, seed=3)
    assert got.lo <= 0.02 <= got.hi


def test_rg_interval_refuses_a_non_binary_sample() -> None:
    with pytest.raises(StatisticsInputError, match="1/0"):
        rg_interval([1, 0, 2], [0, 1], [1, 0])


def test_rg_interval_accepts_numpy_input() -> None:
    """Comment: ``not cal_pos`` raised numpy's ambiguous-truth error."""
    import numpy as np

    got = rg_interval(
        np.array([1] * 18 + [0] * 2),
        np.array([1] * 2 + [0] * 18),
        np.array([1] * 30 + [0] * 70),
        n_resamples=999,
        seed=0,
    )
    assert 0.0 <= got.lo < got.hi <= 1.0


def test_rg_interval_counts_the_draws_it_leaves_out() -> None:
    """A draw with TPR <= 1 - TNR has no corrected rate.

    The bootstrap used to substitute the raw judge rate for such a resample, mixing a second
    estimand into the distribution without saying so. A J=0.29 judge at 35 per class puts a few
    of its draws there; they are counted, and the interval is still reported because they are
    fewer than one tail's 2.5%.
    """
    got = rg_interval(
        [1] * 23 + [0] * 12,  # TPR 0.657
        [1] * 13 + [0] * 22,  # 1 - TNR 0.371
        [1] * 40 + [0] * 60,
        n_resamples=9999,
        seed=4,
    )
    assert got.n_draws == 9999
    assert got.n_nonidentifiable == 60
    assert 0.0 <= got.lo < got.hi <= 1.0


def test_rg_interval_widens_each_tail_by_the_draws_it_left_out() -> None:
    """A left-out draw could be in either tail, so each percentile moves outward by its share."""
    import numpy as np

    pos, neg, ev = [1] * 23 + [0] * 12, [1] * 13 + [0] * 22, [1] * 40 + [0] * 60
    got = rg_interval(pos, neg, ev, n_resamples=9999, seed=4)
    rng = np.random.default_rng(4)
    t = rng.beta(23.5, 12.5, 9999)
    f = rng.beta(13.5, 22.5, 9999)
    m = rng.beta(40.5, 60.5, 9999)
    ok = t - f > 0
    theta = np.clip((m[ok] - f[ok]) / (t[ok] - f[ok]), 0.0, 1.0)
    share = (~ok).mean()
    lo, hi = np.quantile(theta, [(0.025 - share) / (1 - share), 0.975 / (1 - share)])
    assert (got.lo, got.hi) == (float(lo), float(hi))
    plain_lo, plain_hi = np.quantile(theta, [0.025, 0.975])
    assert got.lo <= plain_lo
    assert got.hi >= plain_hi


def test_rg_interval_refuses_when_the_left_out_draws_exceed_a_tail() -> None:
    """The ticket's J=0.20 judge at 35 per class: over 2.5% of draws are not identifiable.

    Past one tail's allowance the percentile at that tail is not determined by the identifiable
    draws, so no interval is reported.
    """
    with pytest.raises(NonIdentifiableRateError, match=r"of 9999 draws"):
        rg_interval([1] * 21 + [0] * 14, [1] * 14 + [0] * 21, [1] * 40 + [0] * 60, seed=0)


def test_cluster_design_effect_of_perfectly_clustered_verdicts() -> None:
    """50 traces of 4 identical verdicts, half the traces FAIL.

    Each cluster residual is +/-2, so var_cl = (50/49) * 50 * 4 / 200^2 and var_srs = 0.25/200:
    the design effect is 4 * 50/49, the cluster size with the small-sample factor.
    """
    from calibrated_agent_evaluation.stats.rogan_gladen import cluster_design_effect

    values = [1] * 100 + [0] * 100
    ids = [f"t{i // 4}" for i in range(200)]
    assert cluster_design_effect(values, ids) == pytest.approx(4 * 50 / 49, abs=1e-12)


def test_cluster_design_effect_is_floored_at_one() -> None:
    """Clusters that balance exactly (one FAIL and one PASS each) are not credited below 1."""
    from calibrated_agent_evaluation.stats.rogan_gladen import cluster_design_effect

    values = [1, 0] * 50
    ids = [f"t{i // 2}" for i in range(100)]
    assert cluster_design_effect(values, ids) == 1.0


def test_cluster_design_effect_refuses_bad_cluster_input() -> None:
    from calibrated_agent_evaluation.stats.rogan_gladen import cluster_design_effect

    with pytest.raises(StatisticsInputError, match="cluster ids"):
        cluster_design_effect([1, 0, 1], ["a", "b"])
    with pytest.raises(StatisticsInputError, match="at least 2 clusters"):
        cluster_design_effect([1, 0, 1], ["a", "a", "a"])


def test_rg_interval_widens_for_clustered_eval_verdicts() -> None:
    """Verdicts that agree within a trace carry less information than as many independent ones.

    A tight 85/85 calibration (162 per class) leaves the eval sample's share of the width visible.
    With traces of 4 identical verdicts the interval is computed at about a quarter of the eval
    sample and is wider; with balanced clusters it is the unclustered interval, draw for draw.
    """
    cal_pos = [1] * 138 + [0] * 24
    cal_neg = [1] * 24 + [0] * 138
    eval_preds = [1] * 100 + [0] * 100
    clustered = [f"t{i // 4}" for i in range(200)]
    iid = rg_interval(cal_pos, cal_neg, eval_preds, n_resamples=4999, seed=5)
    grouped = rg_interval(
        cal_pos, cal_neg, eval_preds, n_resamples=4999, seed=5, eval_cluster_ids=clustered
    )
    assert grouped.eval_design_effect == pytest.approx(4 * 50 / 49, abs=1e-12)
    assert iid.eval_design_effect == 1.0
    assert grouped.hi - grouped.lo > 1.3 * (iid.hi - iid.lo)
    balanced = rg_interval(
        cal_pos,
        [1] * 24 + [0] * 138,
        [1, 0] * 100,
        n_resamples=4999,
        seed=5,
        eval_cluster_ids=[f"t{i // 2}" for i in range(200)],
    )
    plain = rg_interval(cal_pos, [1] * 24 + [0] * 138, [1, 0] * 100, n_resamples=4999, seed=5)
    assert (balanced.lo, balanced.hi) == (plain.lo, plain.hi)


# --- multiplicity -------------------------------------------------------


def test_multiplicity_holm_known_rejects_and_reports_m() -> None:
    # Holm on 4 p-values: only the smallest (0.01) survives at alpha=0.05; m is reported as 4.
    reject, m = multiplicity([0.01, 0.04, 0.03, 0.20], "holm")
    assert reject == [True, False, False, False]
    assert m == 4


def test_multiplicity_fdr_by_available_and_distinct() -> None:
    # The exploratory dependence-aware family exists and is distinct from Holm; m still reported.
    reject, m = multiplicity([0.01, 0.04, 0.03, 0.20], "fdr_by")
    assert m == 4
    assert len(reject) == 4
    # Distinct from Holm on the same input: BY is more conservative here, rejecting fewer.
    holm_reject, _ = multiplicity([0.01, 0.04, 0.03, 0.20], "holm")
    assert reject != holm_reject


def test_multiplicity_rejects_unknown_method() -> None:
    # Negative path: an unknown method (incl. the independence-assuming fdr_bh we forbid) is a typed
    # error, never a silent pass-through to statsmodels.
    with pytest.raises(StatisticsInputError):
        multiplicity([0.01, 0.2], "fdr_bh")
    with pytest.raises(StatisticsInputError):
        multiplicity([0.01, 0.2], "bonferroni")


def test_multiplicity_rejects_empty_and_out_of_range() -> None:
    with pytest.raises(StatisticsInputError):
        multiplicity([], "holm")
    with pytest.raises(StatisticsInputError):
        multiplicity([0.5, 1.5], "holm")


# --- cohen_kappa (sklearn oracle) ----------------------------------------


def test_cohen_kappa_matches_sklearn_oracle() -> None:
    # The reimplementation (~15 lines to avoid a 30 MiB sklearn runtime dep) must equal the
    # sklearn oracle on shared fixtures — the check that keeps the reimplementation honest.
    # A hard import: sklearn is in the dev group, and a missing oracle must fail, not skip.
    from sklearn.metrics import cohen_kappa_score

    y1 = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y2 = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1]
    assert cohen_kappa(y1, y2) == pytest.approx(cohen_kappa_score(y1, y2))


def test_cohen_kappa_perfect_and_degenerate() -> None:
    # Perfect agreement → 1.0; both raters constant-and-agreeing → 1.0 (degenerate convention).
    assert cohen_kappa([1, 0, 1], [1, 0, 1]) == pytest.approx(1.0)
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == pytest.approx(1.0)


def test_hide_raw_agreement_under_imbalance() -> None:
    # Edge: under heavy imbalance (minority < 20%) raw agreement is hidden → show κ.
    imbalanced = [0] * 95 + [1] * 5  # 5% minority
    balanced = [0] * 50 + [1] * 50
    assert should_hide_raw_agreement(imbalanced) is True
    assert should_hide_raw_agreement(balanced) is False
    assert minority_fraction(imbalanced) == pytest.approx(0.05)


def test_cohen_kappa_rejects_mismatched_lengths() -> None:
    with pytest.raises(StatisticsInputError):
        cohen_kappa([1, 0, 1], [1, 0])
    with pytest.raises(StatisticsInputError):
        cohen_kappa([], [])


# --- youden_j ------------------------------------------------------------


def test_youden_j_values() -> None:
    assert youden_j(0.90, 0.90) == pytest.approx(0.80)
    # A coin-flip judge (the boundary the J-guard keys on) carries no information: J == 0.
    assert youden_j(0.50, 0.50) == pytest.approx(0.0)


def test_youden_j_rejects_out_of_domain() -> None:
    with pytest.raises(StatisticsInputError):
        youden_j(1.2, 0.5)


# --- Newcombe's paired hybrid-score interval ---------------------


def test_newcombe_paired_diff_matches_the_ncss_reference_example() -> None:
    """NCSS "Two Correlated Proportions" Example 1: A=30, B=6, C=5, D=9 (N=50).

    p1 = 36/50 = 0.72, p2 = 35/50 = 0.70; the published Wilson-score (Newcombe method 10) 95%
    limits for p1 - p2 are -0.1145 and 0.1537.
    """
    from calibrated_agent_evaluation.stats.proportions import newcombe_paired_diff

    diff, lo, hi = newcombe_paired_diff(30, 6, 5, 9)
    assert diff == pytest.approx(0.02, abs=1e-12)
    assert lo == pytest.approx(-0.1145, abs=5e-5)
    assert hi == pytest.approx(0.1537, abs=5e-5)


def test_newcombe_paired_diff_is_not_degenerate_with_an_empty_discordant_cell() -> None:
    """b=5, c=0, a=d=0: the Wald interval was [1, 1]; the score interval is [1 - 0.614, 1].

    p1 = 1, p2 = 0, phi = 0 (a zero margin). Wilson 95% for 5/5 is [0.565518, 1] and for 0/5
    [0, 0.434482], so delta = sqrt(0.434482^2 + 0.434482^2) = 0.614451.
    """
    from calibrated_agent_evaluation.stats.proportions import newcombe_paired_diff

    diff, lo, hi = newcombe_paired_diff(0, 5, 0, 0)
    assert diff == 1.0
    assert hi == 1.0
    assert lo == pytest.approx(1.0 - 2**0.5 * 0.434482, abs=1e-5)
    assert hi - lo > 0.6


@given(st.integers(0, 30), st.integers(0, 30), st.integers(0, 30), st.integers(0, 30))
def test_newcombe_paired_diff_stays_in_range_and_brackets_the_point(
    a: int, b: int, c: int, d: int
) -> None:
    from calibrated_agent_evaluation.stats.proportions import newcombe_paired_diff

    if a + b + c + d == 0:
        return
    diff, lo, hi = newcombe_paired_diff(a, b, c, d)
    assert -1.0 <= lo <= diff <= hi <= 1.0


# --- kappa interval and the degenerate imbalance case ---------


def test_cohen_kappa_interval_is_the_union_of_the_fce_wald_and_the_agreement_score_interval() -> (
    None
):
    """[[20, 5], [10, 15]]: kappa 0.4; hand ASE = sqrt(0.0504 / (50 * 0.5^4)) = 0.126996.

    The Wald half of the union matches statsmodels' ``cohens_kappa`` (0.1511, 0.6489). The other
    half is the Wilson interval for agreement 35/50 mapped through kappa at p_e = 0.5.
    """
    from statsmodels.stats.inter_rater import cohens_kappa
    from statsmodels.stats.proportion import proportion_confint

    from calibrated_agent_evaluation.stats.kappa import cohen_kappa_interval

    gold = ["F"] * 25 + ["P"] * 25
    verdict = ["F"] * 20 + ["P"] * 5 + ["F"] * 10 + ["P"] * 15
    ci = cohen_kappa_interval(gold, verdict)
    assert ci.point == pytest.approx(0.4, abs=1e-12)
    half = 1.959963984540054 * math.sqrt(0.0504 / (50 * 0.5**4))
    oracle = cohens_kappa([[20, 5], [10, 15]])
    assert 0.4 - half == pytest.approx(float(oracle["kappa_low"]), abs=1e-9)
    agree_lo, agree_hi = proportion_confint(35, 50, alpha=0.05, method="wilson")
    assert ci.lo == pytest.approx(min(0.4 - half, (agree_lo - 0.5) / 0.5), abs=1e-9)
    assert ci.hi == pytest.approx(max(0.4 + half, (agree_hi - 0.5) / 0.5), abs=1e-9)
    assert ci.lo <= float(oracle["kappa_low"])
    assert ci.hi >= float(oracle["kappa_upp"])


def test_perfect_agreement_keeps_a_width() -> None:
    """Five agreeing items used to read kappa 1.0 [1.0, 1.0]: the Wald variance is 0 there."""
    from calibrated_agent_evaluation.stats.kappa import cohen_kappa_interval

    ci = cohen_kappa_interval(["F", "P", "F", "P", "F"], ["F", "P", "F", "P", "F"])
    assert ci.point == pytest.approx(1.0)
    assert ci.hi == pytest.approx(1.0)
    assert ci.lo < 0.3


def test_cohen_kappa_interval_refuses_the_constant_case() -> None:
    from calibrated_agent_evaluation.stats.agreement import UndefinedAgreementError
    from calibrated_agent_evaluation.stats.kappa import cohen_kappa_interval

    with pytest.raises(UndefinedAgreementError, match="0/0"):
        cohen_kappa_interval(["FAIL"] * 5, ["FAIL"] * 5)
    with pytest.raises(StatisticsInputError):
        cohen_kappa_interval([], [])


def test_single_valued_label_set_hides_raw_agreement() -> None:
    """An all-FAIL set is maximally imbalanced; empty has nothing."""
    assert should_hide_raw_agreement(["FAIL"] * 184) is True
    assert should_hide_raw_agreement([]) is False


# --- determinism and properties ------------------------------------


def test_rg_interval_same_seed_same_interval_and_pinned() -> None:
    """A seed fixes the interval; the pinned value catches an RNG-stream or resample change.

    Re-pin deliberately when the estimator changes (the Jeffreys switch did). The lower
    edge at 0.0 for a 0.06 point is not a fallback artifact: at 80 per class 1 - TNR is known only
    to about +/-0.066, which alone moves the corrected rate by more than 0.06, so the data cannot
    rule out a true rate of 0.
    """
    cal_pos = [1] * 72 + [0] * 8
    cal_neg = [1] * 8 + [0] * 72
    eval_preds = [1] * 148 + [0] * 852
    first = rg_interval(cal_pos, cal_neg, eval_preds, n_resamples=999, seed=1)
    assert first == rg_interval(cal_pos, cal_neg, eval_preds, n_resamples=999, seed=1)
    assert (first.lo, first.hi) == pytest.approx((0.0, 0.12460219428504386), abs=1e-12)
    assert first.n_nonidentifiable == 0


@settings(max_examples=12, deadline=None)
@given(
    st.integers(30, 60),
    st.integers(30, 60),
    st.floats(0.85, 0.98),
    st.floats(0.02, 0.15),
    st.floats(0.15, 0.5),
    st.integers(0, 10_000),
)
def test_rg_interval_contains_its_point(
    n_pos: int, n_neg: int, tpr: float, fpr: float, measured: float, seed: int
) -> None:
    """Within the identifiable region the interval brackets the corrected point."""
    k_pos, k_neg = round(tpr * n_pos), round(fpr * n_neg)
    sens, spec = k_pos / n_pos, 1 - k_neg / n_neg
    k_ev = round(measured * 200)
    if not (1 - spec) < k_ev / 200 < sens:
        return
    point = rogan_gladen(k_ev / 200, sens, spec)
    got = rg_interval(
        [1] * k_pos + [0] * (n_pos - k_pos),
        [1] * k_neg + [0] * (n_neg - k_neg),
        [1] * k_ev + [0] * (200 - k_ev),
        n_resamples=199,
        seed=seed,
    )
    assert got.lo <= point <= got.hi


@given(st.floats(0.0, 1.0), st.floats(0.6, 1.0), st.floats(0.6, 1.0))
def test_rogan_gladen_inverts_the_misclassification_forward_model(
    true_rate: float, tpr: float, tnr: float
) -> None:
    """measured = pi * TPR + (1 - pi)(1 - TNR); correcting it recovers pi."""
    if tpr + tnr - 1 <= 0.2:
        return
    measured = true_rate * tpr + (1 - true_rate) * (1 - tnr)
    if not (1 - tnr) < measured < tpr:
        return
    assert rogan_gladen(measured, tpr, tnr) == pytest.approx(true_rate, abs=1e-9)


@given(
    st.lists(st.sampled_from(["PASS", "FAIL", "SKIP"]), min_size=2, max_size=40),
    st.randoms(use_true_random=False),
)
def test_cohen_kappa_symmetric_and_at_most_one(gold: list[str], rnd: random.Random) -> None:
    verdict = [rnd.choice(["PASS", "FAIL", "SKIP"]) for _ in gold]
    assert cohen_kappa(gold, verdict) == pytest.approx(cohen_kappa(verdict, gold), abs=1e-12)
    assert cohen_kappa(gold, verdict) <= 1.0 + 1e-12
