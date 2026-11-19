# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rogan-Gladen prevalence correction and the interval reported with it.

A "good" judge still reports the wrong rate: a 90%-TPR/90%-TNR judge pointed at a true 6%
prevalence reports ~15%, a floor that never moves as the agent improves. The fix
is to invert the judge's known error rates:

    true ~= (measured - (1 - TNR)) / (TPR - (1 - TNR))          # Rogan & Gladen (1978)

and to report an interval that carries **both** the calibration-sample uncertainty (in TPR/TNR)
**and** the eval-sample uncertainty (in ``measured``). The ``judgy`` package resamples only the
calibration set and holds the eval rate fixed, which leaves the eval share out.

:func:`rg_interval` draws TPR, 1 - TNR and ``measured`` from independent Jeffreys posteriors
``Beta(x + 1/2, n - x + 1/2)`` and takes the central percentile range of the corrected rate
over the draws. It replaced a joint three-sample percentile bootstrap
because the bootstrap cannot represent uncertainty in a count it observed as zero: a 95/95 judge
shows no false positive in 20 true passes about 36% of the time, every resample of that class is
then all zeros, and the interval sat entirely above the true rate. Measured over the coverage
study's planner grid (2000 replicates per cell), the bootstrap covered as low as 0.192 at a
nominal 0.95. This interval's lowest cell is 0.9345, its misses fall on both sides of the truth,
and where it is off nominal it is mostly conservative, at the planner's smallest calibration sizes
(``fixtures/expected/coverage/interval_candidates.json`` compares the two with basic,
studentized, delta-method and Fieller alternatives on the same data).

The draws are clipped into [0, 1] before the percentiles are taken, so the interval is always
finite and in range. When clipping collapses the interval to a single boundary point (almost every
draw puts the measured rate below the judge's false-positive floor, or above its true-positive
ceiling), the interval is refused: the sample contradicts the judge's calibrated error rates, and
a zero-width interval at 0 would claim the true rate is exactly 0.

At a boundary count the interval follows the modified Jeffreys rule of Brown, Cai and DasGupta
(2001): a Jeffreys posterior puts no mass on a rate of exactly 0 or 1, so a Jeffreys interval for
a count of 0 excludes a true rate of 0 and undercovers there, and the modified interval sets that
limit to the boundary. The clipped corrected rate never increases with TPR or 1 - TNR and never
decreases with the measured rate, so each boundary moves exactly one limit of the corrected rate:
no false positive (1 - TNR observed as 0) takes the upper limit with 1 - TNR at 0, every true
positive caught (TPR observed as 1) takes the lower limit with TPR at 1, and a measured count of 0
or of every verdict takes the lower or upper limit with the measured rate at 0 or 1. The other
inputs keep their draws. The interval only widens: a judge with no false positives has a true
false-positive rate that may be 0, and the unmodified interval reported a corrected rate that
excluded the truth whenever it was.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from calibrated_agent_evaluation.calibrate.errors import (
    TRANSFER_REFUSAL,
    ErrorRatesDoNotTransferError,
    NonIdentifiableRateError,
    StatisticsInputError,
)


def youden_j(tpr: float, tnr: float) -> float:
    """Return Youden's J = TPR + TNR - 1.

    J ranges from 0 (a coin-flip judge, no information) to 1 (perfect). The gate refuses the
    Rogan-Gladen point correction when J < 0.5 -- below that the denominator ``TPR - (1 - TNR) = J``
    shrinks toward 0 and the correction divides by ~0 (the widening bound is the honest signal).
    """
    _check_prob(tpr, "tpr")
    _check_prob(tnr, "tnr")
    return tpr + tnr - 1.0


def rogan_gladen(measured: float, tpr: float, tnr: float) -> float:
    """Return the Rogan-Gladen prevalence-corrected rate.

    Raises :class:`NonIdentifiableRateError` when the estimate is outside the identifiable
    region: the denominator ``tpr - (1 - tnr) <= 0`` (a judge no better than chance, J <= 0),
    ``measured < 1 - tnr`` (the observed positives are fewer than the judge's false-positive floor,
    so the true rate would be negative), or ``measured > tpr`` (more observed positives than the
    judge's true-positive ceiling, so the true rate would exceed 1). The identifiable region is the
    closed interval ``[1 - tnr, tpr]``: a measured rate exactly at an end returns 0 or 1, the true
    corrected value there, and only a rate outside it is refused, never silently clamped.
    """
    _check_prob(measured, "measured")
    _check_prob(tpr, "tpr")
    _check_prob(tnr, "tnr")
    denom = tpr - (1.0 - tnr)  # == youden_j
    if denom <= 0:
        raise NonIdentifiableRateError(
            f"Rogan-Gladen denominator TPR-(1-TNR)={denom:.4f} <= 0 (judge no better than chance); "
            "the corrected rate is not identifiable"
        )
    if measured < (1.0 - tnr):
        raise NonIdentifiableRateError(
            f"measured rate {measured:.4f} < false-positive floor (1-TNR)={1 - tnr:.4f}; "
            "the corrected rate would be negative -- not identifiable"
        )
    if measured > tpr:
        raise NonIdentifiableRateError(
            f"measured rate {measured:.4f} > true-positive ceiling TPR={tpr:.4f}; "
            "the corrected rate would exceed 1 -- not identifiable"
        )
    # Identifiable region [1-TNR, TPR] guarantees corrected in [0, 1]; no clamping needed.
    return (measured - (1.0 - tnr)) / denom


@dataclass(frozen=True)
class RGInterval:
    """The corrected-rate interval and what went into it.

    ``n_draws`` is the Monte Carlo draw count; ``n_nonidentifiable`` of those had
    ``TPR <= 1 - TNR`` and so no corrected rate. They are left out of the percentiles and counted
    here, never replaced by another quantity.
    """

    lo: float
    hi: float
    n_draws: int
    n_nonidentifiable: int
    eval_design_effect: float = 1.0


def cluster_design_effect(
    values: Sequence[int] | NDArray[np.integer], cluster_ids: Sequence[str]
) -> float:
    """Design effect of the mean of a 0/1 sample whose items come in clusters (spans of a trace).

    The clustered variance of the mean ``m = sum(y) / N`` treats each cluster as one draw (the
    ratio-estimator variance with ``C`` clusters, cluster FAIL counts ``y_c`` and sizes ``n_c``)::

        var_cl = C / (C - 1) * sum_c (y_c - m * n_c)^2 / N^2

    and the design effect is ``var_cl / (m (1 - m) / N)``, the factor by which clustering inflates
    the variance over independent items (Kish 1965, *Survey Sampling*). It is floored at 1: a sample
    whose clusters happen to balance is not credited with more information than independent items.
    A sample with ``m`` at 0 or 1 shows no variance to estimate the design effect from, so it gets
    the worst case ``N / C``: each cluster counts as one independent draw. Taking 1 there would
    read 0 FAILs in 1500 spans of 150 traces as 1500 independent clean items, an interval about ten
    times too tight exactly where a reader concludes the agent almost never fails.

    Raises :class:`StatisticsInputError` when the lengths differ or there are fewer than two
    clusters (one cluster has no between-cluster variance to estimate).
    """
    if len(values) != len(cluster_ids):
        raise StatisticsInputError(
            f"cluster_design_effect: {len(values)} values but {len(cluster_ids)} cluster ids"
        )
    k, n = _successes(values, "values")
    totals: dict[str, list[int]] = {}
    for v, c in zip(np.asarray(values, dtype=int).tolist(), cluster_ids, strict=True):
        cell = totals.setdefault(c, [0, 0])
        cell[0] += v
        cell[1] += 1
    clusters = len(totals)
    if clusters < 2:
        raise StatisticsInputError(
            f"cluster_design_effect: need at least 2 clusters for a variance, got {clusters}"
        )
    m = k / n
    if m in (0.0, 1.0):
        return n / clusters
    resid = sum((y - m * size) ** 2 for y, size in totals.values())
    var_cl = clusters / (clusters - 1) * resid / n**2
    return max(1.0, var_cl / (m * (1.0 - m) / n))


def rg_interval(
    cal_pos: Sequence[int] | NDArray[np.integer],
    cal_neg: Sequence[int] | NDArray[np.integer],
    eval_preds: Sequence[int] | NDArray[np.integer],
    *,
    conf: float = 0.95,
    n_resamples: int = 9999,
    seed: int = 0,
    eval_cluster_ids: Sequence[str] | None = None,
    cal_pos_design_effect: float = 1.0,
    cal_neg_design_effect: float = 1.0,
) -> RGInterval:
    """Two-sided interval for the Rogan-Gladen corrected rate from three independent samples.

    TPR, 1 - TNR and ``measured`` each get an independent Jeffreys posterior
    ``Beta(x + 1/2, n - x + 1/2)``. ``n_resamples`` Monte Carlo draws of the three are pushed
    through the Rogan-Gladen inversion, clipped into [0, 1], and the interval is the central
    ``conf`` percentile range of the result. ``seed`` fixes the draws.

    ``eval_cluster_ids`` -- when several eval verdicts come from one trace (a span- or
    process-level judge), the trace id of each verdict. The verdicts are then not independent, and
    the measured rate's posterior is taken at the effective sample size ``n / deff``, with ``deff``
    from :func:`cluster_design_effect` (the effective-sample-size approach of Korn and Graubard
    1998), so the interval widens by the clustering the sample shows.
    The design effect used is returned. ``cal_pos_design_effect`` and ``cal_neg_design_effect`` do
    the same for calibration gold whose items cluster by trace (the record's
    ``tpr_design_effect`` / ``tnr_design_effect``); at 1 the calibration items are independent.

    When the interval collapses to the single point 0 (or 1), almost every draw has the measured
    rate beyond the judge's false-positive floor (or true-positive ceiling).
    :class:`~calibrated_agent_evaluation.calibrate.errors.ErrorRatesDoNotTransferError` is raised
    then, since the sample contradicts the error rates measured on the calibration gold.

    A draw whose denominator ``TPR - (1 - TNR)`` is not positive has no corrected rate. Such draws
    are left out and counted. Because a left-out draw could have
    belonged in either tail, each tail's percentile moves outward by the left-out share ``f``:
    the bounds are the identifiable draws' quantiles at ``(a - f) / (1 - f)`` and
    ``(1 - a) / (1 - f)``, with ``a = (1 - conf) / 2``. When ``f > a`` a tail is not determined
    and :class:`NonIdentifiableRateError` is raised instead of an interval.

    A boundary count (no false positive, every positive caught, or a measured count of 0 or of
    every verdict) takes the limit it governs with that rate at its boundary, the modified
    Jeffreys rule described in the module docstring.

    Args:
        cal_pos: The judge's verdicts (1 = FAIL) on the hunted true-FAIL class; their mean is TPR.
        cal_neg: The judge's verdicts (1 = FAIL) on the hunted true-PASS class; ``1 - mean`` is TNR.
        eval_preds: The judge's verdicts on the separate rate sample; their mean is the measured
            rate.
        conf: The interval's two-sided coverage target.
        n_resamples: The number of Monte Carlo draws.
        seed: Fixes the draws.
        eval_cluster_ids: The trace id of each eval verdict, when one trace contributes several.
        cal_pos_design_effect: The design effect of the clustered true-FAIL calibration class.
        cal_neg_design_effect: The design effect of the clustered true-PASS calibration class.

    Returns:
        The interval, the share of draws with no corrected rate, and the design effect used.

    Raises:
        StatisticsInputError: A sample is empty or holds a value other than 0 or 1, or ``conf``
            or ``n_resamples`` is out of range.
        NonIdentifiableRateError: Too many draws have no corrected rate for a tail to be determined.
        ErrorRatesDoNotTransferError: The interval collapses to 0 or to 1.
    """
    if len(cal_pos) == 0 or len(cal_neg) == 0 or len(eval_preds) == 0:
        raise StatisticsInputError(
            "rg_interval: cal_pos, cal_neg, and eval_preds must all be non-empty"
        )
    if not 0.0 < conf < 1.0:
        raise StatisticsInputError(f"rg_interval: conf must be in (0, 1), got {conf}")
    if n_resamples < 1:
        raise StatisticsInputError(f"rg_interval: n_resamples must be >= 1, got {n_resamples}")
    for name, value in (
        ("cal_pos_design_effect", cal_pos_design_effect),
        ("cal_neg_design_effect", cal_neg_design_effect),
    ):
        if not value >= 1.0:
            raise StatisticsInputError(f"rg_interval: {name} must be >= 1, got {value}")
    tp, n1 = _successes(cal_pos, "cal_pos")
    fp, n0 = _successes(cal_neg, "cal_neg")
    k, n_eval = _successes(eval_preds, "eval_preds")
    deff = 1.0 if eval_cluster_ids is None else cluster_design_effect(eval_preds, eval_cluster_ids)

    rng = np.random.default_rng(seed)
    dp, dn = cal_pos_design_effect, cal_neg_design_effect
    tpr = rng.beta(tp / dp + 0.5, (n1 - tp) / dp + 0.5, n_resamples)
    fpr = rng.beta(fp / dn + 0.5, (n0 - fp) / dn + 0.5, n_resamples)
    measured = rng.beta(k / deff + 0.5, (n_eval - k) / deff + 0.5, n_resamples)
    tail = (1.0 - conf) / 2.0
    counts = f"TPR {tp}/{n1}, 1 - TNR {fp}/{n0}"
    lower, bad_lo = _tail_quantile(
        np.ones(n_resamples) if tp == n1 else tpr,
        fpr,
        np.zeros(n_resamples) if k == 0 else measured,
        tail,
        upper=False,
        counts=counts,
    )
    upper, bad_hi = _tail_quantile(
        tpr,
        np.zeros(n_resamples) if fp == 0 else fpr,
        np.ones(n_resamples) if k == n_eval else measured,
        tail,
        upper=True,
        counts=counts,
    )
    if upper <= 0.0 or lower >= 1.0:
        edge, side = (
            ("0", "below the judge's false-positive floor")
            if upper <= 0.0
            else (
                "1",
                "above the judge's true-positive ceiling",
            )
        )
        raise ErrorRatesDoNotTransferError(
            f"{TRANSFER_REFUSAL}: the measured rate {k}/{n_eval} sits "
            f"{side} ({counts}) in almost every draw, so the corrected rate collapses to exactly "
            f"{edge}. A zero-width interval would claim the true rate is exactly {edge}; what the "
            "data show is that the judge's error rates on the calibration gold do not hold here. "
            "Check that the rate sample comes from the population the judge was calibrated on, "
            "or calibrate the judge on gold drawn from this population"
        )
    return RGInterval(
        lower,
        upper,
        n_draws=n_resamples,
        n_nonidentifiable=max(bad_lo, bad_hi),
        eval_design_effect=deff,
    )


def _tail_quantile(
    tpr: NDArray[np.floating],
    fpr: NDArray[np.floating],
    measured: NDArray[np.floating],
    tail: float,
    *,
    upper: bool,
    counts: str,
) -> tuple[float, int]:
    """One limit of the corrected rate over the draws, and the non-identifiable draw count.

    A left-out draw could have belonged in either tail, so the percentile moves outward by the
    left-out share ``f``: ``(a - f) / (1 - f)`` for the lower limit and ``(1 - a) / (1 - f)`` for
    the upper, with ``a`` the tail. When ``f > a`` the tail is not determined.
    """
    denom = tpr - fpr
    ok = denom > 0
    n = denom.size
    n_bad = int(n - ok.sum())
    share = n_bad / n
    if share > tail:
        raise NonIdentifiableRateError(
            f"rg_interval: {n_bad} of {n} draws have TPR <= 1 - TNR, more than one tail's "
            f"{tail:.1%}; the interval is not determined ({counts})"
        )
    corrected = np.clip((measured[ok] - fpr[ok]) / denom[ok], 0.0, 1.0)
    q = (1.0 - tail) / (1.0 - share) if upper else (tail - share) / (1.0 - share)
    return float(np.quantile(corrected, q)), n_bad


def _successes(values: Sequence[int] | NDArray[np.integer], name: str) -> tuple[int, int]:
    """Return ``(count of 1s, length)`` of a 1/0 sample, refusing anything that is not 1/0."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or not np.all((arr == 0.0) | (arr == 1.0)):
        raise StatisticsInputError(f"rg_interval: {name} must be a flat sequence of 1/0 verdicts")
    return int(arr.sum()), int(arr.size)


def _check_prob(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise StatisticsInputError(f"{name} must be a probability in [0, 1], got {value}")
