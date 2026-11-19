# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input-distribution drift contracts.

The three drift statistics for **input-distribution** drift -- PSI, Jensen-Shannon
divergence, and Wasserstein distance -- deliberately NOT the KS test (univariate, continuous-only,
pathologically oversensitive at scale). A sibling ``calibrated_agent_evaluation.stats``
block (the ``good_turing``/``kish``, ``mcnemar`` precedent): pure functions, no vendor
trace type, no I/O.

  * :func:`psi` -- Population Stability Index over bins cut at the **baseline's** quantiles
    (deciles by default), adjacent bins merged until each holds at least 5 baseline values, and
    an add-0.5 count correction so an empty fresh bin adds a bounded amount. Refused below
    ``5 * bins`` values per side. The equal-width pooled-range bins with a 1e-6 proportion floor
    this replaced flagged a "major shift" on 99% of same-distribution draws at n=30.
  * :func:`quantile_histogram` -- the same baseline-quantile bin counts, for JSD over raw values.
  * :func:`jensen_shannon` -- the JS **divergence** (0..1, base 2) of two frequency/probability
    vectors -- ``scipy.spatial.distance.jensenshannon`` returns the JS *distance* (its square root),
    so this squares it back to the divergence the drift report reads.
  * :func:`wasserstein` -- the 1-D Wasserstein (earth-mover) distance between two raw-value samples
    (``scipy.stats.wasserstein_distance``, verified surface scipy==1.18), in the data's units.
  * :func:`wasserstein_normalized` -- the same distance divided by the pooled sample's IQR, so a
    threshold on it does not depend on the units.
  * :func:`permutation_p` / :func:`psi_permutation_p` -- a seeded permutation-test p-value for any
    of these statistics against its own null at the sample sizes in hand, since no fixed PSI or
    Wasserstein bar is calibrated for an arbitrary bin scheme and ``n``.

The scipy write paths raise a raw ``ValueError`` on an empty / non-numeric input and can yield
``nan`` on a degenerate (all-zero) vector. Per the "wrap external errors at the boundary" +
"no silent failure" conventions, every such case is caught and re-raised as the typed
:class:`~calibrated_agent_evaluation.calibrate.errors.StatisticsInputError` (the shared stats-input
error, reused not redefined) -- no raw scipy exception and no ``nan`` ever escapes this module.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from statsmodels.stats.proportion import confint_proportions_2indep

from calibrated_agent_evaluation.calibrate.errors import StatisticsInputError

# PSI's minimum baseline values per bin, and the per-side floor of MIN_BIN_COUNT * bins: the
# conventional minimum expected count of 5 for a chi-square-family statistic.
MIN_BIN_COUNT = 5
# The add-0.5 count correction applied to every PSI bin before taking proportions.
_PSI_SMOOTHING = 0.5
# Permutations for a drift p-value: p = (1 + #{null >= observed}) / (1 + n_perm).
DEFAULT_N_PERMUTATIONS = 999

ArrayStatistic = Callable[[np.ndarray, np.ndarray], float]


def _as_float_array(values: Sequence[float], *, name: str) -> np.ndarray:
    """Coerce a sequence to a 1-D float array, raising a typed error on empty / non-numeric input.

    Wraps the ``ValueError`` / ``TypeError`` numpy raises on a non-numeric element so no raw stdlib
    exception escapes the drift boundary.
    """
    if len(values) == 0:
        raise StatisticsInputError(f"drift: {name} must be non-empty")
    try:
        arr = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise StatisticsInputError(
            f"drift: {name} must be all-numeric, got a non-numeric element ({exc})"
        ) from exc
    if not np.all(np.isfinite(arr)):
        raise StatisticsInputError(f"drift: {name} contains NaN or infinite values")
    return arr


def _finite(value: float, *, metric: str) -> float:
    """Return ``value`` as a float, or raise if it is NaN/infinite (no NaN leaks)."""
    if not math.isfinite(value):
        raise StatisticsInputError(
            f"drift: {metric} produced a non-finite value ({value}); refusing to report a "
            "spurious drift number"
        )
    return float(value)


def _bin_counts(base: np.ndarray, fut: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Counts of both samples over bins cut at the baseline's quantiles, sparse bins merged.

    Interior edges are the baseline's ``1/bins .. (bins-1)/bins`` quantiles (duplicates from tied
    values collapse); the outer bins are open, so a fresh value outside the baseline's range still
    lands in a bin. Adjacent bins are merged left to right until each holds ``MIN_BIN_COUNT``
    baseline values, and a short remainder joins the last bin.
    """
    edges = np.unique(np.quantile(base, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    k = len(edges) + 1
    base_raw = np.bincount(np.searchsorted(edges, base, side="right"), minlength=k)
    fresh_raw = np.bincount(np.searchsorted(edges, fut, side="right"), minlength=k)
    base_out: list[int] = []
    fresh_out: list[int] = []
    acc_b = acc_f = 0
    for b, f in zip(base_raw.tolist(), fresh_raw.tolist(), strict=True):
        acc_b += b
        acc_f += f
        if acc_b >= MIN_BIN_COUNT:
            base_out.append(acc_b)
            fresh_out.append(acc_f)
            acc_b = acc_f = 0
    if acc_b or acc_f:
        if base_out:
            base_out[-1] += acc_b
            fresh_out[-1] += acc_f
        else:
            base_out.append(acc_b)
            fresh_out.append(acc_f)
    return np.asarray(base_out, dtype=float), np.asarray(fresh_out, dtype=float)


def _check_bins(bins: int, base: np.ndarray, fut: np.ndarray, *, metric: str) -> None:
    if bins < 2:
        raise StatisticsInputError(f"drift: {metric} bins must be >= 2, got {bins}")
    floor = MIN_BIN_COUNT * bins
    if min(len(base), len(fut)) < floor:
        raise StatisticsInputError(
            f"drift: {metric} over {bins} bins needs at least {floor} values per side (5 per "
            f"bin), got {len(base)} baseline and {len(fut)} fresh; pass fewer bins or more data "
            "rather than read a number the bins cannot support"
        )


def _psi_arrays(base: np.ndarray, fut: np.ndarray, bins: int) -> float:
    base_counts, fresh_counts = _bin_counts(base, fut, bins)
    k = len(base_counts)
    b = (base_counts + _PSI_SMOOTHING) / (base_counts.sum() + _PSI_SMOOTHING * k)
    f = (fresh_counts + _PSI_SMOOTHING) / (fresh_counts.sum() + _PSI_SMOOTHING * k)
    return float(np.sum((f - b) * np.log(f / b)))


def psi(baseline: Sequence[float], fresh: Sequence[float], *, bins: int = 10) -> float:
    """Population Stability Index between two raw-value samples over baseline-quantile bins.

    ``PSI = sum_i (f_i - b_i) * ln(f_i / b_i)`` over the bin proportions ``b`` (baseline) and
    ``f`` (fresh), each computed from counts plus 0.5. Bins are the baseline's quantiles (deciles
    for ``bins=10``), merged until each holds 5 baseline values. ``0`` for identical samples.

    Raises :class:`StatisticsInputError` on an empty / non-numeric / non-finite input, on
    ``bins < 2``, when either side has fewer than ``5 * bins`` values, or on a non-finite result.
    """
    base = _as_float_array(baseline, name="baseline")
    fut = _as_float_array(fresh, name="fresh")
    _check_bins(bins, base, fut, metric="psi")
    return _finite(_psi_arrays(base, fut, bins), metric="psi")


def quantile_histogram(
    baseline: Sequence[float], fresh: Sequence[float], *, bins: int = 10
) -> tuple[list[float], list[float]]:
    """Both samples' counts over the PSI bins (baseline quantiles, sparse bins merged).

    The aligned frequency vectors JSD reads when only raw values are available. Refuses the same
    inputs :func:`psi` refuses.
    """
    base = _as_float_array(baseline, name="baseline")
    fut = _as_float_array(fresh, name="fresh")
    _check_bins(bins, base, fut, metric="quantile_histogram")
    b, f = _bin_counts(base, fut, bins)
    return b.tolist(), f.tolist()


def _jsd_arrays(pa: np.ndarray, qa: np.ndarray) -> float:
    try:
        distance = float(jensenshannon(pa, qa, base=2))
    except (ValueError, TypeError) as exc:
        raise StatisticsInputError(
            f"drift: jensen_shannon failed on the given vectors ({exc})"
        ) from exc
    return distance * distance


def jensen_shannon(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon **divergence** (0..1, base 2) of two frequency/probability vectors.

    ``scipy.spatial.distance.jensenshannon`` returns the JS *distance* (the square root of the
    divergence); this squares it back so the drift report reads a divergence bounded in ``[0, 1]``.
    ``p`` and ``q`` need not be normalized -- ``jensenshannon`` normalizes each to a probability
    vector -- but they must be the same length and not both all-zero (which yields NaN).

    Raises :class:`StatisticsInputError` on an empty / mismatched / non-numeric input or a
    non-finite result (an all-zero vector) -- never a raw exception, never a NaN.
    """
    pa = _as_float_array(p, name="p")
    qa = _as_float_array(q, name="q")
    if pa.shape != qa.shape:
        raise StatisticsInputError(
            f"drift: jensen_shannon needs equal-length vectors, got {pa.shape[0]} and {qa.shape[0]}"
        )
    if pa.sum() <= 0 or qa.sum() <= 0:
        raise StatisticsInputError(
            "drift: jensen_shannon needs a positive-mass vector on each side (an all-zero "
            "distribution has no probability mass)"
        )
    return _finite(_jsd_arrays(pa, qa), metric="jensen_shannon")


def jsd_of_samples(baseline: np.ndarray, fresh: np.ndarray, *, bins: int = 10) -> float:
    """JSD of two raw-value samples over their PSI bins (the statistic the report permutes)."""
    b, f = _bin_counts(baseline, fresh, bins)
    return _jsd_arrays(b, f)


def wasserstein(baseline: Sequence[float], fresh: Sequence[float]) -> float:
    """1-D Wasserstein (earth-mover) distance between two raw-value samples, in the data's units.

    Wraps ``scipy.stats.wasserstein_distance`` (verified surface). ``0`` for identical samples,
    growing with the shift in location/shape. Use :func:`wasserstein_normalized` for anything
    thresholded. Raises :class:`StatisticsInputError` on an empty / non-numeric / non-finite input
    or a non-finite result -- never a raw scipy exception.
    """
    base = _as_float_array(baseline, name="baseline")
    fut = _as_float_array(fresh, name="fresh")
    try:
        value = float(wasserstein_distance(base, fut))
    except (ValueError, TypeError) as exc:
        raise StatisticsInputError(
            f"drift: wasserstein failed on the given samples ({exc})"
        ) from exc
    return _finite(value, metric="wasserstein")


def wasserstein_normalized_arrays(base: np.ndarray, fut: np.ndarray) -> float:
    """:func:`wasserstein_normalized` over validated float arrays (the permuted statistic)."""
    pooled = np.concatenate([base, fut])
    q75, q25 = np.percentile(pooled, [75.0, 25.0])
    scale = float(q75 - q25)
    if scale <= 0.0:
        scale = float(np.std(pooled))
    if scale <= 0.0:  # every value identical: no spread, no shift
        return 0.0
    return float(wasserstein_distance(base, fut)) / scale


def wasserstein_normalized(baseline: Sequence[float], fresh: Sequence[float]) -> float:
    """Wasserstein distance divided by the pooled sample's interquartile range.

    Scale-free: multiplying both samples by a constant leaves it unchanged, so one threshold means
    the same thing for a [0, 1] score and for a token count. Falls back to the pooled standard
    deviation when the IQR is 0, and returns 0.0 when every value is identical.
    """
    base = _as_float_array(baseline, name="baseline")
    fut = _as_float_array(fresh, name="fresh")
    try:
        value = wasserstein_normalized_arrays(base, fut)
    except (ValueError, TypeError) as exc:
        raise StatisticsInputError(
            f"drift: wasserstein_normalized failed on the given samples ({exc})"
        ) from exc
    return _finite(value, metric="wasserstein_normalized")


def permutation_p(
    statistic: ArrayStatistic,
    baseline: Sequence[float],
    fresh: Sequence[float],
    *,
    n_perm: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 0,
) -> float:
    """One-sided permutation p-value that ``statistic(baseline, fresh)`` is no larger than chance.

    The pooled sample is relabelled ``n_perm`` times with ``numpy.random.default_rng(seed)``
    (group sizes kept), and ``p = (1 + #{null >= observed}) / (1 + n_perm)``, which is never 0 and
    is a valid p-value at any ``n``. ``statistic`` receives float arrays and returns a float.

    Raises :class:`StatisticsInputError` on empty / non-numeric input or ``n_perm < 1``.
    """
    if n_perm < 1:
        raise StatisticsInputError(f"drift: permutation_p needs n_perm >= 1, got {n_perm}")
    base = _as_float_array(baseline, name="baseline")
    fut = _as_float_array(fresh, name="fresh")
    observed = _finite(statistic(base, fut), metric="permutation statistic")
    pooled = np.concatenate([base, fut])
    nb = len(base)
    rng = np.random.default_rng(seed)
    # A relative tolerance so a relabelling that reproduces the observed value counts as >=.
    bar = observed - 1e-12 * max(1.0, abs(observed))
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if statistic(perm[:nb], perm[nb:]) >= bar:
            hits += 1
    return (1 + hits) / (1 + n_perm)


def psi_permutation_p(
    baseline: Sequence[float],
    fresh: Sequence[float],
    *,
    bins: int = 10,
    n_perm: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 0,
) -> float:
    """Permutation p-value for :func:`psi`, recutting the bins on each relabelled baseline.

    Refuses the inputs :func:`psi` refuses.
    """
    base = _as_float_array(baseline, name="baseline")
    fut = _as_float_array(fresh, name="fresh")
    _check_bins(bins, base, fut, metric="psi")
    return permutation_p(
        lambda b, f: _psi_arrays(b, f, bins), baseline, fresh, n_perm=n_perm, seed=seed
    )


def delta_proportion_ci(
    count_after: int,
    nobs_after: int,
    count_before: int,
    nobs_before: int,
    *,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """CI for the change in a proportion (after - before), for judge-drift ΔTPR/ΔTNR.

    Returns ``(point, lo, hi)`` where ``point`` is ``count_after/nobs_after -
    count_before/nobs_before`` and ``(lo, hi)`` is the 95%-by-default CI for the difference of two
    independent proportions (``statsmodels.stats.proportion.confint_proportions_2indep``, verified
    surface -- a difference-of-proportions CI, not a novel statistic). A zero change yields a CI
    straddling 0. Raises :class:`StatisticsInputError` on a non-positive denominator or an
    out-of-range count rather than dividing by zero or letting a raw statsmodels error escape.
    """
    if nobs_after <= 0 or nobs_before <= 0:
        raise StatisticsInputError(
            "drift: delta_proportion_ci needs a positive denominator on each side"
        )
    if not (0 <= count_after <= nobs_after and 0 <= count_before <= nobs_before):
        raise StatisticsInputError(
            "drift: delta_proportion_ci counts must lie in [0, nobs] on each side"
        )
    point = count_after / nobs_after - count_before / nobs_before
    try:
        lo, hi = confint_proportions_2indep(
            count_after, nobs_after, count_before, nobs_before, compare="diff", alpha=alpha
        )
    except (ValueError, ZeroDivisionError) as exc:
        raise StatisticsInputError(
            f"drift: delta_proportion_ci failed on the given counts ({exc})"
        ) from exc
    return (
        _finite(point, metric="delta"),
        _finite(float(lo), metric="delta_lo"),
        _finite(float(hi), metric="delta_hi"),
    )
