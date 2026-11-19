# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saturation estimators: Good-Turing sample coverage and Chao1 richness, each with its interval.

``counts`` is the per-failure-mode observation count. A zero entry is a mode with no observed
member, which is not an observed mode, so every estimator drops zeros before counting ``S_obs``
or singletons.

  * :func:`good_turing_coverage` -- the Good-Turing coverage point ``C = 1 - f1/n`` (Good 1953).
  * :func:`good_turing_coverage_bound` -- the same point with its one-sided 95% lower bound from
    Esty's (1983) normal approximation, ``Var(C) ~ (f1/n (1 - f1/n) + 2 f2/n) / n``, with ``f1``
    floored at 1 in the variance so a sample without singletons still has one.
  * :func:`chao1_estimate` -- the Chao1 lower-bound richness estimator (Chao 1984): classic
    ``S_obs + f1^2 / (2 f2)`` when ``f2 > 0``, bias-corrected ``S_obs + f1 (f1 - 1) / 2`` when
    ``f2 = 0``. It was once named ``chapman_estimate``; the Chapman
    (1951) mark-recapture estimator is a different statistic and is not implemented.
  * :func:`chao1_interval` -- Chao's (1987) log-transformed 95% interval for Chao1, with the
    variance and the ``f1 <= 1`` special case as tabulated in the EstimateS User's Guide
    (Colwell 2013, Appendix B, eqs. 5, 7, 8, 13, 14), without the ``(n - 1)/n`` factor because the
    point estimator here does not carry it either.
  * :func:`saturation_check` -- the verdict, gated on the coverage *lower bound* (never the point)
    and refusing a single-mode collapse.
"""

from __future__ import annotations

import math

from calibrated_agent_evaluation.calibrate.record import Bound, Interval
from calibrated_agent_evaluation.cluster.errors import ClusteringError
from calibrated_agent_evaluation.cluster.types import SaturationResult

METHOD_ESTY = "good_turing_esty_normal"
METHOD_CHAO1_LOG = "chao1_log_transformed"

# One-sided 95% (coverage lower bound) and two-sided 95% (Chao1 interval) normal quantiles.
_Z_ONE_SIDED_95 = 1.6448536269514722
_Z_TWO_SIDED_95 = 1.959963984540054


def _validate_counts(counts: list[int]) -> list[int]:
    """Validate ``counts`` and return it with the zero (unobserved) modes dropped."""
    if not isinstance(counts, list):
        raise ClusteringError(
            f"counts must be a list of non-negative integers; got {type(counts).__name__}"
        )
    for i, c in enumerate(counts):
        if not isinstance(c, int) or isinstance(c, bool):
            raise ClusteringError(
                f"counts must be a list of non-negative integers; "
                f"element {i} is {type(c).__name__}={c!r}"
            )
        if c < 0:
            raise ClusteringError(f"counts must be non-negative integers; element {i} is {c}")
    return [c for c in counts if c > 0]


def good_turing_coverage(counts: list[int]) -> float:
    """Good-Turing coverage point estimate: ``1 - singletons/total``.

    Returns a float in [0.0, 1.0], and 0.0 when nothing was observed. Use
    :func:`good_turing_coverage_bound` for the number a gate may read.
    """
    observed = _validate_counts(counts)
    total = sum(observed)
    if total == 0:
        return 0.0
    singletons = sum(1 for c in observed if c == 1)
    return 1.0 - (singletons / total)


def good_turing_coverage_bound(counts: list[int]) -> Bound:
    """Good-Turing coverage with its one-sided 95% lower bound (Esty 1983).

    ``lb95 = C - z_0.95 * sqrt((f1'/n (1 - f1'/n) + 2 f2/n) / n)``, clipped to [0, 1], with
    ``f1' = max(f1, 1)`` in the variance. Without the floor, no singletons and no doubletons give a
    variance of zero and a bound equal to the point 1.0, so 30 labels split [15, 15] read as proven
    saturation; a sample that saw no singleton still cannot rule out an unseen mode, the same
    reasoning behind Chao's ``f1 (f1 - 1)`` form when ``f2 = 0``.

    Raises :class:`ClusteringError` when nothing was observed: there is no coverage to bound.
    """
    observed = _validate_counts(counts)
    n = sum(observed)
    if n == 0:
        raise ClusteringError("good_turing_coverage_bound: no observations, nothing to bound")
    f1 = sum(1 for c in observed if c == 1)
    f2 = sum(1 for c in observed if c == 2)
    point = 1.0 - f1 / n
    f1_var = max(f1, 1)
    var = (f1_var / n * (1.0 - f1_var / n) + 2.0 * f2 / n) / n
    lb = min(1.0, max(0.0, point - _Z_ONE_SIDED_95 * math.sqrt(max(var, 0.0))))
    return Bound(point=point, lb95=lb, n=n, method=METHOD_ESTY)


def chao1_estimate(counts: list[int]) -> tuple[float, int, float]:
    """Chao1 lower-bound species-richness estimator (Chao 1984).

    Returns ``(estimated_total_modes, modes_found, fraction_discovered)``. Zero-count modes are not
    observed modes and do not enter ``modes_found``.
    """
    observed = _validate_counts(counts)
    if not observed:
        return (0.0, 0, 0.0)

    s_obs = len(observed)
    n1 = sum(1 for c in observed if c == 1)  # singletons
    n2 = sum(1 for c in observed if c == 2)  # doubletons

    if n1 == 0:
        # no singletons -- all modes observed >=2x, no unseen predicted
        estimated = float(s_obs)
    elif n2 == 0:
        # Bias-corrected form when no doubletons
        estimated = s_obs + (n1 * (n1 - 1)) / 2.0
    else:
        estimated = s_obs + (n1 * n1) / (2.0 * n2)

    fraction = s_obs / estimated if estimated > 0 else 1.0
    return (estimated, s_obs, fraction)


def _var_s_obs(observed: list[int]) -> tuple[float, float]:
    """EstimateS eq. 8: ``Var(S_obs)`` and ``P = sum_i F_i e^-i / S_obs`` (the ``T = 0`` case)."""
    n = sum(observed)
    s_obs = len(observed)
    var = sum(math.exp(-c) - math.exp(-2 * c) for c in observed)
    var -= sum(c * math.exp(-c) for c in observed) ** 2 / n
    p = sum(math.exp(-c) for c in observed) / s_obs
    return max(var, 0.0), p


def chao1_interval(counts: list[int]) -> Interval:
    """Chao1 with Chao's (1987) log-transformed two-sided 95% interval.

    With ``T = Chao1 - S_obs`` and ``K = exp(z sqrt(log(1 + Var/T^2)))`` the interval is
    ``[S_obs + T/K, S_obs + T*K]`` (eq. 13), so it never falls below ``S_obs``. The variance is
    eq. 5 (``f2 > 0``) or eq. 7 (``f2 = 0``). When ``T = 0`` (``f1 = 0``, or ``f1 = 1`` with
    ``f2 = 0``) the log transform is undefined and eq. 14 is used instead, built on ``Var(S_obs)``.

    Raises :class:`ClusteringError` when nothing was observed.
    """
    observed = _validate_counts(counts)
    if not observed:
        raise ClusteringError("chao1_interval: no observed modes, nothing to estimate")
    estimated, s_obs, _ = chao1_estimate(observed)
    n = sum(observed)
    f1 = sum(1 for c in observed if c == 1)
    f2 = sum(1 for c in observed if c == 2)
    t = estimated - s_obs
    if t > 0:
        if f2 > 0:
            r = f1 / f2
            var = f2 * (0.5 * r**2 + r**3 + 0.25 * r**4)
        else:
            var = f1 * (f1 - 1) / 2.0 + f1 * (2 * f1 - 1) ** 2 / 4.0 - f1**4 / (4.0 * estimated)
        k = math.exp(_Z_TWO_SIDED_95 * math.sqrt(math.log(1.0 + max(var, 0.0) / (t * t))))
        lo, hi = s_obs + t / k, s_obs + t * k
    else:
        var_obs, p = _var_s_obs(observed)
        centre = s_obs / (1.0 - p)
        half = _Z_TWO_SIDED_95 * math.sqrt(var_obs) / (1.0 - p)
        lo, hi = max(float(s_obs), centre - half), centre + half
    return Interval(
        point=estimated,
        lo=lo,
        hi=max(hi, estimated),
        method=METHOD_CHAO1_LOG,
        level=0.95,
        sided="two-sided",
        n=n,
    )


def saturation_check(
    counts: list[int],
    *,
    min_failures: int = 30,
    coverage_threshold: float = 0.95,
    min_modes: int = 2,
) -> SaturationResult:
    """Full saturation verdict.

    Saturated only when ``total >= min_failures``, at least ``min_modes`` distinct modes were
    observed, and the Good-Turing coverage one-sided 95% **lower bound** clears
    ``coverage_threshold``. A point estimate at the threshold does not certify, and a single mode
    holding every observation (coverage exactly 1.0, nothing learned about the landscape) is never
    saturated. Zero-count modes are dropped first.

    Args:
        counts: How many labeled failures each failure mode has.
        min_failures: The fewest labeled failures that can count as saturated.
        coverage_threshold: The lower bound the Good-Turing coverage must clear.
        min_modes: The fewest distinct modes that can count as saturated.

    Returns:
        The verdict, the coverage bound and the Chao1 estimate of the total number of modes.
    """
    observed = _validate_counts(counts)

    total = sum(observed)
    n_modes = len(observed)
    singletons = sum(1 for c in observed if c == 1)
    coverage = (
        good_turing_coverage_bound(observed)
        if total > 0
        else Bound(point=0.0, lb95=0.0, n=0, method=METHOD_ESTY)
    )
    chao1 = chao1_interval(observed) if observed else None

    def _result(saturated: bool, reason: str | None) -> SaturationResult:
        return SaturationResult(
            saturated=saturated,
            coverage=coverage,
            total_failures=total,
            total_modes=n_modes,
            singletons=singletons,
            reason=reason,
            chao1=chao1,
        )

    if total < min_failures:
        return _result(
            False, f"insufficient_observations: {total} < {min_failures}, the minimum failure count"
        )

    if n_modes < min_modes:
        return _result(
            False,
            f"too_few_modes: {n_modes} observed mode(s) < {min_modes}; every observation fell in "
            "one cluster, which says nothing about the unseen modes",
        )

    if coverage.lb95 < coverage_threshold:
        return _result(
            False,
            f"coverage_below_threshold: Good-Turing coverage lower bound {coverage.lb95:.4f} "
            f"(point {coverage.point:.4f}) < {coverage_threshold} ({singletons} singletons in "
            f"{total} observations)",
        )

    return _result(True, None)
