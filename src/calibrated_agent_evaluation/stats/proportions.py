# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wilson score confidence bounds on a proportion.

The calibration gate is a **lower-bound** gate: a judge is certified only if the 95% *one-sided*
lower confidence bound on both TPR and TNR clears the threshold -- a point
estimate at the threshold does not certify. This module wraps the verified
``statsmodels.stats.proportion.proportion_confint`` surface.

One-sided lower bound: ``proportion_confint(k, n, alpha=0.10, method="wilson")`` returns a
two-sided 90% interval whose LOW element is exactly the 95% one-sided lower bound. Verified equal
to a hand-computed Wilson score lower bound (z_0.95) to 1e-9 at k=72, n=80 -> 0.831099. The
statsmodels default ``method`` is the Wald interval -- we ALWAYS pass ``method=`` explicitly so the
gate never silently uses the wrong (anticonservative) bound.
"""

from __future__ import annotations

from statsmodels.stats.proportion import proportion_confint

from calibrated_agent_evaluation.calibrate.errors import StatisticsInputError


def wilson_lower(k: float, n: float, alpha: float = 0.10) -> float:
    """Return the one-sided 95%-by-default Wilson score lower bound for ``k`` of ``n`` successes.

    ``alpha`` is the two-sided level passed to ``proportion_confint``; the returned value is that
    interval's lower edge, which equals the one-sided lower bound at confidence ``1 - alpha/2``
    (so the default ``alpha=0.10`` gives the 95% one-sided lower bound the gate uses).

    Raises :class:`StatisticsInputError` on ``n <= 0`` or ``k`` outside ``[0, n]`` -- a proportion
    from a non-count is a caller bug, surfaced as a typed error rather than a NaN from statsmodels.
    ``k`` and ``n`` may be fractional effective counts (``k / deff`` of ``n / deff``) for clustered
    items.

    Args:
        k: The number of successes.
        n: The number of trials.
        alpha: The two-sided level; 0.10 gives the one-sided 95% bound.

    Returns:
        The lower bound, in [0, 1].

    Raises:
        StatisticsInputError: ``n <= 0`` or ``k`` is outside ``[0, n]``.
    """
    if n <= 0:
        raise StatisticsInputError(f"wilson_lower: n must be positive, got {n}")
    if not 0 <= k <= n:
        raise StatisticsInputError(f"wilson_lower: k={k} must lie in [0, n={n}]")
    if not 0.0 < alpha < 1.0:
        raise StatisticsInputError(f"wilson_lower: alpha must be in (0, 1), got {alpha}")
    low = proportion_confint(k, n, alpha=alpha, method="wilson")[0]
    # Clamp the theoretical [0,1] range: statsmodels can return a tiny negative near k=0.
    return max(0.0, float(low))


def wilson_upper(k: int, n: int, alpha: float = 0.10) -> float:
    """Return the one-sided 95%-by-default Wilson score upper bound for ``k`` of ``n`` successes.

    The mirror of :func:`wilson_lower`, reading the same Wilson score interval's upper edge at the
    same ``alpha`` convention.

    Which edge is conservative depends on which direction is *bad*, so the two are not
    interchangeable. A rate that must clear a **floor** (TPR, TNR: large is good) is gated on the
    lower bound. A rate that must stay under a **ceiling** (bias magnitude: small is good) is gated
    on the upper bound; gating a ceiling on the lower bound reads as "promote unless we are
    confident the judge IS biased", which inverts the burden of proof a safety gate exists to hold.

    Raises :class:`StatisticsInputError` on ``n <= 0`` or ``k`` outside ``[0, n]``.
    """
    if n <= 0:
        raise StatisticsInputError(f"wilson_upper: n must be positive, got {n}")
    if not 0 <= k <= n:
        raise StatisticsInputError(f"wilson_upper: k={k} must lie in [0, n={n}]")
    if not 0.0 < alpha < 1.0:
        raise StatisticsInputError(f"wilson_upper: alpha must be in (0, 1), got {alpha}")
    high = proportion_confint(k, n, alpha=alpha, method="wilson")[1]
    # Clamp the theoretical [0,1] range: statsmodels can overshoot 1.0 near k=n.
    return min(1.0, float(high))


def newcombe_paired_diff(
    a: int, b: int, c: int, d: int, *, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Newcombe's (1998) hybrid-score interval for a paired difference of proportions (method 10).

    For the paired 2x2 table ``[[a, b], [c, d]]`` with ``p1 = (a + b)/n`` and ``p2 = (a + c)/n``,
    returns ``(p1 - p2, lower, upper)`` at two-sided level ``1 - alpha``. The limits combine the
    two marginal Wilson score intervals ``(l1, u1)``, ``(l2, u2)`` with a correlation correction::

        lower = (p1 - p2) - sqrt((p1 - l1)^2 - 2 phi (p1 - l1)(u2 - p2) + (u2 - p2)^2)
        upper = (p1 - p2) + sqrt((u1 - p1)^2 - 2 phi (u1 - p1)(p2 - l2) + (p2 - l2)^2)

    where ``phi = max(ad - bc - n/2, 0) / sqrt((a+b)(c+d)(a+c)(b+d))`` when ``ad > bc``, else
    ``(ad - bc) / sqrt(...)``, and 0 when the denominator is 0 (Newcombe 1998, Statistics in
    Medicine 17:2635-2650, p. 2639; as tabulated in the NCSS "Two Correlated Proportions"
    procedure). Unlike the matched-pairs Wald interval it does not collapse to zero width when a
    discordant cell is empty, and it stays inside ``[-1, 1]``.

    Raises :class:`StatisticsInputError` on a negative cell or an empty table.

    Args:
        a: Pairs where both observations are successes.
        b: Pairs where only the first is a success.
        c: Pairs where only the second is a success.
        d: Pairs where neither is a success.
        alpha: The two-sided level of the interval.

    Returns:
        ``(difference, lower, upper)`` for ``p1 - p2``.

    Raises:
        StatisticsInputError: A cell is negative or the table is empty.
    """
    if min(a, b, c, d) < 0:
        raise StatisticsInputError(f"newcombe_paired_diff: cells must be >= 0, got {(a, b, c, d)}")
    n = a + b + c + d
    if n == 0:
        raise StatisticsInputError("newcombe_paired_diff: the table is empty")
    x1, x2 = a + b, a + c
    p1, p2 = x1 / n, x2 / n
    l1, u1 = wilson_lower(x1, n, alpha=alpha), wilson_upper(x1, n, alpha=alpha)
    l2, u2 = wilson_lower(x2, n, alpha=alpha), wilson_upper(x2, n, alpha=alpha)
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        if num > 0:
            num = max(num - n / 2.0, 0.0)
        phi = num / denom**0.5
    delta = ((p1 - l1) ** 2 - 2.0 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2) ** 0.5
    eps = ((u1 - p1) ** 2 - 2.0 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2) ** 0.5
    diff = p1 - p2
    return diff, max(-1.0, diff - delta), min(1.0, diff + eps)
