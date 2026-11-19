# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibration sample-size planner.

*Before* an Owner hunts and labels traces, tell them how many per class they need. Two jobs set
that number, and the planner returns the larger:

* **Certifying the judge.** The gate certifies only when the 95% one-sided Wilson lower bound
  clears the threshold ``tau``, so :func:`wilson_calibration_size` inverts that same bound: the
  smallest ``n`` from which a judge right ``p_hat`` of the time (count rounded down) clears
  ``tau`` at that size and every larger one. At ``p_hat=0.90, tau=0.80`` this is **55 per class**
  (``wilson_lower(49, 55) ~= 0.803``; ``wilson_lower(48, 54) ~= 0.799`` falls short). The bound
  first clears at 48, but 51 to 54 fall short again, so a plan of 48 would let more labels
  un-certify a judge as good as planned.
* **Estimating the rate.** The corrected rate's interval is computed from the same per-class
  samples, and its coverage depends on how many errors the judge makes in them, not on ``tau``.
  :func:`coverage_floor` is the smallest per-class ``n`` at which that interval was measured to
  cover nominally.

The floor flattens the certification size's steep fall with ``p_hat``, and above 0.92 reverses
it: 28 per class certifies a 0.95 judge, but the interval at 28 per class overcovers (up to 0.986
across the coverage study's cells), and the first size that covers nominally there and at every
larger scanned size is 80. A more accurate judge makes fewer errors per class, so its error
rates are estimated from smaller counts.

A judge expected to be more accurate on one class than the other (``tnr_hat`` differs from
``p_hat``) is planned with the certification size at the lower of its two rates, since the gate
certifies both, and the floor at the higher, since the class with fewer errors is the one whose
rate is estimated from the smaller count. Both classes are hunted at that size. The rule was
checked by simulation rather than assumed: at the planned sizes, over every ordered pair of
distinct rates from 0.85, 0.88, 0.90, 0.92, 0.95 and 0.98 crossed with the coverage study's
prevalences and eval sizes (840 cells, 2000 replicates, the shipped interval), no cell is off
0.95 after Holm (``fixtures/expected/coverage/rg_interval_coverage_asymmetric.json``). Taking the
floor at the lower rate instead leaves 5 of the 840 cells off nominal, mostly over-covering.

A certified record is rate-unsafe when its gold is below the floor at the accuracy the judge
showed, not only the planned one. A judge that certifies at the planned size may
show any accuracy that clears the gate there, and at 55 per class that is 49 to 55 correct, whose
floors run from 11 to 162. A caller who wants a corrected rate plans with
``rate_safe=True``: :func:`rate_safe_calibration_size` is the smallest size at
which every certifying accuracy with a measured floor meets it, 162 per class at every planned
accuracy from 0.88 to 0.99 (at 0.85 certifying alone needs 208).

Gold that two or more raters will double-code has a third job: the raters' agreement gate needs
:data:`~calibrated_agent_evaluation.review.gate.MINIMUM_OVERLAP` shared traces (130) before it can
call the gold usable, and the traces to double-code are the calibration gold itself, balanced per
class. A plan made with ``raters=2`` or more is therefore at least :func:`double_coded_per_class`
(65 per class), so the gold that certifies the judge is also enough to measure its raters. One
rater's plan is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from calibrated_agent_evaluation.calibrate.errors import InfeasiblePlanError
from calibrated_agent_evaluation.review.gate import MINIMUM_OVERLAP
from calibrated_agent_evaluation.stats.proportions import wilson_lower

# A generous upper bound on the search: at p_hat only marginally above tau the required n can grow
# large, but if it exceeds this the target is effectively infeasible for a practical hunt.
_MAX_N_PER_CLASS = 100_000

# The coverage floor per measured expected accuracy. Derived by
# calibrated_agent_evaluation.stats.studies.planner_floor from
# fixtures/expected/coverage/planner_floor_scan.json: rg_interval's coverage at TPR = TNR = p_hat
# (an asymmetric judge reads the floor at its higher rate; see the module docstring) over prevalence
# 0.02-0.50 and eval sizes 100 to 100,000, 1000 replicates per cell. An entry is the smallest
# scanned per-class size whose configuration, and every larger scanned one, shows no cell outside
# the acceptance band (0.9405-0.9595) after Holm across the configuration's cells. None: no scanned
# size up to 260 qualifies. tests/test_calibrate_plan.py checks this table against the derived file.
#
# Known limit: rescanned at 3000 replicates per cell (seeds 20261014-20261016), the interval
# under-covers where a class shows about 2 to 3 errors (down to 0.917), most visibly with large rate
# samples at low prevalence, including rows this table admits (0.98 at 130, 0.99 at 260).
# fixtures/expected/coverage/planner_floor_uniform.json holds that evidence; the table is kept at
# its 1000-replicate derivation and the limit is disclosed with the rate rather than hidden.
#
# The rows above 0.99 come from a separate derivation: planner_floor.py over
# fixtures/expected/coverage/planner_floor_boundary_scan.json, written to
# planner_floor_boundary.json.
# Those configurations (symmetric 0.995, 0.999 and 1.0, and TPR 0.95 with TNR 0.995 to 1.0) were
# scanned with the boundary rule at seeds 20261014-20261016, 3000 replicates per cell pooled, and
# tested for under-coverage only: the boundary rule widens the interval there by design, so
# over-coverage is accepted. Every scanned size from 11 to 800 passes.
COVERAGE_FLOOR: tuple[tuple[float, int | None], ...] = (
    (0.85, 28),
    (0.88, 28),
    (0.90, 35),
    (0.92, 80),
    (0.95, 80),
    (0.98, 130),
    (0.99, 162),
    (0.995, 11),
    (0.999, 11),
    (1.00, 11),
)


# The smallest rate sample a corrected rate is computed from. Derived by
# calibrated_agent_evaluation.stats.studies.rate_sample_minimum from the 3000-replicate scans: over
# every cell the coverage floor admits, a Holm-adjusted one-sided test for under-coverage rejects
# one cell at 100 traces (0.917 at prevalence 0.02) and none from 150 to 1000 at any prevalence 0.02
# to 0.50. The under-coverage at larger samples is the calibration-side limit above and does not
# move with more traces. Prevalence below 0.02 was not measured.
MIN_RATE_SAMPLE = 150


def wilson_calibration_size(p_hat: float, tau: float, alpha: float = 0.10) -> int:
    """Return the smallest per-class ``n`` from which the gate's bound at ``p_hat`` clears ``tau``.

    The certification size alone. :func:`plan_calibration_size` is what to hunt, because it also
    covers the corrected rate's interval.

    ``p_hat`` -- the judge accuracy you expect to observe (e.g. 0.90).
    ``tau``   -- the gate threshold the lower bound must clear (e.g. 0.80).
    ``alpha`` -- the two-sided level whose lower edge is the one-sided bound (default 0.10 -> 95%
    LB).

    The returned ``n`` holds from there on: for every ``n' >= n``, a judge that gets
    ``floor(p_hat * n')`` of ``n'`` right clears the bound. The first ``n`` that clears is not
    enough, because the bound is a sawtooth in ``n``: at ``p_hat=0.90`` it first clears at 48, then
    51 to 54 fall short again (51 labels at 90% is 45 correct, whose bound is 0.788). Collecting
    more labels than planned must never un-certify a judge as accurate as planned.

    The search is finite. With ``z`` the one-sided normal quantile and ``k = floor(p_hat * n)``, so
    ``k / n >= p_hat - 1 / n``, the Wilson lower bound is at least
    ``(p_hat - 1 / n - z / (2 sqrt(n))) / (1 + z^2 / n)``, using ``k (n - k) / n^2 <= 1 / 4``. That
    clears ``tau`` once ``p_hat - tau >= z / (2 sqrt(n)) + (1 + tau z^2) / n``, whose right side
    falls as ``n`` grows, so every size from the first ``n`` meeting it clears the bound. The search
    checks the sizes below that one.

    Raises :class:`InfeasiblePlanError` when ``tau >= p_hat`` (the point estimate never clears the
    threshold, so its lower bound cannot either) or when the guarantee needs more than
    100,000 per class.
    """
    if not 0.0 < p_hat <= 1.0:
        raise InfeasiblePlanError(f"wilson_calibration_size: p_hat must be in (0, 1], got {p_hat}")
    if not 0.0 < tau < 1.0:
        raise InfeasiblePlanError(f"wilson_calibration_size: tau must be in (0, 1), got {tau}")
    if tau >= p_hat:
        raise InfeasiblePlanError(
            f"wilson_calibration_size: tau={tau} >= p_hat={p_hat}; no finite sample clears the "
            "bound (the point estimate itself never reaches the threshold, so its lower bound "
            "cannot)"
        )
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    margin = p_hat - tau
    beyond = next(
        (
            n
            for n in range(2, _MAX_N_PER_CLASS + 1)
            if margin >= z / (2.0 * math.sqrt(n)) + (1.0 + tau * z * z) / n
        ),
        None,
    )
    if beyond is None:
        raise InfeasiblePlanError(
            f"wilson_calibration_size: no size up to {_MAX_N_PER_CLASS} is known to keep the bound "
            f"above tau={tau} at p_hat={p_hat} (the target is too close to p_hat for a practical "
            "hunt)"
        )
    planned = beyond
    for n in range(beyond - 1, 1, -1):
        if wilson_lower(_correct_at(p_hat, n), n, alpha=alpha) < tau:
            break
        planned = n
    return planned


def gate_set_size(expected_pass: float, threshold: float) -> int:
    """The traces per failure mode the run gate needs to pass a suite that passes as expected.

    ``calibrated-agent-evaluation run`` passes a mode when the 95% one-sided Wilson lower bound of
    its pass rate reaches ``threshold``, the same bound the calibration gate reads. So the size is
    :func:`wilson_calibration_size` at ``expected_pass``: the smallest ``n`` from which a suite
    passing ``expected_pass`` of the time (count rounded down) clears the gate, at ``n`` and at
    every larger size, so adding traces never turns a suite as good as expected red.

    Raises:
        InfeasiblePlanError: ``threshold`` is at or above ``expected_pass``, or the size exceeds
            100,000.
    """
    return wilson_calibration_size(expected_pass, threshold)


def gate_fails_allowed(n: int, threshold: float, alpha: float = 0.10) -> int:
    """The most FAIL verdicts among ``n`` that still clear the run gate; -1 when none do."""
    allowed = -1
    for fails in range(n + 1):
        if wilson_lower(n - fails, n, alpha=alpha) < threshold:
            break
        allowed = fails
    return allowed


def _correct_at(p_hat: float, n: int) -> int:
    """The correct count a judge at accuracy ``p_hat`` is planned to show on ``n``, rounded down.

    The tolerance keeps a product such as ``0.29 * 100 = 28.999999999999996`` at 29.
    """
    return math.floor(p_hat * n + 1e-9)


def coverage_floor(p_hat: float) -> int:
    """Return the smallest per-class ``n`` at which the corrected-rate interval covers nominally.

    Read from :data:`COVERAGE_FLOOR`, which was measured at a set of ``p_hat`` values. Between two
    measured values the larger of their floors applies; below the lowest measured value (0.85) its
    floor applies, an extrapolation that rests on coverage improving as the per-class error count
    grows. Depends on ``p_hat`` only: coverage was measured across prevalence 0.02 to 0.50 and
    taken at its worst, and prevalence is what the rate sample estimates, so it cannot be an input.

    Raises :class:`InfeasiblePlanError` when ``p_hat`` is outside (0, 1] or no measured size
    covers nominally there.
    """
    if not 0.0 < p_hat <= 1.0:
        raise InfeasiblePlanError(f"coverage_floor: p_hat must be in (0, 1], got {p_hat}")
    grid = COVERAGE_FLOOR
    if p_hat <= grid[0][0] + 1e-12:
        candidates = [grid[0]]
    else:
        upper = next(i for i, (p, _) in enumerate(grid) if p_hat <= p + 1e-12)
        on_point = abs(grid[upper][0] - p_hat) <= 1e-12
        candidates = [grid[upper]] if on_point else [grid[upper - 1], grid[upper]]
    if any(n is None for _, n in candidates):
        measured = ", ".join(f"{p:.2f}" for p, _ in candidates)
        raise InfeasiblePlanError(
            "coverage_floor: the corrected rate's interval was measured at p_hat="
            f"{measured} and did not cover nominally at any scanned per-class size up to 260; "
            "plan at an accuracy with a measured floor"
        )
    return max(n for _, n in candidates if n is not None)


def _highest_measured() -> tuple[float, int]:
    """The highest accuracy in :data:`COVERAGE_FLOOR` with a measured floor, and the largest floor.

    Read from the table each call, so the rate-safe size follows the committed floor data.
    """
    measured = [(p, n) for p, n in COVERAGE_FLOOR if n is not None]
    return max(p for p, _ in measured), max(n for _, n in measured)


def rate_safe_at(n: int, tau: float, alpha: float = 0.10) -> bool:
    """Whether ``n`` per class is rate-safe at every accuracy that would certify at ``n``.

    Certification marks a judge rate-unsafe when its gold is below the coverage floor at
    the TPR and TNR it showed. At ``n`` per class the gate certifies a class on ``k`` correct when
    the Wilson lower bound of ``k`` of ``n`` reaches ``tau``, so a judge can show any such ``k/n``.
    ``n`` is rate-safe when the floor at every one of them is at most ``n``. Accuracies above the
    highest one with a measured floor are skipped: no size is known to make the interval cover
    there, so no plan can help, and certify marks those records rate-unsafe at any size. With the
    committed table every accuracy has a measured floor, so nothing is skipped.
    """
    highest, _ = _highest_measured()
    for k in range(n, -1, -1):
        if wilson_lower(k, n, alpha=alpha) < tau:
            break  # the bound rises with k, so no smaller count certifies either
        if k / n > highest + 1e-12:
            continue
        if coverage_floor(k / n) > n:
            return False
    return True


def rate_safe_calibration_size(
    p_hat: float, tau: float, alpha: float = 0.10, *, tnr_hat: float | None = None
) -> int:
    """The smallest per-class ``n`` to hunt when the caller wants a corrected rate.

    At least the certification-only plan (:func:`plan_calibration_size`), and rate-safe
    (:func:`rate_safe_at`) there and at every larger size, so a judge more accurate than planned
    does not make its own record rate-unsafe. Every size from the largest measured floor up is
    rate-safe, since no certifying accuracy with a measured floor needs more, so the search runs
    down from there. With the committed table this is 162 per class at every planned accuracy from
    0.88 to 0.99 at ``tau=0.80``, because below 162 some count of errors always gives an accuracy
    between 0.98 and 0.99, whose floor is 162; at 0.85 certifying alone needs 208.

    Raises :class:`InfeasiblePlanError` where the certification-only plan does: ``tau`` the
    expected accuracy cannot reach, or a planned rate with no measured floor.
    """
    start = plan_calibration(p_hat, tau, alpha, tnr_hat=tnr_hat).per_class
    n = max(start, _highest_measured()[1])
    while n > start and rate_safe_at(n - 1, tau, alpha):
        n -= 1
    return n


def double_coded_per_class(minimum_overlap: int = MINIMUM_OVERLAP) -> int:
    """The per-class gold that holds ``minimum_overlap`` double-coded traces, 65 at the default.

    The calibration gold is balanced per class, so the agreement gate's overlap splits evenly
    across FAIL and PASS (rounded up).
    """
    if minimum_overlap < 1:
        raise InfeasiblePlanError(
            f"double_coded_per_class: minimum_overlap must be at least 1, got {minimum_overlap}"
        )
    return math.ceil(minimum_overlap / 2)


@dataclass(frozen=True)
class CalibrationPlan:
    """The per-class sizes :func:`plan_calibration` derives, and the one to hunt.

    ``certification`` certifies the judge; ``coverage_floor`` is the smallest size at which the
    corrected rate's interval covers nominally at the planned accuracy; ``per_class`` is the
    larger, the size to hunt, or the rate-safe size for a plan made with ``rate_safe=True``. A
    judge certified on fewer than ``coverage_floor`` per class is certified, but its corrected rate
    is refused. ``double_coded`` is :func:`double_coded_per_class` for gold two or more raters
    will code, and ``per_class`` is then at least that; ``None`` for one rater.
    """

    per_class: int
    certification: int
    coverage_floor: int
    double_coded: int | None = None


def plan_calibration(
    p_hat: float,
    tau: float,
    alpha: float = 0.10,
    *,
    tnr_hat: float | None = None,
    rate_safe: bool = False,
    raters: int = 1,
) -> CalibrationPlan:
    """The sizes behind :func:`plan_calibration_size`, each kept; see that function.

    ``rate_safe`` -- for a caller who wants a corrected rate: ``per_class`` is then
    :func:`rate_safe_calibration_size`, and ``certification`` and ``coverage_floor`` are the same
    as without it.

    ``raters`` -- how many raters will code the gold. From two up the agreement gate measures
    them on the gold itself, so ``per_class`` is raised to :func:`double_coded_per_class` when it
    is smaller.

    Args:
        p_hat: The accuracy you expect the judge to have on each class (or on TPR, with
            ``tnr_hat`` set).
        tau: The lower bound each class must clear, normally ``GATE_THRESHOLD`` (0.80).
        alpha: The two-sided level of the Wilson bound; 0.10 gives a one-sided 95% bound.
        tnr_hat: The expected TNR when it differs from ``p_hat``.
        rate_safe: Size the plan for a corrected rate as well as certification.
        raters: How many raters will code the gold (1 or more).

    Returns:
        The per-class label count, the certification size, the coverage floor and, for
        double-coded gold, the per-class size the agreement overlap needs.

    Raises:
        InfeasiblePlanError: No label count can clear ``tau`` at the expected accuracy, or
            ``raters`` is below 1.
    """
    if tnr_hat is not None and not 0.0 < tnr_hat <= 1.0:
        raise InfeasiblePlanError(
            f"plan_calibration_size: tnr_hat must be in (0, 1], got {tnr_hat}"
        )
    if raters < 1:
        raise InfeasiblePlanError(f"plan_calibration: raters must be at least 1, got {raters}")
    tpr, tnr = p_hat, p_hat if tnr_hat is None else tnr_hat
    certification = wilson_calibration_size(min(tpr, tnr), tau, alpha)
    floor = coverage_floor(max(tpr, tnr))
    per_class = max(certification, floor)
    if rate_safe:
        per_class = rate_safe_calibration_size(p_hat, tau, alpha, tnr_hat=tnr_hat)
    double_coded = double_coded_per_class() if raters >= 2 else None
    if double_coded is not None:
        per_class = max(per_class, double_coded)
    return CalibrationPlan(
        per_class=per_class,
        certification=certification,
        coverage_floor=floor,
        double_coded=double_coded,
    )


def plan_calibration_size(
    p_hat: float,
    tau: float,
    alpha: float = 0.10,
    *,
    tnr_hat: float | None = None,
    raters: int = 1,
) -> int:
    """Return the per-class ``n`` to hunt: enough to certify the judge and to estimate the rate.

    The larger of :func:`wilson_calibration_size` (the gate's own bound, which depends on ``tau``
    and ``alpha``) and :func:`coverage_floor` (the corrected-rate interval's measured minimum,
    which depends on the expected accuracy only). The certification size is computed first, so a
    ``tau`` the expected accuracy cannot reach is refused as before; the floor can only raise a
    feasible plan or refuse it, never make an infeasible one feasible.

    ``tnr_hat`` -- for a judge expected to be more accurate on one class than the other, its
    expected TNR; ``p_hat`` is then its expected TPR. Both classes are hunted at the returned
    size. The gate must certify both rates, so the certification size is taken at the lower of the
    two; the floor is taken at the higher, the rule the coverage simulations checked (see the
    module docstring). Omitted, the judge is
    expected to be equally accurate on both classes and the plan is the one for ``p_hat`` alone.

    ``raters`` -- from two up, the gold is also the raters' agreement overlap, and the size is at
    least :func:`double_coded_per_class` (65 per class at the defaults).

    Raises :class:`InfeasiblePlanError` when either job has no answer.
    """
    return plan_calibration(p_hat, tau, alpha, tnr_hat=tnr_hat, raters=raters).per_class
