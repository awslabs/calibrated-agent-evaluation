# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""How many double-coded traces the agreement gate needs before it can say ``usable``.

The gate (:func:`calibrated_agent_evaluation.review.agreement.alpha_gate`) passes a gold set when
the bootstrap lower bound of Krippendorff's alpha reaches 0.80. On a small overlap that bound is not
evidence: two raters who happen to agree on all 20 shared traces get a lower bound of 1.0. So the
gate also needs a minimum overlap, and below it the verdict is never ``usable``: it is
``top_up_overlap`` unless alpha's upper bound is already under the bar of ``revise_codebook`` (0.80)
or ``codebook_broken`` (0.667), which more shared traces could not change.

The minimum uses the rule the label planner uses for the judge gate
(:func:`calibrated_agent_evaluation.calibrate.plan.wilson_calibration_size`): plug the expected
agreement in as if it were observed, with counts rounded against the raters, and take the smallest
overlap ``n`` at which the gate's lower bound clears 0.80 at ``n`` and at every larger size. The
first size that clears is not enough, because the bound is a sawtooth in ``n`` (one more discordant
trace appears every few sizes), and labeling more shared traces must never turn raters who agree as
expected unusable.

The bootstrap at each size is computed exactly rather than by resampling: two raters coding
PASS/FAIL give each unit one of three outcomes (both FAIL, both PASS, split), so a bootstrap
resample is a multinomial draw of those counts and alpha is a function of the counts. The
distribution is summed over every count, which makes the size deterministic. One difference from
the label planner's rule: the gate estimates its bound from 2000 resamples, and near 0.80 the
distribution is lumpy enough that the estimate moves by about 0.007 between seeds. A size counts
as clearing only when the exact share of resamples below 0.80 is two Monte Carlo standard errors
under the 2.5% tail (:func:`clears_planned`), so the gate's own estimate clears at about 98% of
seeds. Without that margin the answer would be 123; with it, 130.

The default prevalence is 0.5 because the overlap to double-code is the calibration gold, which
the label planner balances per class.
"""

from __future__ import annotations

import math

import numpy as np

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.review.errors import ReviewError
from calibrated_agent_evaluation.review.gate import ALPHA_LOWER_BOUND_GATE, MINIMUM_OVERLAP

__all__ = [
    "DEFAULT_EXPECTED_ALPHA",
    "DEFAULT_PREVALENCE",
    "GATE_RESAMPLES",
    "MINIMUM_OVERLAP",
    "SCAN_LIMIT",
    "OverlapPlan",
    "alpha_lower_bound_at",
    "clears_planned",
    "plan_overlap",
    "plug_in_counts",
    "share_below_gate",
]

#: The alpha raters working from a clear binary codebook are planned to reach.
DEFAULT_EXPECTED_ALPHA = 0.90
#: The share of FAIL among the double-coded traces: the calibration gold is balanced per class.
DEFAULT_PREVALENCE = 0.5
#: The largest size the guarantee is checked at.
SCAN_LIMIT = 1000
#: The resamples the gate's bootstrap draws (``inter_rater_agreement``'s default).
GATE_RESAMPLES = 2000


class OverlapPlan(CalibratedAgentEvaluationModel):
    """The minimum overlap for raters expected to agree at ``expected_alpha``.

    ``n_units`` is the smallest overlap from which the plug-in counts clear the gate, with the
    margin for its resampling, at every size up to ``scan_limit``.
    """

    n_units: int
    expected_alpha: float
    prevalence: float
    gate_lower_bound: float
    confidence: float
    gate_resamples: int
    scan_limit: int


def plug_in_counts(
    n: int, expected_alpha: float = DEFAULT_EXPECTED_ALPHA, prevalence: float = DEFAULT_PREVALENCE
) -> tuple[int, int, int]:
    """``(both FAIL, both PASS, split)`` among ``n`` units at the expected agreement.

    With equal marginals ``p`` the chance disagreement is ``2p(1-p)``, so alpha fixes the share of
    split units at ``(1 - alpha) 2p(1-p)``. The split count is rounded up and the both-FAIL count
    down, against the raters, the way the label planner rounds the correct count down.

    Raises:
        ReviewError: ``n`` < 2, or an expected alpha and prevalence that give no valid joint
            distribution.
    """
    if n < 2:
        raise ReviewError(f"plug_in_counts: needs n >= 2 shared units, got {n}")
    if not 0.0 < prevalence < 1.0 or not 0.0 <= expected_alpha <= 1.0:
        raise ReviewError(
            f"plug_in_counts: prevalence must be in (0, 1) and expected_alpha in [0, 1], got "
            f"{prevalence}, {expected_alpha}"
        )
    split_share = (1.0 - expected_alpha) * 2.0 * prevalence * (1.0 - prevalence)
    if prevalence - split_share / 2.0 < 0.0 or 1.0 - prevalence - split_share / 2.0 < 0.0:
        raise ReviewError(
            f"plug_in_counts: alpha {expected_alpha} at prevalence {prevalence} is not a valid "
            "joint distribution for two raters"
        )
    split = n - math.floor((1.0 - split_share) * n + 1e-9)
    both_fail = math.floor((prevalence - split_share / 2.0) * n + 1e-9)
    return both_fail, n - split - both_fail, split


def _alpha(both_fail: np.ndarray, both_pass: np.ndarray, split: np.ndarray, n: int) -> np.ndarray:
    # Nominal alpha for two raters on n units: 1 - (2n - 1) * split / (n_FAIL * n_PASS), where
    # n_FAIL and n_PASS count values, so a split unit adds one to each.
    n_fail = 2 * both_fail + split
    n_pass = 2 * both_pass + split
    return 1.0 - (2 * n - 1) * split / (n_fail * n_pass)


def _bootstrap_distribution(counts: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Alpha on every bootstrap resample of units with ``counts``, and each one's probability.

    Sorted by alpha; the probabilities sum to one over the resamples where alpha is defined.
    """
    if min(counts) < 0 or sum(counts) < 2:
        raise ReviewError(f"overlap: needs two or more units and no negative count, got {counts}")
    n = sum(counts)
    log_fact = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, n + 1)))))
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    keep = i + j <= n
    i, j = i[keep], j[keep]
    k = n - i - j
    log_p = log_fact[n] - log_fact[i] - log_fact[j] - log_fact[k]
    for draws, count in ((i, counts[0]), (j, counts[1]), (k, counts[2])):
        if count == 0:
            log_p = np.where(draws > 0, -np.inf, log_p)
        else:
            log_p = log_p + draws * math.log(count / n)
    weight = np.exp(log_p)
    defined = ((2 * i + k) * (2 * j + k) > 0) & (weight > 0)
    if not defined.any():
        raise ReviewError(f"overlap: alpha is undefined on every resample of counts {counts}")
    values = _alpha(i[defined], j[defined], k[defined], n)
    order = np.argsort(values, kind="stable")
    weight = weight[defined][order]
    return values[order], weight / weight.sum()


def alpha_lower_bound_at(counts: tuple[int, int, int], *, confidence: float = 0.95) -> float:
    """The exact percentile-bootstrap lower bound of alpha over units with these counts.

    ``counts`` is ``(both FAIL, both PASS, split)``. A resample in which every value is the same
    category leaves alpha undefined; it is left out, as the gate's bootstrap leaves it out.

    Raises:
        ReviewError: Fewer than two units, a negative count, or every resample undefined.
    """
    values, prob = _bootstrap_distribution(counts)
    tail = (1.0 - confidence) / 2.0
    return float(values[np.searchsorted(np.cumsum(prob), tail - 1e-12)])


def share_below_gate(counts: tuple[int, int, int]) -> float:
    """The probability that a bootstrap resample of these units has alpha below the gate.

    The gate's lower bound clears 0.80 when this is under the lower tail (2.5%).

    Raises:
        ReviewError: As :func:`alpha_lower_bound_at`.
    """
    values, prob = _bootstrap_distribution(counts)
    return float(prob[values < ALPHA_LOWER_BOUND_GATE - 1e-12].sum())


def clears_planned(
    counts: tuple[int, int, int],
    *,
    confidence: float = 0.95,
    gate_resamples: int = GATE_RESAMPLES,
) -> bool:
    """Whether the gate's resampled lower bound clears 0.80 on these units, with a margin.

    The gate estimates the bound from ``gate_resamples`` draws, so the share of draws below 0.80
    it sees varies by about ``sqrt(t (1 - t) / R)`` around the exact share ``t``. The units count
    as clearing when the exact share is at least two of those standard errors under the tail.
    """
    tail = (1.0 - confidence) / 2.0
    margin = 2.0 * math.sqrt(tail * (1.0 - tail) / gate_resamples)
    return share_below_gate(counts) <= tail - margin


def plan_overlap(
    *,
    expected_alpha: float = DEFAULT_EXPECTED_ALPHA,
    prevalence: float = DEFAULT_PREVALENCE,
    confidence: float = 0.95,
    gate_resamples: int = GATE_RESAMPLES,
    scan_limit: int = SCAN_LIMIT,
) -> OverlapPlan:
    """The smallest overlap from which raters at ``expected_alpha`` clear the gate for good.

    Every size from 2 to ``scan_limit`` is checked with :func:`clears_planned` on its plug-in
    counts, and the answer is one more than the largest size that does not clear, so it clears
    at every checked size above it. With the defaults this is :data:`MINIMUM_OVERLAP`.

    Raises:
        ReviewError: ``expected_alpha`` at or below the gate (no overlap clears it), an invalid
            joint distribution, or no clearing size up to ``scan_limit``.
    """
    if expected_alpha <= ALPHA_LOWER_BOUND_GATE:
        raise ReviewError(
            f"plan_overlap: expected alpha {expected_alpha} does not exceed the gate's "
            f"{ALPHA_LOWER_BOUND_GATE}, so no overlap size clears it"
        )
    short = [
        n
        for n in range(2, scan_limit + 1)
        if not clears_planned(
            plug_in_counts(n, expected_alpha, prevalence),
            confidence=confidence,
            gate_resamples=gate_resamples,
        )
    ]
    if short and short[-1] == scan_limit:
        raise ReviewError(
            f"plan_overlap: at expected alpha {expected_alpha} the gate does not clear at "
            f"{scan_limit} shared units"
        )
    return OverlapPlan(
        n_units=(short[-1] + 1) if short else 2,
        expected_alpha=expected_alpha,
        prevalence=prevalence,
        gate_lower_bound=ALPHA_LOWER_BOUND_GATE,
        confidence=confidence,
        gate_resamples=gate_resamples,
        scan_limit=scan_limit,
    )
