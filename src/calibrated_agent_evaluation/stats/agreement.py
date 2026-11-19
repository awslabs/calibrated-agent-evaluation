# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agreement statistics: Cohen's κ, Krippendorff's alpha, Fleiss' κ, and per-class agreement.

Cohen's κ is the secondary judge-vs-gold agreement metric, reported alongside the
headline TPR/TNR. Raw percent agreement is misleading under class imbalance (a judge that always
says PASS "agrees" 94% of the time on a 6%-fail stream while catching nothing), so raw agreement is
hidden under imbalance and κ shown instead.

κ is reimplemented here (~15 lines) rather than depending on scikit-learn -- the installed
``sklearn`` is ~30 MiB for this one function. The reimplementation is kept
honest
by a test that cross-checks it against ``sklearn.metrics.cohen_kappa_score`` (a dev-only
oracle).

Human-human reliability on the gold set uses the multi-rater block at the end of this module:
Krippendorff's alpha (the headline, since it handles incomplete overlap), Fleiss' κ, per-category
specific agreement, and :func:`bootstrap_agreement`, which puts a unit-bootstrap interval on any of
them. All of it is stdlib only.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

from calibrated_agent_evaluation.calibrate.errors import StatisticsInputError

# Below this minority-class fraction, raw percent agreement is misleading -> hide it, show κ.
IMBALANCE_THRESHOLD = 0.20


def cohen_kappa(gold: Sequence[object], verdict: Sequence[object]) -> float:
    """Return Cohen's κ between two equal-length label sequences.

    κ = (p_o - p_e) / (1 - p_e), where ``p_o`` is observed agreement and ``p_e`` is the agreement
    expected by chance from the marginals. Returns 1.0 for perfect agreement; 0.0 when observed
    equals chance. When ``p_e == 1`` (both raters gave a single constant label) κ is undefined --
    returns 1.0 if they fully agree, else 0.0 (the standard degenerate-case convention).

    Raises :class:`StatisticsInputError` on empty or unequal-length inputs.
    """
    if len(gold) != len(verdict):
        raise StatisticsInputError(
            f"cohen_kappa: gold and verdict must be equal length, "
            f"got {len(gold)} and {len(verdict)}"
        )
    n = len(gold)
    if n == 0:
        raise StatisticsInputError("cohen_kappa: inputs must be non-empty")

    labels = set(gold) | set(verdict)
    observed = sum(1 for g, v in zip(gold, verdict, strict=True) if g == v) / n
    expected = sum(
        (sum(1 for g in gold if g == lab) / n) * (sum(1 for v in verdict if v == lab) / n)
        for lab in labels
    )
    if expected >= 1.0:
        # Degenerate: both sequences are a single constant label. κ is undefined; by convention
        # 1.0 if they agree everywhere, else 0.0.
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def minority_fraction(labels: Sequence[object]) -> float:
    """Return the fraction of the least common label value, or 0.0 for fewer than two values."""
    if not labels:
        return 0.0
    counts: dict[object, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    if len(counts) < 2:
        return 0.0
    return min(counts.values()) / len(labels)


def should_hide_raw_agreement(labels: Sequence[object]) -> bool:
    """Return whether raw percent agreement should be hidden for this label set.

    True when the minority class is below :data:`IMBALANCE_THRESHOLD` -- under imbalance raw
    agreement flatters a do-nothing judge, so the report surfaces κ instead. A set with a single
    label value is the extreme case (the other class has 0% of the labels) and is hidden: a judge
    that answers FAIL to everything scores 100% raw agreement on an all-FAIL set. An empty set has
    nothing to hide and returns False; the callers refuse an empty pairing themselves.
    """
    return len(labels) > 0 and minority_fraction(labels) < IMBALANCE_THRESHOLD


# --- Multi-rater reliability: Krippendorff's alpha, Fleiss' kappa, specific agreement --------
#
# Reliability data are one row per unit (a trace, or a span): the values the unit's raters
# assigned, with ``None`` where a rater did not code it. Only the nominal metric is implemented,
# because every axis Calibrated Agent Evaluation labels (binary ``outcome``, categorical
# ``failure_mode``) is nominal.
#
# Krippendorff's alpha is reimplemented because its PyPI package is GPL-3.0, which the
# license policy excludes. Fleiss' kappa is reimplemented rather than calling statsmodels'
# ``fleiss_kappa``: that
# function wants a numpy count table, which would put numpy inside the bootstrap loop, and it
# returns ``nan`` with a RuntimeWarning when only one category is observed, where a typed refusal
# is required ("no silent failure"). statsmodels stays the dev-only oracle the tests check against.

Units = Sequence[Sequence[Hashable | None]]


class UndefinedAgreementError(StatisticsInputError):
    """The agreement statistic is 0/0 on this data, so no value is reported.

    Raised when no unit has two or more values (nothing to agree on), or when every pairable value
    is the same category (expected disagreement is zero). Reporting 1.0 in the second case would
    certify a codebook that the data never tested.
    """


def _pairable(units: Units, who: str) -> list[list[Hashable]]:
    """Return the coded values of every unit with at least two of them."""
    if len(units) == 0:
        raise StatisticsInputError(f"{who}: reliability data must be non-empty")
    rows = [[v for v in unit if v is not None] for unit in units]
    pairable = [row for row in rows if len(row) >= 2]
    if not pairable:
        raise UndefinedAgreementError(
            f"{who}: no unit was coded by two or more raters, so there is nothing to agree on"
        )
    return pairable


def _coincidences(units: Units, who: str) -> tuple[dict[Hashable, float], dict[Hashable, int]]:
    """Return the diagonal of Krippendorff's coincidence matrix and the per-category totals.

    Each unit with ``m`` values contributes every ordered pair of its values from different raters
    with weight ``1 / (m - 1)``, so a unit contributes exactly ``m`` pairable values in total.
    """
    agreeing: dict[Hashable, float] = {}
    totals: dict[Hashable, int] = {}
    for row in _pairable(units, who):
        weight = 1.0 / (len(row) - 1)
        for value, count in Counter(row).items():
            agreeing[value] = agreeing.get(value, 0.0) + count * (count - 1) * weight
            totals[value] = totals.get(value, 0) + count
    return agreeing, totals


def krippendorff_alpha(units: Units) -> float:
    """Return Krippendorff's alpha (nominal metric) for reliability data with missing values.

    alpha = 1 - D_o / D_e, computed from the coincidence matrix (Krippendorff 2004, ch. 11;
    Hayes and Krippendorff 2007). With ``n`` pairable values, ``n_c`` of them in category ``c`` and
    ``o_cc`` agreeing coincidences in ``c``::

        alpha = 1 - (n - 1) * (n - sum_c o_cc) / (n**2 - sum_c n_c**2)

    Units with fewer than two values are not pairable and drop out, so any number of raters and
    any pattern of gaps is handled. 1.0 is perfect agreement, 0.0 is chance, negative is
    systematic disagreement.

    Raises :class:`StatisticsInputError` on empty data and :class:`UndefinedAgreementError` when
    no unit is pairable or only one category is observed.
    """
    agreeing, totals = _coincidences(units, "krippendorff_alpha")
    n = sum(totals.values())
    expected = n * n - sum(t * t for t in totals.values())
    if expected == 0:
        raise UndefinedAgreementError(
            f"krippendorff_alpha: every pairable value is {next(iter(totals))!r}; with one "
            "observed category expected disagreement is zero and alpha is undefined"
        )
    observed = n - sum(agreeing.values())
    return 1.0 - (n - 1) * observed / expected


def specific_agreement(units: Units) -> dict[Hashable, float]:
    """Return per-category specific agreement, ``o_cc / n_c``, for every observed category.

    Of the pairable values placed in category ``c``, the share whose pair partners in the same unit
    were also ``c``. For two raters this is Cicchetti and Feinstein's (1990) positive and negative
    agreement, ``2a / (2a + b + c)`` and ``2d / (2d + b + c)``. It is reported per class because a
    pooled kappa or alpha on imbalanced labels can look acceptable while the raters never agree on
    the rare class (Feinstein and Cicchetti 1990, the prevalence paradox).

    Raises :class:`StatisticsInputError` on empty data and :class:`UndefinedAgreementError` when
    no unit is pairable.
    """
    agreeing, totals = _coincidences(units, "specific_agreement")
    return {value: agreeing.get(value, 0.0) / total for value, total in totals.items()}


def fleiss_kappa(units: Units) -> float:
    """Return Fleiss' kappa for units that each carry the same number (>= 2) of ratings.

    kappa = (P_bar - P_e) / (1 - P_e), where ``P_bar`` is the mean per-unit share of agreeing rater
    pairs and ``P_e`` the chance agreement from the pooled category shares (Fleiss 1971). Fleiss
    has no missing-data treatment, so a gap or an unequal row length is refused; use
    :func:`krippendorff_alpha` for incomplete overlap.

    Raises :class:`StatisticsInputError` on empty, incomplete, or unequal rows and on fewer than
    two ratings per unit, and :class:`UndefinedAgreementError` when only one category is observed.
    """
    if len(units) == 0:
        raise StatisticsInputError("fleiss_kappa: reliability data must be non-empty")
    m = len(units[0])
    if m < 2:
        raise StatisticsInputError(f"fleiss_kappa: each unit needs >= 2 ratings, got {m}")
    for i, unit in enumerate(units):
        if len(unit) != m or any(v is None for v in unit):
            raise StatisticsInputError(
                f"fleiss_kappa: unit {i} has {sum(v is not None for v in unit)} of the {m} "
                "ratings the first unit has; Fleiss needs complete data (use krippendorff_alpha)"
            )
    totals: Counter[Hashable] = Counter()
    p_bar = 0.0
    for unit in units:
        counts = Counter(unit)
        totals.update(counts)
        p_bar += (sum(c * c for c in counts.values()) - m) / (m * (m - 1))
    p_bar /= len(units)
    grand = len(units) * m
    p_e = sum((t / grand) ** 2 for t in totals.values())
    if len(totals) < 2:
        raise UndefinedAgreementError(
            "fleiss_kappa: only one category is observed; chance agreement is 1 and kappa is "
            "undefined"
        )
    return (p_bar - p_e) / (1.0 - p_e)


@dataclass(frozen=True)
class AgreementInterval:
    """An agreement point estimate with its percentile bootstrap interval over units.

    ``n_undefined`` counts the resamples where the statistic was 0/0 (for example, a resample that
    drew no unit of a rare class). They are excluded from the percentiles rather than scored, and
    the count is kept so a reader can see how much of the distribution that was. For alpha and
    kappa a resample with one observed category is one where every rater agreed, so excluding it
    moves the interval down, never up.
    """

    point: float
    lower: float
    upper: float
    confidence: float
    n_units: int
    n_resamples: int
    n_undefined: int
    seed: int


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Return the ``q`` quantile (0 <= q <= 1) by linear interpolation between order statistics."""
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_agreement(
    units: Units,
    statistic: Callable[[Units], float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> AgreementInterval:
    """Return ``statistic`` on ``units`` with a two-sided percentile bootstrap interval.

    Units are resampled with replacement and each unit keeps all its ratings, because the unit
    (the trace) is what was sampled, not the rater pair. A seeded ``random.Random`` drives the
    draw, so one seed and one data set give one interval, and two statistics bootstrapped with the
    same seed over the same data see the same resamples.

    Raises :class:`StatisticsInputError` for ``n_resamples < 1`` or ``confidence`` outside (0, 1),
    and :class:`UndefinedAgreementError` when the point estimate, or every resample, is undefined.
    """
    if n_resamples < 1:
        raise StatisticsInputError(
            f"bootstrap_agreement: n_resamples must be >= 1, got {n_resamples}"
        )
    if not 0.0 < confidence < 1.0:
        raise StatisticsInputError(
            f"bootstrap_agreement: confidence must be in (0, 1), got {confidence}"
        )
    point = statistic(units)
    rng = random.Random(seed)  # noqa: S311  # nosec B311 - a seeded statistical resample, not crypto
    k = len(units)
    draws: list[float] = []
    for _ in range(n_resamples):
        resample = [units[rng.randrange(k)] for _ in range(k)]
        try:
            draws.append(statistic(resample))
        except UndefinedAgreementError:
            continue  # excluded, and counted below as n_resamples - len(draws)
    if not draws:
        raise UndefinedAgreementError(
            f"bootstrap_agreement: the statistic was undefined on all {n_resamples} resamples"
        )
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    return AgreementInterval(
        point=point,
        lower=_percentile(draws, tail),
        upper=_percentile(draws, 1.0 - tail),
        confidence=confidence,
        n_units=k,
        n_resamples=n_resamples,
        n_undefined=n_resamples - len(draws),
        seed=seed,
    )
