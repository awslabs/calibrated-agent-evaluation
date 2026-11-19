# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cohen's kappa with its large-sample confidence interval.

The interval is ``kappa +- z * ASE`` with the asymptotic standard error of Fleiss, Cohen and Everitt
(1969), "Large sample standard errors of kappa and weighted kappa", Psychological Bulletin 72(5).
With cell proportions ``p_ij``, margins ``p_i.`` / ``p_.j``, observed agreement ``p_o`` and chance
agreement ``p_e``::

    Var = [ sum_i p_ii ((1 - p_e) - (p_i. + p_.i)(1 - p_o))^2
            + (1 - p_o)^2 sum_{i != j} p_ij (p_.i + p_j.)^2
            - (p_o p_e - 2 p_e + p_o)^2 ] / (n (1 - p_e)^4)

the same ASE ``statsmodels.stats.inter_rater.cohens_kappa`` reports (the tests use it as an oracle).

That Wald interval collapses near perfect agreement: at ``p_o = 1`` the variance is exactly 0, so
5 agreeing items read as kappa 1.0 [1.0, 1.0], and close to 1 it under-covers badly. The reported
interval is therefore the union of the Wald interval and a score interval on agreement: the Wilson
(1927) 95% interval for ``p_o``, mapped through ``kappa = (p_o - p_e) / (1 - p_e)`` with ``p_e``
held at its observed value. Over 4000 replicates per cell (seed 41, two raters, binary labels at
prevalence 0.3 or 0.5) the Wald interval alone covered the true kappa 17.4% of the time at true
kappa 0.96 and n = 10, 44.5% at n = 30 and 87.5% at kappa 0.80 and n = 20; the union covered 0.967
to 1.000 in every cell, conservative near the ceiling where the Wald interval fails. The limits are
clipped to [-1, 1].

When both raters used one and the same single label throughout, ``p_e = 1`` and kappa is 0/0;
:class:`~calibrated_agent_evaluation.stats.agreement.UndefinedAgreementError` is raised instead of
the 1.0 that :func:`~calibrated_agent_evaluation.stats.agreement.cohen_kappa` returns by convention,
because a do-nothing judge on a one-class gold set would otherwise read as perfect.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from calibrated_agent_evaluation.calibrate.errors import StatisticsInputError
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.stats.agreement import UndefinedAgreementError

METHOD_KAPPA_FCE = "cohen_kappa_fleiss_cohen_everitt_ase"
METHOD_KAPPA_FCE_WILSON = "cohen_kappa_fce_ase_union_wilson_agreement"
_Z_95 = 1.959963984540054


def _wilson(p: float, n: int) -> tuple[float, float]:
    """The two-sided 95% Wilson score interval for a proportion ``p`` observed over ``n`` items."""
    z2 = _Z_95 * _Z_95
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = _Z_95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def cohen_kappa_interval(gold: Sequence[object], verdict: Sequence[object]) -> Interval:
    """Cohen's kappa between two equal-length label sequences with its two-sided 95% interval.

    The interval is the union of the Fleiss-Cohen-Everitt Wald interval and the Wilson interval on
    observed agreement mapped through kappa (module docstring), so it keeps a width at perfect
    agreement.

    Raises :class:`StatisticsInputError` on empty or unequal-length input and
    :class:`UndefinedAgreementError` when chance agreement is 1 (both raters constant).
    """
    if len(gold) != len(verdict):
        raise StatisticsInputError(
            f"cohen_kappa_interval: gold and verdict must be equal length, "
            f"got {len(gold)} and {len(verdict)}"
        )
    n = len(gold)
    if n == 0:
        raise StatisticsInputError("cohen_kappa_interval: inputs must be non-empty")
    labels = sorted(set(gold) | set(verdict), key=repr)
    cell: dict[tuple[object, object], float] = {}
    for g, v in zip(gold, verdict, strict=True):
        cell[(g, v)] = cell.get((g, v), 0.0) + 1.0 / n
    row = {i: sum(cell.get((i, j), 0.0) for j in labels) for i in labels}
    col = {j: sum(cell.get((i, j), 0.0) for i in labels) for j in labels}
    p_o = sum(cell.get((i, i), 0.0) for i in labels)
    p_e = sum(row[i] * col[i] for i in labels)
    if p_e >= 1.0 - 1e-15:
        raise UndefinedAgreementError(
            "cohen_kappa_interval: both raters gave one and the same label throughout, so chance "
            "agreement is 1 and kappa is 0/0"
        )
    kappa = (p_o - p_e) / (1.0 - p_e)
    diag = sum(
        cell.get((i, i), 0.0) * ((1.0 - p_e) - (row[i] + col[i]) * (1.0 - p_o)) ** 2 for i in labels
    )
    off = (1.0 - p_o) ** 2 * sum(
        cell.get((i, j), 0.0) * (col[i] + row[j]) ** 2 for i in labels for j in labels if i != j
    )
    var = (diag + off - (p_o * p_e - 2.0 * p_e + p_o) ** 2) / (n * (1.0 - p_e) ** 4)
    half = _Z_95 * math.sqrt(max(var, 0.0))
    agree_lo, agree_hi = _wilson(p_o, n)
    lo = min(kappa - half, (agree_lo - p_e) / (1.0 - p_e))
    hi = max(kappa + half, (agree_hi - p_e) / (1.0 - p_e))
    return Interval(
        point=kappa,
        lo=max(-1.0, lo),
        hi=min(1.0, hi),
        method=METHOD_KAPPA_FCE_WILSON,
        level=0.95,
        sided="two-sided",
        n=n,
    )
