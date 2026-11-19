# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired before/after significance -- McNemar's test.

The paired test for a before/after improvement on the SAME inputs. Pure functions, no vendor
trace type, no I/O.

  * :func:`mcnemar_paired` -- McNemar over a 2x2 table ``[[a, b], [c, d]]`` where ``b`` and ``c``
    are the discordant (flipped) cells. Uses ``statsmodels.stats.contingency_tables.mcnemar`` for
    exact (binomial) and asymptotic (uncorrected chi-square) variants, and adds the **mid-p**
    adjustment that statsmodels does not provide (verified: its ``mcnemar`` exposes only
    ``exact`` / ``correction``). ``exact=None`` auto-selects the exact path when the discordant
    count ``b + c < 25`` and the asymptotic path otherwise; ``midp=True`` applies mid-p
    correction to the exact path.

The statistical facts (verified against ``statsmodels==0.14.6``):

  * Exact two-sided p-value: with ``k = min(b, c)`` and ``n = b + c``, ``p_exact =
    min(1, 2 * Binom(n, 0.5).cdf(k))`` -- equals ``statsmodels`` ``mcnemar(exact=True)``.
  * Mid-p (Lancaster): ``p_midp = min(1, 2 * Binom(n, 0.5).cdf(k) - Binom(n, 0.5).pmf(k))`` --
    always ``<= p_exact`` (removes half the atom at ``k``, the standard less-conservative small
    correction). This is the "build it (~10 lines)" piece.
  * Asymptotic: the **uncorrected** statistic ``(b - c)^2 / (b + c)`` on one degree of freedom.
    The continuity-corrected form is over-conservative, and switching from mid-p below 25
    discordant pairs to it above made ``p`` jump away from significance as ``n`` grew at a fixed
    effect (``b=17, c=7`` mid-p 0.0433, then ``b=18, c=8`` corrected 0.0776). Fagerland, Lydersen
    and Laake (2014), "Recommended tests and confidence intervals for paired binomial proportions",
    Statistics in Medicine 33(16):2850-2875, recommend mid-p or the uncorrected asymptotic test;
    with the uncorrected form ``b=18, c=8`` gives 0.0499, so the switch is smooth.
  * A **zero-discordant** table (``b + c == 0``) has no paired information: McNemar is undefined.
    Rather than fabricate a p-value, this raises :class:`StatisticsInputError`; the improve-stage
    caller maps that to an explicit ``method="undefined"`` result.

Every raw ``statsmodels`` / ``numpy`` failure (a non-2x2, negative, or malformed table) is caught at
the boundary and re-raised as the shared
:class:`~calibrated_agent_evaluation.calibrate.errors.StatisticsInputError` ("wrap external errors
at the boundary"; "no silent failure") -- no raw vendor exception escapes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import binom
from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar

from calibrated_agent_evaluation.calibrate.errors import StatisticsInputError

# Below this many discordant pairs the asymptotic chi-square is unreliable, so ``exact=None`` uses
# the exact/mid-p path ("exact / mid-p when discordant pairs < 25").
_SMALL_DISCORDANT = 25

McNemarMethod = Literal["exact", "midp", "asymptotic"]


@dataclass(frozen=True)
class McNemarStat:
    """The pure McNemar computation (the improve-stage :class:`McNemarResult` wraps it).

    ``statistic`` is ``min(b, c)`` for the exact/mid-p path (the binomial statistic statsmodels
    reports) or the uncorrected chi-square ``(b - c)^2 / (b + c)`` for the asymptotic path.
    ``n_discordant`` is ``b + c``.
    """

    statistic: float
    p_value: float
    method: McNemarMethod
    n_discordant: int


def _validate_table(table: Sequence[Sequence[int]]) -> tuple[int, int, int, int]:
    """Coerce ``table`` to a 2x2 of non-negative ints ``(a, b, c, d)``; raise a typed error else."""
    try:
        arr = np.asarray(table, dtype=int)
    except (ValueError, TypeError) as exc:
        raise StatisticsInputError(
            f"mcnemar_paired: table must be a 2x2 integer matrix, got {table!r} ({exc})"
        ) from exc
    if arr.shape != (2, 2):
        raise StatisticsInputError(f"mcnemar_paired: table must be 2x2, got shape {arr.shape}")
    if bool(np.any(arr < 0)):
        raise StatisticsInputError(
            f"mcnemar_paired: table cells must be non-negative counts, got {arr.tolist()}"
        )
    a, b, c, d = int(arr[0, 0]), int(arr[0, 1]), int(arr[1, 0]), int(arr[1, 1])
    return a, b, c, d


def _midp(b: int, c: int) -> float:
    """The two-sided mid-p McNemar p-value for discordant cells ``b``, ``c`` (BUILT here).

    ``p_midp = 2 * cdf(k) - pmf(k)`` capped at 1, with ``k = min(b, c)`` and ``n = b + c`` under
    ``Binomial(n, 0.5)`` -- the exact two-sided p-value minus half the probability atom at ``k``, so
    ``p_midp <= p_exact`` always (the standard less-conservative small-sample correction).
    """
    n = b + c
    k = min(b, c)
    # Wrap the scipy calls at the boundary (as the asymptotic path does) so a vendor error surfaces
    # as the typed StatisticsInputError, and refuse a non-finite result rather than leak it.
    try:
        cdf = float(binom.cdf(k, n, 0.5))
        pmf = float(binom.pmf(k, n, 0.5))
    except (ValueError, TypeError) as exc:
        raise StatisticsInputError(
            f"mcnemar_paired: mid-p computation failed on discordant (b={b}, c={c}) ({exc})"
        ) from exc
    p_midp = min(1.0, 2.0 * cdf - pmf)
    if not math.isfinite(p_midp):
        raise StatisticsInputError(
            f"mcnemar_paired: mid-p produced a non-finite p-value on discordant (b={b}, c={c})"
        )
    return p_midp


def mcnemar_paired(
    table: Sequence[Sequence[int]],
    *,
    exact: bool | None = None,
    midp: bool = False,
) -> McNemarStat:
    """Run McNemar's paired test over a 2x2 contingency ``table``.

    ``table`` is ``[[a, b], [c, d]]``; the discordant cells ``b`` (before-fail/after-pass) and ``c``
    (before-pass/after-fail) are what the test counts. ``exact`` selects the variant: ``None``
    (default) auto-selects the exact path when ``b + c < 25`` and the asymptotic path otherwise;
    ``True`` forces exact; ``False`` forces asymptotic. ``midp=True`` applies the mid-p adjustment
    to the exact path (ignored on the asymptotic path).

    Raises :class:`StatisticsInputError` on a non-2x2 / negative table, or on a **zero-discordant**
    table (``b + c == 0``) -- McNemar is undefined with no discordant pairs, so this refuses rather
    than fabricate a p-value (the caller maps it to an explicit ``undefined`` result).
    """
    a, b, c, d = _validate_table(table)
    n_discordant = b + c
    if n_discordant == 0:
        raise StatisticsInputError(
            "mcnemar_paired: the table has zero discordant pairs (b + c == 0); McNemar's test is "
            "undefined with no discordant pairs -- refusing to fabricate a p-value"
        )

    use_exact = (n_discordant < _SMALL_DISCORDANT) if exact is None else exact

    if use_exact and midp:
        return McNemarStat(
            statistic=float(min(b, c)),
            p_value=_midp(b, c),
            method="midp",
            n_discordant=n_discordant,
        )

    try:
        # statsmodels returns a private ``_Bunch`` with dynamic ``pvalue`` / ``statistic`` attrs
        # (no static type surface); read them via getattr so the vendor's untyped bunch does not
        # leak a pyright attribute error into this pure block.
        # correction=False: the uncorrected asymptotic test (Fagerland et al. 2013); the corrected
        # one made p non-monotone in n across the 25-discordant switch from mid-p.
        result = _sm_mcnemar(np.array([[a, b], [c, d]]), exact=use_exact, correction=False)
        p_value = float(getattr(result, "pvalue"))  # noqa: B009 - dynamic vendor _Bunch attr
        statistic = float(getattr(result, "statistic"))  # noqa: B009 - dynamic vendor _Bunch attr
    except (ValueError, TypeError, ZeroDivisionError, AttributeError) as exc:
        raise StatisticsInputError(
            f"mcnemar_paired: statsmodels failed on table {[[a, b], [c, d]]} ({exc})"
        ) from exc
    if not math.isfinite(p_value):
        raise StatisticsInputError(
            f"mcnemar_paired: produced a non-finite p-value ({p_value}) on table {[[a, b], [c, d]]}"
        )
    return McNemarStat(
        statistic=statistic,
        p_value=p_value,
        method="exact" if use_exact else "asymptotic",
        n_discordant=n_discordant,
    )
