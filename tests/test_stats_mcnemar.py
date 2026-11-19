# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the McNemar paired-test block.

Every expected p-value below is hand-computed from its closed form, not by calling statsmodels the
way the implementation does:

  * exact two-sided: ``2 * Binom(b + c, 0.5).cdf(min(b, c))``
  * mid-p: the exact value minus ``Binom(b + c, 0.5).pmf(min(b, c))``
  * asymptotic: ``chi2(1).sf((b - c)^2 / (b + c))``, uncorrected

Malformed and zero-discordant tables surface the typed :class:`StatisticsInputError`.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from calibrated_agent_evaluation.calibrate.errors import StatisticsInputError
from calibrated_agent_evaluation.stats import mcnemar_paired


def _chi2_1_sf(x: float) -> float:
    """Survival function of chi-square(1): ``erfc(sqrt(x / 2))``."""
    return math.erfc(math.sqrt(x / 2.0))


def _exact(b: int, c: int) -> float:
    n, k = b + c, min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def test_exact_reference_b12_c5() -> None:
    """b=12, c=5: exact = 2 * sum_{i<=5} C(17, i) / 2^17 = 0.1434631."""
    got = mcnemar_paired([[0, 12], [5, 0]], midp=False)
    assert got.method == "exact"
    assert got.p_value == pytest.approx(0.143463134765625, abs=1e-12)
    assert got.p_value == pytest.approx(_exact(12, 5), abs=1e-12)


def test_midp_reference_b12_c5() -> None:
    """b=12, c=5: mid-p = 0.1434631 - C(17, 5) / 2^17 = 0.0962524 (a no-op mid-p would fail)."""
    got = mcnemar_paired([[0, 12], [5, 0]], midp=True)
    assert got.method == "midp"
    assert got.p_value == pytest.approx(0.09625244140625, abs=1e-12)
    assert got.p_value == pytest.approx(_exact(12, 5) - math.comb(17, 5) / 2**17, abs=1e-12)


def test_asymptotic_is_uncorrected_reference_b18_c8() -> None:
    """b=18, c=8: statistic 100/26 = 3.846154, p = 0.0498602 (corrected would be 0.0776)."""
    got = mcnemar_paired([[0, 18], [8, 0]])
    assert got.method == "asymptotic"
    assert got.statistic == pytest.approx(100 / 26, abs=1e-12)
    assert got.p_value == pytest.approx(0.04986020375690691, abs=1e-12)


def test_asymptotic_reference_b30_c15() -> None:
    """b=30, c=15: statistic 225/45 = 5.0, p = erfc(sqrt(2.5)) = 0.0253473."""
    got = mcnemar_paired([[3, 30], [15, 7]])
    assert got.statistic == pytest.approx(5.0, abs=1e-12)
    assert got.p_value == pytest.approx(_chi2_1_sf(5.0), abs=1e-12)
    assert got.p_value == pytest.approx(0.025347318677468252, abs=1e-12)


def test_p_monotone_across_the_25_discordant_switch() -> None:
    """At a fixed effect b - c = 10, more pairs never move p away from significance at the switch.

    b=17, c=7 (24 discordant, mid-p) is 0.0433; b=18, c=8 (26, asymptotic) is 0.0499. With the
    continuity-corrected asymptotic test the second was 0.0776 and crossed alpha = 0.05.
    """
    small = mcnemar_paired([[0, 17], [7, 0]], midp=True)
    large = mcnemar_paired([[0, 18], [8, 0]], midp=True)
    assert small.method == "midp"
    assert large.method == "asymptotic"
    assert small.p_value == pytest.approx(0.04328525066375728, abs=1e-12)
    assert small.p_value < 0.05
    assert large.p_value < 0.05


def test_forced_exact_and_asymptotic_flags_honored() -> None:
    """Exact=True/False overrides the auto-select regardless of the discordant count."""
    big = [[10, 15], [20, 10]]  # 35 discordant
    assert mcnemar_paired(big, exact=True).method == "exact"
    small = [[10, 3], [12, 10]]  # 15 discordant
    forced = mcnemar_paired(small, exact=False)
    assert forced.method == "asymptotic"
    assert forced.p_value == pytest.approx(_chi2_1_sf(81 / 15), abs=1e-12)


@given(st.integers(0, 60), st.integers(0, 60), st.booleans())
def test_symmetric_in_b_and_c(b: int, c: int, midp: bool) -> None:
    """The two-sided test does not depend on which discordant cell is which."""
    if b + c == 0:
        return
    one = mcnemar_paired([[1, b], [c, 1]], midp=midp)
    two = mcnemar_paired([[1, c], [b, 1]], midp=midp)
    assert one.p_value == pytest.approx(two.p_value, abs=1e-12)
    assert 0.0 <= one.p_value <= 1.0


def test_zero_discordant_raises_typed_error() -> None:
    """A zero-discordant table is undefined -> typed refusal, never a fabricated p-value."""
    with pytest.raises(StatisticsInputError):
        mcnemar_paired([[10, 0], [0, 10]])


@pytest.mark.parametrize(
    "bad",
    [
        [[1, 2, 3], [4, 5, 6]],  # not 2x2
        [[1, 2], [3, 4], [5, 6]],  # not 2x2
        [[1, -2], [3, 4]],  # negative cell
        [[1, 2]],  # not 2x2
    ],
)
def test_malformed_table_raises_typed_error(bad) -> None:
    """A non-2x2 / negative table surfaces StatisticsInputError, not a raw numpy error."""
    with pytest.raises(StatisticsInputError):
        mcnemar_paired(bad)
