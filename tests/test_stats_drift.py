# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the input-distribution drift block.

Behavioural: assert the drift statistics behave (zero on identity, monotone in the shift, bounded)
and that every invalid input surfaces the typed :class:`StatisticsInputError` -- never a raw scipy
exception, never a NaN (the "wrap external errors / no NaN leaks" convention).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from calibrated_agent_evaluation.calibrate.errors import StatisticsInputError
from calibrated_agent_evaluation.stats import jensen_shannon, psi, wasserstein
from calibrated_agent_evaluation.stats.drift import (
    delta_proportion_ci,
    permutation_p,
    psi_permutation_p,
    quantile_histogram,
    wasserstein_normalized,
)


def test_drift_zero_for_identical_distributions() -> None:
    """Identical baseline/fresh yields zero drift on all three metrics."""
    sample = [float(x) for x in range(50)]
    assert psi(sample, sample) == 0.0
    assert wasserstein(sample, sample) == 0.0
    assert wasserstein_normalized(sample, sample) == 0.0
    assert jensen_shannon([3, 1, 1], [3, 1, 1]) == 0.0


def test_drift_positive_and_monotone_in_the_shift() -> None:
    """Each metric is > 0 under a shift and grows as the shift widens (monotone)."""
    base = [1.0, 2.0, 3.0, 4.0, 5.0]
    near = [2.0, 3.0, 4.0, 5.0, 6.0]
    far = [10.0, 11.0, 12.0, 13.0, 14.0]
    assert 0.0 < wasserstein(base, near) < wasserstein(base, far)

    b = [float(x) for x in range(50)]
    assert 0.0 < psi(b, [x + 5 for x in b]) < psi(b, [x + 25 for x in b])
    # JSD over frequency vectors: a small reshuffle drifts less than a full mass swap.
    assert 0.0 < jensen_shannon([5, 3, 2], [4, 3, 3]) < jensen_shannon([8, 1, 1], [1, 1, 8])


def test_psi_known_answer_on_baseline_deciles() -> None:
    """Baseline 0..49 cut at its deciles gives 10 bins of 5; fresh = baseline + 25.

    Fresh fills bins 5-8 with 5 each and puts 30 in the open top bin, leaving bins 0-4 empty.
    With add-0.5 counts over 10 bins (denominator 55 on both sides) the baseline proportions are
    0.1, and PSI = 5 * (0.5/55 - 0.1) ln(0.5/5.5) + (30.5/55 - 0.1) ln(30.5/5.5).
    """
    b = [float(x) for x in range(50)]
    expected = 5 * (0.5 / 55 - 0.1) * math.log(0.5 / 5.5) + (30.5 / 55 - 0.1) * math.log(30.5 / 5.5)
    assert expected == pytest.approx(1.868579, abs=1e-6)
    assert psi(b, [x + 25 for x in b]) == pytest.approx(expected, abs=1e-12)
    base_counts, fresh_counts = quantile_histogram(b, [x + 25 for x in b])
    assert base_counts == [5.0] * 10
    assert fresh_counts == [0.0] * 5 + [5.0] * 4 + [30.0]


def test_wasserstein_known_answers_and_scale_invariance() -> None:
    """[1..5] vs [2..6] moves every unit of mass by 1: W = 1.0.

    The pooled sample [1,2,2,3,3,4,4,5,5,6] has linear-interpolated quartiles 2.25 and 4.75, so
    the IQR is 2.5 and the normalized distance is 0.4, unchanged by a change of units.
    """
    assert wasserstein([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]) == pytest.approx(1.0)
    assert wasserstein_normalized([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]) == pytest.approx(0.4)
    assert wasserstein_normalized(
        [1000, 2000, 3000, 4000, 5000], [2000, 3000, 4000, 5000, 6000]
    ) == pytest.approx(0.4)


def test_psi_refuses_below_five_per_bin() -> None:
    """PSI is never computed where its bins cannot hold 5 each."""
    with pytest.raises(StatisticsInputError, match="at least 50 values per side"):
        psi([float(x) for x in range(49)], [float(x) for x in range(60)])
    with pytest.raises(StatisticsInputError, match="bins must be >= 2"):
        psi([1.0] * 50, [1.0] * 50, bins=1)


def test_permutation_p_floor_and_validity() -> None:
    """p = (1 + hits) / (1 + n_perm): never 0, and 1.0 when the samples are identical."""
    b = [float(x) for x in range(50)]
    assert psi_permutation_p(b, [x + 100 for x in b], n_perm=99) == pytest.approx(0.01)
    assert psi_permutation_p(b, b, n_perm=99) == 1.0
    assert psi_permutation_p(b, [x + 3 for x in b], n_perm=99, seed=4) == psi_permutation_p(
        b, [x + 3 for x in b], n_perm=99, seed=4
    )
    with pytest.raises(StatisticsInputError, match="n_perm >= 1"):
        permutation_p(lambda x, y: 0.0, b, b, n_perm=0)


@pytest.mark.parametrize("n", [30, 200, 1000])
def test_psi_null_false_positive_rate_is_near_alpha(n: int) -> None:
    """Same-distribution draws: the permutation-tested PSI rejects at about its nominal 5%.

    The equal-width scheme this replaced read PSI > 0.25 ("major shift") on 99.2% of null draws at
    n=30. 60 replicates x 99 permutations keeps this cheap; the bound 0.15 is about 0.05 plus four
    binomial standard errors at 60 draws.
    """
    rng = np.random.default_rng(n)
    bins = 5 if n < 50 else 10
    hits = 0
    reps = 60
    for r in range(reps):
        x = rng.normal(size=n).tolist()
        y = rng.normal(size=n).tolist()
        hits += psi_permutation_p(x, y, bins=bins, n_perm=99, seed=r) < 0.05
    assert hits / reps <= 0.15


@given(
    st.lists(st.floats(-1e3, 1e3), min_size=3, max_size=20),
    st.lists(st.floats(-1e3, 1e3), min_size=3, max_size=20),
    st.lists(st.floats(-1e3, 1e3), min_size=3, max_size=20),
)
def test_wasserstein_is_a_metric(x: list[float], y: list[float], z: list[float]) -> None:
    """Symmetry, identity and the triangle inequality (W is a true metric; PSI and JSD are not)."""
    assert wasserstein(x, y) == pytest.approx(wasserstein(y, x), abs=1e-9)
    assert wasserstein(x, x) == 0.0
    assert wasserstein(x, z) <= wasserstein(x, y) + wasserstein(y, z) + 1e-7


@given(
    st.lists(st.integers(0, 50), min_size=2, max_size=8),
    st.lists(st.integers(0, 50), min_size=2, max_size=8),
)
def test_jsd_symmetric_bounded_and_zero_on_identity(p: list[int], q: list[int]) -> None:
    k = min(len(p), len(q))
    p, q = p[:k], q[:k]
    if sum(p) == 0 or sum(q) == 0:
        return
    assert jensen_shannon(p, q) == pytest.approx(jensen_shannon(q, p), abs=1e-12)
    assert 0.0 <= jensen_shannon(p, q) <= 1.0 + 1e-12
    assert jensen_shannon(p, p) == pytest.approx(0.0, abs=1e-12)


@given(st.lists(st.floats(-1e3, 1e3), min_size=50, max_size=80))
def test_psi_zero_on_identity_and_nonnegative(x: list[float]) -> None:
    assert psi(x, x) == pytest.approx(0.0, abs=1e-12)
    assert psi(x, list(reversed(x))) == pytest.approx(0.0, abs=1e-12)


def test_delta_proportion_ci_reference_values() -> None:
    """delta_proportion_ci: 30/50 after vs 20/50 before is a +0.2 change.

    The reference interval is statsmodels' default ``confint_proportions_2indep`` method
    ("newcomb", Newcombe's hybrid score for independent proportions): Wilson 95% for 30/50 is
    [0.461814, 0.723916] and for 20/50 [0.276084, 0.538186], so the limits are
    0.2 - sqrt(0.138186^2 + 0.138186^2) and 0.2 + sqrt(0.123916^2 + 0.123916^2).
    """
    point, lo, hi = delta_proportion_ci(30, 50, 20, 50)
    assert point == pytest.approx(0.2, abs=1e-12)
    assert lo == pytest.approx(0.2 - 2**0.5 * 0.138186, abs=1e-5)
    assert hi == pytest.approx(0.2 + 2**0.5 * 0.123916, abs=1e-5)
    with pytest.raises(StatisticsInputError):
        delta_proportion_ci(1, 0, 1, 5)


def test_jsd_bounded_in_unit_interval() -> None:
    """The JS divergence (base 2) is bounded in [0, 1]; disjoint support hits 1.0."""
    assert jensen_shannon([1, 0], [0, 1]) == pytest.approx(1.0)
    value = jensen_shannon([7, 2, 1], [1, 2, 7])
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (psi, ([], [1.0, 2.0])),
        (psi, ([1.0, 2.0], [])),
        (wasserstein, ([], [1.0])),
        (wasserstein, ([1.0], [])),
        (jensen_shannon, ([], [1.0])),
    ],
)
def test_empty_input_raises_typed_error(fn, args) -> None:
    """An empty sample raises StatisticsInputError, not a raw ValueError."""
    with pytest.raises(StatisticsInputError):
        fn(*args)


def test_non_numeric_input_raises_typed_error() -> None:
    """A non-numeric element is wrapped into the typed family, not raw."""
    with pytest.raises(StatisticsInputError):
        wasserstein(["a", "b"], [1.0, 2.0])  # type: ignore[list-item]
    with pytest.raises(StatisticsInputError):
        psi([1.0, 2.0], ["x", "y"], bins=2)  # type: ignore[list-item]


def test_degenerate_all_zero_vector_raises_not_nan() -> None:
    """An all-zero JSD vector (scipy would return NaN) raises rather than leaking NaN."""
    with pytest.raises(StatisticsInputError):
        jensen_shannon([0, 0], [1, 1])


def test_nan_or_infinite_input_raises() -> None:
    """A NaN / infinite input value is refused rather than silently propagated."""
    with pytest.raises(StatisticsInputError):
        wasserstein([1.0, math.nan], [1.0, 2.0])
    with pytest.raises(StatisticsInputError):
        psi([1.0, math.inf], [1.0, 2.0])


def test_jsd_length_mismatch_raises() -> None:
    """JSD requires equal-length vectors; a mismatch is a typed refusal (edge)."""
    with pytest.raises(StatisticsInputError):
        jensen_shannon([1, 2, 3], [1, 2])
