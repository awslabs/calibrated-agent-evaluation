# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for repeated-trial robustness aggregation.

Covers the tau-bench pass^k estimator and the Clopper-Pearson rate interval that both directions
carry.
"""

from __future__ import annotations

import math

import pytest

from calibrated_agent_evaluation.probe import ProbeError, capability_robustness, security_result
from calibrated_agent_evaluation.probe.robustness import (
    all_trials_held,
    clopper_pearson,
    pass_hat_k,
)


def test_pass_k_holds_only_when_all_trials_held() -> None:
    result = capability_robustness([True, True, True, True, True])
    assert result.passed is True
    assert result.pass_hat_k == 1.0


def test_pass_k_fails_when_one_trial_fails() -> None:
    # Negative: a single flaky failure means not robust at k = n (pass^5 = C(4,5)/C(5,5) = 0).
    result = capability_robustness([True, True, False, True, True])
    assert result.passed is False
    assert result.pass_hat_k == 0.0


def test_empty_trials_rejected() -> None:
    with pytest.raises(ProbeError):
        capability_robustness([])


def test_pass_k_is_the_tau_bench_estimator_for_k_below_n() -> None:
    # 4 of 5 held. pass^1 = 4/5, pass^2 = C(4,2)/C(5,2) = 6/10, pass^3 = C(4,3)/C(5,3) = 4/10.
    trials = [True, True, False, True, True]
    assert capability_robustness(trials, k=1).pass_hat_k == pytest.approx(0.8)
    assert capability_robustness(trials, k=2).pass_hat_k == pytest.approx(0.6)
    assert capability_robustness(trials, k=3).pass_hat_k == pytest.approx(0.4)


def test_one_held_trial_gives_a_rate_strictly_inside_zero_one_at_k1() -> None:
    result = capability_robustness([True, False, False, False], k=1)
    assert result.pass_hat_k == pytest.approx(0.25)
    assert 0.0 < (result.pass_hat_k or 0.0) < 1.0


def test_pass_hat_k_averages_over_tasks() -> None:
    # Task A 3/4 held, task B 1/4 held. pass^2 = (C(3,2)/C(4,2) + C(1,2)/C(4,2)) / 2 = (3/6 + 0)/2.
    tasks = [[True, True, True, False], [False, True, False, False]]
    assert pass_hat_k(tasks, 2) == pytest.approx(0.25)
    assert pass_hat_k(tasks, 1) == pytest.approx((0.75 + 0.25) / 2)


@pytest.mark.parametrize("k", [0, 6])
def test_k_out_of_range_refused(k: int) -> None:
    with pytest.raises(ProbeError, match="must lie in"):
        capability_robustness([True] * 5, k=k)
    with pytest.raises(ProbeError, match="must lie in"):
        pass_hat_k([[True] * 5], k)


def test_all_trials_held_is_the_strict_flag() -> None:
    assert all_trials_held([True, True]) is True
    assert all_trials_held([True, False]) is False
    with pytest.raises(ProbeError):
        all_trials_held([])


@pytest.mark.parametrize(
    ("x", "n", "lo", "hi"),
    [
        # Reference values: the exact beta-quantile form, Beta(x, n-x+1).ppf(0.025) and
        # Beta(x+1, n-x).ppf(0.975); 0/3 gives the familiar 1 - 0.025^(1/3) = 0.707598.
        (0, 3, 0.0, 1.0 - 0.025 ** (1 / 3)),
        (3, 3, 0.025 ** (1 / 3), 1.0),
        (1, 5, 0.0050507634, 0.7164179361),
        (7, 10, 0.3475471499, 0.9332604888),
        (12, 40, 0.1656272044, 0.4653162853),
    ],
)
def test_clopper_pearson_reference_values(x: int, n: int, lo: float, hi: float) -> None:
    ci = clopper_pearson(x, n)
    assert ci.lo == pytest.approx(lo, abs=1e-9)
    assert ci.hi == pytest.approx(hi, abs=1e-9)
    assert (ci.method, ci.level, ci.sided, ci.n) == ("clopper_pearson", 0.95, "two-sided", n)


def test_clopper_pearson_matches_statsmodels_beta_oracle() -> None:
    from statsmodels.stats.proportion import proportion_confint

    for x, n in [(0, 1), (2, 9), (5, 17), (30, 31)]:
        lo, hi = proportion_confint(x, n, alpha=0.05, method="beta")
        ci = clopper_pearson(x, n)
        assert ci.lo == pytest.approx(float(lo), abs=1e-9)
        assert ci.hi == pytest.approx(float(hi), abs=1e-9)


def test_rate_interval_widens_as_trials_shrink() -> None:
    wide = capability_robustness([True, False] * 2).rate
    narrow = capability_robustness([True, False] * 20).rate
    assert (wide.hi - wide.lo) > (narrow.hi - narrow.lo)
    assert math.isclose(wide.point or 0.0, 0.5)
    assert math.isclose(narrow.point or 0.0, 0.5)


def test_security_flags_vulnerable_if_any_trial_breaks() -> None:
    result = security_result([False, False, True, False, False])
    assert result.vulnerable_if_any is True
    assert result.passed is False  # vulnerable
    assert result.rate.point == pytest.approx(0.2)


def test_security_not_vulnerable_when_all_trials_safe() -> None:
    # Negative: all-safe -> not vulnerable, but the break rate is only bounded, not zero.
    result = security_result([False, False, False])
    assert result.passed is True
    assert result.rate.hi == pytest.approx(1.0 - 0.025 ** (1 / 3), abs=1e-9)


@pytest.mark.parametrize(("x", "n"), [(600, 1200), (0, 5000), (5000, 5000), (3, 20000)])
def test_clopper_pearson_stays_finite_at_large_n(x: int, n: int) -> None:
    """``math.comb(n, i) * p**i`` raised a raw OverflowError from about n = 1030."""
    from scipy.stats import beta

    ci = clopper_pearson(x, n)
    lo = 0.0 if x == 0 else float(beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(0.975, x + 1, n - x))
    assert ci.lo == pytest.approx(lo, abs=1e-10)
    assert ci.hi == pytest.approx(hi, abs=1e-10)
