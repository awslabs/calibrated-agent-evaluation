# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The corrected-rate interval at a boundary count follows the modified Jeffreys rule."""

from __future__ import annotations

import numpy as np
import pytest

from calibrated_agent_evaluation.stats.rogan_gladen import rg_interval


def _unmodified(tp: int, n1: int, fp: int, n0: int, k: int, ne: int) -> tuple[float, float]:
    """The interval as it was before the boundary rule: one draw set, both tails from it."""
    rng = np.random.default_rng(0)
    t = rng.beta(tp + 0.5, n1 - tp + 0.5, 9999)
    f = rng.beta(fp + 0.5, n0 - fp + 0.5, 9999)
    m = rng.beta(k + 0.5, ne - k + 0.5, 9999)
    ok = t - f > 0
    share = 1 - ok.mean()
    theta = np.clip((m[ok] - f[ok]) / (t[ok] - f[ok]), 0.0, 1.0)
    lo, hi = np.quantile(theta, [(0.025 - share) / (1 - share), 0.975 / (1 - share)])
    return float(lo), float(hi)


def _arrays(tp: int, n1: int, fp: int, n0: int, k: int, ne: int):
    return [1] * tp + [0] * (n1 - tp), [1] * fp + [0] * (n0 - fp), [1] * k + [0] * (ne - k)


def test_away_from_the_boundary_the_interval_is_unchanged() -> None:
    counts = (150, 162, 6, 162, 30, 400)
    got = rg_interval(*_arrays(*counts))
    assert (got.lo, got.hi) == pytest.approx(_unmodified(*counts), rel=1e-12)


def test_no_false_positive_takes_the_upper_limit_with_the_false_positive_rate_at_zero() -> None:
    """A judge with no false positive in 162 passes may have none at all. At a true rate of 2%
    measured by that judge, the unmodified upper limit sits below the truth more often than its
    5% share; the modified one is computed as if the false-positive rate were exactly 0."""
    counts = (154, 162, 0, 162, 19, 1000)
    got = rg_interval(*_arrays(*counts))
    lo, hi = _unmodified(*counts)
    assert got.lo == pytest.approx(lo, rel=1e-12)
    assert got.hi > hi
    assert got.hi == pytest.approx(
        float(np.quantile(np.clip(_measured_over(154, 162, 19, 1000), 0, 1), 0.975)), rel=1e-12
    )


def _measured_over(tp: int, n1: int, k: int, ne: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = rng.beta(tp + 0.5, n1 - tp + 0.5, 9999)
    rng.beta(0.5, 162.5, 9999)  # the false-positive draws, unused at the boundary
    m = rng.beta(k + 0.5, ne - k + 0.5, 9999)
    return m / t


def test_every_positive_caught_takes_the_lower_limit_with_tpr_at_one() -> None:
    counts = (162, 162, 8, 162, 150, 500)
    got = rg_interval(*_arrays(*counts))
    lo, hi = _unmodified(*counts)
    assert got.hi == pytest.approx(hi, rel=1e-12)
    assert got.lo < lo


def test_the_interval_only_widens_at_a_boundary() -> None:
    for counts in [(154, 162, 0, 162, 3, 100), (162, 162, 0, 162, 0, 100), (20, 20, 0, 20, 5, 50)]:
        got = rg_interval(*_arrays(*counts))
        lo, hi = _unmodified(*counts)
        assert got.lo <= lo + 1e-12
        assert got.hi >= hi - 1e-12
