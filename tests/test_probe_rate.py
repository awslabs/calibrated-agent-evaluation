# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the judge-biased upper-bound rate."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.probe import (
    ProbeError,
    ProbeUncorrectableError,
    UpperBoundRate,
    upper_bound_rate,
)


def test_rate_is_an_upper_bound() -> None:
    rate = UpperBoundRate(upper_bound=0.42, k=5)
    assert rate.upper_bound == 0.42
    assert rate.k == 5


def test_require_point_refuses_a_corrected_estimate() -> None:
    # Negative: a black-box target has no calibration set → no corrected point exists.
    with pytest.raises(ProbeUncorrectableError):
        UpperBoundRate(upper_bound=0.42, k=5).require_point()


def test_the_producer_bounds_the_rate_from_trial_outcomes() -> None:
    """0 of 5 bounds the rate at 1 - 0.05 ** (1/5) = 0.4507, the exact one-sided bound."""
    rate = upper_bound_rate([False] * 5)
    assert rate.k == 5
    assert rate.failures == 0
    assert rate.level == 0.95
    assert rate.upper_bound == pytest.approx(1 - 0.05 ** (1 / 5), abs=1e-12)
    with pytest.raises(ProbeUncorrectableError):
        rate.require_point()


def test_the_producer_matches_the_beta_quantile_oracle() -> None:
    from scipy.stats import beta

    rate = upper_bound_rate([True] * 3 + [False] * 17)
    assert rate.upper_bound == pytest.approx(float(beta.ppf(0.95, 4, 17)), abs=1e-12)
    assert upper_bound_rate([True] * 4).upper_bound == 1.0


def test_the_producer_refuses_no_trials_and_a_bad_level() -> None:
    with pytest.raises(ProbeError, match="at least one trial"):
        upper_bound_rate([])
    with pytest.raises(ProbeError, match="level"):
        upper_bound_rate([False], level=1.0)
