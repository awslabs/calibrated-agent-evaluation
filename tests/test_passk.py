# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the pass@k point estimator, its bootstrap interval, and its input guard."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.probe.robustness import pass_hat_k
from calibrated_agent_evaluation.scorers import pass_at_k
from calibrated_agent_evaluation.scorers.errors import ScorerError


def test_pass_at_k_matches_the_unbiased_formula() -> None:
    # One task, 2 of 5 samples pass, k=2: 1 - C(3,2)/C(5,2) = 1 - 3/10 = 0.7.
    assert pass_at_k([[True, True, False, False, False]], k=2).point == pytest.approx(0.7)


def test_pass_at_k_averages_over_tasks() -> None:
    # Task A (2 of 4, k=2): 1 - C(2,2)/C(4,2) = 1 - 1/6. Task B (all 3 pass, k=2): 1.
    samples = [[True, True, False, False], [True, True, True]]
    expected = ((1.0 - 1 / 6) + 1.0) / 2
    assert pass_at_k(samples, k=2).point == pytest.approx(expected)


def test_pass_at_k_is_a_different_quantity_from_pass_hat_k() -> None:
    # Same input, opposite estimators: pass@k = "at least one of k passes",
    # pass^k = "all k hold". For 3 of 4 with k=n=4, pass@k is 1.0 while pass^4 is 0.0.
    samples = [[True, True, True, False]]
    assert pass_at_k(samples, k=4).point == pytest.approx(1.0)
    assert pass_hat_k(samples, k=4) == pytest.approx(0.0)


def test_the_interval_brackets_the_point_and_is_two_sided_95() -> None:
    result = pass_at_k([[True, True, False, False, False, True, False, True, False, True]], k=2)
    assert result.interval.lo <= result.point <= result.interval.hi
    assert (result.interval.level, result.interval.sided) == (0.95, "two-sided")
    assert result.interval.method == "bootstrap-percentile"


def test_a_single_task_still_gets_a_non_degenerate_interval() -> None:
    # Within-task resampling means one task does not collapse to a zero-width interval.
    result = pass_at_k([[True, True, False, False, False]], k=2)
    assert result.interval.hi > result.interval.lo


def test_the_interval_is_deterministic_for_a_given_seed() -> None:
    samples = [[True, False, True, False, True]]
    first = pass_at_k(samples, k=2, seed=7)
    second = pass_at_k(samples, k=2, seed=7)
    assert (first.interval.lo, first.interval.hi) == (second.interval.lo, second.interval.hi)


def test_pass_at_k_refuses_k_below_one() -> None:
    with pytest.raises(ScorerError, match="task 0"):
        pass_at_k([[True, False]], k=0)


def test_pass_at_k_refuses_k_above_the_sample_count() -> None:
    with pytest.raises(ScorerError, match="n=2"):
        pass_at_k([[True, False]], k=3)


def test_pass_at_k_refuses_no_tasks_and_an_empty_task() -> None:
    with pytest.raises(ScorerError, match="at least one task"):
        pass_at_k([], k=1)
    with pytest.raises(ScorerError):
        pass_at_k([[]], k=1)
