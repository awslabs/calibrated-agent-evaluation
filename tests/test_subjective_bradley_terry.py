# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Bradley-Terry estimator of record."""

from __future__ import annotations

import math
import random
from typing import Literal

import numpy as np
import pytest

from calibrated_agent_evaluation.subjective import (
    PreferencePair,
    SubjectiveError,
    bradley_terry,
    compute_ratings,
)

ELO_PER_NAT = 400.0 / math.log(10.0)


def _pair(a: str, b: str, winner: Literal["a", "b"] = "a") -> PreferencePair:
    return PreferencePair(
        trace_id_a=a, trace_id_b=b, dimension="tone", winner=winner, judge_kind="human"
    )


def test_two_traces_hand_computed_mle_and_wald_interval() -> None:
    """A beats B 3 times of 4: the MLE odds are 3, so theta_A - theta_B = ln 3.

    Fisher information for the difference is 4 * 0.75 * 0.25 = 0.75, so Var(theta_A - theta_B) =
    4/3, and under sum-to-zero theta_A = (theta_A - theta_B)/2 has SE sqrt(1/3) = 0.577350.
    """
    result = bradley_terry("tone", [_pair("A", "B")] * 3 + [_pair("A", "B", "b")])
    a, b = result.ratings
    assert a.trace_id == "A"
    assert a.rating.point == pytest.approx(1500 + ELO_PER_NAT * math.log(3) / 2, abs=1e-6)
    assert b.rating.point == pytest.approx(1500 - ELO_PER_NAT * math.log(3) / 2, abs=1e-6)
    half = 1.959963984540054 * math.sqrt(1 / 3) * ELO_PER_NAT
    assert a.rating.hi - (a.rating.point or 0.0) == pytest.approx(half, abs=1e-6)
    assert (a.rating.method, a.rating.level, a.rating.sided, a.rating.n) == (
        "bradley_terry_mle_wald",
        0.95,
        "two-sided",
        4,
    )


def test_matches_a_logistic_regression_fit() -> None:
    """The MLE equals the no-intercept logistic regression on pairwise indicators (statsmodels)."""
    import statsmodels.api as sm

    rng = random.Random(5)
    true = {"A": 1.0, "B": 0.4, "C": 0.0, "D": -0.8}
    names = sorted(true)
    pairs = []
    for _ in range(300):
        a, b = rng.sample(names, 2)
        p = 1 / (1 + math.exp(true[b] - true[a]))
        pairs.append(_pair(a, b, "a" if rng.random() < p else "b"))
    got = {
        r.trace_id: ((r.rating.point or 0.0) - 1500) / ELO_PER_NAT
        for r in bradley_terry("tone", pairs).ratings
    }

    # Reference coding: drop A's column, so each coefficient is theta_i - theta_A.
    rows, y = [], []
    for pair in pairs:
        row = [0.0] * len(names)
        row[names.index(pair.trace_id_a)] += 1.0
        row[names.index(pair.trace_id_b)] -= 1.0
        rows.append(row[1:])
        y.append(1.0 if pair.winner == "a" else 0.0)
    fit = sm.Logit(np.asarray(y), np.asarray(rows)).fit(disp=0, tol=1e-12, maxiter=200)
    ref = np.concatenate([[0.0], np.asarray(fit.params)])
    centred = ref - ref.mean()
    for i, name in enumerate(names):
        assert got[name] == pytest.approx(float(centred[i]), abs=1e-6)


def test_order_invariant_unlike_sequential_elo() -> None:
    pairs = [_pair("A", "B"), _pair("B", "C"), _pair("C", "A"), _pair("A", "C"), _pair("B", "A")]
    shuffled = list(reversed(pairs))
    assert bradley_terry("tone", pairs) == bradley_terry("tone", shuffled)
    elo = {r.trace_id: r.rating for r in compute_ratings("tone", pairs).ratings}
    elo_shuffled = {r.trace_id: r.rating for r in compute_ratings("tone", shuffled).ratings}
    assert elo != elo_shuffled  # the reason Elo is display-only


def test_interval_narrows_with_more_comparisons() -> None:
    few = [_pair("A", "B")] * 3 + [_pair("A", "B", "b")]
    many = few * 4
    w_few = bradley_terry("tone", few).ratings[0].rating
    w_many = bradley_terry("tone", many).ratings[0].rating
    assert w_few.point == pytest.approx(w_many.point, abs=1e-6)
    # Four times the data halves the standard error.
    assert (w_few.hi - w_few.lo) == pytest.approx(2 * (w_many.hi - w_many.lo), rel=1e-6)


def test_unbeaten_trace_refused_not_reported_as_infinite() -> None:
    with pytest.raises(SubjectiveError, match="not strongly connected"):
        bradley_terry("tone", [_pair("A", "B"), _pair("A", "B"), _pair("B", "C"), _pair("C", "B")])


def test_bad_inputs_refused() -> None:
    with pytest.raises(SubjectiveError, match="no preference pairs"):
        bradley_terry("tone", [])
    with pytest.raises(SubjectiveError, match="self-comparison"):
        bradley_terry("tone", [_pair("A", "A")])
    with pytest.raises(SubjectiveError, match="dimension"):
        bradley_terry("style", [_pair("A", "B"), _pair("A", "B", "b")])


def test_elo_records_its_hyperparameters() -> None:
    result = compute_ratings("tone", [_pair("A", "B")], k_factor=16.0, initial_rating=1000.0)
    assert (result.k_factor, result.initial_rating) == (16.0, 1000.0)
