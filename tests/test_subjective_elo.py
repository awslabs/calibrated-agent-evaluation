# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Elo rating computation."""

from __future__ import annotations

from typing import Literal

import pytest

from calibrated_agent_evaluation.subjective import PreferencePair, compute_ratings


def _pair(a: str, b: str, winner: Literal["a", "b"] = "a") -> PreferencePair:
    return PreferencePair(
        trace_id_a=a, trace_id_b=b, dimension="tone", winner=winner, judge_kind="human"
    )


class TestComputeRatings:
    """Elo ratings computed correctly."""

    def test_ten_pairs_all_traces_rated(self) -> None:
        pairs = [_pair(f"t{i}", f"t{i + 1}") for i in range(10)]
        result = compute_ratings("tone", pairs)
        trace_ids = {r.trace_id for r in result.ratings}
        assert len(trace_ids) == 11
        assert result.n_total_comparisons == 10
        # Trace with most wins (t0 wins all) has highest rating
        assert result.ratings[0].trace_id == "t0"

    def test_known_answer_ordering(self) -> None:
        # A beats B, B beats C, A beats C → A > B > C
        pairs = [_pair("A", "B"), _pair("B", "C"), _pair("A", "C")]
        result = compute_ratings("tone", pairs)
        ids = [r.trace_id for r in result.ratings]
        assert ids.index("A") < ids.index("B") < ids.index("C")

    def test_treatment_always_tracked(self) -> None:
        pairs = [_pair("t1", "t2")]
        result = compute_ratings("tone", pairs)
        assert result.treatment == "tracked"
        assert not hasattr(result, "confidence_interval")
        assert not hasattr(result, "ci_lower")
        assert not hasattr(result, "ci_upper")

    def test_zero_pairs_empty_ratings(self) -> None:
        result = compute_ratings("tone", [])
        assert result.dimension == "tone"
        assert result.ratings == ()
        assert result.n_total_comparisons == 0
        assert result.treatment == "tracked"

    def test_custom_k_factor(self) -> None:
        pairs = [_pair("t1", "t2")]
        r_default = compute_ratings("tone", pairs, k_factor=32.0)
        r_custom = compute_ratings("tone", pairs, k_factor=16.0)
        # Different k-factors produce different rating spreads
        spread_default = abs(r_default.ratings[0].rating - r_default.ratings[1].rating)
        spread_custom = abs(r_custom.ratings[0].rating - r_custom.ratings[1].rating)
        assert spread_default != spread_custom
        assert spread_default > spread_custom  # higher K → larger update

    def test_custom_initial_rating(self) -> None:
        pairs = [_pair("t1", "t2")]
        result = compute_ratings("tone", pairs, initial_rating=1000.0)
        # Both ratings should be centered around 1000, not 1500
        avg = sum(r.rating for r in result.ratings) / len(result.ratings)
        assert abs(avg - 1000.0) < 1.0

    def test_default_trajectory_hand_computed(self) -> None:
        # Defaults k=32, initial 1500. t1 beats t2 from 1500/1500: expected 0.5, so 1516 / 1484.
        # t2 (1484) then beats t3 (1500): E = 1 / (1 + 10^(16/400)) = 0.4769904, so
        # t2 = 1484 + 32 * 0.5230096 = 1500.73631 and t3 = 1500 - 16.73631 = 1483.26369.
        pairs = [_pair("t1", "t2"), _pair("t2", "t3")]
        got = {r.trace_id: r.rating for r in compute_ratings("tone", pairs).ratings}
        e = 1.0 / (1.0 + 10.0 ** (16.0 / 400.0))
        assert e == pytest.approx(0.4769904, abs=1e-7)
        assert got["t1"] == pytest.approx(1516.0, abs=1e-9)
        assert got["t2"] == pytest.approx(1484.0 + 32.0 * (1.0 - e), abs=1e-9)
        assert got["t3"] == pytest.approx(1500.0 - 32.0 * (1.0 - e), abs=1e-9)
        assert got["t2"] == pytest.approx(1500.73631, abs=1e-5)

    def test_custom_initial_shifts_every_rating(self) -> None:
        # One win from 1000/1000 with k=32 moves each side by 16.
        got = {
            r.trace_id: r.rating
            for r in compute_ratings("tone", [_pair("t1", "t2")], initial_rating=1000.0).ratings
        }
        assert got == {"t1": 1016.0, "t2": 984.0}
