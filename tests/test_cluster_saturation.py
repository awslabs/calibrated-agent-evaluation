# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for saturation estimators: Good-Turing coverage and Chao1, with intervals."""

from __future__ import annotations

import math

import pytest

from calibrated_agent_evaluation.cluster import (
    ClusteringError,
    chao1_estimate,
    chao1_interval,
    good_turing_coverage,
    good_turing_coverage_bound,
    saturation_check,
)

# --- saturation functions accept list[int], return floats ---


class TestGoodTuringSignature:
    """Saturation estimators operate on mode-frequency counts."""

    def test_accepts_list_int_returns_float(self) -> None:
        result = good_turing_coverage([5, 3, 2, 1])
        assert isinstance(result, float)


# --- (negative) — non-integer counts → ClusteringError ---


class TestCountsValidation:
    """Negative: invalid count types raise ClusteringError."""

    def test_float_counts_rejected(self) -> None:
        with pytest.raises(ClusteringError, match="non-negative integers"):
            good_turing_coverage([1.5, 2.0, 3.0])  # type: ignore[list-item]

    def test_nested_list_rejected(self) -> None:
        with pytest.raises(ClusteringError, match="non-negative integers"):
            good_turing_coverage([[1, 2], [3, 4]])  # type: ignore[list-item]

    def test_negative_int_rejected(self) -> None:
        with pytest.raises(ClusteringError, match="non-negative"):
            good_turing_coverage([5, -1, 3])

    def test_chao1_validates(self) -> None:
        with pytest.raises(ClusteringError, match="non-negative integers"):
            chao1_estimate([1.0, 2.0])  # type: ignore[list-item]


# --- known-answer below threshold ---


class TestGoodTuringKnownAnswers:
    """Good-Turing coverage known-answer tests."""

    def test_below_threshold(self) -> None:
        # counts=[10,8,5,3,1,1,1] → 3 singletons, 29 total → coverage ≈ 0.8966
        counts = [10, 8, 5, 3, 1, 1, 1]
        cov = good_turing_coverage(counts)
        expected = 1.0 - 3.0 / 29.0
        assert abs(cov - expected) < 0.0001
        result = saturation_check(counts)
        assert result.saturated is False

    def test_no_singletons_still_leaves_variance_in_the_bound(self) -> None:
        # counts=[15,12,10,8,5,3,2] -> 0 singletons, 55 total -> coverage point 1.0. The variance
        # floors f1 at 1: SE = sqrt((1/55 * 54/55 + 2/55) / 55) = 0.031397; lb = 0.948357 < 0.95.
        counts = [15, 12, 10, 8, 5, 3, 2]
        assert good_turing_coverage(counts) == 1.0
        result = saturation_check(counts)
        se = ((1 / 55 * 54 / 55 + 2 / 55) / 55) ** 0.5
        assert result.coverage.lb95 == pytest.approx(1.0 - 1.6448536269514722 * se, 1e-9)
        assert result.saturated is False

    def test_saturated(self) -> None:
        # Twice the evidence: SE = sqrt((1/110 * 109/110 + 2/110) / 110) -> lb 0.974 clears 0.95.
        result = saturation_check([40, 30, 20, 10, 5, 3, 2])
        assert result.coverage.lb95 > 0.95
        assert result.saturated is True

    def test_empty_counts(self) -> None:
        assert good_turing_coverage([]) == 0.0


# --- (negative) — total < 30 → never saturated ---


class TestSaturationFloor:
    """Saturation refuses below the minimum labeled-failure count."""

    def test_below_min_failures(self) -> None:
        # Total = 9, even with zero singletons
        counts = [5, 3, 1]
        result = saturation_check(counts)
        assert result.saturated is False
        assert "insufficient_observations" in (result.reason or "")

    def test_thirty_labels_without_singletons_are_not_proven_saturated(self) -> None:
        # [10, 10, 10] used to read lb95 = point = 1.0. Drawing 30 from two common modes plus 10%
        # rare mass over two modes declared saturation 15.5% of the time; the floored variance
        # bounds 30 such labels at 0.946.
        for counts in ([10, 10, 10], [15, 15], [27, 3]):
            result = saturation_check(counts)
            assert result.coverage.point == 1.0
            assert result.coverage.lb95 == pytest.approx(0.9460931, abs=1e-6)
            assert result.saturated is False

    def test_enough_labels_without_singletons_saturate(self) -> None:
        assert saturation_check([20, 20, 20]).saturated is True


# --- coverage < 0.95 → diagnostic ---


class TestSaturationDiagnostic:
    """Saturation returns reason when coverage < threshold."""

    def test_coverage_shortfall_reason(self) -> None:
        # Many singletons → low coverage; total > 30
        counts = [10, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert sum(counts) == 31  # above floor
        result = saturation_check(counts)
        assert result.saturated is False
        assert "coverage_below_threshold" in (result.reason or "")


# --- Chao1 with singletons predicts more modes ---


class TestChao1Estimate:
    """Chao1 species-richness estimator."""

    def test_singletons_predict_more(self) -> None:
        counts = [5, 3, 2, 1, 1]
        estimated, found, fraction = chao1_estimate(counts)
        assert estimated > found
        assert found == 5
        assert 0 < fraction < 1.0

    def test_known_answer(self) -> None:
        # S_obs=5, n1=2, n2=1 → N̂ = 5 + (4/2) = 7.0
        counts = [5, 3, 2, 1, 1]
        estimated, _found, fraction = chao1_estimate(counts)
        assert abs(estimated - 7.0) < 0.001
        assert abs(fraction - 5.0 / 7.0) < 0.001


# --- (negative/edge) — zero singletons → total == found ---


class TestChao1ZeroSingletons:
    """Zero singletons → estimated == found."""

    def test_no_singletons(self) -> None:
        counts = [5, 3, 2]
        estimated, found, fraction = chao1_estimate(counts)
        assert estimated == found == 3
        assert fraction == 1.0

    def test_empty_counts(self) -> None:
        estimated, found, _fraction = chao1_estimate([])
        assert estimated == 0.0
        assert found == 0


# --- gate on the coverage lower bound, refuse a one-mode collapse ---


class TestCoverageLowerBound:
    """The gate reads the Esty (1983) one-sided 95% lower bound, never the point."""

    def test_esty_bound_hand_computed(self) -> None:
        # counts [10,8,5,3,2,1]: n=29, f1=1, f2=1.
        # Var = (1/29 * 28/29 + 2/29) / 29 = (0.0332937 + 0.0689655) / 29 = 0.00352618
        # lb = (1 - 1/29) - 1.6448536 * sqrt(0.00352618) = 0.9655172 - 0.0976736 = 0.8678436
        bound = good_turing_coverage_bound([10, 8, 5, 3, 2, 1])
        assert bound.point == pytest.approx(28 / 29, abs=1e-12)
        assert bound.lb95 == pytest.approx(0.8678436, abs=1e-6)
        assert (bound.method, bound.level, bound.sided, bound.n) == (
            "good_turing_esty_normal",
            0.95,
            "lower",
            29,
        )

    def test_point_above_threshold_but_bound_below_is_not_saturated(self) -> None:
        # n=40, f1=1, f2=3: point = 0.975 >= 0.95, but
        # Var = (0.025 * 0.975 + 0.15) / 40 = 0.00435938
        # lb = 0.975 - 1.6448536 * 0.0660256 = 0.866398
        counts = [12, 10, 8, 2, 2, 2, 1, 3]
        result = saturation_check(counts)
        assert result.coverage.point == pytest.approx(0.975)
        assert result.coverage.lb95 == pytest.approx(0.8663976, abs=1e-6)
        assert result.saturated is False
        assert "coverage_below_threshold" in (result.reason or "")

    def test_single_mode_collapse_is_never_saturated(self) -> None:
        # One mode holding all 30 observations has coverage exactly 1.0 and taught nothing.
        result = saturation_check([30])
        assert result.coverage.point == 1.0
        assert result.saturated is False
        assert "too_few_modes" in (result.reason or "")

    def test_empty_bound_refused(self) -> None:
        with pytest.raises(ClusteringError, match="no observations"):
            good_turing_coverage_bound([])


# --- Chao1 by name, zeros dropped, Chao (1987) interval ---


class TestChao1:
    """Chao1 drops unobserved modes and carries its log-transformed interval."""

    def test_zero_count_modes_do_not_inflate_s_obs(self) -> None:
        padded = chao1_estimate([5, 3, 2, 1, 0, 0, 0])
        assert padded == chao1_estimate([5, 3, 2, 1])
        assert padded == pytest.approx((4.5, 4, 4 / 4.5))
        assert saturation_check([10, 10, 10, 0, 0]).total_modes == 3

    def test_log_transformed_interval_hand_computed(self) -> None:
        # S_obs=5, f1=2, f2=1 -> Chao1 = 7, T = 2. Chao (1987) / EstimateS eq. 5:
        # Var = f2 * (r^2/2 + r^3 + r^4/4) with r = 2 -> 2 + 8 + 4 = 14.
        # K = exp(1.959964 * sqrt(ln(1 + 14/4))) = exp(1.959964 * 1.2264083) = 11.064216
        # interval = [5 + 2/K, 5 + 2K] = [5.180763, 27.128433]
        ci = chao1_interval([5, 3, 2, 1, 1])
        k = math.exp(1.959963984540054 * math.sqrt(math.log(4.5)))
        assert k == pytest.approx(11.064216, abs=1e-5)
        assert ci.point == 7.0
        assert ci.lo == pytest.approx(5.180763, abs=1e-6)
        assert ci.hi == pytest.approx(27.128433, abs=1e-6)
        assert (ci.method, ci.level, ci.sided, ci.n) == (
            "chao1_log_transformed",
            0.95,
            "two-sided",
            12,
        )

    def test_no_doubletons_uses_bias_corrected_variance(self) -> None:
        # [5,1,1,1]: S_obs=4, f1=3, f2=0 -> Chao1 = 4 + 3*2/2 = 7, T = 3.
        # EstimateS eq. 7: Var = 3*2/2 + 3*25/4 - 81/(4*7) = 3 + 18.75 - 2.892857 = 18.857143
        # K = exp(1.959964 * sqrt(ln(1 + 18.857143/9))) = exp(1.959964 * 1.0626245) = 8.022...
        ci = chao1_interval([5, 1, 1, 1])
        var = 3 + 18.75 - 81 / 28
        k = math.exp(1.959963984540054 * math.sqrt(math.log(1 + var / 9)))
        assert ci.point == 7.0
        assert ci.lo == pytest.approx(4 + 3 / k, abs=1e-9)
        assert ci.hi == pytest.approx(4 + 3 * k, abs=1e-9)
        assert ci.lo == pytest.approx(4.373539, abs=1e-6)

    def test_no_singletons_interval_never_below_s_obs(self) -> None:
        # T = 0: eq. 14 on Var(S_obs). The lower edge is clamped at S_obs.
        ci = chao1_interval([5, 3, 2])
        assert ci.point == 3.0
        assert ci.lo == 3.0
        assert ci.hi > 3.0

    def test_interval_brackets_point(self) -> None:
        for counts in ([5, 3, 2, 1, 1], [5, 1, 1, 1], [5, 3, 2], [1, 1, 1, 2], [7, 1]):
            ci = chao1_interval(counts)
            assert ci.lo <= (ci.point or 0.0) <= ci.hi

    def test_empty_interval_refused(self) -> None:
        with pytest.raises(ClusteringError, match="no observed modes"):
            chao1_interval([0, 0])


def test_flagship_cluster_stage_reports_saturation_at_the_floor() -> None:
    """The flagship cluster stage keeps the saturation verdict and no longer lowers the floor."""
    from typing import cast

    from calibrated_agent_evaluation.flagship import cluster_stage
    from calibrated_agent_evaluation.flagship.orchestrator import StageContext

    critiques = ["refund over window", "refund over window", "wrong tool", "wrong tool"]

    def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if t.startswith("refund") else [0.0, 1.0] for t in texts]

    stage = cluster_stage(["over_refund"] * 4, critiques, embed, threshold=0.5)
    result = stage(cast(StageContext, None))
    # 4 failures is below the floor of 30, so the stage must say so (it used to pass
    # min_failures=1 and discard the result).
    assert "saturated=False" in result.note
    assert "insufficient_observations: 4 < 30" in result.note
