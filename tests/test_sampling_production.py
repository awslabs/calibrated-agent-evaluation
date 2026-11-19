# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for calibrated_agent_evaluation.sampling.production."""

from __future__ import annotations

from collections import Counter

import pytest

from calibrated_agent_evaluation.sampling.errors import SamplingConfigError
from calibrated_agent_evaluation.sampling.production import production_frame
from calibrated_agent_evaluation.sampling.types import TaxonomyGrid, TrafficDistribution


@pytest.fixture
def grid_3x3() -> TaxonomyGrid:
    return TaxonomyGrid(
        dimensions={
            "feature": ["refund", "cancel", "status"],
            "scenario": ["routine", "edge", "high_stakes"],
        }
    )


@pytest.fixture
def dist_80_15_5() -> TrafficDistribution:
    return TrafficDistribution(
        strata_weights={
            "routine": 0.80,
            "edge": 0.15,
            "high_stakes": 0.05,
        }
    )


class TestProductionFrame:
    """Proportional stratified sample."""

    def test_proportions_match_distribution(
        self, grid_3x3: TaxonomyGrid, dist_80_15_5: TrafficDistribution
    ) -> None:
        result = production_frame(grid_3x3, dist_80_15_5, n=200)

        assert len(result.rows) == 200
        assert result.frame == "production-representative"

        strata_counts: Counter[str] = Counter()
        for row in result.rows:
            assert row.frame == "production-representative"
            strata_counts[row.cell["scenario"]] += 1

        # Within ±5% of target proportion or ±1 row
        for stratum, target_prop in dist_80_15_5.strata_weights.items():
            actual_prop = strata_counts[stratum] / 200
            tolerance = max(0.05, 1.0 / 200)
            assert abs(actual_prop - target_prop) <= tolerance, (
                f"Stratum {stratum}: expected ~{target_prop:.2f}, got {actual_prop:.2f}"
            )

    def test_n_less_than_strata_raises(
        self, grid_3x3: TaxonomyGrid, dist_80_15_5: TrafficDistribution
    ) -> None:
        with pytest.raises(SamplingConfigError, match="less than the number of strata"):
            production_frame(grid_3x3, dist_80_15_5, n=2)

    def test_uneven_allocation_totals_n_and_keeps_every_stratum(self) -> None:
        # B-1: a dominant stratum + several tiny ones at small n. The old "last stratum
        # absorbs the remainder" scheme overshot n and drove the last stratum negative
        # (→ 0 rows). Largest-remainder allocation must return exactly n rows with ≥1 per
        # stratum (the "minimum 1 row per stratum" contract).
        grid = TaxonomyGrid(
            dimensions={"feature": ["a", "b", "c", "d", "e"], "scenario": ["x", "y"]}
        )
        dist = TrafficDistribution(
            strata_weights={"a": 0.60, "b": 0.10, "c": 0.10, "d": 0.10, "e": 0.10}
        )
        result = production_frame(grid, dist, n=5)

        assert len(result.rows) == 5
        counts: Counter[str] = Counter(r.cell["feature"] for r in result.rows)
        assert set(counts) == {"a", "b", "c", "d", "e"}
        assert all(c >= 1 for c in counts.values())

    def test_unmatched_stratum_key_raises(self, grid_3x3: TaxonomyGrid) -> None:
        # M-1: a stratum key that names no taxonomy value must be refused, not silently
        # sampled from the whole grid (which would corrupt the sampled population).
        bad_dist = TrafficDistribution(strata_weights={"routine": 0.5, "typo_key": 0.5})
        with pytest.raises(SamplingConfigError, match="match no value in any grid dimension"):
            production_frame(grid_3x3, bad_dist, n=100)


class TestProductionFrameOversampling:
    """Oversampled strata have adjusted weights."""

    def test_oversampled_rows_have_lower_weight(
        self, grid_3x3: TaxonomyGrid, dist_80_15_5: TrafficDistribution
    ) -> None:
        result = production_frame(grid_3x3, dist_80_15_5, n=200, oversample={"high_stakes": 2.0})

        # Oversampled stratum has more rows than natural proportion
        strata_counts: Counter[str] = Counter()
        for row in result.rows:
            strata_counts[row.cell["scenario"]] += 1

        # high_stakes natural = round(0.05 * 200) = 10, oversampled = 20
        assert strata_counts["high_stakes"] > 10

        # Oversampled rows have weight < 1.0 relative to non-oversampled
        oversampled_weights = [r.weight for r in result.rows if r.cell["scenario"] == "high_stakes"]
        non_oversampled_weights = [
            r.weight for r in result.rows if r.cell["scenario"] != "high_stakes"
        ]
        assert all(w < max(non_oversampled_weights) for w in oversampled_weights)

    def test_weighted_sum_recovers_proportion(
        self, grid_3x3: TaxonomyGrid, dist_80_15_5: TrafficDistribution
    ) -> None:
        result = production_frame(grid_3x3, dist_80_15_5, n=200, oversample={"high_stakes": 2.0})

        # Weighted stratum proportions should match target
        total_weight = sum(r.weight for r in result.rows)
        for stratum, target_prop in dist_80_15_5.strata_weights.items():
            stratum_weight = sum(r.weight for r in result.rows if r.cell["scenario"] == stratum)
            actual_prop = stratum_weight / total_weight
            assert abs(actual_prop - target_prop) < 0.05, (
                f"Weighted {stratum}: expected ~{target_prop:.2f}, got {actual_prop:.3f}"
            )

    def test_no_oversampling_all_weights_1(
        self, grid_3x3: TaxonomyGrid, dist_80_15_5: TrafficDistribution
    ) -> None:
        result = production_frame(grid_3x3, dist_80_15_5, n=200)

        assert all(r.weight == 1.0 for r in result.rows)
        assert result.deff is None
        assert result.n_eff is None


class TestProductionFrameKish:
    """Oversampled frame reports deff > 1.0 and n_eff < n."""

    def test_oversampled_reports_deff_and_neff(
        self, grid_3x3: TaxonomyGrid, dist_80_15_5: TrafficDistribution
    ) -> None:
        result = production_frame(grid_3x3, dist_80_15_5, n=200, oversample={"high_stakes": 2.0})

        assert result.deff is not None
        assert result.n_eff is not None
        assert result.deff > 1.0
        assert result.n_eff < len(result.rows)


class TestProductionFrameNegative:
    """Invalid distribution raises SamplingConfigError."""

    def test_distribution_summing_to_0_7_raises(self, grid_3x3: TaxonomyGrid) -> None:
        bad_dist = TrafficDistribution(strata_weights={"routine": 0.5, "edge": 0.2})
        with pytest.raises(SamplingConfigError, match="sum to"):
            production_frame(grid_3x3, bad_dist, n=100)


class TestAllocationArithmetic:
    """The allocation sums to exactly n and refuses weights that cannot be apportioned."""

    def test_weights_summing_above_one_still_allocate_exactly_n(self) -> None:
        from calibrated_agent_evaluation.sampling.production import _proportional_allocation

        weights = {"a": 1.0099 / 3, "b": 1.0099 / 3, "c": 1.0099 / 3}
        allocation = _proportional_allocation(weights, 122)
        assert sum(allocation.values()) == 122
        assert all(rows >= 1 for rows in allocation.values())

    def test_a_negative_weight_is_refused_by_the_validator(self) -> None:
        from calibrated_agent_evaluation.sampling.validate import validate_distribution

        with pytest.raises(SamplingConfigError, match="non-negative"):
            validate_distribution(TrafficDistribution(strata_weights={"a": 1.2, "b": -0.2}))

    def test_a_negative_weight_is_refused_by_the_allocation(self) -> None:
        from calibrated_agent_evaluation.sampling.production import _proportional_allocation

        with pytest.raises(SamplingConfigError, match="non-negative"):
            _proportional_allocation({"a": 1.2, "b": -0.2}, 10)

    def test_a_zero_total_is_refused(self) -> None:
        from calibrated_agent_evaluation.sampling.production import _proportional_allocation

        with pytest.raises(SamplingConfigError, match="zero"):
            _proportional_allocation({"a": 0.0, "b": 0.0}, 10)


class TestStrataPartitionThePopulation:
    """Strata from one dimension only, weighted by the allocation actually drawn."""

    def test_strata_from_two_dimensions_are_refused(self) -> None:
        # {"refund": .5, "en": .5} drew refund-en rows for both keys: refund 70%, en 80%.
        grid = TaxonomyGrid({"intent": ["refund", "order", "other"], "lang": ["en", "es"]})
        with pytest.raises(SamplingConfigError, match="more than one dimension"):
            production_frame(grid, TrafficDistribution({"refund": 0.5, "en": 0.5}), n=10)

    def test_a_key_naming_values_in_two_dimensions_is_refused(self) -> None:
        grid = TaxonomyGrid({"intent": ["refund", "other"], "channel": ["web", "other"]})
        with pytest.raises(SamplingConfigError, match="more than one dimension, so the cells"):
            production_frame(grid, TrafficDistribution({"refund": 0.5, "other": 0.5}), n=10)

    def test_a_stratum_raised_to_one_row_is_weighted_back_to_its_declared_share(self) -> None:
        # 2% of 20 rows rounds to 0 and is repaired to 1 row (5%); its weight undoes that.
        grid = TaxonomyGrid({"intent": ["refund", "order", "other"], "lang": ["en", "es"]})
        dist = TrafficDistribution({"refund": 0.49, "order": 0.49, "other": 0.02})
        result = production_frame(grid, dist, n=20)
        total = sum(r.weight for r in result.rows)
        for stratum, declared in dist.strata_weights.items():
            share = sum(r.weight for r in result.rows if r.cell["intent"] == stratum) / total
            assert share == pytest.approx(declared)
        other = [r for r in result.rows if r.cell["intent"] == "other"]
        assert len(other) == 1
        assert other[0].weight == pytest.approx(0.4)
        assert result.deff is not None
        assert result.deff > 1.0

    def test_cells_of_a_stratum_come_only_from_its_value(self) -> None:
        grid = TaxonomyGrid({"intent": ["refund", "order"], "lang": ["en", "es"]})
        result = production_frame(grid, TrafficDistribution({"refund": 0.5, "order": 0.5}), n=8)
        counts = Counter(r.cell["intent"] for r in result.rows)
        assert counts == {"refund": 4, "order": 4}
