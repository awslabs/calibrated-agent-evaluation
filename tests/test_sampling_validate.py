# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for calibrated_agent_evaluation.sampling.validate."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.sampling.errors import FrameContaminationError, SamplingConfigError
from calibrated_agent_evaluation.sampling.types import FrameRow, TaxonomyGrid, TrafficDistribution
from calibrated_agent_evaluation.sampling.validate import (
    assert_production_frame,
    validate_distribution,
    validate_grid,
)


class TestValidateGrid:
    """Grid must have ≥2 dimensions each with ≥2 values."""

    def test_rejects_empty_grid(self) -> None:
        grid = TaxonomyGrid(dimensions={})
        with pytest.raises(SamplingConfigError, match="empty grid"):
            validate_grid(grid)

    def test_rejects_single_dimension(self) -> None:
        grid = TaxonomyGrid(dimensions={"feature": ["refund", "cancel"]})
        with pytest.raises(SamplingConfigError, match="1 dimension"):
            validate_grid(grid)

    def test_rejects_dimension_with_single_value(self) -> None:
        grid = TaxonomyGrid(dimensions={"feature": ["refund", "cancel"], "scenario": ["only_one"]})
        with pytest.raises(SamplingConfigError, match="dimension 'scenario' has 1 value"):
            validate_grid(grid)

    def test_accepts_minimal_valid_grid(self) -> None:
        grid = TaxonomyGrid(
            dimensions={"feature": ["refund", "cancel"], "scenario": ["happy", "sad"]}
        )
        validate_grid(grid)  # should not raise


class TestValidateDistribution:
    """Distribution weights must sum to 1.0 ±0.01."""

    def test_rejects_weights_summing_to_0_8(self) -> None:
        dist = TrafficDistribution(strata_weights={"a": 0.5, "b": 0.3})
        with pytest.raises(SamplingConfigError, match=r"sum to 0\.8000"):
            validate_distribution(dist)

    def test_rejects_weights_summing_to_1_05(self) -> None:
        dist = TrafficDistribution(strata_weights={"a": 0.6, "b": 0.45})
        with pytest.raises(SamplingConfigError, match=r"expected 1\.0"):
            validate_distribution(dist)

    def test_accepts_weights_summing_to_0_995(self) -> None:
        dist = TrafficDistribution(strata_weights={"a": 0.5, "b": 0.495})
        validate_distribution(dist)  # 0.995 is within ±0.01

    def test_accepts_weights_summing_to_1_005(self) -> None:
        dist = TrafficDistribution(strata_weights={"a": 0.5, "b": 0.505})
        validate_distribution(dist)  # 1.005 is within ±0.01


class TestAssertProductionFrame:
    """Refuse headline-rate on non-production-representative data."""

    def test_mixed_dataset_raises(self) -> None:
        dataset = [
            FrameRow(cell={"a": "1"}, frame="production-representative"),
            FrameRow(cell={"a": "2"}, frame="discovery"),
        ]
        with pytest.raises(FrameContaminationError, match="tagged 'discovery'"):
            assert_production_frame(dataset)

    def test_pure_discovery_raises(self) -> None:
        dataset = [
            FrameRow(cell={"a": "1"}, frame="discovery"),
            FrameRow(cell={"a": "2"}, frame="discovery"),
        ]
        with pytest.raises(FrameContaminationError, match="production-representative"):
            assert_production_frame(dataset)

    def test_pure_production_passes(self) -> None:
        dataset = [
            FrameRow(cell={"a": "1"}, frame="production-representative"),
            FrameRow(cell={"a": "2"}, frame="production-representative"),
        ]
        assert_production_frame(dataset)  # should not raise
