# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for taxonomy coverage report."""

from __future__ import annotations

from calibrated_agent_evaluation.cluster import coverage_report
from calibrated_agent_evaluation.sampling.types import TaxonomyGrid


def _make_grid() -> TaxonomyGrid:
    """3x3 grid, 9 total cells."""
    return TaxonomyGrid(
        dimensions={
            "feature": ["refund", "cancel", "track"],
            "scenario": ["happy", "edge", "error"],
        }
    )


# --- 7 of 9 cells covered ---


class TestCoverageReport:
    """Coverage report surfaces blind spots."""

    def test_partial_coverage(self) -> None:
        grid = _make_grid()
        observations = [
            {"feature": "refund", "scenario": "happy"},
            {"feature": "refund", "scenario": "edge"},
            {"feature": "refund", "scenario": "error"},
            {"feature": "cancel", "scenario": "happy"},
            {"feature": "cancel", "scenario": "edge"},
            {"feature": "track", "scenario": "happy"},
            {"feature": "track", "scenario": "edge"},
        ]
        result = coverage_report(grid, observations)
        assert result.total_cells == 9
        assert result.covered_cells == 7
        assert len(result.blind_spots) == 2
        # The blind spots should be cancel/error and track/error
        blind_spot_cells = [tuple(sorted(bs.items())) for bs in result.blind_spots]
        assert (("feature", "cancel"), ("scenario", "error")) in blind_spot_cells
        assert (("feature", "track"), ("scenario", "error")) in blind_spot_cells

    # --- (negative/edge) — empty observations ---

    def test_empty_observations_all_blind(self) -> None:
        grid = _make_grid()
        result = coverage_report(grid, [])
        assert result.total_cells == 9
        assert result.covered_cells == 0
        assert len(result.blind_spots) == 9

    # --- all cells covered ---

    def test_all_covered(self) -> None:
        grid = _make_grid()
        observations = [
            {"feature": f, "scenario": s}
            for f in ["refund", "cancel", "track"]
            for s in ["happy", "edge", "error"]
        ]
        result = coverage_report(grid, observations)
        assert result.total_cells == 9
        assert result.covered_cells == 9
        assert len(result.blind_spots) == 0

    # --- (edge) — an out-of-grid observation is surfaced, not silently dropped ---

    def test_out_of_grid_observation_is_surfaced_not_dropped(self) -> None:
        # An observed FAIL whose taxonomy_cell value is not in the grid (unknown/mis-tagged) must
        # be reported as `unclassified`, never silently discarded — dropping it would under-count
        # coverage and could paint a genuinely-observed area as a phantom blind spot.
        grid = _make_grid()
        observations = [
            {"feature": "refund", "scenario": "happy"},  # in grid
            {"feature": "UNKNOWN", "scenario": "happy"},  # feature not in grid
        ]
        result = coverage_report(grid, observations)
        assert result.covered_cells == 1
        assert len(result.unclassified) == 1
        assert result.unclassified[0] == {"feature": "UNKNOWN", "scenario": "happy"}
        # every in-grid count + the unclassified tail accounts for all observations (nothing lost)
        assert sum(result.cell_counts.values()) + len(result.unclassified) == len(observations)
