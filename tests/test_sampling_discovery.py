# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for calibrated_agent_evaluation.sampling.discovery."""

from __future__ import annotations

import os
import subprocess
import sys
from itertools import combinations, product

import pytest

from calibrated_agent_evaluation.sampling.discovery import discovery_frame
from calibrated_agent_evaluation.sampling.errors import SamplingConfigError
from calibrated_agent_evaluation.sampling.types import TaxonomyGrid

# Emits the strength=3 discovery frame as `count|cell,cell,...` — run under two different
# PYTHONHASHSEED values to prove the t>2 supplement is hash-seed-independent (reproducibility).
_DETERMINISM_SNIPPET = """
from calibrated_agent_evaluation.sampling.discovery import discovery_frame
from calibrated_agent_evaluation.sampling.types import TaxonomyGrid
grid = TaxonomyGrid(dimensions={
    "a": ["a1", "a2", "a3", "a4"],
    "b": ["b1", "b2", "b3", "b4"],
    "c": ["c1", "c2", "c3", "c4"],
    "d": ["d1", "d2", "d3", "d4"],
})
res = discovery_frame(grid, strength=3)
print(str(len(res.rows)) + "|" + "|".join(",".join(r.cell.values()) for r in res.rows))
"""


def _all_t_way_tuples(grid: TaxonomyGrid, t: int) -> set[tuple[tuple[str, str], ...]]:
    """Generate all t-way tuples that must be covered."""
    dim_names = list(grid.dimensions.keys())
    tuples: set[tuple[tuple[str, str], ...]] = set()
    for dim_combo in combinations(dim_names, t):
        value_lists = [grid.dimensions[d] for d in dim_combo]
        for values in product(*value_lists):
            tuples.add(tuple(zip(dim_combo, values, strict=True)))
    return tuples


def _covered_t_way(
    rows: list[dict[str, str]], dims: list[str], t: int
) -> set[tuple[tuple[str, str], ...]]:
    """Collect all t-way tuples covered by the given rows."""
    covered: set[tuple[tuple[str, str], ...]] = set()
    for row in rows:
        for dim_combo in combinations(dims, t):
            covered.add(tuple((d, row[d]) for d in dim_combo))
    return covered


class TestDiscoveryFramePairwise:
    """4x4x3 grid -- every 2-way pair appears in at least 1 row."""

    def test_all_pairwise_combinations_covered(self) -> None:
        grid = TaxonomyGrid(
            dimensions={
                "feature": ["refund", "cancel", "status", "billing"],
                "scenario": ["happy", "sad", "edge", "timeout"],
                "persona": ["impatient", "friendly", "confused"],
            }
        )
        result = discovery_frame(grid)

        needed = _all_t_way_tuples(grid, 2)
        covered = _covered_t_way([r.cell for r in result.rows], list(grid.dimensions.keys()), 2)
        assert needed <= covered

    def test_all_rows_tagged_discovery_weight_1(self) -> None:
        grid = TaxonomyGrid(
            dimensions={
                "feature": ["refund", "cancel", "status", "billing"],
                "scenario": ["happy", "sad", "edge", "timeout"],
                "persona": ["impatient", "friendly", "confused"],
            }
        )
        result = discovery_frame(grid)

        assert result.frame == "discovery"
        for row in result.rows:
            assert row.frame == "discovery"
            assert row.weight == 1.0


class TestDiscoveryFrameStrength:
    """T-way coverage for t > 2."""

    def test_strength_3_covers_all_3way(self) -> None:
        grid = TaxonomyGrid(
            dimensions={
                "a": ["a1", "a2", "a3"],
                "b": ["b1", "b2", "b3"],
                "c": ["c1", "c2", "c3"],
            }
        )
        result = discovery_frame(grid, strength=3)

        needed = _all_t_way_tuples(grid, 3)
        covered = _covered_t_way([r.cell for r in result.rows], list(grid.dimensions.keys()), 3)
        assert needed <= covered

    def test_strength_exceeds_dimensions_raises(self) -> None:
        grid = TaxonomyGrid(
            dimensions={
                "a": ["a1", "a2", "a3"],
                "b": ["b1", "b2", "b3"],
                "c": ["c1", "c2", "c3"],
            }
        )
        with pytest.raises(SamplingConfigError, match="5-way coverage with only 3 dimensions"):
            discovery_frame(grid, strength=5)


class TestDiscoveryFrameFilter:
    """Filter function excludes invalid combinations."""

    def test_filter_excludes_invalid_combination(self) -> None:
        grid = TaxonomyGrid(
            dimensions={
                "a": ["a1", "a2"],
                "b": ["b1", "b2"],
                "c": ["c1", "c2"],
            }
        )

        # Exclude the combination where a=a1 AND b=b1
        def exclude_a1_b1(row: list[str]) -> bool:
            if len(row) >= 2:
                return not (row[0] == "a1" and row[1] == "b1")
            return True

        result = discovery_frame(grid, filter_func=exclude_a1_b1)

        for row in result.rows:
            assert not (row.cell["a"] == "a1" and row.cell["b"] == "b1")

    def test_filter_rejects_everything_raises(self) -> None:
        grid = TaxonomyGrid(
            dimensions={
                "a": ["a1", "a2"],
                "b": ["b1", "b2"],
                "c": ["c1", "c2"],
            }
        )
        with pytest.raises(SamplingConfigError, match="no valid combinations remain"):
            discovery_frame(grid, filter_func=lambda _: False)


class TestDiscoveryFramePreviouslyTested:
    """Previously tested configurations are excluded."""

    def test_previously_tested_excluded(self) -> None:
        grid = TaxonomyGrid(
            dimensions={
                "a": ["a1", "a2"],
                "b": ["b1", "b2"],
                "c": ["c1", "c2"],
            }
        )
        # Generate full set first
        full_result = discovery_frame(grid)

        # Now pass some as previously tested
        prev = [list(full_result.rows[0].cell.values())]
        incr_result = discovery_frame(grid, previously_tested=prev)

        # Incremental result is smaller or equal (some configs excluded)
        assert len(incr_result.rows) <= len(full_result.rows)

    def test_all_configs_already_tested_returns_empty(self) -> None:
        grid = TaxonomyGrid(
            dimensions={
                "a": ["a1", "a2"],
                "b": ["b1", "b2"],
                "c": ["c1", "c2"],
            }
        )
        # Generate all combinations as "previously tested"
        all_combos = [
            [a, b, c]
            for a in grid.dimensions["a"]
            for b in grid.dimensions["b"]
            for c in grid.dimensions["c"]
        ]
        result = discovery_frame(grid, previously_tested=all_combos)

        assert len(result.rows) == 0


class TestDiscoveryFrameNegative:
    """Invalid grid raises SamplingConfigError."""

    def test_invalid_grid_raises(self) -> None:
        grid = TaxonomyGrid(dimensions={"only_one": ["a", "b"]})
        with pytest.raises(SamplingConfigError):
            discovery_frame(grid)


class TestDiscoveryFrameReproducibility:
    """Deterministic output across runs."""

    def test_same_grid_same_output(self) -> None:
        grid = TaxonomyGrid(
            dimensions={
                "feature": ["refund", "cancel", "status", "billing"],
                "scenario": ["happy", "sad", "edge", "timeout"],
                "persona": ["impatient", "friendly", "confused"],
            }
        )
        result1 = discovery_frame(grid)
        result2 = discovery_frame(grid)

        assert len(result1.rows) == len(result2.rows)
        for r1, r2 in zip(result1.rows, result2.rows, strict=True):
            assert r1.cell == r2.cell
            assert r1.frame == r2.frame
            assert r1.weight == r2.weight

    def test_tway_supplement_deterministic_across_processes(self) -> None:
        # H-1: the t>2 supplement iterated a str-keyed set, so its output (and row count)
        # varied with PYTHONHASHSEED. Two subprocesses with different hash seeds must now
        # produce byte-identical frames. An in-process double-call can't catch this — both
        # calls share one interpreter's hash seed.
        def _run(seed: str) -> str:
            proc = subprocess.run(
                [sys.executable, "-c", _DETERMINISM_SNIPPET],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "PYTHONHASHSEED": seed},
            )
            return proc.stdout.strip()

        out_a = _run("0")
        out_b = _run("1")
        assert out_a == out_b
        assert out_a  # non-empty guard: the snippet actually produced rows
