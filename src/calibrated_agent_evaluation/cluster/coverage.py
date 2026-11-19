# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Taxonomy coverage report: surfaces blind spots."""

from __future__ import annotations

from itertools import product

from calibrated_agent_evaluation.cluster.types import CoverageResult
from calibrated_agent_evaluation.sampling.types import TaxonomyGrid


def coverage_report(
    grid: TaxonomyGrid,
    observations: list[dict[str, str]],
) -> CoverageResult:
    """Report per-cell observation counts and blind spots.

    Args:
        grid: The taxonomy dimensions and their values.
        observations: The taxonomy cell of each FAIL-labeled trace.

    Returns:
        The count per cell, the cells with no failures, and observations outside the grid.
    """
    dim_names = sorted(grid.dimensions.keys())
    dim_values = [grid.dimensions[d] for d in dim_names]

    # Build all cells as canonical key strings
    all_cells: list[dict[str, str]] = []
    for combo in product(*dim_values):
        cell = dict(zip(dim_names, combo, strict=True))
        all_cells.append(cell)

    def _cell_key(cell: dict[str, str]) -> str:
        return "|".join(f"{k}={cell.get(k, '')}" for k in dim_names)

    # Count observations per cell. An observation whose taxonomy_cell is not a grid cell (an
    # unknown/extra dimension value) is recorded as UNCLASSIFIED, never silently discarded;
    # dropping it would under-count coverage and mislabel a genuinely-observed cell as a blind spot.
    cell_counts: dict[str, int] = {_cell_key(c): 0 for c in all_cells}
    unclassified: list[dict[str, str]] = []
    for obs in observations:
        key = _cell_key(obs)
        if key in cell_counts:
            cell_counts[key] += 1
        else:
            unclassified.append(obs)

    blind_spots: list[dict[str, str]] = []
    for cell in all_cells:
        if cell_counts[_cell_key(cell)] == 0:
            blind_spots.append(cell)

    covered = sum(1 for v in cell_counts.values() if v > 0)

    return CoverageResult(
        cell_counts=cell_counts,
        blind_spots=tuple(blind_spots),
        total_cells=len(all_cells),
        covered_cells=covered,
        unclassified=tuple(unclassified),
    )
