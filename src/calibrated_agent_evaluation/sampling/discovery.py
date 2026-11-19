# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery frame: covering-array generation via allpairspy."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations, product

from calibrated_agent_evaluation.sampling.errors import SamplingConfigError
from calibrated_agent_evaluation.sampling.types import FrameResult, FrameRow, TaxonomyGrid
from calibrated_agent_evaluation.sampling.validate import validate_grid


def _covered_tuples(
    rows: list[list[str]], dim_count: int, strength: int
) -> set[tuple[tuple[int, str], ...]]:
    """Collect all t-way tuples covered by a set of rows."""
    covered: set[tuple[tuple[int, str], ...]] = set()
    for row in rows:
        for indices in combinations(range(dim_count), strength):
            covered.add(tuple((i, row[i]) for i in indices))
    return covered


def _all_required_tuples(
    dim_values: list[list[str]], strength: int
) -> set[tuple[tuple[int, str], ...]]:
    """Generate all t-way tuples that must be covered."""
    required: set[tuple[tuple[int, str], ...]] = set()
    for indices in combinations(range(len(dim_values)), strength):
        for values in product(*(dim_values[i] for i in indices)):
            required.add(tuple((idx, val) for idx, val in zip(indices, values, strict=True)))
    return required


def discovery_frame(
    grid: TaxonomyGrid,
    *,
    strength: int = 2,
    filter_func: Callable[[list[str]], bool] | None = None,
    previously_tested: list[list[str]] | None = None,
) -> FrameResult:
    """Produce a covering-array discovery frame from a taxonomy grid.

    Uses allpairspy's ``AllPairs`` as the base generator and supplements with
    additional rows to guarantee full t-way interaction coverage.
    """
    # Imported here, not at module top: `calibrated_agent_evaluation.run` reaches this package for
    # the frame check, and `calibrated-agent-evaluation run` must work with only the [stats] extra
    # installed.
    from allpairspy import AllPairs

    validate_grid(grid)

    dim_names = list(grid.dimensions.keys())
    dim_values = list(grid.dimensions.values())

    if strength > len(dim_names):
        raise SamplingConfigError(
            f"Cannot achieve {strength}-way coverage with only {len(dim_names)} dimensions"
        )

    # Wrap the allpairspy generator: any failure crossing this public surface becomes a typed
    # SamplingConfigError so callers can catch the CalibratedAgentEvaluationDataError family
    # (conventions: "wrap external errors at the boundary"; no raw vendor exception escapes).
    try:
        tuples: list[list[str]] = [
            list(row)
            for row in AllPairs(
                dim_values,
                filter_func=filter_func if filter_func is not None else (lambda _: True),
                previously_tested=previously_tested,
                n=strength,
            )
        ]
    except SamplingConfigError:
        raise
    except Exception as exc:  # allpairspy raised on the (pre-validated) grid; surface it typed
        raise SamplingConfigError(f"covering-array generation failed: {exc}") from exc

    if not tuples and filter_func is not None:
        raise SamplingConfigError("no valid combinations remain after filtering")

    if strength > 2 and tuples:
        tuples += _supplement_coverage(
            tuples, dim_values, strength, filter_func, previously_tested or []
        )

    rows = [
        FrameRow(
            cell={dim_names[i]: t[i] for i in range(len(dim_names))},
            frame="discovery",
            weight=1.0,
        )
        for t in tuples
    ]

    return FrameResult(rows=rows, frame="discovery", coverage_strength=strength)


def _supplement_coverage(
    rows: list[list[str]],
    dim_values: list[list[str]],
    strength: int,
    filter_func: Callable[[list[str]], bool] | None,
    previously_tested: list[list[str]],
) -> list[list[str]]:
    """The extra rows that close every t-way interaction ``rows`` (and earlier runs) leave open.

    The target tuple is picked in sorted order, not set-iteration order, so the supplemented rows
    and the row count are identical across processes (set iteration over string tuples varies
    with PYTHONHASHSEED). A tuple the filter makes unachievable is dropped.
    """
    n_dims = len(dim_values)
    covered = _covered_tuples(previously_tested + rows, n_dims, strength)
    missing = _all_required_tuples(dim_values, strength) - covered
    added: list[list[str]] = []
    while missing:
        target = min(missing)
        best_row = _best_row(target, missing, dim_values, strength, filter_func)
        if filter_func and not filter_func(best_row):
            missing.discard(target)
            continue
        added.append(best_row)
        missing -= {
            tuple((i, best_row[i]) for i in indices)
            for indices in combinations(range(n_dims), strength)
        }
    return added


def _best_row(
    target: tuple[tuple[int, str], ...],
    missing: set[tuple[tuple[int, str], ...]],
    dim_values: list[list[str]],
    strength: int,
    filter_func: Callable[[list[str]], bool] | None,
) -> list[str]:
    """The allowed row that covers ``target`` and the most other missing tuples (greedy)."""
    n_dims = len(dim_values)
    best_row = [dim_values[i][0] for i in range(n_dims)]
    for idx, val in target:
        best_row[idx] = val
    best_covers = 0
    for combo in product(*dim_values):
        row = list(combo)
        if not all(row[idx] == val for idx, val in target):
            continue
        if filter_func and not filter_func(row):
            continue
        covers = sum(
            1
            for indices in combinations(range(n_dims), strength)
            if tuple((i, row[i]) for i in indices) in missing
        )
        if covers > best_covers:
            best_covers = covers
            best_row = row
    return best_row
