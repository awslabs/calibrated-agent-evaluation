# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production-representative frame: proportional stratified sampling."""

from __future__ import annotations

import math
from itertools import product

from calibrated_agent_evaluation.sampling.errors import SamplingConfigError
from calibrated_agent_evaluation.sampling.stats import DEFAULT_DEFF_THRESHOLD, kish_deff
from calibrated_agent_evaluation.sampling.types import (
    FrameResult,
    FrameRow,
    TaxonomyGrid,
    TrafficDistribution,
)
from calibrated_agent_evaluation.sampling.validate import validate_distribution, validate_grid


def _strata_cells(
    grid: TaxonomyGrid, distribution: TrafficDistribution
) -> dict[str, list[dict[str, str]]]:
    """Map each stratum key to the grid cells it selects.

    Strata must partition the population, so every key must name a value of one and the same
    dimension: keys from two dimensions select overlapping cells (``{"refund": .5, "en": .5}``
    drew refund-en rows for both and gave refund a 70% share), and a key that names a value in two
    dimensions is ambiguous. A key that matches no cell is a config error, not a silent fall-back
    to the whole grid: sampling the wrong population would corrupt every downstream rate.
    """
    dim_names = list(grid.dimensions.keys())
    owners: dict[str, list[str]] = {
        stratum: [dim for dim in dim_names if stratum in grid.dimensions[dim]]
        for stratum in distribution.strata_weights
    }
    unmatched = sorted(s for s, dims in owners.items() if not dims)
    if unmatched:
        raise SamplingConfigError(
            f"stratum key(s) {unmatched!r} match no value in any grid dimension "
            f"{dim_names!r}; every strata_weights key must name a taxonomy value"
        )
    ambiguous = {s: dims for s, dims in owners.items() if len(dims) > 1}
    if ambiguous:
        raise SamplingConfigError(
            f"stratum key(s) {ambiguous!r} name a value in more than one dimension, so the cells "
            "they select overlap; rename the values so each stratum key belongs to one dimension"
        )
    used = sorted({dims[0] for dims in owners.values()})
    if len(used) > 1:
        by_dim = {d: sorted(s for s, dims in owners.items() if dims[0] == d) for d in used}
        raise SamplingConfigError(
            f"strata_weights keys come from more than one dimension {by_dim!r}; strata from two "
            "dimensions overlap (a cell matches one key of each), so each stratum's share could "
            "not match its declared weight. Declare the distribution over one dimension's values"
        )
    (dim,) = used
    all_cells = [
        dict(zip(dim_names, combo, strict=True)) for combo in product(*grid.dimensions.values())
    ]
    return {
        stratum: [cell for cell in all_cells if cell[dim] == stratum]
        for stratum in distribution.strata_weights
    }


def _stratum_weights(
    distribution: TrafficDistribution, allocation: dict[str, int]
) -> dict[str, float]:
    """Per-row design weight for each stratum, so the weighted shares recover the declared ones.

    Weight = (declared proportion * total rows) / rows drawn for the stratum, from the allocation
    actually drawn. A stratum raised to its 1-row minimum, or oversampled, carries the weight that
    undoes it; a stratum drawn at exactly its quota has weight 1.
    """
    total = sum(allocation.values())
    declared = sum(distribution.strata_weights.values())
    weights: dict[str, float] = {}
    for stratum, count in allocation.items():
        weight = distribution.strata_weights[stratum] / declared * total / count
        weights[stratum] = 1.0 if math.isclose(weight, 1.0, rel_tol=1e-9) else weight
    return weights


def _proportional_allocation(strata_weights: dict[str, float], n: int) -> dict[str, int]:
    """Apportion *n* rows across strata proportionally, ≥1 per stratum, summing to exactly *n*.

    Uses the largest-remainder (Hamilton) method so the row counts are the closest
    integers to the ideal quotas ``w_s * n`` whose sum is exactly ``n``; then repairs
    any zero-allocated stratum up to 1 by borrowing from the currently-largest stratum.
    Both passes iterate strata in sorted key order, so the result is deterministic
    and never negative (the naive "last stratum absorbs the remainder" scheme
    could drive it negative). The caller guarantees ``n >= len(strata_weights)``,
    so a ≥1-per-stratum solution always exists (minimum 1 row per stratum).

    Weights are normalized by their sum first: a distribution may sum to anything within the
    validator's tolerance, and quotas on unnormalized weights can floor to more than *n* rows.
    Raises ``SamplingConfigError`` for a negative or non-finite weight or a zero total.
    """
    strata_sorted = sorted(strata_weights)
    if any(not math.isfinite(w) or w < 0 for w in strata_weights.values()):
        raise SamplingConfigError(
            f"strata weights must be finite and non-negative; got {strata_weights}"
        )
    total = sum(strata_weights.values())
    if total <= 0:
        raise SamplingConfigError("strata weights sum to zero; nothing to apportion")

    # Largest-remainder: floor the quotas, then hand the leftover rows to the largest
    # fractional remainders (ties broken by sorted key for determinism).
    quotas = {s: strata_weights[s] / total * n for s in strata_sorted}
    alloc = {s: int(quotas[s]) for s in strata_sorted}
    leftover = n - sum(alloc.values())
    by_remainder = sorted(strata_sorted, key=lambda s: (-(quotas[s] - int(quotas[s])), s))
    for s in by_remainder[:leftover]:
        alloc[s] += 1

    # Repair: raise any zero-allocated stratum to 1, borrowing a row from the current
    # max-allocation stratum. n >= num_strata guarantees a donor with ≥2 always exists.
    for s in strata_sorted:
        if alloc[s] == 0:
            donor = max(strata_sorted, key=lambda d: (alloc[d], d))
            alloc[donor] -= 1
            alloc[s] += 1

    return alloc


def production_frame(
    grid: TaxonomyGrid,
    distribution: TrafficDistribution,
    n: int,
    *,
    oversample: dict[str, float] | None = None,
    deff_threshold: float = DEFAULT_DEFF_THRESHOLD,
) -> FrameResult:
    """Produce a proportional stratified sample.

    The strata are the values of one grid dimension. Rows are allocated in proportion to the
    declared distribution, at least one per stratum, and every row carries the design weight of its
    stratum's realised allocation, so the weighted shares match the declared distribution even
    where rounding, the 1-row minimum or *oversample* moved the row counts. When any weight differs
    from 1 the Kish design effect is reported, with its warning threshold forwarded via
    *deff_threshold*.

    Raises:
        SamplingConfigError: The grid or distribution is invalid, a stratum key names no value or
            a value in two dimensions, the keys come from more than one dimension, or ``n`` is
            smaller than the number of strata.
    """
    validate_grid(grid)
    validate_distribution(distribution)

    num_strata = len(distribution.strata_weights)
    if n < num_strata:
        raise SamplingConfigError(
            f"n={n} is less than the number of strata ({num_strata}); "
            f"minimum 1 row per stratum required"
        )
    strata_to_cells = _strata_cells(grid, distribution)

    # Proportional allocation (largest-remainder, >=1 per stratum, sums to exactly n), then the
    # oversampling multipliers.
    allocation = _proportional_allocation(distribution.strata_weights, n)
    if oversample:
        allocation = {
            stratum: max(1, round(count * oversample.get(stratum, 1.0)))
            for stratum, count in allocation.items()
        }
    weights = _stratum_weights(distribution, allocation)

    rows: list[FrameRow] = []
    for stratum in sorted(distribution.strata_weights):
        cells = strata_to_cells[stratum]
        weight = weights[stratum]
        rows.extend(
            FrameRow(cell=cells[i % len(cells)], frame="production-representative", weight=weight)
            for i in range(allocation[stratum])
        )

    deff_val: float | None = None
    n_eff_val: float | None = None
    if any(w != 1.0 for w in weights.values()):
        deff_val, n_eff_val = kish_deff([r.weight for r in rows], threshold=deff_threshold)

    return FrameResult(
        rows=rows,
        frame="production-representative",
        deff=deff_val,
        n_eff=n_eff_val,
    )
