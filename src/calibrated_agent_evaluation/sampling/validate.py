# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input validation guards for the sampling engine."""

from __future__ import annotations

import math
from collections.abc import Sequence

from calibrated_agent_evaluation.sampling.errors import FrameContaminationError, SamplingConfigError
from calibrated_agent_evaluation.sampling.types import FrameRow, TaxonomyGrid, TrafficDistribution


def validate_grid(grid: TaxonomyGrid) -> None:
    """Raise on < 2 dims or any dim with < 2 values."""
    dims = grid.dimensions
    if not dims:
        raise SamplingConfigError(
            "Grid must have ≥2 dimensions each with ≥2 values; got an empty grid"
        )
    if len(dims) < 2:
        raise SamplingConfigError(
            f"Grid must have ≥2 dimensions each with ≥2 values; got {len(dims)} dimension(s)"
        )
    for name, values in dims.items():
        if len(values) < 2:
            raise SamplingConfigError(
                f"Grid must have ≥2 dimensions each with ≥2 values; "
                f"dimension {name!r} has {len(values)} value(s)"
            )


def validate_distribution(dist: TrafficDistribution) -> None:
    """Raise ``SamplingConfigError`` unless weights are finite, non-negative and sum to 1 ±0.01."""
    bad = {s: w for s, w in dist.strata_weights.items() if not math.isfinite(w) or w < 0}
    if bad:
        raise SamplingConfigError(
            f"Distribution weights must be finite and non-negative; got {bad}"
        )
    total = sum(dist.strata_weights.values())
    if abs(total - 1.0) > 0.01:
        raise SamplingConfigError(f"Distribution weights sum to {total:.4f}, expected 1.0 ±0.01")


def assert_production_frame(dataset: Sequence[FrameRow]) -> None:
    """Raise if any row has frame != 'production-representative'."""
    for row in dataset:
        if row.frame != "production-representative":
            raise FrameContaminationError(
                f"Headline rates require frame='production-representative'; "
                f"this dataset contains row(s) tagged '{row.frame}'"
            )
