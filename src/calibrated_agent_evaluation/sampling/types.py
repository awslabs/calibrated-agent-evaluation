# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data model for the sampling engine.

Pure frozen dataclasses; no I/O, no DB (the sampling engine is stateless computation).
Types use the normalized schema's ``Frame`` literal (no vendor type).
"""

from __future__ import annotations

from dataclasses import dataclass

from calibrated_agent_evaluation.schema.types import Frame


@dataclass(frozen=True)
class TaxonomyGrid:
    """Input grid: dimension name → list of possible values."""

    dimensions: dict[str, list[str]]


@dataclass(frozen=True)
class TrafficDistribution:
    """Declared stratum weights for production-representative sampling."""

    strata_weights: dict[str, float]


@dataclass(frozen=True)
class FrameRow:
    """One row in a generated frame: a specific taxonomy-cell assignment."""

    cell: dict[str, str]
    frame: Frame
    weight: float = 1.0


@dataclass(frozen=True)
class FrameResult:
    """Output of a frame generation operation."""

    rows: list[FrameRow]
    frame: Frame
    coverage_strength: int = 2
    deff: float | None = None
    n_eff: float | None = None
