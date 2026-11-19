# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data types for the clustering, saturation & coverage module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from calibrated_agent_evaluation.calibrate.record import Bound, Interval


@dataclass(frozen=True)
class DraftCluster:
    """One proposed failure-mode cluster before human ratification."""

    name: str
    member_indices: tuple[int, ...]
    member_critiques: tuple[str, ...]


@dataclass(frozen=True)
class DraftCodebook:
    """The output of clustering: NOT usable for tagging."""

    clusters: tuple[DraftCluster, ...]
    embedding_model: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class FailureMode:
    """One named failure mode in a ratified codebook."""

    name: str
    description: str = ""
    example_critiques: tuple[str, ...] = ()


@dataclass(frozen=True)
class Codebook:
    """A versioned, human-ratified catalog of failure modes.

    Only a Codebook with ``ratified_by`` set may tag Labels.
    """

    version: str
    modes: tuple[FailureMode, ...]
    ratified_by: str
    ratified_at: datetime
    parent_version: str | None = None


@dataclass(frozen=True)
class SaturationResult:
    """Output of saturation_check(): answers 'should I keep labeling?'.

    ``coverage`` is the Good-Turing coverage with its one-sided 95% Esty lower bound (the gate
    reads ``coverage.lb95``). ``chao1`` is the Chao1 richness estimate with its two-sided 95%
    log-transformed interval, ``None`` when nothing was observed.
    """

    saturated: bool
    coverage: Bound
    total_failures: int
    total_modes: int
    singletons: int
    reason: str | None = None
    chao1: Interval | None = None


@dataclass(frozen=True)
class CoverageResult:
    """Output of coverage_report(): surfaces taxonomy blind spots.

    ``unclassified`` are observed FAIL cells whose taxonomy_cell does not match any grid cell
    (an unknown/extra dimension value). They are surfaced, not silently dropped; a genuinely
    observed failure that falls outside the declared grid is a signal (the grid is incomplete or
    the trace is mis-tagged), not something to under-count into a phantom blind spot.
    """

    cell_counts: dict[str, int] = field(default_factory=dict)
    blind_spots: tuple[dict[str, str], ...] = ()
    total_cells: int = 0
    covered_cells: int = 0
    unclassified: tuple[dict[str, str], ...] = ()
