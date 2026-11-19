# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame-assembly engine.

Turns a taxonomy grid into two dataset frames:
- **Discovery frame** -- pairwise covering array (every 2-way interaction at least 1x).
- **Production-representative frame** -- proportional stratified sampling.

Public API re-exports from submodules.
"""

from __future__ import annotations

from calibrated_agent_evaluation.sampling.discovery import discovery_frame
from calibrated_agent_evaluation.sampling.errors import (
    DeffWarning,
    FrameContaminationError,
    SamplingConfigError,
)
from calibrated_agent_evaluation.sampling.production import production_frame
from calibrated_agent_evaluation.sampling.stats import kish_deff
from calibrated_agent_evaluation.sampling.types import (
    FrameResult,
    FrameRow,
    TaxonomyGrid,
    TrafficDistribution,
)
from calibrated_agent_evaluation.sampling.validate import (
    assert_production_frame,
    validate_distribution,
    validate_grid,
)

__all__ = [
    "DeffWarning",
    "FrameContaminationError",
    "FrameResult",
    "FrameRow",
    "SamplingConfigError",
    "TaxonomyGrid",
    "TrafficDistribution",
    "assert_production_frame",
    "discovery_frame",
    "kish_deff",
    "production_frame",
    "validate_distribution",
    "validate_grid",
]
