# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clustering, saturation & coverage engine.

Public API re-exports; import from ``calibrated_agent_evaluation.cluster``.
"""

from calibrated_agent_evaluation.cluster.codebook import ratify, tag_label
from calibrated_agent_evaluation.cluster.coverage import coverage_report
from calibrated_agent_evaluation.cluster.draft import build_draft
from calibrated_agent_evaluation.cluster.errors import ClusteringError, UnratifiedCodebookError
from calibrated_agent_evaluation.cluster.group import hierarchical_cluster
from calibrated_agent_evaluation.cluster.saturation import (
    chao1_estimate,
    chao1_interval,
    good_turing_coverage,
    good_turing_coverage_bound,
    saturation_check,
)
from calibrated_agent_evaluation.cluster.types import (
    Codebook,
    CoverageResult,
    DraftCluster,
    DraftCodebook,
    FailureMode,
    SaturationResult,
)

__all__ = [
    "ClusteringError",
    "Codebook",
    "CoverageResult",
    "DraftCluster",
    "DraftCodebook",
    "FailureMode",
    "SaturationResult",
    "UnratifiedCodebookError",
    "build_draft",
    "chao1_estimate",
    "chao1_interval",
    "coverage_report",
    "good_turing_coverage",
    "good_turing_coverage_bound",
    "hierarchical_cluster",
    "ratify",
    "saturation_check",
    "tag_label",
]
