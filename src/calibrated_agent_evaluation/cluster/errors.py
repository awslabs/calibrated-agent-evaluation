# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error hierarchy for the clustering module."""

from __future__ import annotations

from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError


class ClusteringError(CalibratedAgentEvaluationDataError):
    """Invalid input or state in the clustering pipeline."""


class UnratifiedCodebookError(ClusteringError):
    """Attempt to use a draft codebook without human ratification."""
