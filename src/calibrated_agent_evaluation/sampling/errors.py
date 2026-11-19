# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error hierarchy for the sampling engine.

Every sampling failure is a typed, explained refusal (conventions: "no silent failure").
All errors subclass
:class:`~calibrated_agent_evaluation.schema.errors.CalibratedAgentEvaluationDataError` so callers
can catch the whole data-plane family at the boundary.
"""

from __future__ import annotations

from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError


class SamplingConfigError(CalibratedAgentEvaluationDataError):
    """Invalid grid or distribution input."""


class FrameContaminationError(CalibratedAgentEvaluationDataError):
    """Attempt to use a discovery frame where production-representative is required."""


class DeffWarning(UserWarning):
    """Non-fatal design-effect warning when Deff exceeds a threshold."""
