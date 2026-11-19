# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error hierarchy for the crew-attribution layer.

Every attribution failure is a typed, explained refusal (no silent failure).
All errors subclass
:class:`~calibrated_agent_evaluation.schema.errors.CalibratedAgentEvaluationDataError` so callers
can catch the whole data-plane family at the boundary.
"""

from __future__ import annotations

from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError


class AttributionError(CalibratedAgentEvaluationDataError):
    """A walk-back that doesn't fit the trace.

    It names a span absent from the trace, orders the culprit after the manifestation, or
    implicates an undeclared handoff seam.
    """


class CrewDefinitionError(CalibratedAgentEvaluationDataError):
    """An invalid crew Define artifact, or a scenario with no declared expected process."""


class CodebookError(CalibratedAgentEvaluationDataError):
    """A codebook entry is missing its definition or a positive/boundary example."""


class AgentRecordMismatchError(CalibratedAgentEvaluationDataError):
    """Per-agent correction was handed another agent's judge calibration record."""
