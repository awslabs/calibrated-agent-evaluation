# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AgentCore ingest error family is typed and importable on a bare install."""

from __future__ import annotations

from calibrated_agent_evaluation.agentcore import errors
from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError


def test_every_ingest_refusal_is_in_the_agentcore_error_family() -> None:
    refusals = [
        errors.AgentCoreContentMissing,
        errors.AgentCorePartialResult,
        errors.AgentCoreTruncatedResult,
        errors.AgentCoreIndexingTooLow,
        errors.AgentCoreSettlingLag,
        errors.AgentCoreMixedSession,
        errors.AgentCoreScoreIncomplete,
    ]
    for refusal in refusals:
        assert issubclass(refusal, errors.AgentCoreError)
        assert issubclass(refusal, CalibratedAgentEvaluationDataError)


def test_a_refusal_carries_its_message() -> None:
    raised = errors.AgentCorePartialResult("omni query returned a partial result")
    assert "partial" in str(raised)
