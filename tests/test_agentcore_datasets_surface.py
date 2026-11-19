# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The codec's field lists match the installed AgentCore SDK's dataset types."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.agentcore.datasets_codec import (
    ACTOR_FIELDS,
    PREDEFINED_FIELDS,
    SIMULATED_FIELDS,
    TURN_FIELDS,
)

dataset_types = pytest.importorskip("bedrock_agentcore.evaluation.runner.dataset_types")


def _fields(model: type) -> set[str]:
    return set(model.model_fields) - {"schema_type"}  # the SDK's discriminator, not a document key


def test_codec_fields_match_the_installed_sdk_dataset_types() -> None:
    assert _fields(dataset_types.PredefinedScenario) == PREDEFINED_FIELDS
    assert _fields(dataset_types.SimulatedScenario) == SIMULATED_FIELDS
    assert _fields(dataset_types.Turn) == TURN_FIELDS
    assert _fields(dataset_types.ActorProfile) == ACTOR_FIELDS
