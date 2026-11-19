# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The arm reader's span attribute names are the installed AgentCore SDK's own."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.agentcore.arms import ARM_ARN_ATTRIBUTE, ARM_VARIANT_ATTRIBUTE


def test_arm_attribute_names_match_the_installed_sdk() -> None:
    models = pytest.importorskip("bedrock_agentcore.runtime.models")

    assert models.BAGGAGE_KEY_EXPERIMENT_ARN == ARM_ARN_ATTRIBUTE
    assert models.BAGGAGE_KEY_EXPERIMENT_VARIANT == ARM_VARIANT_ATTRIBUTE
