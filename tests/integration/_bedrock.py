# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one Bedrock model knob the live tests read: ``CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID``,
a bare model id."""

from __future__ import annotations

import os

import pytest

from calibrated_agent_evaluation.flagship import FlagshipError
from calibrated_agent_evaluation.flagship.runner import live_bedrock_model_id


def bedrock_model_id() -> str:
    """The bare Bedrock model id; a retired or malformed setting fails the test, never skips it."""
    try:
        return live_bedrock_model_id(os.environ)
    except FlagshipError as exc:
        pytest.fail(str(exc))


def inspect_bedrock_model() -> str:
    """The same model as an Inspect model string."""
    return f"bedrock/{bedrock_model_id()}"
