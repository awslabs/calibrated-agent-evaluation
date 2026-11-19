# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A live Code Interpreter run scores the fixture the same as the local sandbox (opt-in)."""

from __future__ import annotations

import os
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.agentcore]

_OUTPUT = {"code": "def add(a, b):\n    return a + b\n"}
_ORACLE = {"tests": "assert add(2, 3) == 5\n"}


def test_the_live_verdict_matches_the_local_sandbox(
    live_region: str | None, live_budget: Any
) -> None:
    from calibrated_agent_evaluation.agentcore import AwsClients
    from calibrated_agent_evaluation.agentcore.code_interpreter import CodeInterpreterSandbox
    from calibrated_agent_evaluation.execute.sandbox import LocalSubprocessSandbox
    from calibrated_agent_evaluation.scorers import CodeExecutionScorer

    role = os.environ["CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_ROLE_ARN"]
    local = CodeExecutionScorer(LocalSubprocessSandbox()).score(_OUTPUT, _ORACLE, trace_id="live")
    with AwsClients.from_profile(region=live_region, handlers=[live_budget]) as clients:
        backend = CodeInterpreterSandbox(
            region=clients.region or "us-west-2",
            clients=clients,
            budget=live_budget,
            role_arn=role,
        )
        agent = CodeExecutionScorer(backend).score(_OUTPUT, _ORACLE, trace_id="live")
    assert local.outcome == agent.outcome == "PASS"
