# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A billed call past the run's budget is refused before it reaches the account."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.agentcore]


def test_a_zero_budget_refuses_evaluate_before_sending(live_region: str | None) -> None:
    from calibrated_agent_evaluation.agentcore import AwsClients, BilledCallBudget
    from calibrated_agent_evaluation.agentcore.errors import BilledCallBudgetExceeded

    params: dict[str, Any] = {
        "evaluatorId": "Builtin.Helpfulness",
        "evaluationInput": {"sessionSpans": [{}]},
    }
    with AwsClients.from_profile(region=live_region, handlers=[BilledCallBudget(0)]) as clients:
        client = clients.client("bedrock-agentcore")
        with pytest.raises(BilledCallBudgetExceeded, match="Evaluate"):
            client.evaluate(**params)  # pyright: ignore[reportAttributeAccessIssue]
