# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A slice published to a real AgentCore account pulls back unchanged, versions, and is deleted."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from calibrated_agent_evaluation.scenarios import DatasetSlice

pytestmark = [pytest.mark.integration, pytest.mark.agentcore]

_SERVICE = "bedrock-agentcore-control"


def _slice(n: int) -> DatasetSlice:
    from calibrated_agent_evaluation.scenarios import (
        CalibratedAgentEvaluationScenarioMeta,
        DatasetSlice,
        PredefinedScenario,
        Turn,
    )

    return DatasetSlice(
        form="predefined",
        scenarios=[
            PredefinedScenario(
                scenario_id=f"live-{i}",
                turns=[Turn(input={"message": f"refund order {i}"}, expected_response='{"a":1}')],
                expected_trajectory=["lookup_order", "check_refund_policy"],
                assertions=["Agent denies a refund outside the return window"],
                calibrated_agent_evaluation=CalibratedAgentEvaluationScenarioMeta(
                    source_trace_id=f"live-{i}", expected={"a": 1}
                ),
            )
            for i in range(n)
        ],
    )


def test_publish_and_pull_round_trip_against_agentcore(
    live_budget: Any, live_region: str | None
) -> None:
    from calibrated_agent_evaluation.agentcore import (
        AwsClients,
        dataset_guard,
        plan_publish,
        publish,
        pull,
    )

    name = f"calibrated_agent_evaluation_live_{uuid.uuid4().hex[:12]}"
    first, second = _slice(2), _slice(3)
    try:
        with AwsClients.from_profile(
            region=live_region, handlers=[dataset_guard(), live_budget]
        ) as clients:
            plan = plan_publish(clients, first, name)
            assert plan.creates_dataset
            v1 = publish(clients, first, plan)
            pulled_v1 = pull(clients, v1.dataset_id, v1.version)

            # Republish a changed slice: the draft is replaced and the next version is frozen.
            changed = second.model_copy(update={"published": v1})
            v2 = publish(clients, changed, plan_publish(clients, changed, name))
            pulled_v2 = pull(clients, v2.dataset_id, v2.version)
            # Version 1 is unchanged by the republish.
            pulled_v1_again = pull(clients, v1.dataset_id, v1.version)
    finally:
        _delete_by_name(live_region, live_budget, name)

    assert (v1.version, v2.version) == ("1", "2")
    assert pulled_v1.scenarios == first.scenarios
    # A pulled version records exactly what publish recorded, its creation time included, even
    # after a later version moved the dataset's own update time.
    assert pulled_v1_again.published == v1
    assert pulled_v2.published == v2
    assert pulled_v2.scenarios == second.scenarios
    assert pulled_v1_again.scenarios == first.scenarios


def _delete_by_name(region: str | None, budget: Any, name: str) -> None:
    """Delete the test's dataset if it exists, even when the test failed before recording its id.

    The client may list datasets and delete one, and nothing else.
    """
    from calibrated_agent_evaluation.agentcore import AwsClients, ReadOnlyGuard

    guard = ReadOnlyGuard({(_SERVICE, "ListDatasets"), (_SERVICE, "DeleteDataset")})
    with AwsClients.from_profile(region=region, handlers=[guard, budget]) as clients:
        control = clients.client(_SERVICE)
        for page in control.get_paginator("list_datasets").paginate():
            for summary in page.get("datasets", []):
                if summary.get("datasetName") == name:
                    control.delete_dataset(datasetId=summary["datasetId"])  # pyright: ignore[reportAttributeAccessIssue]
