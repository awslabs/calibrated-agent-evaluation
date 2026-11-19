# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A stubbed AgentCore control plane for the dataset publish and pull tests.

Responses go through botocore's ``Stubber``, which validates each one against the installed
service model, so a stub with the wrong shape fails the test instead of passing quietly. Example
documents follow the captured ones in ``fixtures/raw/agentcore/datasets/``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.stub import ANY, Stubber

from calibrated_agent_evaluation.agentcore.datasets import dataset_guard
from calibrated_agent_evaluation.agentcore.session import AwsClients
from calibrated_agent_evaluation.scenarios.types import (
    CalibratedAgentEvaluationScenarioMeta,
    DatasetSlice,
    PredefinedScenario,
    PublishedVersion,
    Scenario,
    Turn,
)

SERVICE = "bedrock-agentcore-control"
REGION = "us-west-2"
DATASET_ID = "orderbot_regressions-AbCdEf1234"
NAME = "orderbot_regressions"
ARN = f"arn:aws:bedrock-agentcore:{REGION}:111122223333:dataset/{DATASET_ID}"
PREDEFINED = "AGENTCORE_EVALUATION_PREDEFINED_V1"
WHEN = datetime(2026, 10, 27, 19, 0, tzinfo=UTC)


def scenario(scenario_id: str, *, assertions: list[str] | None = None) -> PredefinedScenario:
    return PredefinedScenario(
        scenario_id=scenario_id,
        turns=[Turn(input={"message": f"refund {scenario_id}"}, expected_response='{"a":1}')],
        expected_trajectory=["lookup_order", "check_refund_policy"],
        assertions=assertions or [],
        calibrated_agent_evaluation=CalibratedAgentEvaluationScenarioMeta(
            source_trace_id=scenario_id, expected={"a": 1}
        ),
    )


def a_slice(n: int = 2, *, published: PublishedVersion | None = None) -> DatasetSlice:
    scenarios: list[Scenario] = [
        scenario(f"ORD-{i:04d}", assertions=["Agent denies"] if i == 0 else None) for i in range(n)
    ]
    return DatasetSlice(form="predefined", scenarios=scenarios, published=published)


def pointer(version: str = "1", digest: str = "0" * 64) -> PublishedVersion:
    return PublishedVersion(
        dataset_arn=ARN,
        dataset_id=DATASET_ID,
        dataset_name=NAME,
        version=version,
        region=REGION,
        content_digest=digest,
        published_at="2026-10-27T19:00:00Z",
    )


# Every fake's clients, closed after each test by the conftest fixture, as `with` would close them.
_OPEN: list[AwsClients] = []


def close_all() -> None:
    """Close the clients every fake built so far."""
    while _OPEN:
        _OPEN.pop().close()


class Account:
    """Guarded clients for one region with a stubbed control plane.

    ``fail`` maps an operation name to the transport error it raises.
    """

    def __init__(self, *, region: str | None = REGION, fail: dict[str, Exception] | None = None):
        session = boto3.Session(
            region_name=region, aws_access_key_id="testing", aws_secret_access_key="testing"
        )
        failures = fail or {}

        def network(**kwargs: Any) -> None:
            failure = failures.get(kwargs["model"].name)
            if failure is not None:
                raise failure

        self.clients = AwsClients(session, handlers=[dataset_guard(), network])
        _OPEN.append(self.clients)
        # Without a region no client can be built, which is what the region checks rely on.
        self.stub = Stubber(self.clients.client(SERVICE)) if region else Stubber(None)  # pyright: ignore[reportArgumentType]
        if region:
            self.stub.activate()

    def respond(self, method: str, response: dict[str, Any], params: Any = None) -> Account:
        self.stub.add_response(method, response, params)
        return self

    def error(self, method: str, code: str, status: int = 400, message: str = "") -> Account:
        self.stub.add_client_error(
            method, service_error_code=code, service_message=message, http_status_code=status
        )
        return self

    # --- canned responses -------------------------------------------------------------------

    def no_datasets(self, *, next_token: str | None = None) -> Account:
        response: dict[str, Any] = {"datasets": []}
        if next_token is not None:
            response["nextToken"] = next_token
        return self.respond("list_datasets", response)

    def existing(self, *, schema: str = PREDEFINED, dataset_id: str = DATASET_ID) -> Account:
        return self.respond(
            "list_datasets",
            {
                "datasets": [
                    {
                        "datasetArn": ARN,
                        "datasetId": dataset_id,
                        "datasetName": NAME,
                        "status": "ACTIVE",
                        "schemaType": schema,
                        "exampleCount": 2,
                        "createdAt": WHEN,
                        "updatedAt": WHEN,
                    }
                ]
            },
        )

    def versions(self, *numbers: int) -> Account:
        return self.respond(
            "list_dataset_versions",
            {
                "versions": [
                    {"datasetVersion": str(n), "exampleCount": 2, "createdAt": WHEN}
                    for n in numbers
                ]
            },
        )

    def get(
        self,
        status: str = "ACTIVE",
        *,
        draft: str = "UNMODIFIED",
        version: str = "DRAFT",
        schema: str = PREDEFINED,
        reason: str | None = None,
        count: int = 2,
        updated: datetime = WHEN,
    ) -> Account:
        response: dict[str, Any] = {
            "datasetArn": ARN,
            "datasetId": DATASET_ID,
            "datasetVersion": version,
            "datasetName": NAME,
            "status": status,
            "draftStatus": draft,
            "schemaType": schema,
            "exampleCount": count,
            "createdAt": WHEN,
            "updatedAt": updated,
        }
        if reason is not None:
            response["failureReason"] = reason
        return self.respond("get_dataset", response)

    def create(self) -> Account:
        return self.respond(
            "create_dataset",
            {"datasetArn": ARN, "datasetId": DATASET_ID, "status": "CREATING", "createdAt": WHEN},
            {"datasetName": ANY, "schemaType": PREDEFINED, "source": ANY, "clientToken": ANY},
        )

    def examples(
        self,
        docs: list[dict[str, Any]],
        *,
        version: str = "DRAFT",
        next_token: str | None = None,
    ) -> Account:
        response: dict[str, Any] = {
            "datasetArn": ARN,
            "datasetId": DATASET_ID,
            "datasetVersion": version,
            "examples": docs,
        }
        if next_token is not None:
            response["nextToken"] = next_token
        return self.respond("list_dataset_examples", response)

    def delete(self, n: int) -> Account:
        return self.respond(
            "delete_dataset_examples",
            {
                "datasetArn": ARN,
                "datasetId": DATASET_ID,
                "status": "UPDATING",
                "deletedCount": n,
                "updatedAt": WHEN,
            },
        )

    def add(self, n: int) -> Account:
        return self.respond(
            "add_dataset_examples",
            {
                "datasetArn": ARN,
                "datasetId": DATASET_ID,
                "status": "UPDATING",
                "addedCount": n,
                "updatedAt": WHEN,
                "exampleIds": [f"ex-{i}" for i in range(n)],
            },
        )

    def version(self, number: str) -> Account:
        return self.respond(
            "create_dataset_version",
            {
                "datasetArn": ARN,
                "datasetId": DATASET_ID,
                "status": "UPDATING",
                "datasetVersion": number,
                "createdAt": WHEN,
            },
        )

    def done(self) -> None:
        """Assert every queued response was used (no expected call was skipped)."""
        self.stub.assert_no_pending_responses()


def listed(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Documents as ``ListDatasetExamples`` returns them, each with a service ``exampleId``."""
    return [doc | {"exampleId": f"00000000-0000-4000-8000-{i:012d}"} for i, doc in enumerate(docs)]
