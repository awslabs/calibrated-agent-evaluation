# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pulling a dataset version rebuilds its slice exactly, or refuses with a typed error."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentcore_dataset_stubs import DATASET_ID, NAME, Account, a_slice, listed
from calibrated_agent_evaluation.agentcore.datasets import plan_publish, publish, pull
from calibrated_agent_evaluation.agentcore.datasets_codec import encode_example
from calibrated_agent_evaluation.agentcore.errors import (
    DatasetDecodeError,
    DatasetInputError,
    DatasetSchemaUnsupported,
    DatasetVersionNotFound,
)
from calibrated_agent_evaluation.scenarios.types import (
    ActorProfile,
    DatasetSlice,
    SimulatedScenario,
)

CAPTURED = Path(__file__).resolve().parent.parent / "fixtures" / "raw" / "agentcore" / "datasets"


def test_pull_reads_every_page_of_a_version() -> None:
    original = a_slice(3)
    docs = listed([encode_example(s) for s in original.scenarios])
    # The dataset was updated later (version 2 published): the pull still records version 1's time.
    account = Account().get(version="1", updated=datetime(2026, 10, 28, 9, 30, tzinfo=UTC))
    account.examples(docs[:2], version="1", next_token="page-2").examples(docs[2:], version="1")
    account.versions(2, 1)

    pulled = pull(account.clients, DATASET_ID, "1")

    assert pulled.scenarios == original.scenarios
    assert pulled.published is not None
    assert (pulled.published.dataset_id, pulled.published.dataset_name) == (DATASET_ID, NAME)
    assert pulled.published.version == "1"
    # Version 1's own creation time, in UTC with a Z, the same value publish records.
    assert pulled.published.published_at == "2026-10-27T19:00:00Z"
    account.done()


def test_an_empty_version_is_refused() -> None:
    account = Account().get(version="1").examples([], version="1")

    with pytest.raises(DatasetDecodeError, match=f"{DATASET_ID} version 1 has no examples"):
        pull(account.clients, DATASET_ID, "1")


def test_one_bad_example_fails_the_whole_pull() -> None:
    docs = listed([encode_example(s) for s in a_slice(3).scenarios])
    docs[1] = docs[1] | {"expected_output": "not a predefined field"}
    account = Account().get(version="1").examples(docs, version="1")

    with pytest.raises(DatasetDecodeError, match="ORD-0001.*expected_output"):
        pull(account.clients, DATASET_ID, "1")


def test_pull_refuses_unsupported_schemas() -> None:
    got = json.loads((CAPTURED / "get_dataset_third_party.json").read_text())
    # Only GetDataset is answered: listing the examples of a refused schema would fail the stub.
    account = Account().get(version="1", schema=got["schemaType"])

    with pytest.raises(DatasetSchemaUnsupported, match="THIRD_PARTY_EVALUATION_V1"):
        pull(account.clients, DATASET_ID, "1")
    account.done()


def test_missing_or_deleted_versions_are_named() -> None:
    captured = json.loads((CAPTURED / "get_dataset_missing_version_error.json").read_text())
    error = captured["Error"]
    account = Account().error(
        "get_dataset", error["Code"], captured["HTTPStatusCode"], error["Message"]
    )

    with pytest.raises(DatasetVersionNotFound, match="version 99 not found"):
        pull(account.clients, DATASET_ID, "99")


@pytest.mark.parametrize("version", ["DRAFT", "1.0", "", "v2"])
def test_only_published_versions_can_be_pulled(version: str) -> None:
    account = Account()  # nothing queued: any call would fail the stub

    with pytest.raises(DatasetInputError, match="only published versions"):
        pull(account.clients, DATASET_ID, version)


def test_simulated_slices_publish_and_pull_back_equal() -> None:
    simulated = DatasetSlice(
        form="simulated",
        scenarios=[
            SimulatedScenario(
                scenario_id="angry-late-refund",
                actor_profile=ActorProfile(
                    context="Order A-7781 arrived 51 days ago", goal="Get a refund"
                ),
                input="Where is my refund?!",
                max_turns=6,
                assertions=["Agent explains the 30-day return window"],
            )
        ],
    )
    simulated_schema = "AGENTCORE_EVALUATION_SIMULATED_V1"
    account = Account().no_datasets()
    account.respond(
        "create_dataset",
        {
            "datasetArn": "arn:aws:bedrock-agentcore:us-west-2:111122223333:dataset/x",
            "datasetId": DATASET_ID,
            "status": "CREATING",
            "createdAt": "2026-10-27T19:00:00Z",
        },
    )
    account.get("ACTIVE", schema=simulated_schema, count=1).version("1")
    account.get("ACTIVE", schema=simulated_schema).versions(1)
    plan = plan_publish(account.clients, simulated, NAME)
    published = publish(account.clients, simulated, plan, sleep=lambda _s: None)
    account.get(version="1", schema=simulated_schema)
    account.examples(listed([encode_example(s) for s in simulated.scenarios]), version="1")
    account.versions(1)

    pulled = pull(account.clients, published.dataset_id, published.version)

    assert pulled.published == published
    assert pulled.scenarios == simulated.scenarios
    assert pulled.form == "simulated"
    account.done()
