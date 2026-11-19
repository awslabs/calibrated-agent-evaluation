# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publishing a slice plans from reads, writes only when told to, and versions without loss."""

from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import StubResponseError

from agentcore_dataset_stubs import (
    DATASET_ID,
    NAME,
    Account,
    a_slice,
    pointer,
)
from calibrated_agent_evaluation.agentcore.datasets import plan_publish, publish
from calibrated_agent_evaluation.agentcore.datasets_codec import content_digest
from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreError,
    AwsCredentialsMissing,
    DatasetInputError,
    DatasetNameConflict,
    DatasetPublishError,
    ReadOnlyViolation,
)

_NO_SLEEP: dict[str, Any] = {"sleep": lambda _s: None}


class _Clock:
    """A clock that advances by one poll each time it's read."""

    def __init__(self, step: float = 2.0) -> None:
        self.now, self.step = 0.0, step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def test_plan_reports_what_a_first_publish_would_create() -> None:
    account = Account().no_datasets()

    plan = plan_publish(account.clients, a_slice(3), NAME)

    assert plan.creates_dataset
    assert (plan.region, plan.schema_type) == ("us-west-2", "AGENTCORE_EVALUATION_PREDEFINED_V1")
    assert (plan.n_scenarios, plan.n_with_assertions, plan.n_with_trajectory) == (3, 1, 3)
    assert plan.next_version == "1"
    account.done()


def test_first_publish_creates_the_dataset_and_version_one() -> None:
    slice_ = a_slice()
    account = Account().no_datasets().create().get("CREATING").get("ACTIVE").version("1")
    account.get("UPDATING").get("ACTIVE").versions(1)
    plan = plan_publish(account.clients, slice_, NAME)

    published = publish(account.clients, slice_, plan, clock=_Clock(), **_NO_SLEEP)

    assert (published.dataset_id, published.version, published.dataset_name) == (
        DATASET_ID,
        "1",
        NAME,
    )
    assert published.content_digest == content_digest(slice_)
    account.done()


def test_republishing_a_changed_slice_creates_the_next_version() -> None:
    slice_ = a_slice(3, published=pointer("1"))
    account = Account().existing().versions(1)
    account.examples([{"exampleId": "old-0"}, {"exampleId": "old-1"}])
    account.delete(2).get("ACTIVE", count=0).add(3).get("ACTIVE", count=3)
    account.version("2").get("ACTIVE").versions(2, 1)
    plan = plan_publish(account.clients, slice_, NAME)
    assert (plan.creates_dataset, plan.next_version, plan.unchanged) == (False, "2", False)

    published = publish(account.clients, slice_, plan, clock=_Clock(), **_NO_SLEEP)

    assert published.version == "2"
    account.done()


def test_large_drafts_are_cleared_in_several_calls() -> None:
    slice_ = a_slice(1, published=pointer("1"))
    account = Account().existing().versions(1)
    account.examples([{"exampleId": f"old-{i}"} for i in range(1000)], next_token="page-2")
    account.examples([{"exampleId": f"old-{i}"} for i in range(1000, 1500)])
    account.delete(1000).delete(500).get("ACTIVE", count=0).add(1).get("ACTIVE", count=1)
    account.version("2").get("ACTIVE").versions(2, 1)
    plan = plan_publish(account.clients, slice_, NAME)

    assert publish(account.clients, slice_, plan, clock=_Clock(), **_NO_SLEEP).version == "2"
    account.done()


def test_empty_drafts_skip_the_delete_and_a_modified_draft_is_not_published() -> None:
    slice_ = a_slice(1, published=pointer("1"))
    # An empty draft: no delete call is queued, so making one would fail the stub. The first poll
    # after the version call still shows the draft as MODIFIED; the next shows it settled.
    account = Account().existing().versions(1).examples([])
    account.add(1).get("ACTIVE", count=1).version("2")
    account.get("ACTIVE", draft="MODIFIED").get("ACTIVE", draft="UNMODIFIED").versions(2, 1)
    plan = plan_publish(account.clients, slice_, NAME)

    assert publish(account.clients, slice_, plan, clock=_Clock(), **_NO_SLEEP).version == "2"
    account.done()

    stuck = Account().existing().versions(1).examples([]).add(1).get("ACTIVE", count=1)
    stuck.version("2")
    for _ in range(4):
        stuck.get("ACTIVE", draft="MODIFIED")
    plan = plan_publish(stuck.clients, slice_, NAME)
    with pytest.raises(DatasetPublishError, match="draft still differed from the new version"):
        publish(stuck.clients, slice_, plan, clock=_Clock(), max_wait_seconds=6, **_NO_SLEEP)


def test_wait_stops_at_its_bound() -> None:
    slice_ = a_slice()
    account = Account().no_datasets().create()
    for _ in range(4):
        account.get("CREATING")
    plan = plan_publish(account.clients, slice_, NAME)

    with pytest.raises(DatasetPublishError, match="still CREATING after 6s"):
        publish(account.clients, slice_, plan, clock=_Clock(), max_wait_seconds=6, **_NO_SLEEP)


def test_a_thousand_scenarios_is_within_the_limit() -> None:
    account = Account().no_datasets()

    assert plan_publish(account.clients, a_slice(1000), NAME).n_scenarios == 1000


def test_more_than_a_thousand_scenarios_is_refused_before_any_call() -> None:
    account = Account()  # nothing queued: any call fails the stub

    with pytest.raises(DatasetPublishError, match="1001 scenarios.*1000"):
        plan_publish(account.clients, a_slice(1001), NAME)


@pytest.mark.parametrize("name", ["a", "orderbot_regressions", "A" + "b" * 47])
def test_valid_dataset_names_are_accepted(name: str) -> None:
    assert plan_publish(Account().no_datasets().clients, a_slice(), name).name == name


@pytest.mark.parametrize("name", ["orderbot-regressions", "A" + "b" * 48, "1orderbot", ""])
def test_invalid_dataset_names_are_refused_before_any_call(name: str) -> None:
    with pytest.raises(DatasetInputError, match="letters, digits and underscores"):
        plan_publish(Account().clients, a_slice(), name)


def test_name_conflicts_are_refused_before_any_write() -> None:
    # The name exists with a published version, but this slice was never published to it.
    with pytest.raises(DatasetNameConflict, match=DATASET_ID):
        plan_publish(Account().existing().versions(1).clients, a_slice(), NAME)
    # The slice names the dataset, but its schema differs.
    simulated = Account().existing(schema="AGENTCORE_EVALUATION_SIMULATED_V1").versions(1)
    with pytest.raises(DatasetNameConflict, match="SIMULATED_V1"):
        plan_publish(simulated.clients, a_slice(published=pointer()), NAME)


def test_failed_dataset_states_are_reported_and_leave_the_slice_unchanged() -> None:
    slice_ = a_slice()
    created = Account().no_datasets().create().get("CREATE_FAILED", reason="KMS key not usable")
    plan = plan_publish(created.clients, slice_, NAME)
    with pytest.raises(DatasetPublishError, match="CREATE_FAILED.*KMS key not usable") as first:
        publish(created.clients, slice_, plan, clock=_Clock(), **_NO_SLEEP)
    assert "may now differ" not in str(first.value)

    republished = a_slice(published=pointer())
    cleared = Account().existing().versions(1).examples([{"exampleId": "old-0"}]).delete(1)
    cleared.get("UPDATE_FAILED", draft="MODIFIED", reason="internal error")
    plan = plan_publish(cleared.clients, republished, NAME)
    with pytest.raises(DatasetPublishError, match="UPDATE_FAILED") as second:
        publish(cleared.clients, republished, plan, clock=_Clock(), **_NO_SLEEP)
    assert "draft may now differ from its last version" in str(second.value)
    assert DATASET_ID in str(second.value)


def test_failure_midway_through_republish_reports_the_draft_state() -> None:
    slice_ = a_slice(published=pointer())
    account = Account().existing().versions(1).examples([{"exampleId": "old-0"}])
    account.delete(1).get("ACTIVE", count=0)
    account.error("add_dataset_examples", "InternalServerException", 500)
    plan = plan_publish(account.clients, slice_, NAME)

    with pytest.raises(DatasetPublishError, match="draft may now differ") as refused:
        publish(account.clients, slice_, plan, clock=_Clock(), **_NO_SLEEP)
    assert "AddDatasetExamples" in str(refused.value)


def test_only_listed_dataset_operations_are_sent() -> None:
    # A response is queued, so only the guard can stop the call; it is never consumed.
    account = Account().respond(
        "delete_dataset",
        {
            "datasetArn": "arn",
            "datasetId": DATASET_ID,
            "status": "DELETING",
            "datasetVersion": "1",
            "updatedAt": "2026-10-27T19:00:00Z",
        },
    )

    with pytest.raises(ReadOnlyViolation, match="DeleteDataset"):
        account.clients.client("bedrock-agentcore-control").delete_dataset(datasetId=DATASET_ID)  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(AssertionError):
        account.done()  # the queued response is still waiting: the call was never sent


def test_unlisted_operations_are_refused_before_sending() -> None:
    account = Account().respond("list_agent_runtimes", {"agentRuntimes": []})

    with pytest.raises(ReadOnlyViolation, match="ListAgentRuntimes"):
        account.clients.client("bedrock-agentcore-control").list_agent_runtimes()  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(AssertionError):
        account.done()


def test_plan_for_an_existing_dataset_names_the_next_version_without_writing() -> None:
    account = Account().existing().versions(1)

    plan = plan_publish(account.clients, a_slice(published=pointer("1")), NAME)

    assert (plan.creates_dataset, plan.existing_id, plan.next_version) == (False, DATASET_ID, "2")
    account.done()
    # Nothing else was queued, so any write here would have raised.
    with pytest.raises(StubResponseError):
        account.clients.client("bedrock-agentcore-control").get_dataset(datasetId=DATASET_ID)  # pyright: ignore[reportAttributeAccessIssue]


def test_unchanged_slice_creates_no_new_version() -> None:
    unchanged = a_slice()
    slice_ = a_slice(published=pointer("1", digest=content_digest(unchanged)))
    account = Account().existing().versions(1)

    plan = plan_publish(account.clients, slice_, NAME)
    published = publish(account.clients, slice_, plan, **_NO_SLEEP)

    assert plan.unchanged
    assert published == slice_.published
    account.done()


def test_access_denied_names_the_iam_action() -> None:
    account = Account().error("list_datasets", "AccessDeniedException", 403, "not authorized")

    with pytest.raises(AgentCoreError, match="bedrock-agentcore:ListDatasets"):
        plan_publish(account.clients, a_slice(), NAME)


def test_a_stale_active_right_after_a_write_is_not_taken_as_done() -> None:
    slice_ = a_slice(3, published=pointer("1"))
    account = Account().existing().versions(1).examples([])
    # The first poll after the add still shows the old draft (2 examples) as ACTIVE.
    account.add(3).get("ACTIVE", count=2).get("ACTIVE", count=3).version("2").get("ACTIVE")
    account.versions(2, 1)
    plan = plan_publish(account.clients, slice_, NAME)

    assert publish(account.clients, slice_, plan, clock=_Clock(), **_NO_SLEEP).version == "2"
    account.done()


def test_a_version_failure_after_the_draft_changed_says_to_publish_again() -> None:
    slice_ = a_slice(1, published=pointer("1"))
    account = Account().existing().versions(1).examples([]).add(1).get("ACTIVE", count=1)
    account.error("create_dataset_version", "ConflictException", 409, "busy")
    plan = plan_publish(account.clients, slice_, NAME)

    with pytest.raises(DatasetPublishError, match="no new version was created; publish again"):
        publish(account.clients, slice_, plan, clock=_Clock(), **_NO_SLEEP)


def test_an_existing_dataset_on_a_later_page_is_found() -> None:
    account = Account().no_datasets(next_token="page-2").existing().versions(1, 2)

    plan = plan_publish(account.clients, a_slice(published=pointer("2")), NAME)

    assert (plan.existing_id, plan.next_version) == (DATASET_ID, "3")
    account.done()


def test_missing_credentials_are_a_typed_error_chained_to_the_cause() -> None:
    from botocore.exceptions import NoCredentialsError

    account = Account(fail={"ListDatasets": NoCredentialsError()}).no_datasets()

    with pytest.raises(AwsCredentialsMissing, match="AWS_PROFILE") as refused:
        plan_publish(account.clients, a_slice(), NAME)
    assert isinstance(refused.value.__cause__, NoCredentialsError)


def test_a_first_publish_that_stopped_part_way_is_finished_by_publishing_again() -> None:
    slice_ = a_slice()
    first = Account().no_datasets().create().get("ACTIVE")
    first.error("create_dataset_version", "InternalServerException", 500, "try later")
    plan = plan_publish(first.clients, slice_, NAME)
    with pytest.raises(DatasetPublishError, match="publish again to finish") as stopped:
        publish(first.clients, slice_, plan, clock=_Clock(), **_NO_SLEEP)
    assert DATASET_ID in str(stopped.value)

    # The retry finds the dataset with no version, picks it up and versions it.
    retry = Account().existing().versions().examples([{"exampleId": "ex-0"}, {"exampleId": "ex-1"}])
    retry.delete(2).get("ACTIVE", count=0).add(2).get("ACTIVE", count=2)
    retry.version("1").get("ACTIVE").versions(1)
    plan = plan_publish(retry.clients, slice_, NAME)
    assert (plan.creates_dataset, plan.next_version) == (False, "1")

    assert publish(retry.clients, slice_, plan, clock=_Clock(), **_NO_SLEEP).version == "1"
    retry.done()


def test_a_create_wait_failure_names_the_new_dataset() -> None:
    slice_ = a_slice()
    account = Account().no_datasets().create().error("get_dataset", "ThrottlingException", 429)
    plan = plan_publish(account.clients, slice_, NAME)

    with pytest.raises(
        DatasetPublishError, match="was created but holds no version yet"
    ) as refused:
        publish(account.clients, slice_, plan, clock=_Clock(), **_NO_SLEEP)
    assert DATASET_ID in str(refused.value)


def test_a_rejected_add_after_clearing_the_draft_still_reports_the_draft_state() -> None:
    slice_ = a_slice(published=pointer())
    account = Account().existing().versions(1).examples([{"exampleId": "old-0"}])
    account.delete(1).get("ACTIVE", count=0).error("add_dataset_examples", "ValidationException")
    plan = plan_publish(account.clients, slice_, NAME)

    with pytest.raises(DatasetPublishError, match="draft may now differ") as refused:
        publish(account.clients, slice_, plan, clock=_Clock(), **_NO_SLEEP)
    assert "rejected AddDatasetExamples" in str(refused.value)
