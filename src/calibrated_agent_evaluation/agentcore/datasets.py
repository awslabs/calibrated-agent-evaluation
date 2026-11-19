# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish scenario slices as AgentCore dataset versions, and pull a version back.

Every call goes through :class:`~calibrated_agent_evaluation.agentcore.session.AwsClients` with
:func:`dataset_guard`, which refuses any operation outside :data:`DATASET_OPERATIONS` before it is
sent. Dataset writes return before they finish (the dataset goes ``CREATING``/``UPDATING``), so
nothing is reported as published until a bounded ``GetDataset`` poll sees ``ACTIVE`` after the last
write and the draft matches the new version. Republishing replaces the draft wholesale and then
versions it; published versions are never touched. Reads use ``ListDatasetExamples`` (not the
presigned download URL), so they stay on the guarded client and need no HTTP library.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    ReadTimeoutError,
    SSOError,
    TokenRetrievalError,
)

from calibrated_agent_evaluation.agentcore.datasets_codec import (
    VERSION_PATTERN,
    content_digest,
    decode_example,
    encode_example,
    form_for_schema,
    schema_for_form,
    validate_dataset_name,
    validate_version,
)
from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreError,
    AwsCredentialsMissing,
    DatasetDecodeError,
    DatasetNameConflict,
    DatasetPublishError,
    DatasetVersionNotFound,
)
from calibrated_agent_evaluation.agentcore.guards import Operation, ReadOnlyGuard
from calibrated_agent_evaluation.agentcore.session import AwsClients
from calibrated_agent_evaluation.scenarios.errors import ScenarioError
from calibrated_agent_evaluation.scenarios.types import (
    DatasetSlice,
    PredefinedScenario,
    PublishedVersion,
)

__all__ = [
    "DATASET_OPERATIONS",
    "MAX_EXAMPLES_PER_REQUEST",
    "DatasetPlan",
    "dataset_guard",
    "plan_publish",
    "publish",
    "pull",
    "validate_dataset_name",
    "validate_version",
]

_SERVICE = "bedrock-agentcore-control"
MAX_EXAMPLES_PER_REQUEST = 1000
_FAILED = frozenset({"CREATE_FAILED", "UPDATE_FAILED", "DELETE_FAILED"})

DATASET_OPERATIONS: frozenset[Operation] = frozenset(
    (_SERVICE, name)
    for name in (
        "ListDatasets",
        "GetDataset",
        "ListDatasetVersions",
        "ListDatasetExamples",
        "CreateDataset",
        "DeleteDatasetExamples",
        "AddDatasetExamples",
        "CreateDatasetVersion",
    )
)


class _WaitFailed(DatasetPublishError):
    """A bounded wait ended without the dataset ready; its message already says the state."""


def dataset_guard() -> ReadOnlyGuard:
    """The allowlist every publish and pull client carries: the dataset operations, nothing else.

    It is a :class:`ReadOnlyGuard` used as a plain allowlist: the listed operations include the
    dataset writes ``publish`` makes, and every other operation is refused before it is sent.
    """
    return ReadOnlyGuard(DATASET_OPERATIONS)


@dataclass(frozen=True)
class DatasetPlan:
    """What a publish would do, worked out from reads only."""

    name: str
    region: str
    schema_type: str
    n_scenarios: int
    n_with_assertions: int
    n_with_trajectory: int
    digest: str
    existing_id: str | None
    next_version: str
    unchanged: bool

    @property
    def creates_dataset(self) -> bool:
        """Whether the publish creates a new dataset rather than versioning an existing one."""
        return self.existing_id is None


def plan_publish(clients: AwsClients, slice_: DatasetSlice, name: str) -> DatasetPlan:
    """The publish plan for ``slice_`` under ``name``; checks that need no call come first."""
    validate_dataset_name(name)
    n = len(slice_.scenarios)
    if n > MAX_EXAMPLES_PER_REQUEST:
        raise DatasetPublishError(
            f"the slice holds {n} scenarios, more than the {MAX_EXAMPLES_PER_REQUEST} examples "
            "AgentCore accepts in one request; split the slice (S3 uploads are not supported yet)"
        )
    region = _region(clients)
    schema = schema_for_form(slice_.form)
    digest = content_digest(slice_)
    control = clients.client(_SERVICE)
    existing = _find_dataset(control, name)
    if existing is None:
        next_version, existing_id, unchanged = "1", None, False
    else:
        existing_id = str(existing["datasetId"])
        if existing.get("schemaType") != schema:
            raise DatasetNameConflict(
                f"dataset {name!r} ({existing_id}) holds {existing.get('schemaType')} examples, "
                f"but this slice is {schema}"
            )
        versions = [
            int(v["datasetVersion"])
            for page in _pages(control, "list_dataset_versions", datasetId=existing_id)
            for v in page.get("versions", [])
            if VERSION_PATTERN.fullmatch(str(v.get("datasetVersion", "")))
        ]
        pointer = slice_.published
        # A dataset with no version yet is what a first publish that stopped part way leaves
        # behind; publishing again picks it up instead of refusing the name.
        if versions and (pointer is None or pointer.dataset_id != existing_id):
            raise DatasetNameConflict(
                f"a dataset named {name!r} already exists ({existing_id}, "
                f"{existing.get('schemaType')}) and this slice was not published to it; choose "
                "another --name, or pull that dataset to start from it"
            )
        next_version = str(max(versions, default=0) + 1)
        unchanged = pointer is not None and versions != [] and pointer.content_digest == digest
    return DatasetPlan(
        name=name,
        region=region,
        schema_type=schema,
        n_scenarios=n,
        n_with_assertions=sum(1 for s in slice_.scenarios if s.assertions),
        n_with_trajectory=sum(
            1
            for s in slice_.scenarios
            if isinstance(s, PredefinedScenario) and s.expected_trajectory
        ),
        digest=digest,
        existing_id=existing_id,
        next_version=next_version,
        unchanged=unchanged,
    )


def publish(
    clients: AwsClients,
    slice_: DatasetSlice,
    plan: DatasetPlan,
    *,
    poll_seconds: float = 2.0,
    max_wait_seconds: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> PublishedVersion:
    """Create or version the dataset the plan names and return where the slice now lives."""
    if plan.unchanged and slice_.published is not None:
        return slice_.published
    control = clients.client(_SERVICE)
    examples = [encode_example(s) for s in slice_.scenarios]
    source = {"inlineExamples": {"examples": examples}}

    def wait(
        dataset_id: str,
        step: str,
        *,
        draft_touched: bool,
        count: int | None = None,
        unmodified: bool = False,
    ) -> dict[str, Any]:
        return _wait_active(
            control,
            dataset_id,
            step,
            draft_touched,
            count,
            unmodified,
            poll_seconds,
            max_wait_seconds,
            sleep,
            clock,
        )

    if plan.existing_id is None:
        created = _call(
            "CreateDataset",
            control.create_dataset,
            datasetName=plan.name,
            schemaType=plan.schema_type,
            source=source,
            clientToken=_token(),
        )
        dataset_id = str(created["datasetId"])
        try:
            wait(dataset_id, "creating the dataset", draft_touched=False, count=len(examples))
        except _WaitFailed:
            raise
        except AgentCoreError as exc:
            raise DatasetPublishError(
                f"{exc}; dataset {dataset_id} was created but holds no version yet; publish "
                "again to finish (the same --name picks it up)"
            ) from exc
    else:
        dataset_id = plan.existing_id
        draft_ids = [
            str(e["exampleId"])
            for page in _pages(
                control, "list_dataset_examples", datasetId=dataset_id, datasetVersion="DRAFT"
            )
            for e in page.get("examples", [])
            if isinstance(e, dict) and "exampleId" in e
        ]
        try:
            for chunk in _chunks(draft_ids, MAX_EXAMPLES_PER_REQUEST):
                _call(
                    "DeleteDatasetExamples",
                    control.delete_dataset_examples,
                    datasetId=dataset_id,
                    exampleIds=chunk,
                    clientToken=_token(),
                )
            if draft_ids:
                wait(dataset_id, "clearing the draft", draft_touched=True, count=0)
            _call(
                "AddDatasetExamples",
                control.add_dataset_examples,
                datasetId=dataset_id,
                source=source,
                clientToken=_token(),
            )
            wait(dataset_id, "adding the slice's examples", draft_touched=True, count=len(examples))
        except _WaitFailed:
            raise
        except AgentCoreError as exc:
            raise DatasetPublishError(
                f"{exc}; dataset {dataset_id}'s draft may now differ from its last version, which "
                "is unchanged; publish again to finish"
            ) from exc

    try:
        versioned = _call(
            "CreateDatasetVersion",
            control.create_dataset_version,
            datasetId=dataset_id,
            clientToken=_token(),
        )
    except AgentCoreError as exc:
        raise DatasetPublishError(
            f"{exc}; dataset {dataset_id} holds the slice's examples in its draft but no new "
            "version was created; publish again to finish (the same --name picks it up)"
        ) from exc
    # Done only when the draft matches the new version; a poll right after the call can still
    # show the old draft as MODIFIED, so keep polling until the bound instead of failing on it.
    wait(dataset_id, "creating the version", draft_touched=True, unmodified=True)
    return PublishedVersion(
        dataset_arn=str(versioned["datasetArn"]),
        dataset_id=dataset_id,
        dataset_name=plan.name,
        version=str(versioned["datasetVersion"]),
        region=plan.region,
        content_digest=plan.digest,
        # Read back from the version list, as pull does: CreateDatasetVersion's own createdAt is
        # a few seconds earlier than the time AgentCore lists for the finished version.
        published_at=_version_time(
            _version_created(control, dataset_id, str(versioned["datasetVersion"]))
        ),
    )


def pull(clients: AwsClients, dataset_id: str, version: str) -> DatasetSlice:
    """The slice one published version of a dataset holds, decoded strictly."""
    validate_version(version)
    region = _region(clients)
    control = clients.client(_SERVICE)
    got = _call("GetDataset", control.get_dataset, datasetId=dataset_id, datasetVersion=version)
    schema = str(got.get("schemaType", ""))
    form = form_for_schema(schema)
    documents = [
        doc
        for page in _pages(
            control,
            "list_dataset_examples",
            datasetId=dataset_id,
            datasetVersion=version,
            PaginationConfig={"PageSize": MAX_EXAMPLES_PER_REQUEST},
        )
        for doc in page.get("examples", [])
    ]
    if not documents:
        raise DatasetDecodeError(
            f"dataset {dataset_id} version {version} has no examples, and a slice needs one"
        )
    scenarios = [decode_example(doc, schema) for doc in documents]
    # The version's own creation time, the one publish recorded; the dataset's updatedAt moves
    # every time a later version is published.
    version_time = _version_created(control, dataset_id, version)
    try:
        slice_ = DatasetSlice(form=form, scenarios=scenarios)
        pointer = PublishedVersion(
            dataset_arn=str(got["datasetArn"]),
            dataset_id=dataset_id,
            dataset_name=str(got["datasetName"]),
            version=version,
            region=region,
            content_digest=content_digest(slice_),
            published_at=_version_time(version_time),
        )
        return DatasetSlice(form=form, scenarios=slice_.scenarios, published=pointer)
    except ScenarioError as exc:
        raise DatasetDecodeError(f"dataset {dataset_id} version {version}: {exc}") from exc


def _region(clients: AwsClients) -> str:
    if not clients.region:
        raise AgentCoreError(
            "no AWS region configured; pass --region or set AWS_REGION (no default region is "
            "assumed)"
        )
    return clients.region


def _utc(moment: datetime) -> str:
    """One timestamp format for both publish and pull: UTC, seconds, ``Z``."""
    return moment.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _version_created(control: Any, dataset_id: str, version: str) -> object:
    """The creation time AgentCore lists for one version of a dataset."""
    for page in _pages(control, "list_dataset_versions", datasetId=dataset_id):
        for listed in page.get("versions", []):
            if str(listed.get("datasetVersion")) == version:
                return listed.get("createdAt")
    raise DatasetVersionNotFound(f"dataset {dataset_id} lists no version {version}")


def _version_time(created: object) -> str:
    """A version's creation time in the one format both publish and pull record."""
    if isinstance(created, datetime):
        return _utc(created)
    raise DatasetPublishError(f"AgentCore returned no usable version creation time ({created!r})")


def _token() -> str:
    return "calibrated-agent-evaluation-" + uuid.uuid4().hex


def _chunks(items: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _find_dataset(control: Any, name: str) -> dict[str, Any] | None:
    for page in _pages(control, "list_datasets"):
        for summary in page.get("datasets", []):
            if summary.get("datasetName") == name:
                return dict(summary)
    return None


def _pages(control: Any, method: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
    operation = _operation_name(method)
    try:
        yield from control.get_paginator(method).paginate(**kwargs)
    except (ClientError, BotoCoreError) as exc:
        raise _translate(operation, exc) from exc


def _call(operation: str, fn: Callable[..., Any], **kwargs: Any) -> dict[str, Any]:
    try:
        return dict(fn(**kwargs))
    except (ClientError, BotoCoreError) as exc:
        raise _translate(operation, exc) from exc


def _wait_active(
    control: Any,
    dataset_id: str,
    step: str,
    draft_touched: bool,
    count: int | None,
    unmodified: bool,
    poll_seconds: float,
    max_wait_seconds: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> dict[str, Any]:
    deadline = clock() + max_wait_seconds
    while True:
        got = _call("GetDataset", control.get_dataset, datasetId=dataset_id)
        status = str(got.get("status", ""))
        # A poll right after a write can still see the old ACTIVE; the example count and, after
        # a version, the draft status tell.
        ready = (count is None or got.get("exampleCount") == count) and (
            not unmodified or got.get("draftStatus") == "UNMODIFIED"
        )
        if status == "ACTIVE" and ready:
            return got
        if status in _FAILED or clock() >= deadline:
            if status in _FAILED:
                why = f"it reached {status}"
            elif status == "ACTIVE" and unmodified:
                why = f"after {max_wait_seconds:g}s its draft still differed from the new version"
            elif status == "ACTIVE":
                why = (
                    f"after {max_wait_seconds:g}s it held {got.get('exampleCount')} example(s), "
                    f"not {count}"
                )
            else:
                why = f"it was still {status} after {max_wait_seconds:g}s"
            reason = got.get("failureReason")
            draft = (
                " Its draft may now differ from its last version, which is unchanged."
                if draft_touched or got.get("draftStatus") == "MODIFIED"
                else ""
            )
            raise _WaitFailed(
                f"{step} for dataset {dataset_id} did not finish: {why}"
                + (f" ({reason})" if reason else "")
                + f". The slice file was not updated.{draft}"
            )
        sleep(poll_seconds)


def _operation_name(method: str) -> str:
    return "".join(part.title() for part in method.split("_"))


_NOT_FOUND_MEANS_MISSING_VERSION = frozenset({"GetDataset", "ListDatasetExamples"})
_WRITES = frozenset(
    {"CreateDataset", "DeleteDatasetExamples", "AddDatasetExamples", "CreateDatasetVersion"}
)


def _translate(operation: str, exc: ClientError | BotoCoreError) -> AgentCoreError:
    if isinstance(exc, ClientError):
        error = exc.response.get("Error", {})
        code = str(error.get("Code", ""))
        message = str(error.get("Message", "")) or str(exc)
        if code == "ResourceNotFoundException" and operation in _NOT_FOUND_MEANS_MISSING_VERSION:
            return DatasetVersionNotFound(f"{operation}: {message}")
        if code in {"AccessDeniedException", "AccessDenied", "UnauthorizedOperation"}:
            return AgentCoreError(
                f"{operation} was denied ({message}); the credentials need "
                f"bedrock-agentcore:{operation}"
            )
        if code in {"ThrottlingException", "TooManyRequestsException"}:
            return AgentCoreError(f"{operation} was throttled ({message}); wait a minute and retry")
        if code == "ServiceQuotaExceededException":
            return AgentCoreError(
                f"{operation} hit an AgentCore service quota ({message}); delete datasets you no "
                "longer need or request a quota increase"
            )
        if code == "ValidationException":
            kind = DatasetPublishError if operation in _WRITES else AgentCoreError
            return kind(f"AgentCore rejected {operation}: {message}")
        return AgentCoreError(f"{operation} failed ({code}): {message}")
    if isinstance(exc, NoCredentialsError | SSOError | TokenRetrievalError):
        return AwsCredentialsMissing(
            f"{operation} found no usable AWS credentials ({exc}); set AWS_PROFILE or pass "
            "--profile, and refresh an SSO login if it expired"
        )
    if isinstance(exc, EndpointConnectionError | ConnectTimeoutError | ReadTimeoutError):
        return AgentCoreError(
            f"{operation} could not reach AgentCore ({exc}); check the network and the region, "
            "then retry"
        )
    return AgentCoreError(f"{operation} failed before reaching AgentCore: {exc}")
