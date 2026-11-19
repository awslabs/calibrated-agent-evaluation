# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run manifest keeps one record per run in its JSON and CSV pair, and a manifest that can't
be locked, read or written, or that holds records of another format, is refused naming its path."""

from __future__ import annotations

import errno
import json
import multiprocessing
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.agentcore import RunRecord, RuntimeInvocationError
from calibrated_agent_evaluation.agentcore.run_manifest import (
    AGENTCORE_RUN_SCHEMA_VERSION,
    RUN_RECORD_STORE,
    append_run_record,
    manifest_path,
)

_RUNTIME_ARN = (
    "arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/"
    "calibrated_agent_evaluation_fixture-AbCdEf1234"
)
_W3C_TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"


def _record(**overrides: Any) -> RunRecord:
    fields: dict[str, Any] = {
        "trace_id": f"agentcore-{_W3C_TRACE}",
        "w3c_trace_id": _W3C_TRACE,
        "session_id": "calibrated-agent-evaluation-0123456789abcdef0123456789abcdef",
        "runtime_arn": _RUNTIME_ARN,
        "qualifier": "DEFAULT",
        "runtime_version": "3",
        "runtime_version_source": "endpoint",
        "config_bundle_arn": None,
        "config_bundle_version": None,
        "payload_format": "agentcore_sse",
        "started_at": "2026-10-26T10:00:00+00:00",
        "ended_at": "2026-10-26T10:00:01.250000+00:00",
        "duration_ms": 1250,
        "attempts": 1,
        "status_code": 200,
        "content_type": "text/event-stream",
        "response_traceparent": f"00-{_W3C_TRACE}-00f067aa0ba902b7-01",
        "replayed": False,
        "session_stop": "stopped",
    }
    fields.update(overrides)
    return RunRecord(**fields)


def test_run_records_append_and_read_back_equal_from_json_and_csv(tmp_path: Path) -> None:
    directory = tmp_path / "records" / "agentcore"
    succeeded = _record()
    failed = _record(
        trace_id="CLM-4471",
        w3c_trace_id="0af7651916cd43dd8448eb211c80319c",
        session_id="calibrated-agent-evaluation-fedcba9876543210fedcba9876543210",
        runtime_version="7",
        runtime_version_source="latest",
        config_bundle_arn=(
            "arn:aws:bedrock-agentcore:us-west-2:111122223333:"
            "configuration-bundle/calibrated_agent_evaluation_bundle-AbCdEf1234"
        ),
        config_bundle_version="0f8b6c1e-3a2d-4c5b-9e7f-1a2b3c4d5e6f",
        attempts=3,
        status_code=None,
        content_type=None,
        response_traceparent=None,
        session_stop="failed",
        error_code="RetryableConflictException",
    )

    append_run_record(directory, succeeded)
    append_run_record(directory, failed)

    assert RUN_RECORD_STORE.load(directory, fmt="json") == [succeeded, failed]
    assert RUN_RECORD_STORE.load(directory, fmt="csv") == [succeeded, failed]
    stored = json.loads(manifest_path(directory).read_text(encoding="utf-8"))
    assert [doc["trace_id"] for doc in stored] == [succeeded.trace_id, "CLM-4471"]
    assert {doc["schema_version"] for doc in stored} == {AGENTCORE_RUN_SCHEMA_VERSION}


def _block_the_directory(tmp_path: Path) -> Path:
    (tmp_path / "records").write_text("not a directory", encoding="utf-8")
    return tmp_path / "records" / "agentcore"


def _block_the_file(tmp_path: Path) -> Path:
    manifest_path(tmp_path).mkdir()
    return tmp_path


def _read_only_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "logs"
    directory.mkdir()
    directory.chmod(0o500)
    return directory


def _causes(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    cause = error.__cause__
    while cause is not None:
        chain.append(cause)
        cause = cause.__cause__
    return chain


@pytest.mark.parametrize(
    "unwritable",
    [_block_the_directory, _block_the_file, _read_only_directory],
    ids=["a-file-in-place-of-its-directory", "a-directory-in-place-of-its-file", "read-only"],
)
def test_an_unwritable_manifest_is_refused_naming_its_path(
    tmp_path: Path, unwritable: Callable[[Path], Path]
) -> None:
    directory = unwritable(tmp_path)
    try:
        if unwritable is _read_only_directory and os.access(directory, os.W_OK):
            pytest.skip("this user can write to a read-only directory")
        with pytest.raises(RuntimeInvocationError) as refused:
            append_run_record(directory, _record())
    finally:
        if directory.is_dir():
            directory.chmod(0o700)

    assert str(manifest_path(directory)) in str(refused.value)
    assert "log directory" in str(refused.value)
    # A write failure is not a malformed manifest, so the advice is about the directory only.
    assert "aside" not in str(refused.value)
    assert any(isinstance(cause, OSError) for cause in _causes(refused.value))


_STORED = _record().model_dump(mode="json")
_MALFORMED = {
    "not-json": "{not json",
    "not-a-list": '{"trace_id": "CLM-4471"}',
    "not-an-object": "[1]",
    "a-partial-record": json.dumps(
        [{"schema_version": AGENTCORE_RUN_SCHEMA_VERSION, "trace_id": "CLM-4471"}]
    ),
    "a-number-too-long-to-read": "[" + "9" * 5000 + "]",
    # Deep enough that the JSON decoder itself runs out of stack, on every supported Python.
    "nested-too-deep-to-decode": "[" * 100_001 + "]" * 100_001,
}
_OF_ANOTHER_VERSION = {
    "another-schema-version": json.dumps([{**_STORED, "schema_version": "0"}]),
    "no-schema-version": json.dumps([{k: v for k, v in _STORED.items() if k != "schema_version"}]),
}


@pytest.mark.parametrize(
    ("content", "cause"),
    [
        *(pytest.param(content, BaseException, id=name) for name, content in _MALFORMED.items()),
        *(
            pytest.param(content, RuntimeInvocationError, id=name)
            for name, content in _OF_ANOTHER_VERSION.items()
        ),
    ],
)
def test_a_malformed_manifest_or_one_of_another_schema_version_is_refused_naming_its_path(
    tmp_path: Path, content: str, cause: type[BaseException]
) -> None:
    manifest_path(tmp_path).write_text(content, encoding="utf-8")

    with pytest.raises(RuntimeInvocationError) as refused:
        append_run_record(tmp_path, _record(trace_id="CLM-4471"))

    assert str(manifest_path(tmp_path)) in str(refused.value)
    assert "aside" in str(refused.value)
    assert isinstance(refused.value.__cause__, cause)
    # The refused manifest is left as it was, not rewritten without its old records.
    assert manifest_path(tmp_path).read_text(encoding="utf-8") == content


def test_a_manifest_of_another_version_nested_too_deep_to_search_is_still_refused_by_version(
    tmp_path: Path,
) -> None:
    # Deeper than a recursive search could go (two frames a level against a limit of 1,000), and
    # shallow enough for the JSON decoder to read on every supported Python.
    detail: list[object] = []
    for _ in range(700):
        detail = [detail]
    record = _record().model_dump(mode="json")
    stored = json.dumps([{**record, "schema_version": "0", "detail": detail}])
    manifest_path(tmp_path).write_text(stored, encoding="utf-8")
    try:
        json.loads(stored)
    except RecursionError:
        pytest.skip("this Python's JSON decoder can't read 700 levels")

    with pytest.raises(RuntimeInvocationError) as refused:
        append_run_record(tmp_path, _record(trace_id="CLM-4471"))

    assert str(manifest_path(tmp_path)) in str(refused.value)
    assert "schema_version '0'" in str(refused.value)


def test_a_lock_that_cannot_be_taken_is_refused_as_a_lock_failure_naming_its_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import calibrated_agent_evaluation._recordstore as recordstore

    def held_elsewhere(_fd: int) -> None:
        # What msvcrt.locking raises on Windows once another process held the lock too long.
        raise OSError(errno.EDEADLK, "Resource deadlock avoided")

    monkeypatch.setattr(recordstore, "_lock", held_elsewhere)

    with pytest.raises(RuntimeInvocationError) as refused:
        append_run_record(tmp_path, _record())

    message = str(refused.value)
    assert f"could not lock the run manifest {manifest_path(tmp_path)}" in message
    assert "another run may be holding the lock" in message
    assert not manifest_path(tmp_path).exists()
    assert isinstance(refused.value.__cause__, OSError)


def _append_many(directory: str, worker: int, count: int) -> None:
    for i in range(count):
        append_run_record(Path(directory), _record(trace_id=f"worker-{worker}-run-{i}"))


def test_appends_from_two_processes_under_the_lock_keep_every_record(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    per_worker = 15
    workers = [
        ctx.Process(target=_append_many, args=(str(tmp_path), worker, per_worker))
        for worker in range(2)
    ]
    for proc in workers:
        proc.start()
    for proc in workers:
        proc.join(timeout=120)
        assert proc.exitcode == 0

    stored = RUN_RECORD_STORE.load(tmp_path, fmt="json")
    expected = {f"worker-{w}-run-{i}" for w in range(2) for i in range(per_worker)}
    assert sorted(record.trace_id for record in stored) == sorted(expected)
    assert RUN_RECORD_STORE.load(tmp_path, fmt="csv") == stored
