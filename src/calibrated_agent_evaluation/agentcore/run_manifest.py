# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run manifest: one record per run, tying each trace to the agent setup that produced it.

A :class:`RunRecord` says which runtime, qualifier, runtime version and configuration bundle a run
used, which IDs it carried, when it ran, how its session ended and, when it failed, why. Records
are keyed by ``trace_id``: the ID of the ``Trace`` the run returned, or would have returned.

A log directory's manifest is the pair ``agentcore_run_record.json`` and ``.csv``, written through
the shared record store with a ``schema_version``. :func:`append_run_record` rewrites both files
with the new record, each replaced atomically, while holding the directory's lock, so two runs
appending at once both keep theirs. The JSON file, :func:`manifest_path`, is the one read back.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import ConfigDict

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._recordstore import RecordStore, directory_lock
from calibrated_agent_evaluation.agentcore.runtime_errors import RuntimeInvocationError

__all__ = [
    "AGENTCORE_RUN_SCHEMA_VERSION",
    "RUN_RECORD_STEM",
    "RUN_RECORD_STORE",
    "RunRecord",
    "SessionStop",
    "VersionSource",
    "append_run_record",
    "manifest_path",
]

AGENTCORE_RUN_SCHEMA_VERSION = "1"
"""The run record format this version writes and reads; a manifest of another one is refused."""

RUN_RECORD_STEM = "agentcore_run_record"
"""The file name stem of the run manifest's JSON and CSV files inside a log directory."""

VersionSource = Literal["endpoint", "latest"]
"""Where a runtime version came from: the endpoint's live version, or the runtime's latest."""

SessionStop = Literal["stopped", "already_gone", "failed", "not_started", "held_open"]
"""How a run's runtime session ended.

``stopped``: the stop succeeded. ``already_gone``: the service no longer had the session, which
counts as stopped. ``failed``: the stop still failed after the SDK's retries, so the session may
stay billed until it times out. ``not_started``: no invocation was sent, so there was nothing to
stop. ``held_open``: a simulated conversation's turn left the session open for its next turn; the
conversation's end stops it, and a stop that fails is logged as a warning and tried again when the
executor closes.
"""


class RunRecord(CalibratedAgentEvaluationModel):
    """One record of the run manifest: where one run went, what it carried and how it ended.

    Attributes:
        schema_version: The run record format, :data:`AGENTCORE_RUN_SCHEMA_VERSION`.
        trace_id: The ID of the run's ``Trace``; the key the record is found by.
        w3c_trace_id: The W3C trace ID the run's invocations carried.
        session_id: The runtime session the run used.
        runtime_arn: The runtime invoked.
        qualifier: The endpoint invoked.
        runtime_version: The runtime version that served the run, or ``None`` when the run ended
            before the version was known.
        runtime_version_source: ``endpoint`` or ``latest``, or ``None`` with no version.
        config_bundle_arn: The configuration bundle the run named, or ``None``.
        config_bundle_version: That bundle's version, or ``None``.
        payload_format: The request payload format.
        started_at: When the run started, ISO-8601 with its UTC offset.
        ended_at: When the run ended, in the same form.
        duration_ms: How long the run took, in milliseconds.
        attempts: How many invocation attempts the run sent, retries included.
        status_code: The last response's HTTP status, or ``None`` with no response.
        content_type: The last response's content type, or ``None``.
        response_traceparent: The ``traceparent`` the runtime sent back, or ``None``.
        replayed: Whether the run was replayed from a cassette instead of sent.
        session_stop: How the run's session ended.
        error_code: Why the run failed: the AWS error code for an AWS service error, else the
            type name of the error that ended it. ``None`` for a run that succeeded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", defer_build=True)
    invalid_error: ClassVar[type[Exception]] = RuntimeInvocationError

    schema_version: str = AGENTCORE_RUN_SCHEMA_VERSION
    trace_id: str
    w3c_trace_id: str
    session_id: str
    runtime_arn: str
    qualifier: str
    runtime_version: str | None
    runtime_version_source: VersionSource | None
    config_bundle_arn: str | None
    config_bundle_version: str | None
    payload_format: str
    started_at: str
    ended_at: str
    duration_ms: int
    attempts: int
    status_code: int | None
    content_type: str | None
    response_traceparent: str | None
    replayed: bool
    session_stop: SessionStop
    error_code: str | None = None


RUN_RECORD_STORE: RecordStore[RunRecord] = RecordStore(
    RunRecord,
    stem=RUN_RECORD_STEM,
    schema_version=AGENTCORE_RUN_SCHEMA_VERSION,
    error=RuntimeInvocationError,
    noun="AgentCore run record",
)
"""The record store that reads and writes the run manifest of a log directory."""


def manifest_path(directory: Path) -> Path:
    """The run manifest's JSON file in the log directory ``directory``.

    It is the file read back, and the path messages name; its CSV twin sits beside it.
    """
    return directory / f"{RUN_RECORD_STEM}.json"


def append_run_record(directory: Path, record: RunRecord) -> None:
    """Add ``record`` to the run manifest in the log directory ``directory``.

    The directory is created if missing, and a missing manifest is an empty one. Holding the
    directory's lock, it reads the JSON file's records and writes both files back with ``record``
    added. A manifest that is refused is left as it was.

    Args:
        directory: The log directory.
        record: The run to add.

    Raises:
        RuntimeInvocationError: The directory can't be created; it can't be locked (it is not
            writable, or, on Windows, another run held the lock for about 10 seconds); the
            manifest can't be read or written; or it is malformed, or holds records of another or
            no ``schema_version``. The message names :func:`manifest_path`, which step failed and
            what to do next.
    """
    path = manifest_path(directory)
    with contextlib.ExitStack() as held:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeInvocationError(
                f"could not create the log directory of the run manifest {path} "
                f"({type(exc).__name__}); check that no file stands in its place and that its "
                "parent directory is writable, or pass another log directory"
            ) from exc
        try:
            held.enter_context(directory_lock(directory))
        except OSError as exc:
            # Told apart from a failed write: on Windows another run holding the lock for about
            # 10 seconds ends here too, and waiting for it is the way out.
            raise RuntimeInvocationError(
                f"could not lock the run manifest {path} ({type(exc).__name__}): its directory "
                "may not be writable, or another run may be holding the lock. Check that the "
                "directory is writable, or let the other run finish and run again, or pass "
                "another log directory"
            ) from exc
        try:
            stored = RUN_RECORD_STORE.load(directory, fmt="json")
            RUN_RECORD_STORE.dump([*stored, record], directory, fmt="json")
        except (OSError, RuntimeInvocationError) as exc:
            # The store wraps a filesystem failure in its typed error, so the cause tells a read or
            # write failure apart from stored records that are malformed or of another format.
            failure = exc if isinstance(exc, OSError) else exc.__cause__
            if isinstance(failure, OSError):
                raise RuntimeInvocationError(
                    f"could not read or write the run manifest {path} "
                    f"({type(failure).__name__}); check that its directory is writable and holds "
                    "no directory in the manifest's place, or pass another log directory"
                ) from exc
            # The store's message says what is wrong with the stored records, with no values.
            detail = str(exc).rstrip(".")
            raise RuntimeInvocationError(
                f"could not add a run record to the run manifest {path} ({detail}). Move a "
                "malformed manifest or one of another format aside, or pass another log directory"
            ) from exc
