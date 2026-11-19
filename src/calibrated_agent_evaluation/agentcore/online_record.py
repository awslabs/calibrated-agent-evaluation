# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The records an AgentCore online evaluation keeps: the config attached and the sessions to skip.

An :class:`OnlineEvaluationRecord` names one online evaluation config Calibrated Agent Evaluation
created, the evaluators attached to it with the version and calibration each was certified at, and
the session ids its read-back skips. An :class:`AgentCoreSessionLog` lists the session ids one
calibration or bias run sent, so they can be skipped later; it is kept apart from the calibration
and bias records, which take no extra fields.

Neither record holds an AWS type, and this module imports no AWS SDK. Each is written through the
shared record store, with a ``schema_version``, as ``agentcore_online_config.{csv,json}`` and
``agentcore_session_log.{csv,json}``. A stored record of another or no ``schema_version``, with an
unknown field, or that does not parse, is refused with :class:`AgentCoreError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import ConfigDict, Field

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._recordstore import RecordStore
from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
from calibrated_agent_evaluation.agentcore.online_version import ONLINE_SCHEMA_VERSION
from calibrated_agent_evaluation.schema.types import Label

__all__ = [
    "AgentCoreSessionLog",
    "AttachedEvaluator",
    "OnlineEvaluationRecord",
    "OnlineReadRecord",
    "ReadVerdict",
    "SessionRunKind",
    "dump_online_records",
    "dump_read_records",
    "dump_session_logs",
    "load_online_records",
    "load_read_records",
    "load_session_logs",
    "new_read_id",
    "session_log_id",
    "verdict_digest",
]

SessionRunKind = Literal["calibration", "bias"]
"""The kind of run a session log lists the sessions of."""


class AttachedEvaluator(CalibratedAgentEvaluationModel):
    """One evaluator attached to an online config, as it was when it was certified.

    Attributes:
        evaluator_id: The AgentCore evaluator id.
        evaluator_arn: The evaluator's ARN.
        updated_at: The evaluator's ``updatedAt`` when it was attached, as AgentCore reported it.
        fingerprint: The judge fingerprint its calibration was pinned to.
        calib_id: The calibration record that certified it.
        battery_id: The bias battery record that cleared it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", defer_build=True)
    invalid_error: ClassVar[type[Exception]] = AgentCoreError

    evaluator_id: str
    evaluator_arn: str
    updated_at: str
    fingerprint: str
    calib_id: str
    battery_id: str


class OnlineEvaluationRecord(CalibratedAgentEvaluationModel):
    """One AgentCore online evaluation config Calibrated Agent Evaluation created.

    Attributes:
        schema_version: The record format, :data:`ONLINE_SCHEMA_VERSION`.
        config_id: The online config id AgentCore generated; the key the record is found by.
        config_name: The name the config was created with.
        runtime_service_name: The ``service.name`` of the runtime it watches.
        evaluators: The evaluators attached, each as it was certified.
        excluded_session_ids: The calibration and bias sessions its read-back skips.
        sampling_percentage: The share of sessions AgentCore scores, 0.01 to 100 (AgentCore's
            bounds).
        session_timeout_minutes: How long a session is idle before it is scored, 1 to 1440
            minutes (AgentCore's bounds).
        created_at: When the config was created, as AgentCore reported it.
        filtered: Whether the config scores only sessions its filters admit; its sessions are then
            not a random sample of the agent's traffic. The record type and this field ship
            together, so no stored record of a filtered config predates it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", defer_build=True)
    invalid_error: ClassVar[type[Exception]] = AgentCoreError

    schema_version: str = ONLINE_SCHEMA_VERSION
    config_id: str
    config_name: str
    runtime_service_name: str
    evaluators: tuple[AttachedEvaluator, ...]
    excluded_session_ids: tuple[str, ...]
    sampling_percentage: float = Field(ge=0.01, le=100.0)
    session_timeout_minutes: int = Field(ge=1, le=1440)
    created_at: str
    filtered: bool = False


class AgentCoreSessionLog(CalibratedAgentEvaluationModel):
    """The session ids one calibration or bias run sent to AgentCore.

    Attributes:
        schema_version: The record format, :data:`ONLINE_SCHEMA_VERSION`.
        run_kind: ``calibration`` or ``bias``.
        run_id: The run's ``calib_id`` or ``battery_id``.
        judge_id: The judge the run scored with.
        evaluator_fingerprint: The judge fingerprint the run was pinned to.
        session_ids: The session ids the run sent, in the order sent.
        gold_source: For a calibration, where its gold came from (``synthetic-rule``, ``human``,
            or both joined by ``and``), recorded when it ran so a command against another store
            names the same source; ``None`` for a bias run and for a log written before it was
            recorded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", defer_build=True)
    invalid_error: ClassVar[type[Exception]] = AgentCoreError

    schema_version: str = ONLINE_SCHEMA_VERSION
    run_kind: SessionRunKind
    run_id: str
    judge_id: str
    evaluator_fingerprint: str
    session_ids: tuple[str, ...]
    gold_source: str | None = None


class ReadVerdict(CalibratedAgentEvaluationModel):
    """One verdict a ``calibrated-agent-evaluation online read`` stored or found already stored.

    Attributes:
        evaluator_id: The evaluator that produced it.
        trace_id: The trace it is about.
        verdict_digest: :func:`verdict_digest` of the stored label, so the very verdict is named,
            not just its trace: an on-demand verdict on the same trace carries the same rater id.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", defer_build=True)
    invalid_error: ClassVar[type[Exception]] = AgentCoreError

    evaluator_id: str
    trace_id: str
    verdict_digest: str


class OnlineReadRecord(CalibratedAgentEvaluationModel):
    """What one ``online read`` read back for an online config, into which store.

    Attributes:
        schema_version: The record format, :data:`ONLINE_SCHEMA_VERSION`.
        read_id: ``<config_id>:<random hex>``, so two reads of one window are two records.
        config_id: The online config read.
        backend: The store the verdicts went to, ``local`` or ``langfuse``.
        since: The window's start.
        until: The window's end.
        verdicts: Each verdict the read stored or found already stored.
        errors: Per AgentCore error type, the results it could not score.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", defer_build=True)
    invalid_error: ClassVar[type[Exception]] = AgentCoreError

    schema_version: str = ONLINE_SCHEMA_VERSION
    read_id: str
    config_id: str
    backend: Literal["local", "langfuse"]
    since: str
    until: str
    verdicts: tuple[ReadVerdict, ...]
    errors: dict[str, int] = Field(default_factory=dict)


def verdict_digest(label: Label) -> str:
    """A 16-hex SHA-256 of ``label``'s full identity (every field), naming that one verdict."""
    import hashlib

    from calibrated_agent_evaluation.schema.serde import label_identity

    return hashlib.sha256(label_identity(label).encode("utf-8")).hexdigest()[:16]


def new_read_id(config_id: str) -> str:
    """A new read id for ``config_id``."""
    import uuid

    return f"{config_id}:{uuid.uuid4().hex}"


def session_log_id(log: AgentCoreSessionLog) -> str:
    """The key a session log is stored under: ``<run_kind>:<run_id>``.

    A calibration and a bias run can share an id, so the kind is part of the key.
    """
    return f"{log.run_kind}:{log.run_id}"


_ONLINE_STORE = RecordStore(
    OnlineEvaluationRecord,
    stem="agentcore_online_config",
    schema_version=ONLINE_SCHEMA_VERSION,
    error=AgentCoreError,
    noun="AgentCore online evaluation record",
    rerun="calibrated-agent-evaluation online attach",
)

_READ_STORE = RecordStore(
    OnlineReadRecord,
    stem="agentcore_online_read",
    schema_version=ONLINE_SCHEMA_VERSION,
    error=AgentCoreError,
    noun="AgentCore online read record",
    rerun="calibrated-agent-evaluation online read",
)
_SESSION_STORE = RecordStore(
    AgentCoreSessionLog,
    stem="agentcore_session_log",
    schema_version=ONLINE_SCHEMA_VERSION,
    error=AgentCoreError,
    noun="AgentCore session log",
)


def dump_online_records(
    records: Sequence[OnlineEvaluationRecord], path: str | Path, fmt: str = "csv"
) -> None:
    """Write ``records`` to ``agentcore_online_config.{csv,json}`` under ``path``.

    Args:
        records: The online-evaluation records to persist.
        path: The directory to write into.
        fmt: Which file a later :func:`load_online_records` reads back (both are always written).

    Raises:
        AgentCoreError: A bad ``fmt`` or any filesystem failure.
    """
    _ONLINE_STORE.dump(records, path, fmt)


def load_online_records(path: str | Path, fmt: str = "csv") -> list[OnlineEvaluationRecord]:
    """Read online-evaluation records back from ``path``; a missing file is an empty store.

    Args:
        path: The directory :func:`dump_online_records` wrote.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The stored records.

    Raises:
        AgentCoreError: A stored record of another or no ``schema_version``, with an unknown
            field, or any read, parse or validation failure.
    """
    return _ONLINE_STORE.load(path, fmt)


def dump_session_logs(
    logs: Sequence[AgentCoreSessionLog], path: str | Path, fmt: str = "csv"
) -> None:
    """Write ``logs`` to ``agentcore_session_log.{csv,json}`` under ``path``.

    Args:
        logs: The session logs to persist.
        path: The directory to write into.
        fmt: Which file a later :func:`load_session_logs` reads back (both are always written).

    Raises:
        AgentCoreError: A bad ``fmt`` or any filesystem failure.
    """
    _SESSION_STORE.dump(logs, path, fmt)


def load_session_logs(path: str | Path, fmt: str = "csv") -> list[AgentCoreSessionLog]:
    """Read session logs back from ``path``; a missing file is an empty store.

    Args:
        path: The directory :func:`dump_session_logs` wrote.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The stored logs.

    Raises:
        AgentCoreError: A stored log of another or no ``schema_version``, with an unknown field,
            or any read, parse or validation failure.
    """
    return _SESSION_STORE.load(path, fmt)


def dump_read_records(
    records: Sequence[OnlineReadRecord], path: str | Path, fmt: str = "csv"
) -> None:
    """Write ``records`` to ``agentcore_online_read.{csv,json}`` under ``path``.

    Args:
        records: The read records to persist.
        path: The directory to write into.
        fmt: Which file a later :func:`load_read_records` reads back (both are always written).

    Raises:
        AgentCoreError: A bad ``fmt`` or any filesystem failure.
    """
    _READ_STORE.dump(records, path, fmt)


def load_read_records(path: str | Path, fmt: str = "csv") -> list[OnlineReadRecord]:
    """Read read records back from ``path``; a missing file is an empty store.

    Args:
        path: The directory :func:`dump_read_records` wrote.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The stored records.

    Raises:
        AgentCoreError: A stored record of another or no ``schema_version``, with an unknown
            field, or any read, parse or validation failure.
    """
    return _READ_STORE.load(path, fmt)
