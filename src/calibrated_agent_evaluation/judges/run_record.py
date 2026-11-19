# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The JudgeRunRecord: the reproducibility anchor for a judge run.

``Label`` has no field for the frozen judge version, but every reported number must be reproducible
from stored artifacts. So a judge run emits a ``JudgeRunRecord`` recording the
frozen ``(judge_id, model, prompt_version)``, the calibration record it cleared, and exactly which
traces/spans it scored. Verdicts link back by ``trace_id ∈ record.trace_ids``.

Like the ``CalibrationRecord`` (which spans a hunted set + a rate sample), a run record is not
a ``TraceStoreBackend`` object, so it ships its OWN CSV+JSON round-trip reusing the calibration
store *pattern*: flat ``dict[str, str]`` rows, one JSON-encoded value per cell, a ``schema_version``
that refuses-on-mismatch (no coercion). Any filesystem/parse failure is wrapped into
:class:`JudgeRuntimeError`; no raw ``OSError``/``JSONDecodeError`` escapes (conventions: "no silent
failure").
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import ConfigDict, model_validator

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._recordstore import RecordStore
from calibrated_agent_evaluation.judges.errors import JudgeRuntimeError
from calibrated_agent_evaluation.judges.types import JudgeLevel
from calibrated_agent_evaluation.judges.version import JUDGES_SCHEMA_VERSION


class JudgeRunRecord(CalibratedAgentEvaluationModel):
    """One judge run's reproducibility record.

    ``model``/``prompt_version`` are the frozen judge version; ``calib_id`` is the
    :class:`~calibrated_agent_evaluation.calibrate.CalibrationRecord` this run cleared.
    ``trace_ids`` and ``span_ids`` are aligned: ``span_ids[i]`` is the span scored on
    ``trace_ids[i]`` (``None`` ⇒ trace-level). ``n_cached``/``n_computed`` count verdicts served
    from cache vs computed by a model call. A record with an empty ``calib_id`` is refused at
    construction; a run with no cleared calibration is not reproducible and must not be recorded as
    if it were.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = JudgeRuntimeError

    run_id: str
    judge_id: str
    model: str
    prompt_version: str
    calib_id: str
    # An AgentCore session-level evaluator scores whole sessions, a level no prompt judge has.
    level: JudgeLevel | Literal["session"]
    trace_ids: tuple[str, ...]
    span_ids: tuple[str | None, ...]
    n_cached: int = 0
    n_computed: int = 0
    schema_version: str = JUDGES_SCHEMA_VERSION
    judge_fingerprint: str | None = None  # the judge's content hash when it carries one
    # The pre-registration ledger's head (last entry hash, or "none") and length when the run was
    # recorded, so a later evaluation can check the ledger it ran against still holds.
    ledger_head: str | None = None
    ledger_entries: int | None = None

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        self._post_init()
        return self

    def _post_init(self) -> None:
        # A run record must name the calibration it cleared, or it cannot answer
        # "which certified judge produced these verdicts?"; refuse an un-anchored record.
        if not self.calib_id:
            raise JudgeRuntimeError(
                f"JudgeRunRecord {self.run_id!r} has no calib_id; a judge run must record the "
                "calibration record it cleared"
            )
        if len(self.trace_ids) != len(self.span_ids):
            raise JudgeRuntimeError(
                f"JudgeRunRecord {self.run_id!r}: trace_ids ({len(self.trace_ids)}) and span_ids "
                f"({len(self.span_ids)}) must be aligned 1:1"
            )


_STORE = RecordStore(
    JudgeRunRecord,
    stem="judge_run_record",
    schema_version=JUDGES_SCHEMA_VERSION,
    error=JudgeRuntimeError,
    noun="judge run record",
    rerun="calibrated-agent-evaluation judge",
)


def dump_run_records(records: Sequence[JudgeRunRecord], path: str | Path, fmt: str = "csv") -> None:
    """Write ``records`` to ``judge_run_record.{csv,json}`` under ``path``.

    Args:
        records: The judge run records to persist.
        path: The directory to write into.
        fmt: Which file a later :func:`load_run_records` reads back (both are always written).

    Raises:
        JudgeRuntimeError: A bad ``fmt`` or any filesystem failure.
    """
    _STORE.dump(records, path, fmt)


def load_run_records(path: str | Path, fmt: str = "csv") -> list[JudgeRunRecord]:
    """Read judge run records back from ``path``; a missing file is an empty store.

    Args:
        path: The directory :func:`dump_run_records` wrote.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The stored records.

    Raises:
        JudgeRuntimeError: A schema-version mismatch, or any read, parse or validation failure.
    """
    return _STORE.load(path, fmt)
