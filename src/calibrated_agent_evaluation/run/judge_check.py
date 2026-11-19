# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The judge re-check: a certified judge measured again on fresh gold, against its certification.

A judge is certified once, on one locked test. Production moves, and a judge can get worse on it
without anything in its file changing. :class:`JudgeCheckRecord` holds one re-check: the judge's
TPR and TNR on fresh human gold, and :func:`~calibrated_agent_evaluation.run.judge_drift` between
the certifying calibration record and that measurement. The check never certifies, never consumes
the locked test and never alters a calibration record. When the drift is flagged the judge should be
certified again on new gold.

:func:`check_size` says how much fresh gold the check needs per class: the smallest size at which
a judge whose rate fell to the gate threshold, and showed that rate (rounded up), would be flagged
against the certifying record, at that size and at every larger one up to
:data:`MAX_CHECK_PER_CLASS`. It is derived from the record's own counts and the same interval
``judge_drift`` reports, so a judge certified on little gold needs more fresh gold to be checked.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict, Field

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._provenance import ProvenanceStamp
from calibrated_agent_evaluation._recordstore import RecordStore
from calibrated_agent_evaluation.calibrate.errors import InfeasiblePlanError
from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord, Interval
from calibrated_agent_evaluation.run.errors import RunError, RunStoreError
from calibrated_agent_evaluation.run.report import JudgeDriftReport

__all__ = [
    "JUDGE_CHECK_SCHEMA_VERSION",
    "MAX_CHECK_PER_CLASS",
    "JudgeCheckRecord",
    "check_size",
    "drift_flagged",
    "dump_judge_checks",
    "load_judge_checks",
]

JUDGE_CHECK_SCHEMA_VERSION = "1"

# The largest per-class size check_size scans; a size that needs more is refused as infeasible.
MAX_CHECK_PER_CLASS = 10_000


class JudgeCheckRecord(CalibratedAgentEvaluationModel):
    """One re-check of a certified judge on fresh gold.

    ``against_calib_id`` is the certifying record the drift is measured against. ``tpr`` and
    ``tnr`` are the judge's rates on the fresh gold with their 95% lower bounds; ``drift`` is the
    change from the certifying record, each with its interval. ``drifted`` is true when either
    change's interval lies wholly below zero. ``needed_fail`` and ``needed_pass`` are the
    per-class sizes :func:`check_size` asked for.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = RunError

    check_id: str
    judge_id: str
    model: str
    prompt_version: str
    judge_fingerprint: str | None
    against_calib_id: str
    check_set_id: str
    n_pos: int
    n_neg: int
    tpr: Bound
    tnr: Bound
    drift: JudgeDriftReport
    drifted: bool
    needed_fail: int
    needed_pass: int
    agreement_alpha: Interval | None = None
    trace_ids: tuple[str, ...] = Field(default_factory=tuple)
    schema_version: str = JUDGE_CHECK_SCHEMA_VERSION
    provenance: ProvenanceStamp


def drift_flagged(drift: JudgeDriftReport) -> bool:
    """Whether either rate's change has an interval wholly below zero."""
    return drift.delta_tpr.hi < 0.0 or drift.delta_tnr.hi < 0.0


def _class_size(right_before: int, n_before: int, threshold: float, max_n: int) -> int:
    from calibrated_agent_evaluation.stats.drift import delta_proportion_ci

    def flagged(n: int) -> bool:
        right = math.ceil(threshold * n - 1e-9)
        return delta_proportion_ci(right, n, right_before, n_before)[2] < 0.0

    if not flagged(max_n):
        raise InfeasiblePlanError(
            f"check_size: at {right_before} of {n_before} on the certifying record, a drop to "
            f"{threshold:.2f} is not detected at any size up to {max_n} per class; the "
            "certification's own uncertainty is too wide. Certify again on more gold instead"
        )
    size = max_n
    for n in range(max_n - 1, 0, -1):
        if not flagged(n):
            break
        size = n
    return size


def check_size(
    record: CalibrationRecord, *, threshold: float = 0.80, max_n: int = MAX_CHECK_PER_CLASS
) -> tuple[int, int]:
    """The fresh gold per class (FAIL, PASS) a re-check of ``record``'s judge needs.

    For each class, the smallest ``n`` at which a judge showing ``threshold`` (the count rounded
    up, so the shown rate is never below it) is flagged by the drift interval against the record,
    at ``n`` and every larger size up to ``max_n``. At the planned size a judge that truly fell
    to the threshold is flagged about half the time, and one that fell further more often.

    Raises:
        InfeasiblePlanError: No size up to ``max_n`` detects the drop for some class.
    """
    tp = round(record.tpr.point * record.n_pos)
    tn = round(record.tnr.point * record.n_neg)
    return (
        _class_size(tp, record.n_pos, threshold, max_n),
        _class_size(tn, record.n_neg, threshold, max_n),
    )


_STORE = RecordStore(
    JudgeCheckRecord,
    stem="judge_check_record",
    schema_version=JUDGE_CHECK_SCHEMA_VERSION,
    error=RunStoreError,
    noun="judge check record",
    rerun="calibrated-agent-evaluation calibrate --check",
)


def dump_judge_checks(
    records: Sequence[JudgeCheckRecord], path: str | Path, fmt: str = "csv"
) -> None:
    """Write ``records`` to ``judge_check_record.{csv,json}`` under ``path``.

    Raises:
        RunStoreError: A bad ``fmt`` or any filesystem failure.
    """
    _STORE.dump(records, path, fmt)


def load_judge_checks(path: str | Path, fmt: str = "csv") -> list[JudgeCheckRecord]:
    """Read judge check records back from ``path``; a missing file is an empty store.

    Raises:
        RunStoreError: A version mismatch or any read, parse or validation failure.
    """
    return _STORE.load(path, fmt)
