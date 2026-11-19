# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check a judge against human labels, and correct the failure rate it reports.

Works on pairs of a human gold label and the judge's verdict on the same trace, and answers three
questions: is this judge good enough to trust (both its true-positive and true-negative rates
must clear the gate at the low end of their 95% range), what is the real failure rate (the
Rogan-Gladen correction, with a range that carries the uncertainty of both the gold set and the
scored sample), and when must the judge be checked again (version and fingerprint pinning, and
single-use test sets). Running a judge is the judge runtime's job, not this package's.

Reads normalized ``Label`` objects only; the numeric work lives in
:mod:`calibrated_agent_evaluation.stats`. The engine and the planner load lazily because they need
the ``stats`` extra, while the record types (``Interval``, ``Bound``) and the errors need nothing
beyond the base install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from calibrated_agent_evaluation.calibrate.errors import (
    CalibrationError,
    CalibrationStoreError,
    CalibrationVersionError,
    DuplicateGoldError,
    ErrorRatesDoNotTransferError,
    FewShotEvaluationError,
    InfeasiblePlanError,
    LockedTestReuseError,
    NonIdentifiableRateError,
    RateSampleTooSmallError,
    RateUnsafeError,
    SampleMisuseError,
    StatisticsInputError,
    UncalibratedJudgeError,
    UnpairedObservationError,
    UnratifiedGoldError,
    WeakJudgeError,
)
from calibrated_agent_evaluation.calibrate.ledger import (
    InMemoryLockedTestLedger,
    LockedTestLedger,
    locked_set_digest,
    open_for_top_up,
)
from calibrated_agent_evaluation.calibrate.pairing import (
    ConfusionCounts,
    pair_observations,
    refuse_duplicate_targets,
)
from calibrated_agent_evaluation.calibrate.record import (
    CALIB_SCHEMA_VERSION,
    Bound,
    CalibrationRecord,
    CalibrationSet,
    Interval,
    LabelPair,
)
from calibrated_agent_evaluation.calibrate.store import dump_records, load_records

if TYPE_CHECKING:
    from calibrated_agent_evaluation.calibrate.engine import (
        certifies,
        certify,
        corrected_rate,
        fingerprint_of,
        pin_mismatch,
    )
    from calibrated_agent_evaluation.calibrate.plan import (
        MIN_RATE_SAMPLE,
        CalibrationPlan,
        coverage_floor,
        double_coded_per_class,
        gate_fails_allowed,
        gate_set_size,
        plan_calibration,
        plan_calibration_size,
        rate_safe_at,
        rate_safe_calibration_size,
        wilson_calibration_size,
    )

_PLAN_NAMES = (
    "MIN_RATE_SAMPLE",
    "CalibrationPlan",
    "coverage_floor",
    "double_coded_per_class",
    "gate_fails_allowed",
    "gate_set_size",
    "plan_calibration",
    "plan_calibration_size",
    "rate_safe_at",
    "rate_safe_calibration_size",
    "wilson_calibration_size",
)

__all__ = [
    "CALIB_SCHEMA_VERSION",
    "MIN_RATE_SAMPLE",
    "Bound",
    "CalibrationError",
    "CalibrationPlan",
    "CalibrationRecord",
    "CalibrationSet",
    "CalibrationStoreError",
    "CalibrationVersionError",
    "ConfusionCounts",
    "DuplicateGoldError",
    "ErrorRatesDoNotTransferError",
    "FewShotEvaluationError",
    "InMemoryLockedTestLedger",
    "InfeasiblePlanError",
    "Interval",
    "LabelPair",
    "LockedTestLedger",
    "LockedTestReuseError",
    "NonIdentifiableRateError",
    "RateSampleTooSmallError",
    "RateUnsafeError",
    "SampleMisuseError",
    "StatisticsInputError",
    "UncalibratedJudgeError",
    "UnpairedObservationError",
    "UnratifiedGoldError",
    "WeakJudgeError",
    "certifies",
    "certify",
    "corrected_rate",
    "coverage_floor",
    "double_coded_per_class",
    "dump_records",
    "fingerprint_of",
    "gate_fails_allowed",
    "gate_set_size",
    "load_records",
    "locked_set_digest",
    "open_for_top_up",
    "pair_observations",
    "pin_mismatch",
    "plan_calibration",
    "plan_calibration_size",
    "rate_safe_at",
    "rate_safe_calibration_size",
    "refuse_duplicate_targets",
    "wilson_calibration_size",
]


def __getattr__(name: str) -> object:
    if name in ("certifies", "certify", "corrected_rate", "fingerprint_of", "pin_mismatch"):
        from calibrated_agent_evaluation.calibrate import engine

        return getattr(engine, name)
    if name in _PLAN_NAMES:
        from calibrated_agent_evaluation.calibrate import plan

        return getattr(plan, name)
    raise AttributeError(
        f"module 'calibrated_agent_evaluation.calibrate' has no attribute {name!r}"
    )
