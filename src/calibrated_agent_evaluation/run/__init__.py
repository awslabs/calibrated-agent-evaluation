# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run / CI-gate stage -- the ``run`` step of ``probe -> ... -> run -> improve``.

The point where the flat per-trace ``Label``s that ``score_traces`` and
``run_judge`` already produced become a **CI pass/fail gate**, a **per-failure-mode trend report**
(prevalence-corrected rates with CIs, never a single aggregate accuracy number), a **passive
staleness warning**, and **drift detection** (PSI/JSD/Wasserstein + judge ΔTPR/ΔTNR). It covers
multiplicity control and confidence-treatment everywhere.

Pure methodology plane: this package imports no vendor SDK type (a static test guards it). It
*reuses* the stats primitives (`multiplicity`/`corrected_rate`/`wilson_lower`), `enforce_ci_gate`,
the scorer/judge outputs, `assert_production_frame`, and the drift block in
:mod:`calibrated_agent_evaluation.stats.drift` -- it re-implements none of them. The only vendor
import lives in the confined ``backends/dashboard.py`` adapter. The library gate returns a typed
``RunGateDecision`` and a persisted ``RunRecord``; the CLI wires
``calibrated-agent-evaluation run --ci|--cadence`` onto it and maps the decision to a process exit
code.

The record types, errors, store and per-mode aggregation load eagerly and need nothing beyond the
base install. The gate, trend, drift, staleness and multiplicity functions need the
``stats`` extra, so they load lazily; ``backends/dashboard.py`` imports the report types under the
``langfuse`` extra alone.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from calibrated_agent_evaluation.run.aggregate import aggregate_by_mode
from calibrated_agent_evaluation.run.errors import RunError, RunGateError, RunStoreError
from calibrated_agent_evaluation.run.judge_check import (
    JUDGE_CHECK_SCHEMA_VERSION,
    JudgeCheckRecord,
    check_size,
    drift_flagged,
    dump_judge_checks,
    load_judge_checks,
)
from calibrated_agent_evaluation.run.report import (
    RUN_SCHEMA_VERSION,
    DiscoveryEnqueue,
    DriftMetric,
    InputDriftReport,
    JudgeDriftReport,
    ModeCounts,
    ModeGate,
    ModeKey,
    MultiplicityResult,
    RunDecision,
    RunGateDecision,
    RunRecord,
    RunReport,
    TrendPoint,
    run_record_of,
)
from calibrated_agent_evaluation.run.store import dump_run_records, load_run_records

if TYPE_CHECKING:
    from calibrated_agent_evaluation.run.drift_report import (
        DriftThresholds,
        input_drift,
        judge_drift,
        judge_drift_paired,
    )
    from calibrated_agent_evaluation.run.flags import flag_slices
    from calibrated_agent_evaluation.run.gate import (
        CiGateRequest,
        discovery_traces,
        gate_mode_counts,
        run_ci_gate,
    )
    from calibrated_agent_evaluation.run.multi import apply_multiplicity
    from calibrated_agent_evaluation.run.staleness import check_staleness
    from calibrated_agent_evaluation.run.trend import assert_production_rate_frame, trend_report

_LAZY: dict[str, str] = {
    "DriftThresholds": "drift_report",
    "input_drift": "drift_report",
    "judge_drift": "drift_report",
    "judge_drift_paired": "drift_report",
    "flag_slices": "flags",
    "CiGateRequest": "gate",
    "discovery_traces": "gate",
    "gate_mode_counts": "gate",
    "run_ci_gate": "gate",
    "apply_multiplicity": "multi",
    "check_staleness": "staleness",
    "assert_production_rate_frame": "trend",
    "trend_report": "trend",
}

__all__ = [
    "JUDGE_CHECK_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "CiGateRequest",
    "DiscoveryEnqueue",
    "DriftMetric",
    "DriftThresholds",
    "InputDriftReport",
    "JudgeCheckRecord",
    "JudgeDriftReport",
    "ModeCounts",
    "ModeGate",
    "ModeKey",
    "MultiplicityResult",
    "RunDecision",
    "RunError",
    "RunGateDecision",
    "RunGateError",
    "RunRecord",
    "RunReport",
    "RunStoreError",
    "TrendPoint",
    "aggregate_by_mode",
    "apply_multiplicity",
    "assert_production_rate_frame",
    "check_size",
    "check_staleness",
    "discovery_traces",
    "drift_flagged",
    "dump_judge_checks",
    "dump_run_records",
    "flag_slices",
    "gate_mode_counts",
    "input_drift",
    "judge_drift",
    "judge_drift_paired",
    "load_judge_checks",
    "load_run_records",
    "run_ci_gate",
    "run_record_of",
    "trend_report",
]


def __getattr__(name: str) -> object:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'calibrated_agent_evaluation.run' has no attribute {name!r}")
    return getattr(import_module(f"calibrated_agent_evaluation.run.{module}"), name)
