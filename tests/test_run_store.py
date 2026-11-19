# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the RunRecord round-trip store.

Property/round-trip: a RunRecord (with nested trend + drift + multiplicity + decisions) dumps and
loads field-for-field equal in both CSV and JSON; a schema-version mismatch and a malformed file
refuse loudly with the typed RunStoreError; the RunRecord/RunDecision empty-inputs guard raises.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _helpers import TEST_INTERVAL_KIND, TEST_STAMP
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.run.errors import RunError, RunStoreError
from calibrated_agent_evaluation.run.report import (
    DriftMetric,
    InputDriftReport,
    JudgeDriftReport,
    ModeKey,
    MultiplicityResult,
    RunDecision,
    RunRecord,
    TrendPoint,
)
from calibrated_agent_evaluation.run.store import dump_run_records, load_run_records


def _record() -> RunRecord:
    """A fully-populated record exercising every nested type the codec must reconstruct."""
    return RunRecord(
        provenance=TEST_STAMP,
        run_id="run-001",
        mode="cadence",
        gate_passed=True,
        judge_ids=("orderbot-refund",),
        calib_ids=("calib-1",),
        scorer_lanes=("ExactMatchScorer:ci_gate",),
        trend=(
            TrendPoint(
                mode=ModeKey(failure_type="over_refund"),
                n_total=50,
                corrected_rate=Interval(point=0.06, lo=0.02, hi=0.11, **TEST_INTERVAL_KIND),
                calib_id="calib-1",
                raw_rate=Interval(point=0.16, lo=0.08, hi=0.29, **TEST_INTERVAL_KIND),
            ),
            TrendPoint(
                mode=ModeKey(failure_type="dropped_fact", agent="intake"),
                n_total=20,
                corrected_rate=Interval(
                    point=None,
                    lo=0.0,
                    hi=0.3,
                    refusal_reason="youden_j=0.400<0.5",
                    **TEST_INTERVAL_KIND,
                ),
                calib_id="calib-2",
                raw_rate=Interval(point=0.16, lo=0.08, hi=0.29, **TEST_INTERVAL_KIND),
            ),
        ),
        input_drift=InputDriftReport(
            metrics=(
                DriftMetric(
                    name="psi",
                    value=0.31,
                    threshold=0.25,
                    exceeded=True,
                    p_value=0.001,
                    method="test",
                ),
                DriftMetric(
                    name="jsd",
                    value=0.10,
                    threshold=0.20,
                    exceeded=False,
                    p_value=0.001,
                    method="test",
                ),
                DriftMetric(
                    name="wasserstein",
                    value=0.50,
                    threshold=1.0,
                    exceeded=False,
                    p_value=0.001,
                    method="test",
                ),
            ),
            dimension="taxonomy_cell",
        ),
        judge_drift=(
            JudgeDriftReport(
                judge_id="orderbot-refund",
                delta_tpr=Interval(point=-0.02, lo=-0.05, hi=0.01, **TEST_INTERVAL_KIND),
                delta_tnr=Interval(point=0.0, lo=-0.03, hi=0.03, **TEST_INTERVAL_KIND),
            ),
        ),
        multiplicity=(
            MultiplicityResult(family="slice_flags", method="fdr_by", m=2, reject=(False, True)),
            MultiplicityResult(
                family="drift_flags", method="fdr_by", m=3, reject=(True, False, False)
            ),
        ),
        decisions=(
            RunDecision(
                rule="RUN-4",
                tier="warning",
                inputs={"judge_id": "orderbot-refund"},
                outcome="warned",
                message="judge version drifted from its calibration record",
            ),
        ),
        inputs={"run_id": "run-001"},
        gate_n_modes=2,
    )


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_run_record_round_trips_field_for_field(fmt: str, tmp_path: Path) -> None:
    """Dump -> load returns records field-for-field equal, in both CSV and JSON."""
    record = _record()
    dump_run_records([record], tmp_path, fmt=fmt)
    loaded = load_run_records(tmp_path, fmt=fmt)
    assert loaded == [record]


def test_schema_version_mismatch_refuses_load(tmp_path: Path) -> None:
    """A stored schema_version != running raises RunStoreError (no coercion)."""

    stale = replace(_record(), schema_version="999")
    dump_run_records([stale], tmp_path, fmt="json")
    with pytest.raises(RunStoreError):
        load_run_records(tmp_path, fmt="json")


def test_malformed_file_raises_run_store_error_not_raw(tmp_path: Path) -> None:
    """A corrupt JSON payload surfaces as RunStoreError, not raw."""
    (tmp_path / "run_record.json").write_text("{not valid json")
    with pytest.raises(RunStoreError):
        load_run_records(tmp_path, fmt="json")


def test_missing_required_field_raises_run_store_error(tmp_path: Path) -> None:
    """A structurally-malformed record (missing key) is a typed refusal."""
    (tmp_path / "run_record.json").write_text('[{"run_id": "x", "schema_version": "1"}]')
    with pytest.raises(RunStoreError):
        load_run_records(tmp_path, fmt="json")


def test_empty_inputs_guard_raises_run_error() -> None:
    """Constructing a RunRecord/RunDecision with empty inputs raises RunError."""
    with pytest.raises(RunError):
        RunRecord(
            provenance=TEST_STAMP,
            run_id="x",
            mode="ci",
            gate_passed=None,
            judge_ids=(),
            calib_ids=(),
            scorer_lanes=(),
            trend=(),
            input_drift=None,
            judge_drift=(),
            multiplicity=(),
            decisions=(),
            inputs={},
        )
    with pytest.raises(RunError):
        RunDecision(rule="R", tier="warning", inputs={}, outcome="warned", message="m")


def test_missing_files_load_empty(tmp_path: Path) -> None:
    """Loading a directory with no dump returns an empty list, not an error."""
    assert load_run_records(tmp_path, fmt="csv") == []
    assert load_run_records(tmp_path, fmt="json") == []
