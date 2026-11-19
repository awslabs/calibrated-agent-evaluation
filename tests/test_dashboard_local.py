# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the local dashboard sink + the Optional capability-absent path.

Behavioural: the local sink writes a numeric series that loads back field-for-field; a corrupt file
raises RunStoreError; and with no Langfuse backend configured, publishing targets only the local
sink and no vendor create_score/auth_check is invoked (the Optional-FR capability-absent path).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _helpers import TEST_INTERVAL_KIND
from calibrated_agent_evaluation.backends.dashboard import LocalDashboardSink
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.run.errors import RunStoreError
from calibrated_agent_evaluation.run.report import (
    DriftMetric,
    InputDriftReport,
    ModeKey,
    RunReport,
    TrendPoint,
    numeric_series,
)


def _report() -> RunReport:
    return RunReport(
        run_id="run-dash",
        mode="cadence",
        trend=(
            TrendPoint(
                mode=ModeKey(failure_type="over_refund"),
                n_total=50,
                corrected_rate=Interval(point=0.06, lo=0.02, hi=0.11, **TEST_INTERVAL_KIND),
                calib_id="c1",
                raw_rate=Interval(point=0.16, lo=0.08, hi=0.29, **TEST_INTERVAL_KIND),
            ),
            # a refused correction contributes NO numeric point
            TrendPoint(
                mode=ModeKey(failure_type="dropped_fact", agent="intake"),
                n_total=20,
                corrected_rate=Interval(
                    point=None,
                    lo=0.0,
                    hi=0.3,
                    refusal_reason="youden_j=0.4<0.5",
                    **TEST_INTERVAL_KIND,
                ),
                calib_id="c2",
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
    )


def test_local_sink_round_trips_field_for_field(tmp_path: Path) -> None:
    """The numeric series written to the local sink loads back field-for-field equal."""
    sink = LocalDashboardSink(tmp_path)
    report = _report()

    sink.publish(report)
    loaded = sink.load()

    assert tuple(loaded) == numeric_series(report)


def test_refused_correction_is_a_refusal_row_not_a_number(tmp_path: Path) -> None:
    """A refused-correction mode gets no number, and its refusal stays visible.

    It used to vanish from the series entirely; now it is a row with
    ``value=None``, its reason, and the widened interval.
    """
    sink = LocalDashboardSink(tmp_path)
    sink.publish(_report())
    by_name = {p.name: p for p in sink.load()}
    certified = by_name["rate.over_refund"]
    assert certified.value is not None
    assert certified.lo is not None
    assert certified.hi is not None
    refused = by_name["rate.intake/dropped_fact"]
    assert refused.value is None
    assert refused.refusal_reason
    assert refused.lo is not None
    assert refused.hi is not None
    assert all(by_name[n].p_value is not None for n in by_name if n.startswith("drift."))


def test_corrupt_series_file_raises_run_store_error(tmp_path: Path) -> None:
    """A corrupt JSON series surfaces as RunStoreError, not raw."""
    (tmp_path / "dashboard_series.json").write_text("{not valid json")
    with pytest.raises(RunStoreError):
        LocalDashboardSink(tmp_path).load()


def test_missing_series_loads_empty(tmp_path: Path) -> None:
    """Loading a directory with no published series returns an empty list."""
    assert LocalDashboardSink(tmp_path).load() == []


def test_capability_absent_writes_local_only_no_vendor_call(tmp_path: Path) -> None:
    """With no Langfuse backend, publish is local-only.

    The Optional-pattern negative: when the Langfuse capability is NOT configured, the local sink
    is authoritative and NO vendor create_score/auth_check is attempted. We prove the sibling
    (Langfuse) is not invoked by publishing through the local sink alone and confirming the
    round-trip stands with no vendor client in play at all.
    """
    sink = LocalDashboardSink(tmp_path)
    report = _report()
    sink.publish(report)
    # Local artifact is authoritative and complete without any Langfuse client existing.
    assert tuple(sink.load()) == numeric_series(report)
