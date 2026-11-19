# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation results`` rows for policy-decision findings, from recorded
batteries."""

from __future__ import annotations

import pytest

pytest.importorskip("boto3")

from calibrated_agent_evaluation._provenance import capture_provenance
from calibrated_agent_evaluation.dumpload.results import results_table
from calibrated_agent_evaluation.probe import ProbeBatteryRecord
from calibrated_agent_evaluation.probe.framing import LimitsFraming
from calibrated_agent_evaluation.probe.record import ProbeFinding
from test_probe_policy_battery import _run


def _record(*findings: ProbeFinding) -> ProbeBatteryRecord:
    return ProbeBatteryRecord(
        probe_id="probe-1",
        endpoint_fingerprint="fp",
        findings=findings,
        provenance=capture_provenance(),
    )


def _rows(record: ProbeBatteryRecord):
    return [row for row in results_table([record]) if row.source == "probe:probe-1"]


def test_a_policy_finding_gives_one_upper_bound_row_per_count() -> None:
    delete = next(f for f in _run("phase3").findings if f.payload["case_id"] == "delete")
    rows = _rows(_record(delete))
    assert [r.estimand for r in rows] == [
        "policy_observed_allow_rate",
        "policy_gap_rate",
        "policy_denied_rate",
        "policy_over_deny_rate",
    ]
    assert {r.slice for r in rows} == {f"{delete.endpoint_fingerprint}/delete/enforce"}
    assert all(r.sided == "upper" and r.point is None and r.n == 3 for r in rows)
    assert rows[0].hi == float(delete.payload["bounds"]["observed_allow"]["ub95"])


def test_a_case_with_no_scored_trial_gives_unbounded_rows_with_a_reason() -> None:
    delete = next(f for f in _run("silent").findings if f.payload["case_id"] == "delete")
    rows = _rows(_record(delete))
    assert len(rows) == 4
    assert all(r.n == 0 and r.hi == 1.0 for r in rows)
    assert all("held a Policy decision" in (r.refusal_reason or "") for r in rows)


def test_a_policy_finding_is_never_dropped_beside_a_benchmark_finding() -> None:
    delete = next(f for f in _run("phase2").findings if f.payload["case_id"] == "delete")
    benchmark = ProbeFinding(
        kind="benchmark",
        captured_at="2026-11-02T00:00:00+00:00",
        endpoint_fingerprint="fp",
        framing=LimitsFraming(),
        payload={"benchmark": "b", "k": 5, "break_rate_ub95": "0.4507"},
    )
    rows = _rows(_record(delete, benchmark))
    assert len(rows) == 5
    assert [r.estimand for r in rows].count("break_rate") == 1
