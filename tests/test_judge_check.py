# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The judge re-check record, its store, and the fresh-gold size it is planned at.

The size is checked against the interval it is derived from: at the planned size a judge showing
exactly the gate threshold is flagged, at every larger size up to the cap too, and one size below
it is not. A seeded simulation checks the two error rates the size implies: a judge whose rates
did not move is flagged rarely, and one that fell to the threshold about half the time.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from calibrated_agent_evaluation._provenance import capture_provenance
from calibrated_agent_evaluation.calibrate import (
    Bound,
    CalibrationRecord,
    InfeasiblePlanError,
    Interval,
)
from calibrated_agent_evaluation.run import JudgeDriftReport, RunStoreError
from calibrated_agent_evaluation.run.judge_check import (
    JudgeCheckRecord,
    check_size,
    drift_flagged,
    dump_judge_checks,
    load_judge_checks,
)
from calibrated_agent_evaluation.stats.drift import delta_proportion_ci


def _calib(tp: int, n_pos: int, tn: int, n_neg: int) -> CalibrationRecord:
    return CalibrationRecord(
        calib_id="calib-x",
        judge_id="judge-x",
        model="m",
        prompt_version="v1",
        hunted_set_id="gold-abc-1",
        n_pos=n_pos,
        n_neg=n_neg,
        tpr=Bound(point=tp / n_pos, lb95=0.81, n=n_pos),
        tnr=Bound(point=tn / n_neg, lb95=0.81, n=n_neg),
        youden_j=tp / n_pos + tn / n_neg - 1,
        result="CERTIFIED",
        provenance=capture_provenance(),
    )


def _delta(point: float, lo: float, hi: float) -> Interval:
    return Interval(
        point=point, lo=lo, hi=hi, method="newcomb_diff_2indep", level=0.95, sided="two-sided", n=2
    )


_STAMP = capture_provenance()


def _record() -> JudgeCheckRecord:
    return JudgeCheckRecord(
        check_id="check-judge-x-1",
        judge_id="judge-x",
        model="m",
        prompt_version="v1",
        judge_fingerprint="fp",
        against_calib_id="calib-x",
        check_set_id="check-abc-1",
        n_pos=30,
        n_neg=30,
        tpr=Bound(point=0.9, lb95=0.78, n=30),
        tnr=Bound(point=0.9, lb95=0.78, n=30),
        drift=JudgeDriftReport(
            judge_id="judge-x",
            delta_tpr=_delta(-0.05, -0.2, 0.1),
            delta_tnr=_delta(-0.05, -0.2, 0.1),
        ),
        drifted=False,
        needed_fail=26,
        needed_pass=26,
        trace_ids=("t1", "t2"),
        provenance=_STAMP,
    )


def test_the_record_round_trips_through_its_store(tmp_path: Path):
    record = _record()
    dump_judge_checks([record], tmp_path, fmt="json")
    assert load_judge_checks(tmp_path, fmt="json") == [record]
    assert load_judge_checks(tmp_path, fmt="csv") == [record]


def test_a_record_of_another_schema_version_is_refused(tmp_path: Path):
    dump_judge_checks([_record()], tmp_path, fmt="json")
    path = tmp_path / "judge_check_record.json"
    docs = json.loads(path.read_text())
    docs[0]["schema_version"] = "0"
    path.write_text(json.dumps(docs))
    (tmp_path / "judge_check_record.csv").unlink()
    with pytest.raises(RunStoreError, match="calibrated-agent-evaluation calibrate --check"):
        load_judge_checks(tmp_path, fmt="json")


def test_drift_is_flagged_only_when_an_interval_lies_below_zero():
    report = _record().drift
    assert not drift_flagged(report)
    lower = report.model_copy(update={"delta_tnr": _delta(-0.2, -0.3, -0.01)})
    assert drift_flagged(lower)


def _flags_at(right_before: int, n_before: int, n: int, threshold: float = 0.80) -> bool:
    right = math.ceil(threshold * n - 1e-9)
    return delta_proportion_ci(right, n, right_before, n_before)[2] < 0.0


@pytest.mark.parametrize(("tp", "n"), [(52, 55), (72, 80), (150, 162), (50, 55)])
def test_the_size_is_the_smallest_from_which_a_drop_to_the_gate_is_flagged(tp: int, n: int):
    need, _ = check_size(_calib(tp, n, tp, n), max_n=3000)
    assert _flags_at(tp, n, need)
    assert not _flags_at(tp, n, need - 1)
    assert all(_flags_at(tp, n, m) for m in range(need, 3001))


def test_the_size_derives_from_the_record_not_a_constant():
    small = check_size(_calib(52, 55, 52, 55))
    large = check_size(_calib(150, 162, 150, 162))
    assert small != large
    assert check_size(_calib(52, 55, 75, 80))[0] == small[0]


def test_a_record_too_close_to_the_gate_is_refused_as_infeasible():
    with pytest.raises(InfeasiblePlanError, match="Certify again on more gold"):
        check_size(_calib(48, 55, 52, 55), max_n=2000)


def test_the_planned_size_rarely_flags_a_stable_judge_and_often_flags_a_drop():
    """Seed 7, 2000 replicates per arm, at the size planned for a 72/80 certification."""
    tp, n_before = 72, 80
    need, _ = check_size(_calib(tp, n_before, tp, n_before))
    rng = random.Random(7)
    reps = 2000

    def rate(p: float) -> float:
        hits = 0
        for _ in range(reps):
            before = sum(rng.random() < tp / n_before for _ in range(n_before))
            after = sum(rng.random() < p for _ in range(need))
            hits += delta_proportion_ci(after, need, before, n_before)[2] < 0.0
        return hits / reps

    stable, dropped, far = rate(tp / n_before), rate(0.80), rate(0.70)
    assert stable < 0.05
    assert 0.25 < dropped < 0.75
    assert far > 0.9
