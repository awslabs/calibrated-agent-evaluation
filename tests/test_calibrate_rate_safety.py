# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A certification on gold below the coverage floor is marked rate-unsafe.

The Wilson bound is valid at any n, so the judge stays certified. But the corrected rate's interval
was measured to cover nominally only from the coverage floor per class up
, so the record says the gold is below it and ``corrected_rate``
refuses an interval built from it, naming the planned and actual sizes.

The floor rises with accuracy, so the record keeps two: the floor at the planned accuracy and the
floor at the TPR and TNR the judge showed on the locked test. The larger one applies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from _helpers import make_label, rate_ids
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.calibrate import (
    CalibrationPlan,
    InMemoryLockedTestLedger,
    plan_calibration,
    plan_calibration_size,
)
from calibrated_agent_evaluation.calibrate.engine import (
    GATE_THRESHOLD,
    certifies,
    certify,
    corrected_rate,
)
from calibrated_agent_evaluation.calibrate.errors import CalibrationVersionError, RateUnsafeError
from calibrated_agent_evaluation.calibrate.record import (
    CALIB_SCHEMA_VERSION,
    CalibrationSet,
    LabelPair,
)
from calibrated_agent_evaluation.calibrate.store import dump_records, load_records


@dataclass(frozen=True)
class _Judge:
    judge_id: str = "intake-extraction"
    model: str = "bedrock/claude-sonnet-4"
    prompt_version: str = "sha1"


def _locked(
    n_per_class: int,
    set_id: str = "locked",
    *,
    errors: int = 0,
    pass_errors: int | None = None,
) -> CalibrationSet:
    """A locked test of ``n_per_class`` items per class.

    The judge gets ``errors`` FAIL items wrong and ``pass_errors`` PASS items wrong (``errors`` if
    omitted).
    """
    wrong = {"FAIL": errors, "PASS": errors if pass_errors is None else pass_errors}
    flip = {"FAIL": "PASS", "PASS": "FAIL"}
    pairs = []
    for cls in ("FAIL", "PASS"):
        for i in range(n_per_class):
            tid = f"{cls}-{i}"
            gold = make_label(trace_id=tid, kind="human", outcome=cls)
            said = flip[cls] if i < wrong[cls] else cls
            verdict = make_label(trace_id=tid, kind="llm-judge", outcome=said)
            pairs.append(LabelPair(gold=gold, verdict=verdict))
    return CalibrationSet(pairs=tuple(pairs), split="locked_test", set_id=set_id)


_RATE = [1] * 20 + [0] * 180
_PLAN = plan_calibration(0.90, GATE_THRESHOLD)


def test_the_plan_names_both_sizes_and_the_larger_is_hunted():
    assert plan_calibration(0.90, GATE_THRESHOLD) == CalibrationPlan(55, 55, 35)
    high = plan_calibration(0.95, GATE_THRESHOLD)
    assert (high.per_class, high.certification, high.coverage_floor) == (80, 28, 80)
    assert plan_calibration_size(0.95, GATE_THRESHOLD) == 80
    asym = plan_calibration(0.95, GATE_THRESHOLD, tnr_hat=0.85)
    assert (asym.per_class, asym.certification, asym.coverage_floor) == (208, 208, 80)


def test_gold_below_the_floor_still_certifies_and_is_recorded_as_rate_unsafe():
    # 19 of 20 right per class: the Wilson bound (0.804) certifies.
    record = certify(_Judge(), _locked(20, errors=1), plan=_PLAN, ledger=InMemoryLockedTestLedger())
    assert record.result == "CERTIFIED"
    assert certifies(record, _Judge())
    assert (record.planned_per_class, record.planned_coverage_floor) == (55, 35)
    assert record.observed_coverage_floor == 80
    assert record.coverage_floor == 80
    assert record.below_coverage_floor


def test_a_corrected_rate_from_a_rate_unsafe_record_is_refused_naming_the_sizes():
    record = certify(_Judge(), _locked(20, errors=1), plan=_PLAN, ledger=InMemoryLockedTestLedger())
    with pytest.raises(RateUnsafeError) as exc:
        corrected_rate(record, _RATE, rate_set_id="prod", rate_trace_ids=rate_ids(_RATE))
    message = str(exc.value)
    assert "20 FAIL and 20 PASS" in message
    assert "coverage floor of 80 per class" in message
    assert "35 at the planned accuracy" in message
    assert "80 at the observed TPR 0.950 and TNR 0.950" in message
    assert "planned 55 per class" in message
    assert "intake-extraction" in message


# --- the floor at the observed accuracy ---------------------------------------


def test_a_judge_more_accurate_than_planned_is_marked_rate_unsafe():
    """At the planned floor, but 33 of 35 per class right: the observed floor is 80."""
    record = certify(_Judge(), _locked(35, errors=2), plan=_PLAN, ledger=InMemoryLockedTestLedger())
    assert record.result == "CERTIFIED"
    assert (record.n_pos, record.n_neg) == (35, 35)
    assert record.planned_coverage_floor == 35
    assert record.observed_coverage_floor == 80
    assert record.coverage_floor == 80
    assert record.below_coverage_floor
    with pytest.raises(RateUnsafeError) as exc:
        corrected_rate(record, _RATE, rate_set_id="prod", rate_trace_ids=rate_ids(_RATE))
    message = str(exc.value)
    assert "35 FAIL and 35 PASS" in message
    assert "coverage floor of 80 per class" in message
    assert "35 at the planned accuracy" in message
    assert "80 at the observed TPR 0.943 and TNR 0.943" in message
    assert "Label up to 80 per class" in message


def test_the_observed_floor_is_read_at_the_higher_of_the_two_rates():
    """33 of 35 FAIL right (0.943) and 34 of 35 PASS right (0.971): the floor at 0.971 is 130."""
    record = certify(
        _Judge(),
        _locked(35, errors=2, pass_errors=1),
        plan=_PLAN,
        ledger=InMemoryLockedTestLedger(),
    )
    assert (round(record.tpr.point, 3), round(record.tnr.point, 3)) == (0.943, 0.971)
    assert record.observed_coverage_floor == 130
    assert record.coverage_floor == 130


def test_gold_at_both_floors_gives_a_corrected_rate():
    """76 of 80 per class right (0.95): the observed floor is 80, the planned 35."""
    record = certify(_Judge(), _locked(80, errors=4), plan=_PLAN, ledger=InMemoryLockedTestLedger())
    assert (record.planned_coverage_floor, record.observed_coverage_floor) == (35, 80)
    assert not record.below_coverage_floor
    scored = corrected_rate(
        record, _RATE, rate_set_id="prod", n_resamples=199, rate_trace_ids=rate_ids(_RATE)
    )
    assert scored.corrected_rate is not None


def test_a_perfect_locked_test_reads_the_boundary_floor_and_is_rate_safe():
    """With the boundary rule the interval covers at least nominally at 1.0 from 11 per class."""
    record = certify(_Judge(), _locked(35), plan=_PLAN, ledger=InMemoryLockedTestLedger())
    assert record.result == "CERTIFIED"
    assert (record.observed_coverage_floor, record.observed_floor_unmeasured) == (11, False)
    assert record.coverage_floor == 35
    assert not record.below_coverage_floor
    assert corrected_rate(
        record, _RATE, rate_set_id="prod", n_resamples=199, rate_trace_ids=rate_ids(_RATE)
    ).corrected_rate


def test_an_accuracy_without_a_measured_floor_is_rate_unsafe(monkeypatch):
    """An accuracy whose floor table entry is None is not known to be covered at any size."""
    from calibrated_agent_evaluation.calibrate import plan as plan_module

    table = dict(plan_module.COVERAGE_FLOOR) | {1.00: None}
    monkeypatch.setattr(plan_module, "COVERAGE_FLOOR", tuple(table.items()))
    record = certify(_Judge(), _locked(35), plan=_PLAN, ledger=InMemoryLockedTestLedger())
    assert record.result == "CERTIFIED"
    assert record.observed_coverage_floor is None
    assert record.observed_floor_unmeasured
    assert record.coverage_floor == 35
    assert record.below_coverage_floor
    with pytest.raises(RateUnsafeError) as exc:
        corrected_rate(record, _RATE, rate_set_id="prod", rate_trace_ids=rate_ids(_RATE))
    message = str(exc.value)
    assert "at the observed TPR 1.000 and TNR 1.000 the corrected rate's interval was measured" in (
        message
    )
    assert "did not cover nominally at any scanned gold size" in message
    assert "35 FAIL and 35 PASS" in message


def test_gold_hunted_to_a_rate_safe_plan_gives_a_corrected_rate_below_0_99():
    """At 162 per class every certifying accuracy up to 0.99 meets its floor."""
    plan = plan_calibration(0.90, GATE_THRESHOLD, rate_safe=True)
    assert plan == CalibrationPlan(162, 55, 35)
    # 160 of 162 right (0.988) is the hardest measured case: its floor is 162.
    record = certify(_Judge(), _locked(162, errors=2), plan=plan, ledger=InMemoryLockedTestLedger())
    assert (record.planned_per_class, record.observed_coverage_floor) == (162, 162)
    assert not record.below_coverage_floor
    assert corrected_rate(
        record, _RATE, rate_set_id="prod", n_resamples=199, rate_trace_ids=rate_ids(_RATE)
    ).corrected_rate


def test_a_rate_safe_plan_covers_accuracies_between_0_99_and_0_995():
    """161 of 162 right (0.994) sits between 0.99 and 0.995 and takes the larger floor, 162."""
    plan = plan_calibration(0.90, GATE_THRESHOLD, rate_safe=True)
    record = certify(
        _Judge(), _locked(162, "locked-161", errors=1), plan=plan, ledger=InMemoryLockedTestLedger()
    )
    assert (record.observed_coverage_floor, record.observed_floor_unmeasured) == (162, False)
    assert not record.below_coverage_floor


def test_a_refused_judge_carries_no_observed_floor():
    record = certify(_Judge(), _locked(20, errors=5), plan=_PLAN, ledger=InMemoryLockedTestLedger())
    assert record.result == "REFUSED"
    assert record.planned_coverage_floor == 35
    assert (record.observed_coverage_floor, record.observed_floor_unmeasured) == (None, False)


def test_one_class_below_the_floor_is_enough_to_refuse():
    record = certify(_Judge(), _locked(80, errors=4), plan=_PLAN, ledger=InMemoryLockedTestLedger())
    short = replace(record, n_neg=79)
    assert short.below_coverage_floor
    with pytest.raises(RateUnsafeError, match="80 FAIL and 79 PASS"):
        corrected_rate(short, _RATE, rate_set_id="prod", rate_trace_ids=rate_ids(_RATE))


def test_a_record_certified_without_a_plan_still_carries_its_observed_floor():
    record = certify(_Judge(), _locked(20), ledger=InMemoryLockedTestLedger())
    assert (record.planned_per_class, record.planned_coverage_floor) == (None, None)
    assert (record.observed_coverage_floor, record.observed_floor_unmeasured) == (11, False)
    assert not record.below_coverage_floor
    assert corrected_rate(
        record, _RATE, rate_set_id="prod", n_resamples=199, rate_trace_ids=rate_ids(_RATE)
    ).corrected_rate


def test_a_library_caller_without_a_plan_gets_no_rate_from_gold_below_the_floor():
    """38 of 40 correct per class certifies, but at 0.95 the interval needs 80 per class."""
    record = certify(_Judge(), _locked(40, errors=2), ledger=InMemoryLockedTestLedger())
    assert record.result == "CERTIFIED"
    assert record.planned_coverage_floor is None
    assert record.observed_coverage_floor == 80
    assert record.below_coverage_floor
    with pytest.raises(RateUnsafeError):
        corrected_rate(
            record, _RATE, rate_set_id="prod", n_resamples=199, rate_trace_ids=rate_ids(_RATE)
        )


def test_the_floors_round_trip_and_a_schema_5_record_is_refused(tmp_path: Path):
    record = certify(_Judge(), _locked(35, errors=2), plan=_PLAN, ledger=InMemoryLockedTestLedger())
    assert CALIB_SCHEMA_VERSION == "9"
    dump_records([record], tmp_path, fmt="json")
    assert load_records(tmp_path, fmt="json") == [record]

    path = tmp_path / "calibration_record.json"
    docs = json.loads(path.read_text())
    for doc in docs:
        doc["schema_version"] = "5"
        doc["coverage_floor"] = doc.pop("planned_coverage_floor")
        del doc["observed_coverage_floor"], doc["observed_floor_unmeasured"]
    path.write_text(json.dumps(docs))
    with pytest.raises(CalibrationVersionError, match="'5'"):
        load_records(tmp_path, fmt="json")
