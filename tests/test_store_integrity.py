# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record stores survive crashes, concurrent writers and hand-damaged files."""

from __future__ import annotations

import csv
import json
import multiprocessing
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.calibrate import CalibrationError, dump_records, load_records
from calibrated_agent_evaluation.calibrate.errors import (
    CalibrationStoreError,
    CalibrationVersionError,
)
from calibrated_agent_evaluation.calibrate.record import Bound, Interval
from calibrated_agent_evaluation.cli.errors import GoldSetReusedError
from calibrated_agent_evaluation.cli.records import append_records, load_records_of, records_dir
from cli_store_helpers import certified_calibration

# --- atomic writes --------------------------------------------------------------------------------


def test_a_crash_mid_write_leaves_the_previous_records_whole(tmp_path, monkeypatch):
    first = certified_calibration(calib_id="calib-1")
    dump_records([first], tmp_path, fmt="json")
    before = (tmp_path / "calibration_record.json").read_bytes()

    def crash(fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("calibrated_agent_evaluation._recordstore.os.fsync", crash)
    second = replace(certified_calibration(calib_id="calib-2"), hunted_set_id="other-gold")
    with pytest.raises(CalibrationStoreError, match="disk full"):
        dump_records([first, second], tmp_path, fmt="json")
    assert (tmp_path / "calibration_record.json").read_bytes() == before
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []
    assert [r.calib_id for r in load_records(tmp_path, fmt="json")] == ["calib-1"]


def test_atomic_writes_keep_the_default_file_mode(tmp_path):
    dump_records([certified_calibration()], tmp_path)
    plain = tmp_path / "plain.txt"
    plain.write_text("x")
    for name in ("calibration_record.json", "calibration_record.csv"):
        assert (tmp_path / name).stat().st_mode == plain.stat().st_mode


# --- concurrent appends ---------------------------------------------------------------------------


def _append_many(root: str, worker: int, count: int) -> None:
    for i in range(count):
        record = replace(
            certified_calibration(calib_id=f"calib-{worker}-{i}"),
            hunted_set_id=f"gold-{worker}-{i}",
        )
        append_records(Path(root), "calibration", [record])


def test_two_processes_appending_at_once_lose_no_record(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    per_worker = 15
    workers = [
        ctx.Process(target=_append_many, args=(str(tmp_path), w, per_worker)) for w in range(2)
    ]
    for proc in workers:
        proc.start()
    for proc in workers:
        proc.join(timeout=120)
        assert proc.exitcode == 0
    stored = {r.calib_id for r in load_records_of(tmp_path, "calibration")}
    assert len(stored) == 2 * per_worker


def test_a_second_record_on_a_consumed_gold_set_is_refused(tmp_path):
    append_records(tmp_path, "calibration", [certified_calibration(calib_id="calib-a")])
    again = certified_calibration(judge_id="copied-judge", calib_id="calib-b")
    with pytest.raises(GoldSetReusedError, match="single-use"):
        append_records(tmp_path, "calibration", [again])
    assert [r.calib_id for r in load_records_of(tmp_path, "calibration")] == ["calib-a"]


def test_re_appending_the_same_record_is_still_a_no_op(tmp_path):
    record = certified_calibration()
    assert append_records(tmp_path, "calibration", [record]) == 1
    assert append_records(tmp_path, "calibration", [record]) == 0


# --- damaged and old files ------------------------------------------------------------------------


def _rewrite_row(tmp_path: Path, change: Callable[[list[str]], list[str]]) -> None:
    dump_records([certified_calibration()], tmp_path, fmt="csv")
    file = tmp_path / "calibration_record.csv"
    with file.open(newline="") as fh:
        header, row = list(csv.reader(fh))
    with file.open("w", newline="") as fh:
        csv.writer(fh).writerows([header, change(row)])


def test_a_csv_row_with_missing_cells_is_a_typed_refusal(tmp_path):
    _rewrite_row(tmp_path, lambda row: row[:-3])
    with pytest.raises(CalibrationStoreError, match="different number of cells"):
        load_records(tmp_path, fmt="csv")


def test_a_csv_row_with_extra_cells_is_a_typed_refusal(tmp_path):
    _rewrite_row(tmp_path, lambda row: [*row, "1", "2"])
    with pytest.raises(CalibrationStoreError, match="different number of cells"):
        load_records(tmp_path, fmt="csv")


def _stored_doc() -> dict:
    return certified_calibration().model_dump(mode="json")


def test_a_record_without_a_schema_version_is_refused(tmp_path):
    doc = _stored_doc()
    del doc["schema_version"]
    (tmp_path / "calibration_record.json").write_text(json.dumps([doc]))
    with pytest.raises(CalibrationStoreError, match="no schema_version"):
        load_records(tmp_path, fmt="json")


def test_a_record_from_another_version_names_both_versions(tmp_path):
    doc = _stored_doc()
    doc["schema_version"] = "99"
    (tmp_path / "calibration_record.json").write_text(json.dumps([doc]))
    with pytest.raises(CalibrationVersionError, match="schema_version '99'"):
        load_records(tmp_path, fmt="json")


def test_the_cli_reads_its_record_directory_through_the_same_checks(tmp_path):
    target = records_dir(tmp_path, "calibration")
    target.mkdir(parents=True)
    doc = _stored_doc()
    doc["schema_version"] = "6"
    (target / "calibration_record.json").write_text(json.dumps([doc]))
    with pytest.raises(CalibrationVersionError):
        load_records_of(tmp_path, "calibration")


# --- gate numbers are numbers ---------------------------------------------------------------------


@pytest.mark.parametrize("bad", [True, "0.9", b"0.9"])
def test_a_bound_refuses_a_bool_or_text_gate_value(bad):
    with pytest.raises(CalibrationError, match="must be a number"):
        Bound(point=0.9, lb95=bad, n=80)


def test_a_bound_refuses_a_bool_class_size():
    with pytest.raises(CalibrationError, match="must be a number"):
        Bound(point=0.9, lb95=0.8, n=True)


def test_an_interval_refuses_a_bool_edge():
    with pytest.raises(CalibrationError, match="must be a number"):
        Interval(point=None, lo=False, hi=0.3, method="m", level=0.95, sided="two-sided", n=3)


def test_numpy_numbers_still_build_gate_values():
    # The estimators hand numpy scalars to these models; they must keep building.
    bound = Bound.model_validate(
        {"point": np.float64(0.9), "lb95": np.float32(0.8), "n": np.int64(80)}
    )
    assert (bound.point, bound.n) == (0.9, 80)


def test_a_stored_bool_gate_value_is_refused_on_load(tmp_path):
    doc = _stored_doc()
    doc["tpr"]["lb95"] = True
    (tmp_path / "calibration_record.json").write_text(json.dumps([doc]))
    with pytest.raises(CalibrationStoreError):
        load_records(tmp_path, fmt="json")


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    monkeypatch.setenv("AWS_PROFILE", "calibrated-agent-evaluation-test-no-such-profile")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
