# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the self-contained CalibrationRecord round-trip.

ships its own CSV+JSON round-trip rather than routing through the data plane's
trace-store dump. These tests prove field-for-field round-trip in both formats and the
refuse-on-version-mismatch / wrap-OSError disciplines.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _helpers import TEST_INTERVAL_KIND, TEST_STAMP
from calibrated_agent_evaluation.calibrate.errors import (
    CalibrationStoreError,
    CalibrationVersionError,
)
from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord, Interval
from calibrated_agent_evaluation.calibrate.store import dump_records, load_records


def _record() -> CalibrationRecord:
    """A fully-populated record (incl. Bound/Interval/trace_ids) to exercise every field."""
    return CalibrationRecord(
        provenance=TEST_STAMP,
        calib_id="calib-intake-v3",
        judge_id="intake-extraction",
        model="bedrock/claude-sonnet-4",
        prompt_version="sha-abc123",
        hunted_set_id="hunted-1",
        n_pos=80,
        n_neg=80,
        tpr=Bound(point=0.90, lb95=0.8311, n=80),
        tnr=Bound(point=0.90, lb95=0.8149, n=80),
        youden_j=0.80,
        result="CERTIFIED",
        rate_set_id="prod-1",
        corrected_rate=Interval(point=0.06, lo=0.03, hi=0.10, **TEST_INTERVAL_KIND),
        locked_test_consumed=True,
        trace_ids=("p1", "p2", "n1"),
    )


# --- round-trip equality (the proof for this artifact) ------------


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_calibration_record_round_trips_field_for_field(tmp_path: Path, fmt: str) -> None:
    original = [_record()]
    dump_records(original, tmp_path, fmt=fmt)
    reloaded = load_records(tmp_path, fmt=fmt)
    assert reloaded == original  # frozen models compare field-for-field, incl. Bound/Interval


def test_round_trip_preserves_a_refused_interval(tmp_path: Path) -> None:
    # An Interval whose point was refused (J<0.5) must survive with point=None + refusal_reason.
    rec = _record()
    refused = CalibrationRecord(
        **{
            **{k: getattr(rec, k) for k in type(rec).model_fields},
            "corrected_rate": Interval(
                point=None,
                lo=0.0,
                hi=0.4,
                refusal_reason="youden_j=0.440<0.5",
                **TEST_INTERVAL_KIND,
            ),
        },
    )
    dump_records([refused], tmp_path, fmt="json")
    reloaded = load_records(tmp_path, fmt="json")
    assert reloaded == [refused]
    assert reloaded[0].corrected_rate is not None
    assert reloaded[0].corrected_rate.point is None
    assert reloaded[0].corrected_rate.refusal_reason == "youden_j=0.440<0.5"


def test_empty_store_round_trips(tmp_path: Path) -> None:
    # Edge: dumping zero records writes loadable (header-only) files.
    dump_records([], tmp_path, fmt="csv")
    assert load_records(tmp_path, fmt="csv") == []


# --- refuse-on-mismatch + wrap-OSError -----------------------------------


def test_load_refuses_on_schema_version_mismatch(tmp_path: Path) -> None:
    # Negative path: a dump whose schema_version is bumped to an unknown value makes load refuse
    # (no coercion) — the precedent.
    bumped = CalibrationRecord(
        **{
            **{k: getattr(_record(), k) for k in type(_record()).model_fields},
            "schema_version": "999",
        },
    )
    dump_records([bumped], tmp_path, fmt="json")
    with pytest.raises(CalibrationVersionError):
        load_records(tmp_path, fmt="json")


def test_load_refuses_a_schema_3_record_carrying_m_comparisons(tmp_path: Path) -> None:
    """A schema "3" record, which carried m_comparisons and gate_family, is
    refused rather than read with those fields dropped."""
    import json

    dump_records([_record()], tmp_path, fmt="json")
    path = tmp_path / "calibration_record.json"
    docs = json.loads(path.read_text())
    for doc in docs:
        doc.update(schema_version="3", m_comparisons=2, gate_family="holm")
    path.write_text(json.dumps(docs))
    with pytest.raises(CalibrationVersionError, match="'3'"):
        load_records(tmp_path, fmt="json")


def test_load_of_missing_dir_returns_empty(tmp_path: Path) -> None:
    # A dir with no dump files is an empty workspace, not an error.
    assert load_records(tmp_path / "nonexistent", fmt="csv") == []


def test_unwritable_path_raises_typed_store_error(tmp_path: Path) -> None:
    # Negative path: a filesystem failure is wrapped into CalibrationStoreError, not a raw OSError.
    # Point the dump at a path whose parent is a FILE (so mkdir fails).
    blocker = tmp_path / "iamafile"
    blocker.write_text("x")
    with pytest.raises(CalibrationStoreError):
        dump_records([_record()], blocker / "sub", fmt="csv")


def test_bad_fmt_raises_typed_error(tmp_path: Path) -> None:
    with pytest.raises(CalibrationStoreError):
        dump_records([_record()], tmp_path, fmt="xml")


def test_malformed_but_valid_json_raises_typed_error(tmp_path: Path) -> None:
    # Negative path: a hand-edited dump that is valid JSON but missing a required field must surface
    # as a typed CalibrationStoreError, never a raw KeyError escaping the public load surface
    # (conventions: no silent failure / wrap external errors at the boundary).
    import json

    partial = [{"calib_id": "c", "judge_id": "j"}]  # missing tpr/tnr/... — structurally malformed
    (tmp_path / "calibration_record.json").write_text(json.dumps(partial))
    with pytest.raises(CalibrationStoreError):
        load_records(tmp_path, fmt="json")


def test_a_non_numeric_interval_cell_raises_typed_error(tmp_path: Path) -> None:
    # The interval decoder converts with float(); a hand-edited "point": "high" is a ValueError,
    # which must surface as CalibrationStoreError like a missing key does.
    import json

    dump_records([_record()], tmp_path, fmt="json")
    path = tmp_path / "calibration_record.json"
    docs = json.loads(path.read_text())
    docs[0]["corrected_rate"]["point"] = "high"
    path.write_text(json.dumps(docs))
    with pytest.raises(CalibrationStoreError, match="malformed"):
        load_records(tmp_path, fmt="json")
