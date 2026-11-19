# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ImproveRecord round-trip codec.

Behavioural: an ImproveRecord (with a pre-registration, a paired result, a guard panel, and a crew
relocation) round-trips field-for-field through both CSV and JSON; a schema-version mismatch and a
structurally-malformed file surface the typed :class:`ImproveStoreError` (never a raw
JSONDecodeError / KeyError); and constructing a record with empty inputs is refused.
"""

from __future__ import annotations

import json

import pytest

from _helpers import TEST_INTERVAL_KIND, TEST_STAMP
from calibrated_agent_evaluation.attribution import RelocationResult
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.improve import (
    GuardPanelReport,
    GuardSlice,
    ImproveError,
    ImproveRecord,
    ImproveStoreError,
    McNemarResult,
    PreRegEntry,
    dump_improve_records,
    load_improve_records,
)
from calibrated_agent_evaluation.improve.report import IMPROVE_SCHEMA_VERSION


def _iv(point: float | None, lo: float, hi: float, reason: str | None = None) -> Interval:
    return Interval(point=point, lo=lo, hi=hi, refusal_reason=reason, **TEST_INTERVAL_KIND)


def _record() -> ImproveRecord:
    """A representative crew record exercising every nested type the codec reconstructs."""
    reloc = RelocationResult(
        workflow_outcome_rate_before=0.30,
        workflow_outcome_rate_after=0.30,
        per_agent_failures_before={"intake": 6, "policy-checker": 3},
        per_agent_failures_after={"intake": 2, "policy-checker": 3},
        verdict="relocation",
        relocated_from="intake",
    )
    panel = GuardPanelReport(
        slices=(
            GuardSlice(
                name="policy-checker",
                kind="agent",
                regression_p=0.4,
                rate_before=_iv(0.1, 0.0, 0.2),
                rate_after=_iv(0.11, 0.0, 0.22),
            ),
            GuardSlice(
                name="intake->policy-checker",
                kind="handoff",
                regression_p=0.7,
                rate_before=_iv(0.05, 0.0, 0.1),
                rate_after=_iv(0.05, 0.0, 0.1),
            ),
        ),
        method="holm",
        m=2,
        regressed=(),
        relocation=reloc,
    )
    paired = McNemarResult(
        method="midp", n_discordant=15, p_value=0.031, effect=_iv(-0.25, -0.4, -0.1)
    )
    return ImproveRecord(
        provenance=TEST_STAMP,
        improve_id="imp-1",
        verdict="rejected",
        prereg=PreRegEntry(
            prereg_id="p1",
            target_metric="workflow_outcome_rate",
            success_threshold=0.10,
            is_crew=True,
            content_hash="deadbeef",
        ),
        paired=paired,
        panel=panel,
        confirmation_slice_id="conf-v1",
        development_set_id="dev-v1",
        inputs={"confirmation_slice_id": "conf-v1", "touched_agent": "intake"},
        reasons=("failure relocation: relocated_from='intake'",),
    )


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_improve_record_round_trips(tmp_path, fmt) -> None:
    """An ImproveRecord dumps -> loads field-for-field equal in both CSV and JSON."""
    original = _record()
    dump_improve_records([original], tmp_path, fmt=fmt)
    loaded = load_improve_records(tmp_path, fmt=fmt)
    assert loaded == [original]


def test_schema_version_mismatch_refuses(tmp_path) -> None:
    """A stored schema_version != the running one raises ImproveStoreError (no coercion)."""
    dump_improve_records([_record()], tmp_path, fmt="json")
    file = tmp_path / "improve_record.json"
    docs = json.loads(file.read_text())
    docs[0]["schema_version"] = "999"
    file.write_text(json.dumps(docs))
    with pytest.raises(ImproveStoreError):
        load_improve_records(tmp_path, fmt="json")


def test_malformed_record_raises_store_error_not_raw(tmp_path) -> None:
    """A structurally-malformed record (missing key) raises ImproveStoreError, not raw."""
    dump_improve_records([_record()], tmp_path, fmt="json")
    file = tmp_path / "improve_record.json"
    docs = json.loads(file.read_text())
    del docs[0]["paired"]  # drop a required nested field
    file.write_text(json.dumps(docs))
    with pytest.raises(ImproveStoreError):
        load_improve_records(tmp_path, fmt="json")


def test_corrupt_json_raises_store_error(tmp_path) -> None:
    """An invalid-JSON file raises ImproveStoreError, not a raw JSONDecodeError."""
    (tmp_path / "improve_record.json").write_text("{not valid json")
    with pytest.raises(ImproveStoreError):
        load_improve_records(tmp_path, fmt="json")


@pytest.mark.parametrize("fmt", ["json", "csv"])
def test_non_utf8_file_raises_store_error_not_raw(tmp_path, fmt) -> None:
    """A non-UTF-8 file raises ImproveStoreError, not a raw UnicodeDecodeError.

    A UnicodeDecodeError is a ValueError (not an OSError), so it would slip past a plain OSError
    catch and escape the surface -- a no-silent-failure violation. The boundary must wrap it.
    """
    (tmp_path / f"improve_record.{fmt}").write_bytes(b"\xff\xfe not utf-8 text")
    with pytest.raises(ImproveStoreError):
        load_improve_records(tmp_path, fmt=fmt)


def test_malformed_csv_raises_store_error_not_raw(tmp_path) -> None:
    """A CSV the parser rejects raises ImproveStoreError, not a raw csv.Error."""
    import csv as _csv

    # Calibrated Agent Evaluation lifts the field-size cap at import, so lower it here to make the
    # parser refuse.
    previous = _csv.field_size_limit(16)
    try:
        (tmp_path / "improve_record.csv").write_text("improve_id\n" + "x" * 17 + "\n")
        with pytest.raises(ImproveStoreError):
            load_improve_records(tmp_path, fmt="csv")
    finally:
        _csv.field_size_limit(previous)


def test_a_csv_field_over_the_default_csv_limit_loads(tmp_path) -> None:
    """A record holding one large value (a fetched page in a critique) is not refused as corrupt."""
    rec = _record()
    big = rec.model_copy(update={"improve_id": "x" * 200_000})
    dump_improve_records([big], tmp_path, fmt="csv")
    assert load_improve_records(tmp_path, fmt="csv") == [big]


def test_empty_inputs_refused_at_construction() -> None:
    """An ImproveRecord with empty inputs is refused (a record with no inputs is
    not reproducible)."""
    rec = _record()
    with pytest.raises(ImproveError):
        ImproveRecord(
            provenance=TEST_STAMP,
            improve_id=rec.improve_id,
            verdict=rec.verdict,
            prereg=rec.prereg,
            paired=rec.paired,
            panel=rec.panel,
            confirmation_slice_id=rec.confirmation_slice_id,
            development_set_id=rec.development_set_id,
            inputs={},
        )


def test_empty_dump_is_loadable(tmp_path) -> None:
    """Dumping zero records writes header-only files that load back as an empty list."""
    dump_improve_records([], tmp_path, fmt="csv")
    assert load_improve_records(tmp_path, fmt="csv") == []
    assert IMPROVE_SCHEMA_VERSION == "6"


def test_the_prereg_ledger_round_trips_and_still_verifies(tmp_path) -> None:
    from calibrated_agent_evaluation.improve import (
        PreRegEntry,
        PreRegLedger,
        dump_prereg_ledger,
        load_prereg_ledger,
        register_prereg,
        verify_ledger,
    )

    ledger = register_prereg(
        PreRegLedger(),
        PreRegEntry(prereg_id="p1", target_metric="over_refund", success_threshold=0.05),
    )
    ledger = register_prereg(
        ledger,
        PreRegEntry(
            prereg_id="p2", target_metric="dropped_fact", success_threshold=0.10, code_ref="abc"
        ),
    )
    dump_prereg_ledger(ledger, tmp_path)
    loaded = load_prereg_ledger(tmp_path)
    assert loaded == ledger
    verify_ledger(loaded)


def test_an_absent_ledger_is_empty_and_a_malformed_one_is_a_store_error(tmp_path) -> None:
    from calibrated_agent_evaluation.improve import load_prereg_ledger

    assert load_prereg_ledger(tmp_path).entries == ()
    (tmp_path / "prereg_ledger.json").write_text('[{"prereg_id": "p1"}]')
    with pytest.raises(ImproveStoreError, match="malformed"):
        load_prereg_ledger(tmp_path)
