# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agreement and codebook records round-trip through their CSV+JSON store, and refuse damage."""

from __future__ import annotations

import json

import pytest

from _helpers import TEST_STAMP
from calibrated_agent_evaluation.review import (
    AgreementRecord,
    CodebookMode,
    CodebookRecord,
    agreement_record_of,
    inter_rater_agreement,
)
from calibrated_agent_evaluation.review.errors import ReviewError
from calibrated_agent_evaluation.review.record import (
    ProposedModeRecord,
    mode_key,
    pending_proposals,
    with_content_id,
)
from calibrated_agent_evaluation.review.store import (
    dump_agreement_records,
    dump_codebook_records,
    dump_proposed_modes,
    load_agreement_records,
    load_codebook_records,
    load_proposed_modes,
)
from calibrated_agent_evaluation.schema import Label


def _labels() -> list[Label]:
    a = ["FAIL", "PASS", "FAIL", "PASS", "PASS", "FAIL", "PASS", "PASS"]
    b = ["FAIL", "PASS", "PASS", "PASS", "PASS", "FAIL", "PASS", "PASS"]
    return [
        Label(
            trace_id=f"u{i}",
            span_id=None,
            kind="human",
            outcome=o,  # type: ignore[arg-type]
            critique="why",
            rater_id=rater,
            codebook_version="1",
        )
        for rater, outcomes in (("alice", a), ("bob", b))
        for i, o in enumerate(outcomes)
    ]


def _record() -> AgreementRecord:
    report = inter_rater_agreement(
        _labels(), raters=("alice", "bob"), codebook_version="1", n_resamples=50
    )
    return agreement_record_of(
        report,
        trace_ids={f"u{i}" for i in range(8)},
        cell={"feature": "refund"},
        provenance=TEST_STAMP,
    )


def _codebook() -> CodebookRecord:
    return with_content_id(
        CodebookRecord(
            codebook_id="",
            version="1",
            modes=(CodebookMode(name="wrong_tool", description="called the wrong tool"),),
            ratified_by="carol",
            ratified_at="2026-10-26T00:00:00+00:00",
            n_tagged=2,
            source="ratify.json",
            provenance=TEST_STAMP,
        ),
        "codebook_id",
        "codebook",
    )


def _proposal(
    name: str = "promised_callback", at: str = "2026-10-26T10:00:00+00:00", version: str = "1"
) -> ProposedModeRecord:
    return with_content_id(
        ProposedModeRecord(
            proposal_id="",
            codebook_version=version,
            name=name,
            note="the agent promised a callback nobody scheduled",
            proposed_by="alice",
            trace_id="ORD-1001",
            proposed_at=at,
            provenance=TEST_STAMP,
        ),
        "proposal_id",
        "proposal",
    )


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_proposed_modes_round_trip(tmp_path, fmt):
    record = _proposal()
    dump_proposed_modes([record], tmp_path, fmt=fmt)
    assert load_proposed_modes(tmp_path, fmt=fmt) == [record]


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_a_codebook_record_with_merged_proposals_round_trips(tmp_path, fmt):
    from calibrated_agent_evaluation._model import replace

    record = replace(_codebook(), merged={"wrong tool used": "wrong_tool"})
    dump_codebook_records([record], tmp_path, fmt=fmt)
    assert load_codebook_records(tmp_path, fmt=fmt) == [record]


def test_a_proposal_needs_a_note_and_a_clean_name():
    with pytest.raises(ReviewError, match="note"):
        ProposedModeRecord(
            proposal_id="p",
            codebook_version="1",
            name="x",
            note="  ",
            proposed_by="alice",
            trace_id="T",
            proposed_at="2026-10-26T10:00:00+00:00",
            provenance=TEST_STAMP,
        )
    with pytest.raises(ReviewError, match="surrounding spaces"):
        _proposal(name=" padded")


def test_mode_names_compare_without_case_or_spacing():
    assert mode_key("  Promised   Callback ") == mode_key("promised callback")
    assert mode_key("promised_callback") != mode_key("promised callback")


def test_pending_proposals_drop_settled_and_repeated_names():
    from calibrated_agent_evaluation._model import replace

    first = _proposal("Promised callback", "2026-10-26T10:00:00+00:00")
    again = _proposal("promised  CALLBACK", "2026-10-26T11:00:00+00:00")
    adopted = _proposal("wrong_tool", "2026-10-26T09:00:00+00:00")
    merged = _proposal("bad tone", "2026-10-26T09:30:00+00:00")
    other_version = _proposal("late", version="2")
    codebook = replace(_codebook(), merged={"Bad Tone": "wrong_tool"})
    pending = pending_proposals([again, adopted, merged, other_version, first], [codebook], "1")
    assert pending == (first,)


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_agreement_records_round_trip(tmp_path, fmt):
    record = _record()
    dump_agreement_records([record], tmp_path, fmt=fmt)
    assert load_agreement_records(tmp_path, fmt=fmt) == [record]


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_codebook_records_round_trip(tmp_path, fmt):
    record = _codebook()
    dump_codebook_records([record], tmp_path, fmt=fmt)
    assert load_codebook_records(tmp_path, fmt=fmt) == [record]


def test_the_record_id_is_a_hash_of_its_content():
    record = _record()
    assert record.agreement_id.startswith("agreement-")
    assert _record().agreement_id == record.agreement_id
    other = agreement_record_of(
        inter_rater_agreement(_labels(), raters=("alice", "bob"), codebook_version="1", seed=1),
        trace_ids=record.trace_ids,
        provenance=TEST_STAMP,
    )
    assert other.agreement_id != record.agreement_id


def test_a_stored_record_with_another_schema_version_is_refused(tmp_path):
    dump_agreement_records([_record()], tmp_path, fmt="json")
    path = tmp_path / "agreement_record.json"
    docs = json.loads(path.read_text())
    docs[0]["schema_version"] = "99"
    path.write_text(json.dumps(docs))
    with pytest.raises(ReviewError):
        load_agreement_records(tmp_path, fmt="json")


def test_a_codebook_record_with_duplicate_modes_is_refused():
    with pytest.raises(ReviewError, match="unique"):
        CodebookRecord(
            codebook_id="c",
            version="1",
            modes=(CodebookMode(name="a"), CodebookMode(name="a")),
            ratified_by="carol",
            ratified_at="2026-10-26T00:00:00+00:00",
            provenance=TEST_STAMP,
        )


def test_results_rows_cover_every_agreement_statistic():
    from calibrated_agent_evaluation.dumpload import results_table

    rows = results_table([_record(), _codebook(), _proposal()])
    assert {r.estimand for r in rows} == {
        "krippendorff_alpha",
        "cohen_kappa",
        "specific_agreement",
    }
    alpha = next(r for r in rows if r.estimand == "krippendorff_alpha")
    assert alpha.slice == "outcome/1/alice+bob"
    assert alpha.sided == "two-sided"
    assert alpha.seed == 0
    assert alpha.n_resamples == 50
