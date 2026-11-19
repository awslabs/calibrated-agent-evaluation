# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Label.rater_id``: the human who applied a label survives every store.

Without a rater identity two raters double-coding one trace produce two ``Label`` rows that cannot
be told apart, so inter-rater agreement (Krippendorff's alpha, Fleiss' kappa) is unrepresentable.
Worse, when both raters PASS with the canonical default critique their labels are byte-identical,
and the read-side identity dedup (``schema.serde.dedup_labels``) collapses them into one. These
tests pin the field end to end: schema validation, the local store (including a store written
before the column existed), the Langfuse adapter's ``Score.metadata`` envelope, and the fixture
gold set.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from _helpers import FIXTURES, make_label, make_trace
from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend, LangfuseLike
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.review.labeling import PASS_DEFAULT_CRITIQUE
from calibrated_agent_evaluation.schema.errors import SchemaError
from calibrated_agent_evaluation.schema.serde import label_from_dict, to_dict
from calibrated_agent_evaluation.schema.validate import validate
from test_backends_langfuse import StubLangfuseClient


def _default_pass(trace_id: str, rater_id: str | None) -> object:
    return make_label(
        trace_id=trace_id,
        kind="human",
        outcome="PASS",
        critique=PASS_DEFAULT_CRITIQUE,
        rater_id=rater_id,
    )


def test_rater_id_round_trips_through_the_serde_mapping() -> None:
    label = make_label(rater_id="alice")
    assert label_from_dict(to_dict(label)).rater_id == "alice"


def test_a_label_doc_written_before_rater_id_existed_reads_back_unattributed() -> None:
    doc = to_dict(make_label())
    del doc["rater_id"]
    assert label_from_dict(doc).rater_id is None


@pytest.mark.parametrize("bad", ["", "   ", 7])
def test_validate_refuses_an_empty_or_non_string_rater_id(bad: object) -> None:
    with pytest.raises(SchemaError, match="rater_id"):
        validate(make_label(rater_id=bad))


def test_two_raters_agreeing_on_a_default_pass_are_two_labels_locally(tmp_path: Path) -> None:
    """The case full-content dedup cannot separate without the field: identical verdict text."""
    backend = LocalBackend(root=tmp_path / "store")
    backend.put_score(_default_pass("R-1", "alice"))  # type: ignore[arg-type]
    backend.put_score(_default_pass("R-1", "bob"))  # type: ignore[arg-type]

    raters = sorted(str(lbl.rater_id) for lbl in backend.get_scores("R-1"))
    assert raters == ["alice", "bob"]
    assert len(backend.all_scores()) == 2


def test_one_rater_re_putting_the_same_label_still_reads_back_once(tmp_path: Path) -> None:
    """The one-row-per-rater boundary still holds: same rater, same label, one row."""
    backend = LocalBackend(root=tmp_path / "store")
    backend.put_score(_default_pass("R-2", "alice"))  # type: ignore[arg-type]
    backend.put_score(_default_pass("R-2", "alice"))  # type: ignore[arg-type]
    assert len(backend.get_scores("R-2")) == 1


def test_a_store_written_before_the_rater_column_is_migrated_on_open(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    legacy_doc = to_dict(make_label(trace_id="OLD-1"))
    del legacy_doc["rater_id"]
    conn = sqlite3.connect(root / "store.sqlite3")
    conn.executescript(
        """
        CREATE TABLE labels (
            rowid_pk  INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id  TEXT NOT NULL,
            span_id   TEXT,
            doc       TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO labels (trace_id, span_id, doc) VALUES (?, ?, ?)",
        ("OLD-1", None, json.dumps(legacy_doc)),
    )
    conn.commit()
    conn.close()

    with LocalBackend(root=root) as backend:
        [legacy] = backend.get_scores("OLD-1")
        assert legacy.rater_id is None  # an old label stays readable, honestly unattributed
        backend.put_score(make_label(trace_id="OLD-1", critique="second look", rater_id="alice"))
        assert {lbl.rater_id for lbl in backend.get_scores("OLD-1")} == {None, "alice"}

    with sqlite3.connect(root / "store.sqlite3") as conn:
        rows = conn.execute("SELECT rater_id FROM labels ORDER BY rowid_pk").fetchall()
    assert rows == [(None,), ("alice",)]


def test_rater_id_rides_score_metadata_and_two_raters_stay_distinct_on_langfuse() -> None:
    """Parity: no mapping change, because the whole Label doc rides Score.metadata."""
    lf = LangfuseBackend(cast("LangfuseLike", StubLangfuseClient()))
    lf.put_trace(make_trace())
    lf.put_score(_default_pass("T-1", "alice"))  # type: ignore[arg-type]
    lf.put_score(_default_pass("T-1", "bob"))  # type: ignore[arg-type]

    raters = sorted(str(lbl.rater_id) for lbl in lf.get_scores("T-1"))
    assert raters == ["alice", "bob"]


def test_every_fixture_gold_label_is_attributed_to_a_rater() -> None:
    for path in (
        FIXTURES / "labels" / "gold.json",
        Path(__file__).resolve().parents[1]
        / "src"
        / "calibrated_agent_evaluation"
        / "_quickstart"
        / "data"
        / "labels"
        / "gold.json",
    ):
        rows = json.loads(path.read_text(encoding="utf-8"))
        unattributed = [r["trace_id"] for r in rows if not r.get("rater_id")]
        assert unattributed == [], f"{path} has unattributed gold labels"
