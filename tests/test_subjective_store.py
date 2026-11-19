# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The stored preference record, its CSV+JSON round-trip, and the atom-set and pairs files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _helpers import TEST_STAMP
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.subjective import (
    PreferencePair,
    PreferenceRecord,
    SubjectiveError,
    SubjectiveStoreError,
    atom_set_from_dict,
    dump_preference_records,
    load_atom_set,
    load_preference_records,
    preference_record,
    read_preference_file,
)

ATOMS = {
    "criterion": "claim letter",
    "objective": [{"name": "cites_exclusion", "description": "names the exclusion applied"}],
    "subjective": [{"name": "tone"}, {"name": "clarity", "description": "plain words"}],
}


def _record() -> PreferenceRecord:
    """A stored preference, shared with the dump/load tests of every record kind."""
    return PreferenceRecord(
        preference_id="pref-0001",
        criterion="claim letter",
        dimension="tone",
        trace_id_a="CLM-4471",
        trace_id_b="CLM-4473",
        winner="b",
        judge_kind="human",
        rater_id="rater-1",
        source="file",
        source_ref="pairs.jsonl:1",
        provenance=TEST_STAMP,
    )


def _pair(**changes: str) -> PreferencePair:
    fields = {
        "trace_id_a": "CLM-4471",
        "trace_id_b": "CLM-4473",
        "dimension": "tone",
        "winner": "a",
        "judge_kind": "human",
        **changes,
    }
    return PreferencePair(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_preference_records_round_trip_losslessly(tmp_path: Path, fmt: str) -> None:
    records = [_record(), replace(_record(), preference_id="pref-0002", winner="a", rater_id=None)]
    dump_preference_records(records, tmp_path, fmt=fmt)
    assert load_preference_records(tmp_path, fmt=fmt) == records


def test_a_missing_store_loads_empty(tmp_path: Path) -> None:
    assert load_preference_records(tmp_path / "nothing") == []


def test_a_record_from_another_schema_version_is_refused(tmp_path: Path) -> None:
    dump_preference_records([_record()], tmp_path, fmt="json")
    doc = json.loads((tmp_path / "preference_record.json").read_text())
    doc[0]["schema_version"] = "0"
    (tmp_path / "preference_record.json").write_text(json.dumps(doc))
    with pytest.raises(SubjectiveStoreError, match="schema_version"):
        load_preference_records(tmp_path, fmt="json")


def test_a_preference_record_is_built_only_for_a_subjective_dimension() -> None:
    atoms = atom_set_from_dict(ATOMS)
    record = preference_record(
        _pair(), atoms, preference_id="p1", source="review", source_ref="ui", rater_id="r"
    )
    assert record.pair() == _pair()
    assert record.criterion == "claim letter"
    with pytest.raises(SubjectiveError, match="objective"):
        preference_record(
            _pair(dimension="cites_exclusion"),
            atoms,
            preference_id="p",
            source="file",
            source_ref="x",
        )
    with pytest.raises(SubjectiveError, match="not found"):
        preference_record(
            _pair(dimension="warmth"), atoms, preference_id="p", source="file", source_ref="x"
        )


def test_an_unknown_judge_kind_is_refused_at_the_record() -> None:
    with pytest.raises(SubjectiveStoreError, match="judge_kind"):
        preference_record(
            _pair(judge_kind="crowd"),
            atom_set_from_dict(ATOMS),
            preference_id="p",
            source="file",
            source_ref="x",
        )


def test_the_atom_set_file_declares_the_two_lanes(tmp_path: Path) -> None:
    path = tmp_path / "atoms.json"
    path.write_text(json.dumps(ATOMS))
    atoms = load_atom_set(path)
    assert [a.name for a in atoms.objective_atoms] == ["cites_exclusion"]
    assert [d.name for d in atoms.subjective_dimensions] == ["tone", "clarity"]


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        ({**ATOMS, "weights": {}}, "weights"),
        ({**ATOMS, "subjective": [{"name": "cites_exclusion"}]}, "both"),
        ({"criterion": " ", "subjective": [{"name": "tone"}]}, "criterion"),
        ({"criterion": "c"}, "at least 1"),
        (["not", "an", "object"], "AtomFile"),
    ],
)
def test_a_malformed_atom_set_file_is_refused(tmp_path: Path, doc: object, match: str) -> None:
    path = tmp_path / "atoms.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(SubjectiveError, match=match):
        load_atom_set(path)


def test_an_unreadable_or_non_json_atom_set_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SubjectiveError, match="cannot read"):
        load_atom_set(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    with pytest.raises(SubjectiveError, match="not valid JSON"):
        load_atom_set(bad)


def _write_pairs(tmp_path: Path, *lines: object) -> Path:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines) + "\n"
    )
    return path


def _line(**changes: str) -> dict[str, str]:
    return {
        "trace_id_a": "CLM-4471",
        "trace_id_b": "CLM-4473",
        "dimension": "tone",
        "winner": "a",
        "judge_kind": "human",
        **changes,
    }


def test_the_pairs_file_yields_each_checked_preference_with_its_line(tmp_path: Path) -> None:
    path = _write_pairs(tmp_path, _line(), "", _line(winner="b", dimension="clarity"))
    pairs = read_preference_file(path, atom_set_from_dict(ATOMS))
    assert [n for n, _ in pairs] == [1, 3]
    assert pairs[1][1].winner == "b"


def test_one_bad_line_refuses_the_pairs_file_naming_every_bad_line(tmp_path: Path) -> None:
    path = _write_pairs(
        tmp_path,
        _line(),
        _line(dimension="cites_exclusion"),
        _line(trace_id_b="CLM-4471"),
        "{broken",
        {**_line(), "score": 4},
        _line(winner="tie"),
    )
    with pytest.raises(SubjectiveError) as info:
        read_preference_file(path, atom_set_from_dict(ATOMS))
    message = str(info.value)
    assert "5 line(s)" in message
    assert "nothing was imported" in message
    for number in ("line 2", "line 3", "line 4", "line 5", "line 6"):
        assert number in message
    assert "line 1:" not in message


def test_an_empty_pairs_file_is_refused(tmp_path: Path) -> None:
    path = _write_pairs(tmp_path, "")
    with pytest.raises(SubjectiveError, match="holds no preference"):
        read_preference_file(path, atom_set_from_dict(ATOMS))
