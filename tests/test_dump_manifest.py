# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A dump is sealed by a manifest written last; ``load`` refuses a directory that does not match.

Every case writes real dump directories from real stores and loads them into a fresh one, which
must stay empty after a refusal.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import calibrated_agent_evaluation.dumpload.io as dumpload_io
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.dumpload import (
    MANIFEST_NAME,
    CorruptDumpError,
    DumpMismatchError,
    IncompleteDumpError,
    UnlistedDumpFileError,
    dump,
    load,
)
from calibrated_agent_evaluation.ingest.core import run_ingest
from calibrated_agent_evaluation.schema.types import Label, Outcome

_OTEL = Path(__file__).resolve().parent.parent / "fixtures" / "raw" / "otel"
_MESSAGES = _OTEL / "messages" / "swebench_tool_call.otlp.json"
_OPENINFERENCE = _OTEL / "openinference" / "weather_tool_call.otlp.json"
_CORE_FILES = ["label.csv", "label.json", "span.csv", "span.json", "trace.csv", "trace.json"]


def _label(trace_id: str, outcome: Outcome = "FAIL") -> Label:
    return Label(trace_id=trace_id, span_id=None, kind="human", outcome=outcome, critique="x")


def _store(path: Path) -> LocalBackend:
    backend = LocalBackend(":memory:")
    (result,) = run_ingest([path], backend=backend).results
    assert result.trace_id is not None
    backend.put_score(_label(result.trace_id))
    return backend


def _assert_refused(directory: Path, fmt: str, error: type[CorruptDumpError], match: str) -> None:
    target = LocalBackend(":memory:")
    with pytest.raises(error, match=match):
        load(target, directory, fmt)
    assert target.all_traces() == []
    assert target.all_scores() == []


def test_the_manifest_lists_every_file_with_its_count_and_digest(tmp_path: Path) -> None:
    manifest = dump(_store(_MESSAGES), tmp_path)

    on_disk = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert on_disk == manifest.model_dump(mode="json")
    assert on_disk["format_version"] == "1"
    assert sorted(on_disk["files"]) == _CORE_FILES
    for name, entry in on_disk["files"].items():
        assert entry["sha256"] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    counts = {name: entry["records"] for name, entry in on_disk["files"].items()}
    spans = counts["span.json"]
    assert spans > 0
    assert counts == {
        "label.csv": 1,
        "label.json": 1,
        "span.csv": spans,
        "span.json": spans,
        "trace.csv": 1,
        "trace.json": 1,
    }


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_a_dump_interrupted_before_its_manifest_is_refused(
    tmp_path: Path, fmt: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(out: Path, manifest: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(dumpload_io, "_write_manifest", interrupted)
    with pytest.raises(KeyboardInterrupt):
        dump(_store(_MESSAGES), tmp_path, fmt)
    assert sorted(p.name for p in tmp_path.iterdir()) == _CORE_FILES

    _assert_refused(tmp_path, fmt, IncompleteDumpError, "no dump-manifest.json")


@pytest.mark.parametrize("fmt", ["csv", "json"])
@pytest.mark.parametrize("cut", ["before the manifest", "between files"])
def test_a_partial_dump_over_a_complete_one_is_refused(
    tmp_path: Path, fmt: str, cut: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    dump(_store(_MESSAGES), tmp_path, fmt)

    if cut == "before the manifest":
        target, original = "_write_manifest", dumpload_io._write_manifest
    else:
        target, original = "_write_csv", dumpload_io._write_csv

    def failing(path: Path, *args: object) -> None:
        if cut == "before the manifest" or path.name == "label.csv":
            raise OSError("disk full")
        original(path, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(dumpload_io, target, failing)
    with pytest.raises(OSError, match="disk full"):
        dump(_store(_OPENINFERENCE), tmp_path, fmt)
    monkeypatch.undo()

    _assert_refused(tmp_path, fmt, IncompleteDumpError, "no dump-manifest.json")


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_files_mixed_from_two_complete_dumps_are_refused(tmp_path: Path, fmt: str) -> None:
    """The traces of one store with the labels of another."""
    old, new = tmp_path / "old", tmp_path / "new"
    dump(_store(_MESSAGES), old, fmt)
    dump(_store(_OPENINFERENCE), new, fmt)
    shutil.copy(old / f"label.{fmt}", new / f"label.{fmt}")

    target = LocalBackend(":memory:")
    with pytest.raises(DumpMismatchError, match=f"label.{fmt}") as info:
        load(target, new, fmt)
    assert info.value.path == f"label.{fmt}"
    assert target.all_traces() == []


@pytest.mark.parametrize("fmt", ["csv", "json"])
@pytest.mark.parametrize("name", ["trace.json", "trace.csv", "label.csv"])
def test_one_changed_byte_in_any_data_file_is_refused(tmp_path: Path, fmt: str, name: str) -> None:
    """Whichever file of the pair the load reads: the other one is part of the dump too."""
    dump(_store(_MESSAGES), tmp_path, fmt)
    path = tmp_path / name
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))

    _assert_refused(tmp_path, fmt, DumpMismatchError, f"{name} in .* differs")


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_a_record_count_that_differs_from_the_manifest_is_refused(tmp_path: Path, fmt: str) -> None:
    dump(_store(_MESSAGES), tmp_path, fmt)
    path = tmp_path / MANIFEST_NAME
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["files"][f"trace.{fmt}"]["records"] = 5
    path.write_text(json.dumps(doc), encoding="utf-8")

    _assert_refused(tmp_path, fmt, DumpMismatchError, "reads as 1 record.* records 5")


def test_a_data_file_missing_from_the_dump_is_refused(tmp_path: Path) -> None:
    dump(_store(_MESSAGES), tmp_path)
    (tmp_path / "span.csv").unlink()

    _assert_refused(tmp_path, "json", DumpMismatchError, "span.csv, listed in .* is missing")


@pytest.mark.parametrize("extra", ["notes.json", "records/run/run_record.json", "old.csv"])
def test_a_file_the_manifest_does_not_list_is_refused(tmp_path: Path, extra: str) -> None:
    dump(_store(_MESSAGES), tmp_path)
    (tmp_path / extra).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / extra).write_text("[]", encoding="utf-8")

    target = LocalBackend(":memory:")
    with pytest.raises(UnlistedDumpFileError, match="does not list") as info:
        load(target, tmp_path, "json")
    assert info.value.paths == (extra,)
    assert target.all_traces() == []


def test_hidden_and_non_data_files_are_not_part_of_a_dump(tmp_path: Path) -> None:
    """Lock files, a crashed write's temporary, Finder's metadata and a README."""
    source = _store(_MESSAGES)
    dump(source, tmp_path)
    (tmp_path / ".DS_Store").write_bytes(b"\0")
    (tmp_path / ".trace.json.abc123.tmp").write_text("{", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "README.md").write_text("The release baseline.\n", encoding="utf-8")

    target = LocalBackend(":memory:")
    load(target, tmp_path, "json")
    assert target.all_traces() == source.all_traces()


def test_a_dump_from_before_manifests_is_refused_as_incomplete(tmp_path: Path) -> None:
    dump(_store(_MESSAGES), tmp_path)
    (tmp_path / MANIFEST_NAME).unlink()

    _assert_refused(tmp_path, "json", IncompleteDumpError, "written before dumps carried")


def test_a_manifest_of_another_format_version_is_refused(tmp_path: Path) -> None:
    dump(_store(_MESSAGES), tmp_path)
    path = tmp_path / MANIFEST_NAME
    doc = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**doc, "format_version": "2"}), encoding="utf-8")

    _assert_refused(tmp_path, "json", CorruptDumpError, "dump format '2'")


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_labels_for_traces_the_store_does_not_hold_round_trip_exactly(
    tmp_path: Path, fmt: str
) -> None:
    source = _store(_MESSAGES)
    source.put_score(_label("never-ingested", "PASS"))
    source.put_score(_label("also-never-ingested"))

    dump(source, tmp_path, fmt)
    target = LocalBackend(":memory:")
    load(target, tmp_path, fmt)

    def by_key(labels: list[Label]) -> list[Label]:
        return sorted(labels, key=lambda lb: (lb.trace_id, lb.span_id or ""))

    assert target.all_traces() == source.all_traces()
    assert by_key(target.all_scores()) == by_key(source.all_scores())
    assert {lb.trace_id for lb in target.all_scores()} >= {"never-ingested", "also-never-ingested"}


def test_a_dump_over_an_earlier_one_replaces_its_files_and_keeps_the_rest(tmp_path: Path) -> None:
    """A file only the earlier dump wrote is removed; a file no dump wrote is left alone."""

    def with_records(out: Path) -> None:
        (out / "records").mkdir()
        (out / "records" / "kind.json").write_text("[1, 2]", encoding="utf-8")

    first = dump(_store(_MESSAGES), tmp_path, extra=with_records)
    assert first.files["records/kind.json"].records == 2

    second = _store(_OPENINFERENCE)
    dump(second, tmp_path)
    assert not (tmp_path / "records" / "kind.json").exists()
    target = LocalBackend(":memory:")
    load(target, tmp_path, "json")
    assert target.all_traces() == second.all_traces()

    (tmp_path / "mine.json").write_text('"not a dump file"', encoding="utf-8")
    dump(second, tmp_path)
    assert (tmp_path / "mine.json").read_text(encoding="utf-8") == '"not a dump file"'
    assert (
        "mine.json"
        not in json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))["files"]
    )
    _assert_refused(tmp_path, "json", UnlistedDumpFileError, "mine.json")
