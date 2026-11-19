# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Store and environment edges: corrupt docs, shared span ids, big CSV cells, torn record writes."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel

from _helpers import make_span, make_trace, reseal_dump
from calibrated_agent_evaluation._errors import CalibratedAgentEvaluationDataError
from calibrated_agent_evaluation._install import detect_install_source
from calibrated_agent_evaluation._provenance import _commit_of
from calibrated_agent_evaluation._recordstore import RecordStore
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.dumpload import dump, load
from calibrated_agent_evaluation.schema.errors import BackendError, UnregisteredArtifactError
from calibrated_agent_evaluation.schema.types import ToolCall


def test_a_corrupt_trace_doc_is_a_backend_error_on_every_read(tmp_path: Path) -> None:
    with LocalBackend(tmp_path) as store:
        store.put_trace(make_trace())
    with sqlite3.connect(tmp_path / "store.sqlite3") as conn:
        conn.execute("UPDATE traces SET doc = '{not json'")
    with LocalBackend(tmp_path) as store:
        for read in (lambda: store.get_trace("T-1"), store.query, store.all_traces):
            with pytest.raises(BackendError, match="not valid JSON"):
                read()


def _span_rows(root: Path) -> set[tuple[str, str]]:
    with sqlite3.connect(root / "store.sqlite3") as conn:
        return set(conn.execute("SELECT trace_id, span_id FROM spans"))


def test_a_span_id_two_traces_share_keeps_both_span_rows(tmp_path: Path) -> None:
    with LocalBackend(tmp_path) as store:
        store.put_trace(make_trace(trace_id="A", spans=[make_span(trace_id="A", span_id="root")]))
        store.put_trace(make_trace(trace_id="B", spans=[make_span(trace_id="B", span_id="root")]))
    assert _span_rows(tmp_path) == {("A", "root"), ("B", "root")}


def test_a_store_keyed_on_span_id_alone_is_rekeyed_from_its_trace_docs(tmp_path: Path) -> None:
    with LocalBackend(tmp_path) as store:
        store.put_trace(make_trace(trace_id="A", spans=[make_span(trace_id="A", span_id="root")]))
        store.put_trace(make_trace(trace_id="B", spans=[make_span(trace_id="B", span_id="root")]))
    with sqlite3.connect(tmp_path / "store.sqlite3") as conn:
        conn.execute("DROP TABLE spans")
        conn.execute(
            "CREATE TABLE spans (span_id TEXT NOT NULL PRIMARY KEY, trace_id TEXT NOT NULL, "
            "seq INTEGER NOT NULL, doc TEXT NOT NULL)"
        )
    LocalBackend(tmp_path).close()
    assert _span_rows(tmp_path) == {("A", "root"), ("B", "root")}


def test_a_trace_with_a_cell_over_the_csv_field_limit_round_trips_through_csv(
    tmp_path: Path,
) -> None:
    page = "x" * 200_000
    span = make_span(tool_calls=[ToolCall(name="fetch", args={}, result={"page": page})])
    trace = make_trace(spans=[span])
    with LocalBackend(tmp_path / "src") as src:
        src.put_trace(trace)
        dump(src, tmp_path / "dump")
    with LocalBackend(tmp_path / "dst") as dst:
        load(dst, tmp_path / "dump", fmt="csv")
        assert dst.get_trace("T-1") == trace


def test_a_dump_with_an_orphan_span_writes_nothing(tmp_path: Path) -> None:
    with LocalBackend(tmp_path / "src") as src:
        src.put_trace(make_trace(trace_id="T-2", ownership="probed", spans=[]))
        dump(src, tmp_path / "dump", fmt="json")
    spans_file = tmp_path / "dump" / "span.json"
    spans = json.loads(spans_file.read_text())
    spans.append(make_span(trace_id="T-9", span_id="T-9/root").model_dump(mode="json"))
    spans_file.write_text(json.dumps(spans))
    reseal_dump(tmp_path / "dump", "span.json")

    with LocalBackend(tmp_path / "dst") as dst:
        with pytest.raises(UnregisteredArtifactError, match="orphan"):
            load(dst, tmp_path / "dump", fmt="json")
        assert dst.all_traces() == []


class _Record(BaseModel):
    schema_version: str
    record_id: str


class _RecordError(CalibratedAgentEvaluationDataError):
    pass


def _store() -> RecordStore[_Record]:
    return RecordStore(_Record, stem="rec", schema_version="1", error=_RecordError, noun="record")


def test_a_torn_record_write_is_refused_rather_than_read_as_the_truth(tmp_path: Path) -> None:
    store = _store()
    first = _Record(schema_version="1", record_id="a")
    store.dump([first], tmp_path)
    # A crash between the two renames: the JSON copy holds a record the CSV copy never got.
    second = _Record(schema_version="1", record_id="b")
    docs = [first.model_dump(mode="json"), second.model_dump(mode="json")]
    (tmp_path / "rec.json").write_text(json.dumps(docs))

    for fmt in ("csv", "json"):
        with pytest.raises(_RecordError, match="disagree"):
            store.load(tmp_path, fmt=fmt)


def test_agreeing_record_copies_load_in_either_format(tmp_path: Path) -> None:
    store = _store()
    records = [_Record(schema_version="1", record_id="a")]
    store.dump(records, tmp_path, fmt="json")
    assert store.load(tmp_path, fmt="csv") == records
    assert store.load(tmp_path, fmt="json") == records


def _git(repo: Path, *argv: str) -> None:
    subprocess.run(["git", "-C", str(repo), *argv], check=True, capture_output=True)


def test_a_checkout_with_uncommitted_changes_reports_a_dirty_commit(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "f.py").write_text("x = 1\n")
    _git(tmp_path, "add", "f.py")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")
    clean = _commit_of(tmp_path)
    assert clean is not None
    assert len(clean) == 40

    (tmp_path / "f.py").write_text("x = 2\n")
    assert _commit_of(tmp_path) == f"{clean}-dirty"

    (tmp_path / "f.py").write_text("x = 1\n")
    (tmp_path / "untracked.txt").write_text("notes")
    assert _commit_of(tmp_path) == clean


class _Dist:
    def __init__(self, direct_url: dict[str, object]) -> None:
        self._direct = direct_url

    def read_text(self, filename: str) -> str | None:
        return json.dumps(self._direct) if filename == "direct_url.json" else None


def test_a_checkout_path_with_a_space_is_recognized_as_the_project_install(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "my projects" / "calibrated_agent_evaluation"
    venv = checkout / ".venv"
    venv.mkdir(parents=True)
    dist = _Dist({"url": checkout.as_uri(), "dir_info": {"editable": True}})
    assert "%20" in checkout.as_uri()
    assert detect_install_source(dist, venv).kind == "project"
