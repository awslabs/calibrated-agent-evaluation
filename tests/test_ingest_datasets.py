# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation ingest` reaches the dataset readers and binary OTLP.

`calibrated_agent_evaluation.normalize.datasets` shipped JSONL, parquet and HuggingFace readers, but
no `calibrated-agent-evaluation ingest` glob or `--format` value led to them: a `.parquet` or
`.jsonl` in a directory was never attempted, and a binary `.otlp` needed `--format otel` named per
file.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli.dispatch import main
from calibrated_agent_evaluation.ingest import run_ingest

_RAW = Path(__file__).resolve().parent.parent / "fixtures" / "raw"
_HF = _RAW / "hf"
# The swebench slice holds three sessions; the JSONL twin holds one of them again.
_SLICE_IDS = {"e3bfdef4aca3_c0427304", "c0d1d84c7898_35f42bfd", "140646846ae7_8d10c195"}
_SWEBENCH_ID = "e3bfdef4aca3_c0427304"


def test_format_parquet_ingests_one_trace_per_row(tmp_path: Path) -> None:
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = run_ingest([_HF / "swebench-slice.parquet"], backend=dest, fmt="parquet")
    assert report.exit_code == 0, report.results
    assert {r.trace_id for r in report.results} == _SLICE_IDS
    assert {t.trace_id for t in dest.query()} == _SLICE_IDS
    assert all((r.adapter or "").startswith("dataset/parquet") for r in report.results)
    # Rows name the record they came from, so a skip is traceable to its row.
    assert {r.source for r in report.results} == {
        f"{_HF / 'swebench-slice.parquet'}[{i}]" for i in range(3)
    }


def test_format_jsonl_ingests_the_session(tmp_path: Path) -> None:
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = run_ingest([_HF / "swebench-session.jsonl"], backend=dest, fmt="jsonl")
    assert report.exit_code == 0, report.results
    assert [t.trace_id for t in dest.query()] == [_SWEBENCH_ID]


def test_dataset_traces_default_to_the_discovery_frame(tmp_path: Path) -> None:
    """A benchmark slice is not production data, so it must not arrive authorized as such."""
    dest = LocalBackend(root=str(tmp_path / "store"))
    run_ingest([_HF / "swebench-session.jsonl"], backend=dest)
    assert dest.get_trace(_SWEBENCH_ID).frame == "discovery"


def test_auto_detects_datasets_by_suffix(tmp_path: Path) -> None:
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = run_ingest([_HF / "swebench-session.jsonl"], backend=dest)
    assert [(r.adapter or "").split(" ")[0] for r in report.results] == ["dataset/jsonl"]


def test_a_row_without_the_spans_column_is_skipped_with_the_columns_named(tmp_path: Path) -> None:
    """The renamed-columns slice needs a mapping; each row says why it was not read."""
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = run_ingest([_HF / "renamed-columns-slice.parquet"], backend=dest)
    assert report.results, "a dataset whose rows all fail must still report them"
    assert {r.status for r in report.results} == {"skipped"}
    assert all("no spans column 'spans'" in (r.reason or "") for r in report.results)
    assert dest.query() == []


def test_one_session_in_two_datasets_is_refused_not_overwritten(tmp_path: Path) -> None:
    """The same session id from two sources in one batch must not silently replace the first."""
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = run_ingest(
        [_HF / "swebench-slice.parquet", _HF / "swebench-session.jsonl"], backend=dest
    )
    by_source = {r.source: r for r in report.results}
    clash = by_source[f"{_HF / 'swebench-session.jsonl'}[0]"]
    assert clash.status == "refused"
    assert _SWEBENCH_ID in (clash.reason or "")
    assert report.exit_code == 1


def test_format_parquet_on_a_file_that_is_not_parquet_is_skipped(tmp_path: Path) -> None:
    bogus = tmp_path / "x.parquet"
    bogus.write_text("not parquet", encoding="utf-8")
    report = run_ingest([bogus], backend=LocalBackend(root=str(tmp_path / "store")))
    [result] = report.results
    assert result.status == "skipped"
    assert result.reason


def test_format_hf_resolves_a_repo_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--format hf` takes a repo id, resolved by the [hf] reader (stubbed here: no network)."""
    from calibrated_agent_evaluation.normalize.datasets import hf, jsonl

    seen: list[str] = []

    def fake_iter_rows(source: Any) -> Any:
        seen.append(source.repo_id)
        return jsonl.iter_rows(_HF / "swebench-session.jsonl")

    monkeypatch.setattr(hf, "iter_rows", fake_iter_rows)
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = run_ingest([Path("Exgentic/agent-llm-traces")], backend=dest, fmt="hf")
    assert seen == ["Exgentic/agent-llm-traces"]
    assert report.exit_code == 0, report.results
    assert (report.results[0].adapter or "").startswith("dataset/hf")
    assert [t.trace_id for t in dest.query()] == [_SWEBENCH_ID]


def _binary_otlp(src: Path, dest: Path) -> Path:
    from google.protobuf.json_format import ParseDict
    from opentelemetry.proto.trace.v1.trace_pb2 import TracesData

    message = ParseDict(json.loads(src.read_text(encoding="utf-8")), TracesData())
    dest.write_bytes(message.SerializeToString())
    return dest


def test_binary_otlp_is_picked_up_by_directory_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("opentelemetry.proto")
    exports = tmp_path / "exports"
    exports.mkdir()
    _binary_otlp(_RAW / "otel" / "ORD-1002.events.otlp.json", exports / "ORD-1002.otlp")
    monkeypatch.chdir(tmp_path)
    code = main(["ingest", "--dry-run", str(exports)])
    out = capsys.readouterr().out
    assert code == 0
    assert "1 would ingest" in out, out
    assert "ORD-1002" in out


def test_calibrated_agent_evaluation_ingest_over_the_hf_fixture_dir_reaches_every_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ticket's observed run: the parquet and JSONL files used to be invisible."""
    copy = tmp_path / "hf"
    shutil.copytree(_HF, copy)
    monkeypatch.chdir(tmp_path)
    main(["ingest", "--dry-run", str(copy)])
    out = capsys.readouterr().out
    for name in (
        "swebench-slice.parquet",
        "renamed-columns-slice.parquet",
        "swebench-session.jsonl",
    ):
        assert name in out, out
