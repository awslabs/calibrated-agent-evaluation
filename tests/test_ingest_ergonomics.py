# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation ingest` ergonomics: directory expansion, the convention in the
summary, and the convention escape hatch."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli.dispatch import EXIT_USAGE, main
from calibrated_agent_evaluation.ingest import run_ingest

_RAW = Path(__file__).resolve().parent.parent / "fixtures" / "raw"


def _printed(capsys: pytest.CaptureFixture[str]) -> str:
    """Everything the command printed; a failing run writes its summary to stderr."""
    captured = capsys.readouterr()
    return captured.out + captured.err


# --- 1. a stray non-trace JSON file in a directory does not fail the batch -----------------------


def test_a_non_trace_json_in_a_directory_is_listed_as_ignored_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    shutil.copy(_RAW / "otel" / "ORD-1002.events.otlp.json", exports)
    (exports / "MANIFEST.json").write_text(json.dumps({"sessions": []}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    code = main(["ingest", str(exports)])
    out = _printed(capsys)
    assert code == 0, out
    assert "1 ingested" in out
    assert "[ignored]" in out
    assert "MANIFEST.json" in out, "an ignored file is still named, never silently dropped"


def test_a_corrupt_json_in_a_directory_is_still_attempted_and_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only a file that parses and is plainly not OTLP is ignored; a broken trace still fails."""
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "run.json").write_text('{"resourceSpans": [', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["ingest", str(exports)]) == 1
    assert "skipped" in _printed(capsys)


def test_a_named_non_trace_json_is_attempted_and_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Naming a file is a request to ingest it, so it is never ignored."""
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["ingest", str(manifest)]) == 1
    assert "[skipped]" in _printed(capsys)


# --- 2. directories are walked recursively, and a quoted glob expands -----------------------------


def test_a_directory_is_walked_recursively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    nested = tmp_path / "exports" / "run-1"
    nested.mkdir(parents=True)
    shutil.copy(_RAW / "otel" / "ORD-1002.events.otlp.json", nested / "trace.json")
    monkeypatch.chdir(tmp_path)
    assert main(["ingest", "--dry-run", "exports"]) == 0
    out = _printed(capsys)
    assert "1 would ingest" in out, out
    assert "run-1" in out


def test_a_quoted_recursive_glob_expands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for run in ("run-1", "run-2"):
        nested = tmp_path / "exports" / run
        nested.mkdir(parents=True)
    shutil.copy(_RAW / "otel" / "ORD-1002.events.otlp.json", tmp_path / "exports" / "run-1")
    shutil.copy(_RAW / "strands" / "CLM-4471.otlp.json", tmp_path / "exports" / "run-2")
    monkeypatch.chdir(tmp_path)
    assert main(["ingest", "--dry-run", "exports/**/*.json"]) == 0
    assert "2 would ingest" in _printed(capsys)


def test_a_glob_matching_nothing_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["ingest", "nothing/**/*.json"]) == 1
    assert "nothing/**/*.json" in _printed(capsys)


# --- 3. the summary names the convention that read the trace --------------------------------------


@pytest.mark.parametrize(
    ("relative", "convention"),
    [
        ("otel/messages/swebench_tool_call.otlp.json", "gen_ai/messages"),
        ("otel/openinference/weather_tool_call.otlp.json", "openinference"),
        ("otel/openllmetry/weather_tool_call.otlp.json", "openllmetry"),
        ("otel/ORD-1002.events.otlp.json", "gen_ai/latest"),
        ("strands/CLM-4471.otlp.json", "strands"),
    ],
)
def test_the_summary_names_the_resolved_convention(
    relative: str, convention: str, tmp_path: Path
) -> None:
    report = run_ingest(
        [_RAW / relative], backend=LocalBackend(root=str(tmp_path / "s")), dry_run=True
    )
    assert [r.adapter for r in report.results] == [convention], report.results


def test_an_inspect_log_reports_a_readable_adapter_name(tmp_path: Path) -> None:
    pytest.importorskip("inspect_ai")
    evals = sorted((_RAW / "inspect").glob("*.eval"))
    if not evals:
        pytest.skip("no committed .eval fixture")
    report = run_ingest([evals[0]], backend=LocalBackend(root=str(tmp_path / "s")), dry_run=True)
    assert {r.adapter for r in report.results} == {"inspect"}


# --- 4. the ambiguity error has a CLI answer ------------------------------------------------------


def _ambiguous(tmp_path: Path) -> Path:
    payload = json.loads(
        (_RAW / "otel" / "messages" / "swebench_tool_call.otlp.json").read_text(encoding="utf-8")
    )
    span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    span.setdefault("attributes", []).append(
        {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}}
    )
    path = tmp_path / "mixed.otlp.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_an_ambiguous_trace_names_the_convention_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _ambiguous(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["ingest", str(path)]) == 1
    assert "--convention" in _printed(capsys)


def test_convention_flag_disambiguates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _ambiguous(tmp_path)
    monkeypatch.chdir(tmp_path)
    code = main(["ingest", "--convention", "gen_ai/messages", str(path)])
    out = _printed(capsys)
    assert code == 0, out
    assert "gen_ai/messages" in out


def test_convention_flag_is_refused_for_a_non_otel_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(
        ["ingest", "--format", "inspect", "--convention", "openinference", str(tmp_path / "x")]
    )
    assert code == EXIT_USAGE
    assert "--convention" in _printed(capsys)


# --- 8. --backend position is stated in the help --------------------------------------------------


def test_ingest_help_states_where_backend_goes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["ingest", "--help"])
    assert "calibrated-agent-evaluation --backend langfuse ingest" in _printed(capsys)
