# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI tests for ``calibrated-agent-evaluation ingest``: the command wires the ingest
feature to the local backend, reports a per-source summary, and returns the report's exit code."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from _helpers import structural_view
from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli.dispatch import main

_RAW = Path(__file__).resolve().parent.parent / "fixtures" / "raw"


def test_calibrated_agent_evaluation_ingest_persists_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`calibrated-agent-evaluation ingest <strands-file>` exits 0 and reports the trace it
    ingested."""
    monkeypatch.chdir(
        tmp_path
    )  # local backend writes ./.calibrated-agent-evaluation — isolate it in a temp cwd
    code = main(["ingest", str(_RAW / "strands" / "CLM-4471.otlp.json")])
    out = capsys.readouterr().out
    assert code == 0
    assert "CLM-4471" in out
    assert "1 ingested" in out


def test_calibrated_agent_evaluation_ingest_exits_nonzero_on_unreadable_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Negative: an unrecognized file is skipped and the command exits non-zero."""
    monkeypatch.chdir(tmp_path)
    junk = tmp_path / "notes.txt"
    junk.write_text("not a trace", encoding="utf-8")
    code = main(["ingest", str(junk)])
    err = capsys.readouterr().err
    assert code != 0
    assert "skipped" in err.lower()


def test_calibrated_agent_evaluation_ingest_dry_run_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--dry-run reports a summary, exits 0, and creates no local store."""
    monkeypatch.chdir(tmp_path)
    code = main(["ingest", "--dry-run", str(_RAW / "strands" / "CLM-4471.otlp.json")])
    out = capsys.readouterr().out
    assert code == 0
    assert "would ingest" in out
    # nothing persisted: a fresh backend over the same store finds no traces
    from calibrated_agent_evaluation.backends.local import LocalBackend

    assert LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).query() == []


# --- a degraded trace is visible in the header and refusable -----------------


def test_probed_trace_is_counted_in_the_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A terminal-I/O-only source still ingests, but the aggregate header says it degraded.

    The defect this pins: the per-source line already printed `probed`, and `ownership` was already
    on the `IngestResult`, but the header reported only ingested/refused/skipped — so a batch that
    degraded end to end read as unqualified success, and the exit code agreed. A reader who scans
    the summary line (which is what a summary line is for) saw nothing.
    """
    monkeypatch.chdir(tmp_path)
    code = main(["ingest", str(_RAW / "otel" / "ORD-DEGRADED.otlp.json")])
    out = capsys.readouterr().out

    assert code == 0, "a probed trace is faithful data, not a failure — it must still ingest"
    assert "1 ingested (1 probed)" in out
    assert "probed" in out.splitlines()[-1], "and still named on the per-source line"


def test_intact_trace_omits_the_probed_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An `owned` batch prints no probed figure at all, so the figure keeps its signal.

    Same reasoning as the contentless count: a number printed on every healthy run is a number
    the reader learns to skip, and then it is not reporting anything when it matters.
    """
    monkeypatch.chdir(tmp_path)
    code = main(["ingest", str(_RAW / "strands" / "CLM-4471.otlp.json")])
    out = capsys.readouterr().out

    assert code == 0
    assert "1 ingested," in out
    assert "probed" not in out


def test_require_owned_refuses_a_degraded_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Negative path: --require-owned turns the same source into a refusal, with a named reason."""
    monkeypatch.chdir(tmp_path)
    code = main(["ingest", "--require-owned", str(_RAW / "otel" / "ORD-DEGRADED.otlp.json")])
    captured = capsys.readouterr()
    text = captured.out + captured.err

    assert code != 0, "refusing must fail the process, not just annotate the line"
    assert "1 refused" in text
    assert "0 ingested" in text
    assert "--require-owned" in text, "the message must name the flag that caused the refusal"
    assert "step-level attribution" in text, "and why an outcome-only trace cannot serve"

    from calibrated_agent_evaluation.backends.local import LocalBackend

    assert LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).query() == [], (
        "refused writes nothing"
    )


def test_require_owned_leaves_an_intact_trace_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--require-owned is not a stricter gate on everything: an `owned` trace passes unchanged."""
    monkeypatch.chdir(tmp_path)
    code = main(["ingest", "--require-owned", str(_RAW / "strands" / "CLM-4471.otlp.json")])
    out = capsys.readouterr().out

    assert code == 0
    assert "1 ingested" in out
    assert "0 refused" in out


# --- two sources must not silently collapse into one stored trace ------------


def _copy(src: Path, dest_dir: Path, name: str) -> Path:
    dest = dest_dir / name
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def test_colliding_stems_with_different_wire_ids_both_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ticket's exact case: `run.json` and `run.otlp.json` both stem to "run".

    `_trace_id_for` takes the first dotted filename segment, and the store persists with
    `INSERT OR REPLACE` keyed on trace id — so the report said `2 ingested`, exit 0, and the store
    held exactly 1 trace. The survivor was whichever file glob order reached second; the other
    file's tool calls were gone with nothing said.

    A colliding stem is now re-resolved from each source's own wire id. Only on collision: the stem
    is what makes ingest reproduce the frozen oracle ids, so preferring the wire id unconditionally
    would turn every fixture id into a hex blob.
    """
    monkeypatch.chdir(tmp_path)
    _copy(_RAW / "otel" / "ORD-1001.attrs.otlp.json", tmp_path, "run.otlp.json")
    _copy(_RAW / "otel" / "ORD-1002.events.otlp.json", tmp_path, "run.json")

    code = main(["ingest", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "2 ingested" in out

    from calibrated_agent_evaluation.backends.local import LocalBackend

    stored = LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).query()
    assert len(stored) == 2, "the reported count and the store count must agree"
    assert len({t.trace_id for t in stored}) == 2


def test_genuinely_ambiguous_collision_is_refused_naming_both_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same stem AND the same wire id: nothing left to disambiguate with, so refuse both.

    This is the case where preferring the wire id cannot help — the sources really are
    indistinguishable to the store — and silently keeping one would be the original defect.
    """
    monkeypatch.chdir(tmp_path)
    source = _RAW / "otel" / "ORD-1001.attrs.otlp.json"
    _copy(source, tmp_path, "run.otlp.json")
    _copy(source, tmp_path, "run.json")

    code = main(["ingest", str(tmp_path)])
    captured = capsys.readouterr()
    text = captured.out + captured.err

    assert code != 0
    assert "2 refused" in text
    assert "run.json" in text, "both colliding paths must be named"
    assert "run.otlp.json" in text
    assert "calibrated_agent_evaluation.trace_id" in text, "and the message must name the fix"

    from calibrated_agent_evaluation.backends.local import LocalBackend

    assert LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).query() == [], (
        "refused writes nothing"
    )


def test_distinct_stems_keep_their_filename_derived_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The determinism the stem default exists for must be untouched when there is no collision.

    Without this, "prefer the wire id" could be over-applied and every frozen-oracle id would become
    a hex string.
    """
    monkeypatch.chdir(tmp_path)
    _copy(_RAW / "otel" / "ORD-1001.attrs.otlp.json", tmp_path, "ORD-1001.otlp.json")
    _copy(_RAW / "otel" / "ORD-1002.events.otlp.json", tmp_path, "ORD-1002.otlp.json")

    code = main(["ingest", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0, out

    from calibrated_agent_evaluation.backends.local import LocalBackend

    stored = {
        t.trace_id
        for t in LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).query()
    }
    assert stored == {"ORD-1001", "ORD-1002"}


def test_a_multi_sample_eval_ingests_one_trace_per_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end: `calibrated-agent-evaluation ingest` on a 3-sample.eval stores 3 traces.

    The adapter read the first sample and drained the rest, and `_ingest_path` returned one
    `IngestResult` per file — so a full-dataset eval log ingested as 1/N of itself, reporting
    `1 ingested` and exit 0.
    """
    pytest.importorskip("inspect_ai")
    monkeypatch.chdir(tmp_path)
    source = _RAW / "inspect" / "multisample-3.eval"

    code = main(["ingest", str(source)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "3 ingested" in out

    from calibrated_agent_evaluation.backends.local import LocalBackend

    stored = {
        t.trace_id
        for t in LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).query()
    }
    assert stored == {"ORD-1001", "ORD-1002", "CLM-4471"}


# --- --from-langfuse names the missing extra or credentials


def test_from_langfuse_without_the_extra_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without the ``[langfuse]`` extra the pull refuses, naming the extra; nothing persisted."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-dev")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-calibrated-agent-evaluation-dev")
    monkeypatch.setitem(sys.modules, "langfuse", None)  # find_spec reports it not installed
    code = main(["ingest", "--from-langfuse", "calibrated-agent-evaluation-dev"])
    err = capsys.readouterr().err
    assert code == 1
    assert install_hint("langfuse") in err
    assert LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).query() == []


def test_from_langfuse_without_credentials_names_the_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no ``LANGFUSE_*`` keys the pull refuses, naming both variables; nothing persisted."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    code = main(["ingest", "--from-langfuse", "calibrated-agent-evaluation-dev"])
    err = capsys.readouterr().err
    assert code == 1
    assert "LANGFUSE_PUBLIC_KEY" in err
    assert "LANGFUSE_SECRET_KEY" in err
    assert LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).query() == []
    # An installed package has no checkout, so the pointer is the docs page, and that page exists.
    assert "docker/" not in err
    site = "https://awslabs.github.io/calibrated-agent-evaluation/"
    assert site in err
    page = err.split(site, 1)[1].split(")", 1)[0].strip("/")
    docs = _RAW.parent.parent / "docs" / "site" / "content"
    assert (docs / f"{page}.md").is_file(), page


# --- calibrated-agent-evaluation ingest reproduces the oracle on the structural standard
# ---------------

_ORACLE = _RAW.parent / "normalized"


def test_calibrated_agent_evaluation_ingest_reproduces_the_oracle_on_the_structural_standard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A directory of OTel captures ingests, and every persisted Trace equals its
    ``fixtures/normalized/<id>.json`` oracle on the structural standard; a summary line per file."""
    captures = tmp_path / "captures"
    captures.mkdir()
    for name in ("ORD-1001.events.otlp.json", "ORD-1002.events.otlp.json"):
        shutil.copy(_RAW / "otel" / name, captures / name)
    monkeypatch.chdir(tmp_path)

    code = main(["ingest", str(captures)])
    out = capsys.readouterr().out

    assert code == 0
    assert "2 ingested" in out
    store = LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation"))
    for trace_id in ("ORD-1001", "ORD-1002"):
        assert trace_id in out
        oracle = json.loads((_ORACLE / f"{trace_id}.json").read_text(encoding="utf-8"))
        assert structural_view(store.get_trace(trace_id)) == structural_view(oracle)


def test_calibrated_agent_evaluation_ingest_dry_run_over_a_directory_reports_every_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--dry-run <dir>`` prints a summary line for every file and writes nothing."""
    captures = tmp_path / "captures"
    captures.mkdir()
    for name in ("ORD-1001.events.otlp.json", "ORD-1002.events.otlp.json"):
        shutil.copy(_RAW / "otel" / name, captures / name)
    monkeypatch.chdir(tmp_path)

    code = main(["ingest", "--dry-run", str(captures)])
    out = capsys.readouterr().out

    assert code == 0
    assert "2 would ingest" in out
    assert "ORD-1001" in out
    assert "ORD-1002" in out
    assert LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).query() == []


@pytest.mark.parametrize("dataset", ["swebench-session.jsonl", "swebench-slice.parquet"])
def test_frame_flag_stamps_every_dataset_record(
    dataset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--frame`` reaches dataset records the same way it reaches a single trace file."""
    monkeypatch.chdir(tmp_path)
    code = main(["ingest", "--frame", "production-representative", str(_RAW / "hf" / dataset)])
    assert code == 0
    traces = LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).query()
    assert traces
    assert {t.frame for t in traces} == {"production-representative"}


def test_a_sampled_dataset_ingest_is_stamped_production_representative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = str(_RAW / "hf" / "swebench-session.jsonl")
    code = main(["ingest", "--sample", "1", "--seed", "7", source])
    assert code == 0
    traces = LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).query()
    assert [t.frame for t in traces] == ["production-representative"]
