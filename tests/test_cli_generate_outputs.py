# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What `calibrated-agent-evaluation generate` produces, and where its taxonomy is from.

The `--run` tests construct the real `InspectExecutor` through the CLI (nothing patched between
the flag and the executor) and run a real Inspect solver on Inspect's offline `mockllm/model`.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from cli_store_helpers import trace

TAXONOMY = {"feature": ["refund", "exchange"], "persona": ["new", "returning"]}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    return tmp_path / "store"


@pytest.fixture
def taxonomy(tmp_path) -> Path:
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(TAXONOMY))
    return path


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_out_writes_every_synthesized_input_as_jsonl(root, taxonomy, tmp_path, capsys):
    out = tmp_path / "inputs.jsonl"
    argv = ["--root", str(root), "generate", "--taxonomy", str(taxonomy), "--out", str(out)]
    assert main(argv) == EXIT_OK
    rows = _lines(out)
    assert len(rows) == 4  # a 2x2 grid's pairwise covering array is the full grid
    for row in rows:
        assert set(row) == {"text", "cell", "frame", "weight"}
        assert row["frame"] == "discovery"
        assert row["text"].strip()
        assert row["cell"]["feature"] in TAXONOMY["feature"]
    assert {tuple(sorted(r["cell"].items())) for r in rows} == {
        (("feature", f), ("persona", p)) for f in TAXONOMY["feature"] for p in TAXONOMY["persona"]
    }


def test_out_is_deterministic(root, taxonomy, tmp_path, capsys):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for out in (a, b):
        main(["--root", str(root), "generate", "--taxonomy", str(taxonomy), "--out", str(out)])
    assert a.read_text() == b.read_text()


def test_the_summary_count_matches_the_file(root, taxonomy, tmp_path, capsys):
    out = tmp_path / "inputs.jsonl"
    main(["--root", str(root), "generate", "--taxonomy", str(taxonomy), "--out", str(out)])
    summary = capsys.readouterr().out
    assert f"synthesized {len(_lines(out))} frame-tagged input(s)" in summary
    assert str(out) in summary


def test_without_out_the_summary_says_the_inputs_were_not_kept(root, taxonomy, capsys):
    assert main(["--root", str(root), "generate", "--taxonomy", str(taxonomy)]) == EXIT_OK
    assert "not kept (pass --out FILE" in capsys.readouterr().out


def test_an_untagged_corpus_gets_guidance_for_its_own_data(root, capsys):
    """An ingested external corpus has no taxonomy cells; the refusal names the
    instrumentation attribute and --taxonomy, not the bundled demo fixture."""
    with LocalBackend(root=root) as backend:
        untagged = trace("EXT-1")
        backend.put_trace(type(untagged)(**{**untagged.__dict__, "taxonomy_cell": {}}))
    assert main(["--root", str(root), "generate"]) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "calibrated_agent_evaluation.taxonomy." in err
    assert "--taxonomy FILE" in err
    assert "seed-fixture" not in err


def test_the_bundled_corpus_still_generates_from_its_own_cells(seeded_local_root, capsys):
    assert main(["generate"]) == EXIT_OK
    assert "taxonomy of the stored traces" in capsys.readouterr().out


def test_a_malformed_taxonomy_file_is_refused(root, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"feature": "refund"}))
    assert main(["--root", str(root), "generate", "--taxonomy", str(bad)]) == EXIT_DOMAIN_ERROR
    assert "list of strings" in capsys.readouterr().err


def test_frame_and_taxonomy_together_are_refused(root, taxonomy, tmp_path, capsys):
    rc = main(
        ["--root", str(root), "generate", "--taxonomy", str(taxonomy), "--frame", str(taxonomy)]
    )
    assert rc == EXIT_USAGE
    assert "not both" in capsys.readouterr().err


@pytest.fixture
def private_tmp(tmp_path, monkeypatch) -> Path:
    """A temp root only this test writes to, so another process's eval dirs cannot show up."""
    path = tmp_path / "tmp"
    path.mkdir()
    monkeypatch.setenv("TMPDIR", str(path))
    monkeypatch.setattr(tempfile, "tempdir", str(path))
    return path


def _eval_tmpdirs(under: Path) -> list[Path]:
    return sorted(under.glob("calibrated_agent_evaluation_eval_*"))


@pytest.mark.slow
def test_run_through_a_real_inspect_agent_persists_discovery_traces(
    root, taxonomy, private_tmp, capsys
):
    """The real executor is built with frame="discovery" (so the
    frame-propagation check passes) and is used as a context manager (no temp log dir left)."""
    rc = main(
        [
            "--root",
            str(root),
            "generate",
            "--run",
            "--executor",
            "inspect",
            "--agent",
            "cli_agents:echo",
            "--agent-model",
            "mockllm/model",
            "--taxonomy",
            str(taxonomy),
        ]
    )
    captured = capsys.readouterr()
    assert rc == EXIT_OK, captured.err
    assert "4 discovery-frame" in captured.out
    with LocalBackend(root=root) as backend:
        traces = backend.query()
    assert len(traces) == 4
    assert {t.frame for t in traces} == {"discovery"}
    assert {t.taxonomy_cell["feature"] for t in traces} == set(TAXONOMY["feature"])
    assert _eval_tmpdirs(private_tmp) == []


def test_inspect_run_without_an_agent_model_is_refused(root, taxonomy, capsys):
    rc = main(
        [
            "--root",
            str(root),
            "generate",
            "--run",
            "--executor",
            "inspect",
            "--agent",
            "cli_agents:echo",
            "--taxonomy",
            str(taxonomy),
        ]
    )
    assert rc == EXIT_USAGE
    assert "--agent-model" in capsys.readouterr().err


def test_a_failed_run_still_releases_the_executor_temp_dir(root, taxonomy, private_tmp, capsys):
    """A cassette model with no cassette fails inside the run; the log dir goes with it."""
    rc = main(
        [
            "--root",
            str(root),
            "generate",
            "--run",
            "--executor",
            "inspect",
            "--agent",
            "cli_agents:echo",
            "--agent-model",
            "cassette/orderbot",
            "--taxonomy",
            str(taxonomy),
        ]
    )
    assert rc == EXIT_DOMAIN_ERROR
    assert "cassette" in capsys.readouterr().err
    assert _eval_tmpdirs(private_tmp) == []
    with LocalBackend(root=root) as backend:
        assert backend.query() == []


def test_the_strands_executor_is_built_for_the_discovery_frame():
    from calibrated_agent_evaluation.cli.stages.generate import _build_executor_and_builder
    from calibrated_agent_evaluation.execute import StrandsExecutor

    executor, _builder = _build_executor_and_builder(
        "strands", lambda model: model, model="bedrock/some-model"
    )
    assert isinstance(executor, StrandsExecutor)
    assert executor._frame == "discovery"
    assert executor._model_id == "bedrock/some-model"
