# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation dump --judges-only` carries a judge to a fresh store, where it is
admitted and scores.

The development store certifies the clean fixture judge and runs its bias battery for real (judge
replies from cassettes). A judges-only dump exports the judge file and the two records; loading it
into an empty root and ingesting one trace gives a CI store in which
`calibrated-agent-evaluation judge` admits the judge and writes its verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.records import load_records_of
from cli_store_helpers import trace
from test_cli_calibrate_bias import (
    BIAS_CASSETTE,
    CLEAN,
    PROBES,
    THRESHOLDS,
    _calibrate,
    _calibrated_agent_evaluation,
    _gold_store,
    _verdict_cassette,
)


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)


@pytest.fixture
def trusted(tmp_path, capsys):
    """A development store whose clean judge is certified and CLEAR; returns (root, cassette)."""
    root = tmp_path / "dev"
    traces = _gold_store(root, 55, 55)
    replies = [(t, "fail" if i < 55 else "pass") for i, t in enumerate(traces)]
    cassette = _verdict_cassette(tmp_path / "judge.cassette.json", CLEAN, replies)
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    bias = ["bias", "--judge", str(CLEAN), "--cassette", str(BIAS_CASSETTE)]
    assert (
        _calibrated_agent_evaluation(
            root, *bias, "--probes", str(PROBES), "--thresholds", str(THRESHOLDS)
        )
        == EXIT_OK
    )
    capsys.readouterr()
    return root, cassette


def test_a_ci_store_built_from_a_judges_only_dump_admits_and_scores(trusted, tmp_path, capsys):
    dev, cassette = trusted
    kit = tmp_path / "kit"
    assert (
        _calibrated_agent_evaluation(dev, "dump", str(kit), "--judges-only", "--judge", str(CLEAN))
        == EXIT_OK
    )
    assert "dumped 1 judge(s) ('fixture-clean-judge')" in capsys.readouterr().out
    assert (kit / "judges" / CLEAN.name).read_bytes() == CLEAN.read_bytes()
    assert not (kit / "trace.json").exists()

    ci = tmp_path / "ci"
    assert _calibrated_agent_evaluation(ci, "load", str(kit)) == EXIT_OK
    out = capsys.readouterr().out
    assert "loaded 1 judge(s)" in out
    restored = ci / "judges" / CLEAN.name
    assert restored.is_file()
    assert [r.calib_id for r in load_records_of(ci, "calibration")] == [
        r.calib_id for r in load_records_of(dev, "calibration")
    ]
    assert len(load_records_of(ci, "bias")) == 1

    with LocalBackend(root=ci) as backend:
        backend.put_trace(trace("G001"))
    judge = ["judge", "--judge", str(restored), "--cassette", str(cassette), "--trace", "G001"]
    assert _calibrated_agent_evaluation(ci, *judge) == EXIT_OK
    assert "admitted" in capsys.readouterr().out
    with LocalBackend(root=ci) as backend:
        (verdict,) = [lb for lb in backend.get_scores("G001") if lb.kind == "llm-judge"]
    assert verdict.outcome == "FAIL"


def test_a_judge_that_would_not_be_admitted_is_not_exported(tmp_path, capsys):
    root = tmp_path / "dev"
    traces = _gold_store(root, 55, 55)
    replies = [(t, "fail" if i < 55 else "pass") for i, t in enumerate(traces)]
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, replies)
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    capsys.readouterr()
    kit = tmp_path / "kit"
    assert _calibrated_agent_evaluation(
        root, "dump", str(kit), "--judges-only", "--judge", str(CLEAN)
    ) == (EXIT_DOMAIN_ERROR)
    assert "has no bias battery" in capsys.readouterr().err
    assert not kit.exists()


def test_judges_only_needs_a_judge_file(tmp_path, capsys):
    assert _calibrated_agent_evaluation(
        tmp_path / "r", "dump", str(tmp_path / "kit"), "--judges-only"
    ) == (EXIT_USAGE)
    assert "needs --judge FILE" in capsys.readouterr().err


def test_a_judge_without_judges_only_is_refused(tmp_path, capsys):
    rc = _calibrated_agent_evaluation(
        tmp_path / "r", "dump", str(tmp_path / "kit"), "--judge", str(CLEAN)
    )
    assert rc == EXIT_USAGE
    assert "add --judges-only" in capsys.readouterr().err


def test_a_non_empty_target_is_refused(trusted, tmp_path, capsys):
    dev, _ = trusted
    kit = tmp_path / "kit"
    kit.mkdir()
    (kit / "other.txt").write_text("x")
    rc = _calibrated_agent_evaluation(dev, "dump", str(kit), "--judges-only", "--judge", str(CLEAN))
    assert rc == EXIT_DOMAIN_ERROR
    assert "not an empty directory" in capsys.readouterr().err


def test_an_edited_judge_file_in_the_dump_is_refused_on_load(trusted, tmp_path, capsys):
    dev, _ = trusted
    kit = tmp_path / "kit"
    assert (
        _calibrated_agent_evaluation(dev, "dump", str(kit), "--judges-only", "--judge", str(CLEAN))
        == EXIT_OK
    )
    doc = json.loads((kit / "judges" / CLEAN.name).read_text())
    doc["prompt_template"] += " Be lenient."
    (kit / "judges" / CLEAN.name).write_text(json.dumps(doc))
    capsys.readouterr()
    ci = tmp_path / "ci"
    assert _calibrated_agent_evaluation(ci, "load", str(kit)) == EXIT_DOMAIN_ERROR
    assert "edited after the dump" in capsys.readouterr().err
    assert not (ci / "records").exists()
    assert not (ci / "judges").exists()


def test_a_different_judge_file_already_in_the_root_is_refused(trusted, tmp_path, capsys):
    dev, _ = trusted
    kit = tmp_path / "kit"
    assert (
        _calibrated_agent_evaluation(dev, "dump", str(kit), "--judges-only", "--judge", str(CLEAN))
        == EXIT_OK
    )
    ci = tmp_path / "ci"
    (ci / "judges").mkdir(parents=True)
    (ci / "judges" / CLEAN.name).write_text("{}")
    capsys.readouterr()
    assert _calibrated_agent_evaluation(ci, "load", str(kit)) == EXIT_DOMAIN_ERROR
    assert "already exists with different content" in capsys.readouterr().err


def test_loading_the_same_dump_twice_adds_nothing(trusted, tmp_path, capsys):
    dev, _ = trusted
    kit: Path = tmp_path / "kit"
    assert (
        _calibrated_agent_evaluation(dev, "dump", str(kit), "--judges-only", "--judge", str(CLEAN))
        == EXIT_OK
    )
    ci = tmp_path / "ci"
    assert _calibrated_agent_evaluation(ci, "load", str(kit)) == EXIT_OK
    capsys.readouterr()
    assert _calibrated_agent_evaluation(ci, "load", str(kit)) == EXIT_OK
    assert "no records" in capsys.readouterr().out
