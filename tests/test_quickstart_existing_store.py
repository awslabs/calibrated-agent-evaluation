# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""quickstart never mixes its sample data into a store directory that already holds some."""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK


def _seed_marker(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mine.txt").write_text("a user's own data")


@pytest.mark.parametrize("certify", [False, True])
def test_quickstart_refuses_a_root_that_already_holds_data(tmp_path, capsys, certify):
    root = tmp_path / "store"
    _seed_marker(root)
    argv = ["--root", str(root), "quickstart", *(["--certify"] if certify else [])]
    assert main(argv) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "already holds data" in err
    assert "--force" in err
    assert sorted(p.name for p in root.iterdir()) == ["mine.txt"]


def test_quickstart_refuses_the_env_root_too(tmp_path, capsys, monkeypatch):
    root = tmp_path / "env-store"
    _seed_marker(root)
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", str(root))
    assert main(["quickstart"]) == EXIT_DOMAIN_ERROR
    assert "already holds data" in capsys.readouterr().err


def test_quickstart_seeds_an_empty_or_new_root(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--root", str(empty), "quickstart"]) == EXIT_OK
    assert main(["--root", str(tmp_path / "new"), "quickstart"]) == EXIT_OK
    assert "already held data" not in capsys.readouterr().out


def test_quickstart_force_seeds_and_says_the_sample_is_mixed_in(tmp_path, capsys):
    root = tmp_path / "store"
    _seed_marker(root)
    assert main(["--root", str(root), "quickstart", "--force"]) == EXIT_OK
    assert "already held data (--force)" in capsys.readouterr().out
    assert (root / "mine.txt").exists()


def test_quickstart_twice_into_one_root_is_refused_the_second_time(tmp_path, capsys):
    root = tmp_path / "store"
    assert main(["--root", str(root), "quickstart", "--certify"]) == EXIT_OK
    kit = root / "quickstart" / "judge-v1.json"
    kit.write_text('{"edited": true}')
    assert main(["--root", str(root), "quickstart", "--certify"]) == EXIT_DOMAIN_ERROR
    assert kit.read_text() == '{"edited": true}'


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    monkeypatch.setenv("AWS_PROFILE", "calibrated-agent-evaluation-test-no-such-profile")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
