# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation doctor` checks the store the flags configure, and refuses flags it
would ignore.

A failing report goes to stderr, as every non-zero outcome does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_USAGE


def test_a_root_that_is_a_file_is_reported_unusable_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "store"
    root.write_text("")
    assert main(["--root", str(root), "doctor"]) == 1
    assert f"Local backend: NOT usable; {root} exists and is not a directory" in (
        capsys.readouterr().err
    )


def test_a_root_under_a_read_only_directory_is_reported_not_writable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        assert main(["--root", str(locked / "store"), "doctor"]) == 1
    finally:
        locked.chmod(0o700)
    assert "Local backend: NOT writable" in capsys.readouterr().err


def test_the_env_root_is_the_one_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", str(tmp_path))
    assert main(["doctor"]) == 0
    assert f"Local backend: writable at {tmp_path}" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == [], "the probe file is removed"


def test_a_root_not_created_yet_is_not_created_by_doctor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "later"
    assert main(["--root", str(root), "doctor"]) == 0
    assert "will be created on first write" in capsys.readouterr().out
    assert not root.exists()


def test_agentcore_flags_without_agentcore_are_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["doctor", "--runtime", "foo", "--region", "us-west-2"]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "--region, --runtime only apply with --agentcore" in err


def test_the_langfuse_backend_without_keys_is_reported_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("langfuse")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert main(["--backend", "langfuse", "--root", str(tmp_path), "doctor"]) == 1
    assert "Langfuse backend: NOT ready" in capsys.readouterr().err


def test_an_unreachable_langfuse_is_reported_not_reachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("langfuse")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:9")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")
    assert main(["--backend", "langfuse", "--root", str(tmp_path), "doctor"]) == 1
    assert "Langfuse backend: NOT reachable" in capsys.readouterr().err
