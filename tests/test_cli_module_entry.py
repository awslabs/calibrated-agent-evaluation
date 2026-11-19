# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``python -m calibrated_agent_evaluation`` runs the same command as the console script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "raw" / "otel"


def _module(*argv: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "calibrated_agent_evaluation", *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=cwd,
        check=False,
        timeout=120,
    )


def test_the_module_prints_the_command_help(tmp_path: Path) -> None:
    result = _module("--help", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: calibrated-agent-evaluation")


def test_the_module_runs_a_stage_and_returns_its_exit_code(tmp_path: Path) -> None:
    ok = _module(
        "--root", str(tmp_path / "store"), "ingest", "--dry-run", str(_FIXTURE), cwd=tmp_path
    )
    assert ok.returncode == 0, ok.stderr
    assert "would ingest" in ok.stdout
    missing = _module(
        "--root", str(tmp_path / "store"), "load", str(tmp_path / "nope"), cwd=tmp_path
    )
    assert missing.returncode == 1
