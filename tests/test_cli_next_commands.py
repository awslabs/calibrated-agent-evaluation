# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The "Next:" command calibrate, bias and judge print pastes into a shell and runs.

The lines used to splice the judge path in unquoted and drop ``--root`` and ``--cassette``, so under
a store path with a space the pasted command split the path in two, and without the cassette it
asked the live judge model instead of the replies the run before it used. Each step here runs the
command the previous one printed, parsed back with ``shlex`` the way a shell would.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from calibrated_agent_evaluation._quickstart import documented_commands
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK


def _next_command(out: str, program: str) -> list[str]:
    """The backquoted command on the summary's "Next:" line, split as a shell would."""
    (line,) = [ln for ln in out.splitlines() if ln.startswith("Next:") and program in ln]
    (command,) = re.findall(r"`([^`]+)`", line)
    argv = shlex.split(command)
    assert argv[0] == "calibrated-agent-evaluation", argv
    return argv[1:]


def _main(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_each_printed_next_command_runs_under_a_root_with_a_space(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_EXPLAIN", raising=False)
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    root = tmp_path / "my store"
    assert _main(capsys, ["--root", str(root), "quickstart", "--certify"])[0] == EXIT_OK
    _, _, calibrate_v2, bias_v2, _, _ = documented_commands(root)
    probes = bias_v2[bias_v2.index("--probes") + 1]
    thresholds = bias_v2[bias_v2.index("--thresholds") + 1]
    cassette = calibrate_v2[calibrate_v2.index("--cassette") + 1]

    code, out = _main(capsys, ["--root", str(root), *calibrate_v2])
    assert code == EXIT_OK, out
    bias = _next_command(out, " bias ")
    assert bias[:2] == ["--root", str(root)]
    assert bias[-2:] == ["--cassette", cassette]
    filled = [probes if w == "DIR" else thresholds if w == "FILE" else w for w in bias]

    code, out = _main(capsys, filled)
    assert code == EXIT_OK, out
    judge = _next_command(out, " judge ")
    assert judge[:2] == ["--root", str(root)]
    assert judge[-2:] == ["--cassette", cassette]

    code, out = _main(capsys, [*judge, "--frame", "production-representative"])
    assert code == EXIT_OK, out
    run = _next_command(out, " run ")
    assert run[:2] == ["--root", str(root)]

    code, out = _main(capsys, [*run, "--ci"])
    # The recorded v2 judge's corrected rate does not clear the default bar; the gate ran.
    assert code == EXIT_DOMAIN_ERROR, out
    assert "CI gate failed: 1 of 1 failure mode(s)" in out
