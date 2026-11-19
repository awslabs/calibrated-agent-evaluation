# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the local subprocess sandbox and the vendor-free sandbox seam."""

from __future__ import annotations

import inspect

import pytest

from calibrated_agent_evaluation.execute import LocalSubprocessSandbox, SandboxBackend, SandboxError


def test_a_clean_program_reports_exit_zero_and_its_output() -> None:
    result = LocalSubprocessSandbox().run_code("print('hello')", timeout_s=10)
    assert result.exit_code == 0
    assert not result.timed_out
    assert "hello" in result.stdout


def test_seeded_files_are_importable_by_the_program() -> None:
    result = LocalSubprocessSandbox().run_code(
        "from solution import answer\nprint(answer())",
        timeout_s=10,
        files={"solution.py": "def answer():\n    return 42\n"},
    )
    assert result.exit_code == 0
    assert "42" in result.stdout


def test_a_program_past_the_timeout_is_terminated_and_marked_timed_out() -> None:
    result = LocalSubprocessSandbox().run_code("import time\ntime.sleep(5)", timeout_s=0.5)
    assert result.timed_out
    assert result.exit_code != 0


def test_outbound_network_is_disabled_inside_the_sandbox() -> None:
    result = LocalSubprocessSandbox().run_code(
        "import socket\nsocket.create_connection(('203.0.113.1', 80), timeout=2)",
        timeout_s=10,
    )
    assert result.exit_code != 0
    assert "network disabled" in result.stderr


def test_a_limitless_host_still_runs_and_records_limits_not_applied(monkeypatch) -> None:
    # Force the no-POSIX-limits path (e.g. Windows) and confirm the run still completes.
    monkeypatch.setattr(
        "calibrated_agent_evaluation.execute.sandbox._posix_limit_setter", lambda *a, **k: None
    )
    result = LocalSubprocessSandbox().run_code("print('ok')", timeout_s=10)
    assert result.exit_code == 0
    assert result.limits_applied is False


def test_an_unsupported_language_is_refused_locally() -> None:
    with pytest.raises(SandboxError, match="Python only"):
        LocalSubprocessSandbox().run_code("console.log(1)", language="javascript", timeout_s=5)


def test_the_sandbox_seam_exposes_no_vendor_type() -> None:
    # Contract: the methodology plane reaches execution only through stdlib/schema types.
    signature = str(inspect.signature(SandboxBackend.run_code))
    assert "boto3" not in signature
    assert "bedrock" not in signature
    assert isinstance(LocalSubprocessSandbox(), SandboxBackend)


def test_runaway_output_is_capped_not_held_whole_in_memory() -> None:
    """A print loop past the output cap returns promptly with truncated output (the child's stream
    is written to a file and read back capped), not buffered whole in the scorer process."""
    result = LocalSubprocessSandbox().run_code("print('x' * (2 * 1024 * 1024))", timeout_s=10)
    assert result.exit_code == 0
    assert "output truncated" in result.stdout
    assert len(result.stdout) <= 1024 * 1024 + 64  # the 1 MiB cap plus the short marker


@pytest.mark.skipif(not hasattr(__import__("os"), "killpg"), reason="process groups are POSIX-only")
def test_a_timeout_kills_grandchild_processes_not_just_the_direct_child(tmp_path) -> None:
    """On timeout the whole process group is killed, so a grandchild the program spawned cannot
    outlive the run. The grandchild records a marker immediately, then tries to write a second one
    after a delay; the second must never appear, proving it was terminated."""
    import time

    started = tmp_path / "grandchild_started.txt"
    survived = tmp_path / "grandchild_survived.txt"
    code = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', "
        f"\"open({str(started)!r}, 'w').write('1'); "
        f"import time; time.sleep(2); open({str(survived)!r}, 'w').write('1')\"])\n"
        "time.sleep(30)\n"
    )
    result = LocalSubprocessSandbox().run_code(code, timeout_s=0.8)
    assert result.timed_out
    time.sleep(3)  # well past the grandchild's 2s delay
    assert started.exists(), "the grandchild should have started before the timeout"
    assert not survived.exists(), "the grandchild outlived the timeout — the group was not killed"


_CANARY = "CALIBRATED_AGENT_EVALUATION_SANDBOX_CANARY"


def test_the_program_does_not_inherit_the_callers_environment(monkeypatch) -> None:
    monkeypatch.setenv(_CANARY, "secret-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "canary-aws-secret")
    result = LocalSubprocessSandbox().run_code(
        "import os\nprint(sorted(os.environ))\nprint(os.environ.get("
        f"{_CANARY!r}), os.environ.get('AWS_SECRET_ACCESS_KEY'))",
        timeout_s=10,
    )
    assert result.exit_code == 0, result.stderr
    assert "secret-value" not in result.stdout
    assert "canary-aws-secret" not in result.stdout
    assert "None None" in result.stdout


def test_the_program_runs_in_isolated_mode() -> None:
    result = LocalSubprocessSandbox().run_code(
        "import sys\nprint(sys.flags.isolated)", timeout_s=10
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "1"


@pytest.mark.parametrize("name", ["../escape.py", "sub/../../escape.py", "ABSOLUTE"])
def test_a_seeded_file_outside_the_sandbox_dir_is_refused(name, tmp_path) -> None:
    if name == "ABSOLUTE":
        name = str(tmp_path / "escape.py")
    with pytest.raises(SandboxError, match="outside the sandbox"):
        LocalSubprocessSandbox().run_code("print(1)", timeout_s=5, files={name: "x = 1\n"})
    assert not (tmp_path / "escape.py").exists()
