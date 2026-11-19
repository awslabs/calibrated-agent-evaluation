# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation doctor` stays offline without --agentcore and refuses cleanly
without credentials."""

from __future__ import annotations

import socket
from typing import Any

import pytest

from calibrated_agent_evaluation.cli import main


def test_doctor_without_the_flag_makes_no_network_call(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    def no_network(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("doctor opened a network connection")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)

    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "Optional extras:" in out
    assert "AgentCore readiness" not in out


def test_missing_credentials_stop_before_any_check_and_name_the_fix(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, tmp_path: Any
) -> None:
    pytest.importorskip("boto3")
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")

    code = main(["doctor", "--agentcore", "--region", "us-west-2"])

    err = capsys.readouterr().err
    assert code == 1
    assert "AWS_PROFILE" in err
    assert "--profile" in err
    assert "[pass]" not in err
    assert "Traceback" not in err


def test_a_nonexistent_profile_is_the_same_plain_refusal(capsys: Any) -> None:
    pytest.importorskip("boto3")

    code = main(
        [
            "doctor",
            "--agentcore",
            "--region",
            "us-west-2",
            "--profile",
            "calibrated-agent-evaluation-nope",
        ]
    )

    err = capsys.readouterr().err
    assert code == 1
    assert "--profile" in err
    assert "Traceback" not in err
