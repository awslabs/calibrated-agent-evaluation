# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live AgentCore tests run only on request, and only as the scoped live-test role."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")

import boto3
from botocore.stub import Stubber

from calibrated_agent_evaluation.agentcore import (
    AwsClients,
    WrongAwsIdentity,
    caller_is_role,
    require_role,
)
from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
from calibrated_agent_evaluation.agentcore.guards import BilledCallBudget

_REPO = Path(__file__).resolve().parents[1]
_LIVE_DIR = _REPO / "tests" / "integration" / "agentcore"
_ROLE = "arn:aws:iam::111122223333:role/calibrated-agent-evaluation-agentcore-live-test"


def _pytest(env_overrides: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("CALIBRATED_AGENT_EVALUATION_AGENTCORE_")
    }
    env.pop("CALIBRATED_AGENT_EVALUATION_EXPLAIN", None)
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-rs", *args],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


@pytest.mark.parametrize("opt_in", [None, "true", "0"])
def test_live_tests_skip_naming_the_opt_in_unless_it_is_exactly_one(opt_in: str | None) -> None:
    env = {} if opt_in is None else {"CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE": opt_in}

    result = _pytest(env, "-m", "integration and agentcore", str(_LIVE_DIR))

    assert result.returncode == 0, result.stdout[-2000:]
    assert " skipped" in result.stdout
    assert " passed" not in result.stdout
    assert "CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE=1" in result.stdout


def test_the_default_run_deselects_the_live_tests_and_the_marker_selects_only_them() -> None:
    default = _pytest({}, "--collect-only", "-q", str(_LIVE_DIR))
    selected = _pytest(
        {}, "--collect-only", "-q", "-m", "integration and agentcore", "tests/integration"
    )

    assert "deselected" in default.stdout
    assert "test_doctor_live.py" not in default.stdout.split("deselected")[0]
    collected = [line for line in selected.stdout.splitlines() if "::" in line]
    assert collected
    assert all(line.startswith("tests/integration/agentcore/") for line in collected)


@pytest.mark.parametrize(
    ("caller", "expected"),
    [
        (
            "arn:aws:sts::111122223333:assumed-role/calibrated-agent-evaluation-agentcore-live-test/alice",
            True,
        ),
        ("arn:aws:sts::111122223333:assumed-role/Admin/alice", False),
        ("arn:aws:iam::111122223333:user/alice", False),
        (
            "arn:aws:sts::999999999999:assumed-role/calibrated-agent-evaluation-agentcore-live-test/alice",
            False,
        ),
        (
            "arn:aws-cn:sts::111122223333:assumed-role/calibrated-agent-evaluation-agentcore-live-test/alice",
            False,
        ),
        ("not-an-arn", False),
    ],
)
def test_only_a_session_of_the_configured_role_matches(caller: str, expected: bool) -> None:
    assert caller_is_role(caller, _ROLE) is expected


def test_a_role_with_a_path_matches_its_sessions() -> None:
    role = (
        "arn:aws:iam::111122223333:role/calibrated_agent_evaluation/ci/"
        "calibrated-agent-evaluation-agentcore-live-test"
    )
    caller = (
        "arn:aws:sts::111122223333:assumed-role/calibrated-agent-evaluation-agentcore-live-test/"
        "run-42"
    )

    assert caller_is_role(caller, role)


def _clients_reporting(caller_arn: str, seen: list[str]) -> tuple[AwsClients, Stubber]:
    def record(**kwargs: Any) -> None:
        seen.append(kwargs["model"].name)

    session = boto3.Session(
        region_name="us-west-2", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    clients = AwsClients(session, handlers=[record])
    stub = Stubber(clients.client("sts"))
    stub.add_response(
        "get_caller_identity", {"UserId": "AROA:x", "Account": "111122223333", "Arn": caller_arn}
    )
    return clients, stub


def test_running_as_admin_is_refused_after_only_the_identity_call() -> None:
    seen: list[str] = []
    clients, stub = _clients_reporting("arn:aws:sts::111122223333:assumed-role/Admin/alice", seen)

    with (
        stub,
        pytest.raises(WrongAwsIdentity, match="calibrated-agent-evaluation-agentcore-live-test"),
    ):
        require_role(clients, _ROLE)

    assert seen == ["GetCallerIdentity"]


def test_running_as_the_role_returns_the_caller() -> None:
    seen: list[str] = []
    caller = (
        "arn:aws:sts::111122223333:assumed-role/calibrated-agent-evaluation-agentcore-live-test/"
        "alice"
    )
    clients, stub = _clients_reporting(caller, seen)

    with stub:
        assert require_role(clients, _ROLE) == caller


@pytest.mark.parametrize(("raw", "limit"), [("", None), ("0", 0), (" 25 ", 25)])
def test_the_budget_comes_from_the_environment(raw: str, limit: int | None) -> None:
    env = {"CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS": raw} if raw else {}

    assert BilledCallBudget.from_env(env).limit == limit


@pytest.mark.parametrize("raw", ["ten", "-1", "2.5"])
def test_an_unusable_budget_value_is_a_typed_error(raw: str) -> None:
    with pytest.raises(
        AgentCoreError, match="CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS"
    ):
        BilledCallBudget.from_env({"CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS": raw})


def test_dataset_live_test_skips_without_opt_in() -> None:
    result = _pytest(
        {}, "-m", "integration and agentcore", str(_LIVE_DIR / "test_datasets_live.py")
    )

    assert result.returncode == 0, result.stdout[-2000:]
    assert "1 skipped" in result.stdout
    assert "CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE=1" in result.stdout
