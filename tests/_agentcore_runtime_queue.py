# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared setup for the AgentCore task-builder and CLI tests: no ambient AWS, queued answers."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from _agentcore_runtime_fakes import RUNTIME_ARN, StubbedAwsClients, sse_body, stop_params

_RUNTIME_ID = "calibrated_agent_evaluation_fixture-AbCdEf1234"
_WHEN = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)


def hide_ambient_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hide the machine's AWS setup, cassette mode, billed-call budget and local root."""
    ambient = [name for name in os.environ if name.startswith("AWS_")]
    for name in (
        *ambient,
        "CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE",
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    empty = tmp_path / "empty-aws-config"
    empty.write_text("")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(empty))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(empty))


def queue_preflight(clients: StubbedAwsClients, qualifier: str = "DEFAULT") -> None:
    """Queue the two read-only preflight answers: endpoint ``qualifier`` serves version 24."""
    control = clients.stub("bedrock-agentcore-control")
    made = {"status": "READY", "createdAt": _WHEN, "lastUpdatedAt": _WHEN}
    control.add_response(
        "get_agent_runtime_endpoint",
        {
            **made,
            "agentRuntimeEndpointArn": f"{RUNTIME_ARN}/runtime-endpoint/{qualifier}",
            "agentRuntimeArn": RUNTIME_ARN,
            "name": qualifier,
            "id": qualifier,
            "liveVersion": "24",
        },
        {"agentRuntimeId": _RUNTIME_ID, "endpointName": qualifier},
    )
    control.add_response(
        "get_agent_runtime",
        {
            **made,
            "agentRuntimeArn": RUNTIME_ARN,
            "agentRuntimeName": "calibrated_agent_evaluation_fixture",
            "agentRuntimeId": _RUNTIME_ID,
            "agentRuntimeVersion": "24",
            "roleArn": "arn:aws:iam::111122223333:role/calibrated-agent-evaluation-fixture-runtime",
            "lifecycleConfiguration": {},
            "protocolConfiguration": {"serverProtocol": "HTTP"},
        },
        {"agentRuntimeId": _RUNTIME_ID, "agentRuntimeVersion": "24"},
    )


def queue_run(clients: StubbedAwsClients, answer: str, qualifier: str = "DEFAULT") -> None:
    """Queue one invocation that answers ``answer``, and the stop of its session."""
    clients.stub("bedrock-agentcore", view=True).add_response(
        "invoke_agent_runtime",
        {
            "response": sse_body({"response": answer}),
            "contentType": "text/event-stream",
            "statusCode": 200,
        },
    )
    clients.stub("bedrock-agentcore").add_response(
        "stop_runtime_session", {"statusCode": 200}, stop_params(qualifier=qualifier)
    )
