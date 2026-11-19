# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-turn `generate` on an AgentCore Runtime agent: one session per conversation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")
pytest.importorskip("strands_evals")

from _agentcore_runtime_fakes import (
    REGION,
    RUNTIME_ARN,
    StubbedAwsClients,
    add_reader_response,
    register_fake_agentcore_span_reader,
    sse_body,
    stop_params,
)
from _agentcore_runtime_queue import queue_preflight
from calibrated_agent_evaluation.agentcore import AwsClients
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, EXIT_USAGE
from test_synth_helpers import FakeActorSimulator

_REPO = Path(__file__).resolve().parent.parent
_FRAME = _REPO / "fixtures" / "raw" / "frames" / "agentcore_CLM-4471.frame.json"
_ANSWER = "Denied: the vacancy exclusion applies."
_LIVE = (
    "generate",
    "--run",
    "--executor",
    "agentcore",
    "--runtime-arn",
    RUNTIME_ARN,
    "--payload-format",
    "agentcore_sse",
)


@pytest.fixture(autouse=True)
def _no_ambient_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in list(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", raising=False)
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", raising=False)
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    empty = tmp_path / "empty-aws-config"
    empty.write_text("")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(empty))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(empty))


class _ConversationRuntime:
    """Stands in for ``AwsClients.from_profile``: a runtime answering ``turns`` invocations, each
    with one span read, and one session stop at the end of the conversation."""

    def __init__(self, turns: int) -> None:
        self.turns = turns
        self.built: list[StubbedAwsClients] = []

    def __call__(self, *, region: str | None = None, **options: Any) -> StubbedAwsClients:
        options.pop("profile", None)
        clients = StubbedAwsClients(region=region or REGION, **options)
        queue_preflight(clients)
        for _ in range(self.turns):
            clients.stub("bedrock-agentcore", view=True).add_response(
                "invoke_agent_runtime",
                {
                    "response": sse_body({"response": _ANSWER}),
                    "contentType": "text/event-stream",
                    "statusCode": 200,
                },
            )
            add_reader_response(clients)
        clients.stub("bedrock-agentcore").add_response(
            "stop_runtime_session", {"statusCode": 200}, stop_params()
        )
        self.built.append(clients)
        return clients


def _fake_actor(monkeypatch: pytest.MonkeyPatch, follow_ups: int) -> None:
    """The strands actor, replaced by a fake that sends ``follow_ups`` messages, then stops."""
    import calibrated_agent_evaluation.synth.borrow.simulator as simulator

    monkeypatch.setattr(
        simulator,
        "_load_actor_simulator",
        lambda: lambda **kw: FakeActorSimulator(n_turns=follow_ups, **kw),
    )


def test_generate_holds_conversations_with_a_runtime_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _ConversationRuntime(turns=2)
    monkeypatch.setattr(AwsClients, "from_profile", runtime)
    register_fake_agentcore_span_reader(monkeypatch)
    _fake_actor(monkeypatch, follow_ups=1)
    root = tmp_path / "store"

    rc = main(
        [
            "--root",
            str(root),
            *_LIVE,
            "--frame",
            str(_FRAME),
            "--multi-turn",
            "--max-turns",
            "3",
            "--yes",
        ]
    )

    assert rc == EXIT_OK, capsys.readouterr().err
    [clients] = runtime.built
    invoked = [c.params["runtimeSessionId"] for c in clients.calls_to("InvokeAgentRuntime")]
    assert len(invoked) == 2
    assert len(set(invoked)) == 1
    assert [c.params["runtimeSessionId"] for c in clients.calls_to("StopRuntimeSession")] == [
        invoked[0]
    ]
    with LocalBackend(root=root) as backend:
        [trace] = backend.query()
    assert trace.input["turns"] == 2
    assert trace.input["simulator"] == "strands"
    assert [m["role"] for m in trace.input["conversation"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    turn_prefixes = {span.span_id.split(":", 1)[0] for span in trace.spans}
    assert turn_prefixes == {"t1", "t2"}
    clients.assert_no_pending_responses()


def test_inspect_still_cannot_hold_a_conversation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "--root",
            str(tmp_path / "store"),
            "generate",
            "--run",
            "--executor",
            "inspect",
            "--agent",
            "x:y",
            "--frame",
            str(_FRAME),
            "--multi-turn",
        ]
    )
    assert rc == EXIT_USAGE
    assert "pass --executor strands or agentcore" in capsys.readouterr().err


def test_a_replayed_multi_turn_generate_never_builds_the_simulated_user(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import calibrated_agent_evaluation.synth.borrow.simulator as simulator

    cassette = tmp_path / "runs.cassette.json"
    argv = [*_LIVE, "--frame", str(_FRAME), "--multi-turn", "--max-turns", "3"]
    runtime = _ConversationRuntime(turns=2)
    monkeypatch.setattr(AwsClients, "from_profile", runtime)
    register_fake_agentcore_span_reader(monkeypatch)
    _fake_actor(monkeypatch, follow_ups=1)
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", "record")
    recorded_root = tmp_path / "recorded"
    rc = main(["--root", str(recorded_root), *argv, "--cassette", str(cassette), "--yes"])
    assert rc == EXIT_OK, capsys.readouterr().err
    assert (tmp_path / "runs.actor.json").exists()

    def no_simulator() -> Any:
        raise AssertionError("a replay built the simulated user")

    def no_aws(**_options: Any) -> Any:
        raise AssertionError("a replay built AWS clients")

    monkeypatch.setattr(simulator, "_load_actor_simulator", no_simulator)
    monkeypatch.setattr(AwsClients, "from_profile", no_aws)
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", "replay")
    replayed_root = tmp_path / "replayed"
    rc = main(["--root", str(replayed_root), *argv, "--cassette", str(cassette)])

    assert rc == EXIT_OK, capsys.readouterr().err
    with LocalBackend(root=recorded_root) as backend:
        [recorded] = backend.query()
    with LocalBackend(root=replayed_root) as backend:
        [replayed] = backend.query()
    assert replayed.input["conversation"] == recorded.input["conversation"]


def test_a_multi_turn_replay_without_a_simulated_user_recording_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", "replay")
    rc = main(
        [
            "--root",
            str(tmp_path / "store"),
            *_LIVE,
            "--frame",
            str(_FRAME),
            "--multi-turn",
            "--cassette",
            str(tmp_path / "runs.cassette.json"),
        ]
    )
    assert rc != EXIT_OK
    assert "simulated-user recording" in capsys.readouterr().err
