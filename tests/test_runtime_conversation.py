# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A simulated conversation on the AgentCore Runtime executor: one session across its turns."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")

from _agentcore_runtime_fakes import (
    RUNTIME_ARN,
    FakeRunSpanReader,
    StubbedAwsClients,
    add_invocation_response,
    add_preflight_responses,
    add_stop_response,
)
from calibrated_agent_evaluation.agentcore import (
    AgentCoreRuntimeExecutor,
    AgentCoreRuntimeTask,
    RuntimeInvocationError,
    RuntimeTarget,
)
from calibrated_agent_evaluation.synth.conversation import ConversationTurn

_CONVERSATION = "ORD-1001-sim"


@pytest.fixture
def clients() -> Iterator[StubbedAwsClients]:
    with StubbedAwsClients() as stubbed:
        yield stubbed


@pytest.fixture(autouse=True)
def _no_ambient_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in list(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", raising=False)
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", raising=False)
    empty = tmp_path / "empty-aws-config"
    empty.write_text("")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(empty))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(empty))


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def _executor(
    clients: StubbedAwsClients, tmp_path: Path, **overrides: Any
) -> AgentCoreRuntimeExecutor:
    options: dict[str, Any] = {
        "span_reader": FakeRunSpanReader(clients),
        "clients": clients,
        "log_dir": tmp_path / "logs",
        "sleep": lambda _s: None,
        "clock": _Clock(),
    }
    options.update(overrides)
    target = RuntimeTarget(runtime_arn=RUNTIME_ARN, payload_format="agentcore_sse")
    return AgentCoreRuntimeExecutor(target, **options)


def _turn(n: int, prompt: str = "", conversation: str = _CONVERSATION) -> AgentCoreRuntimeTask:
    return AgentCoreRuntimeTask(
        prompt=prompt or f"customer message {n}",
        conversation=ConversationTurn(conversation, n),
    )


def _sessions(clients: StubbedAwsClients, operation: str) -> list[str]:
    return [call.params["runtimeSessionId"] for call in clients.calls_to(operation)]


def test_a_conversation_reuses_one_runtime_session_and_stops_it_once(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    add_preflight_responses(clients)
    for _ in range(3):
        add_invocation_response(clients)
    add_stop_response(clients)

    with _executor(clients, tmp_path) as executor:
        for n in (1, 2, 3):
            executor.run(_turn(n))
        executor.end_conversation(_CONVERSATION)

    sent = clients.calls_to("InvokeAgentRuntime")
    sessions = {call.params["runtimeSessionId"] for call in sent}
    assert len(sent) == 3
    assert len(sessions) == 1
    [session] = sessions
    assert session.startswith("calibrated-agent-evaluation-")
    assert len({call.params["traceParent"] for call in sent}) == 3
    assert _sessions(clients, "StopRuntimeSession") == [session]
    clients.assert_no_pending_responses()


def test_the_session_stays_open_between_turns(tmp_path: Path, clients: StubbedAwsClients) -> None:
    add_preflight_responses(clients)
    add_invocation_response(clients)
    add_invocation_response(clients)
    add_stop_response(clients)

    with _executor(clients, tmp_path) as executor:
        executor.run(_turn(1))
        executor.run(_turn(2))
        assert clients.calls_to("StopRuntimeSession") == []
        executor.end_conversation(_CONVERSATION)
    assert len(clients.calls_to("StopRuntimeSession")) == 1


def test_a_failed_turn_still_stops_the_conversation_session(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    add_preflight_responses(clients)
    add_invocation_response(clients)
    clients.stub("bedrock-agentcore", view=True).add_client_error(
        "invoke_agent_runtime",
        service_error_code="ValidationException",
        service_message="bad payload",
        http_status_code=400,
    )
    add_stop_response(clients)

    with _executor(clients, tmp_path) as executor:
        executor.run(_turn(1))
        with pytest.raises(RuntimeInvocationError):
            executor.run(_turn(2))
        assert clients.calls_to("StopRuntimeSession") == []
        executor.end_conversation(_CONVERSATION)
        [session] = set(_sessions(clients, "InvokeAgentRuntime"))
        assert _sessions(clients, "StopRuntimeSession") == [session]
        # Ending it again, or ending a conversation that never ran, stops nothing more.
        executor.end_conversation(_CONVERSATION)
        executor.end_conversation("never-ran")
    assert len(clients.calls_to("StopRuntimeSession")) == 1


def test_closing_the_executor_stops_a_conversation_left_open(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    add_preflight_responses(clients)
    add_invocation_response(clients)
    add_stop_response(clients)

    with _executor(clients, tmp_path) as executor:
        executor.run(_turn(1))
        # Interrupted here: the conversation never reaches end_conversation.
    [session] = _sessions(clients, "InvokeAgentRuntime")
    assert _sessions(clients, "StopRuntimeSession") == [session]


def test_single_turn_runs_keep_their_sessions_and_fingerprints(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    add_preflight_responses(clients)
    add_invocation_response(clients)
    add_stop_response(clients)
    add_invocation_response(clients)
    add_stop_response(clients)

    with _executor(clients, tmp_path) as executor:
        executor.run(AgentCoreRuntimeTask(prompt="one"))
        assert len(clients.calls_to("StopRuntimeSession")) == 1
        executor.run(AgentCoreRuntimeTask(prompt="two"))

    invoked = _sessions(clients, "InvokeAgentRuntime")
    assert len(set(invoked)) == 2
    assert _sessions(clients, "StopRuntimeSession") == invoked


def _record_cassette(clients: StubbedAwsClients, tmp_path: Path, prompts: list[str]) -> Path:
    path = tmp_path / "conversation.cassette.json"
    add_preflight_responses(clients)
    for _ in prompts:
        add_invocation_response(clients)
    add_stop_response(clients)
    with _executor(clients, tmp_path, cassette_path=path, cassette_mode="record") as executor:
        for n, prompt in enumerate(prompts, start=1):
            executor.run(_turn(n, prompt))
        executor.end_conversation(_CONVERSATION)
    return path


def test_repeated_prompts_in_a_conversation_are_recorded_per_turn(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    path = _record_cassette(clients, tmp_path, ["yes", "yes"])
    entries = json.loads(path.read_text())["entries"]
    assert len(entries) == 2

    replayed: list[str] = []
    with _executor(
        clients, tmp_path, cassette_path=path, cassette_mode="replay", span_reader=None
    ) as replay:
        for n in (1, 2):
            [trace] = replay.run(_turn(n, "yes"))
            replayed.append(trace.input["prompt"])
        replay.end_conversation(_CONVERSATION)
    assert replayed == ["yes", "yes"]
    # The replay opened no session, so it stopped none.
    assert len(clients.calls_to("StopRuntimeSession")) == 1


def test_replay_refuses_a_conversation_turn_never_recorded(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    path = _record_cassette(clients, tmp_path, ["yes", "yes"])
    with _executor(
        clients, tmp_path, cassette_path=path, cassette_mode="replay", span_reader=None
    ) as replay:
        replay.run(_turn(1, "yes"))
        replay.run(_turn(2, "yes"))
        with pytest.raises(RuntimeInvocationError, match="cassette miss"):
            replay.run(_turn(3, "yes"))


def test_a_repeated_single_turn_prompt_is_still_refused(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    path = tmp_path / "single.cassette.json"
    add_preflight_responses(clients)
    add_invocation_response(clients)
    add_stop_response(clients)
    with _executor(clients, tmp_path, cassette_path=path, cassette_mode="record") as executor:
        executor.run(AgentCoreRuntimeTask(prompt="yes"))
        with pytest.raises(RuntimeInvocationError, match="the same payload format, builder and"):
            executor.run(AgentCoreRuntimeTask(prompt="yes"))


def test_a_task_with_a_malformed_conversation_is_refused_with_no_aws_call(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    with _executor(clients, tmp_path) as executor:
        for bad in (ConversationTurn("", 1), ConversationTurn(_CONVERSATION, 0)):
            with pytest.raises(RuntimeInvocationError, match="task's conversation"):
                executor.run(AgentCoreRuntimeTask(prompt="hi", conversation=bad))
    assert clients.calls_to("InvokeAgentRuntime") == []


def test_a_conversations_turns_are_recorded_as_held_open_until_it_ends(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    add_preflight_responses(clients)
    add_invocation_response(clients)
    add_invocation_response(clients)
    add_stop_response(clients)
    with _executor(clients, tmp_path) as executor:
        executor.run(_turn(1))
        executor.run(_turn(2))
        assert executor.end_conversation(_CONVERSATION) == "stopped"
        assert [r.session_stop for r in executor.run_records] == ["held_open", "held_open"]
        assert executor.end_conversation(_CONVERSATION) is None


def test_a_turn_on_a_session_id_that_is_not_a_runs_is_refused() -> None:
    from calibrated_agent_evaluation.agentcore.runtime import turn_run_ids

    assert (
        turn_run_ids("calibrated-agent-evaluation-" + "a" * 32).session_id
        == "calibrated-agent-evaluation-" + "a" * 32
    )
    with pytest.raises(RuntimeInvocationError, match="session ID"):
        turn_run_ids("not-a-session")
