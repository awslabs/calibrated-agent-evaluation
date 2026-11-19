# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two simulated users behind the ``SimulatedUser`` port, their extras, and their failures."""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")
pytest.importorskip("bedrock_agentcore")
pytest.importorskip("strands_evals")

from botocore.exceptions import EndpointConnectionError

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
    RuntimeTarget,
)
from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
from calibrated_agent_evaluation.agentcore.simulated_user import AgentCoreSimulatedUser
from calibrated_agent_evaluation.cli import conversation_route
from calibrated_agent_evaluation.cli.errors import CliError
from calibrated_agent_evaluation.synth.borrow.simulator import StrandsSimulatedUser
from calibrated_agent_evaluation.synth.conversation import (
    ConversationTurn,
    UserTurn,
    run_simulated_conversation,
)
from calibrated_agent_evaluation.synth.errors import ConversationError, SynthError


def _sdk_result(message: str | None, *, stop: bool, stop_reason: str | None) -> Any:
    from bedrock_agentcore.evaluation.runner import SimulatorResult

    return SimulatorResult(message=message, reasoning="", stop=stop, stop_reason=stop_reason)


class _FakeAgentCoreActor:
    """Mirrors ``AgentCoreActorSimulator``: built with its keyword arguments, ``act`` per turn."""

    def __init__(self, results: list[Any], **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._results = list(results)

    def act(self, agent_message: str) -> Any:
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _agentcore_user(results: list[Any]) -> AgentCoreSimulatedUser:
    return AgentCoreSimulatedUser(
        {"traits": {}, "context": "bought a blender 51 days ago", "actor_goal": "get a refund"},
        "I want a refund.",
        max_turns=4,
        scenario_description="refund outside the window",
        simulator_cls=lambda **kw: _FakeAgentCoreActor(results, **kw),
    )


class _Structured:
    def __init__(self, message: str | None, stop: bool, stop_reason: str | None) -> None:
        self.message, self.stop, self.stop_reason = message, stop, stop_reason


class _StrandsResult:
    def __init__(self, message: str | None, stop: bool, stop_reason: str | None) -> None:
        self.structured_output = _Structured(message, stop, stop_reason)


class _FakeStrandsActor:
    def __init__(self, results: list[Any], **_kwargs: Any) -> None:
        self._results = list(results)

    def has_next(self) -> bool:
        return bool(self._results)

    def act(self, agent_message: str, **_kwargs: Any) -> Any:
        return self._results.pop(0)


def test_each_simulator_maps_to_the_same_user_turn() -> None:
    agentcore = _agentcore_user(
        [
            _sdk_result("Can you make an exception?", stop=False, stop_reason=None),
            _sdk_result(None, stop=True, stop_reason="goal_completed"),
        ]
    )
    strands = StrandsSimulatedUser(
        None,
        "I want a refund.",
        simulator_cls=lambda **kw: _FakeStrandsActor(
            [
                _StrandsResult("Can you make an exception?", False, None),
                _StrandsResult(None, True, "goal_completed"),
            ]
        ),
    )
    for user in (agentcore, strands):
        assert user.act("No, it is outside the window.") == UserTurn(
            message="Can you make an exception?", stop=False
        )
        assert user.act("Still no.") == UserTurn(
            message=None, stop=True, stop_reason="goal_completed"
        )


def test_each_simulator_keeps_the_last_message_it_stops_with() -> None:
    agentcore = _agentcore_user(
        [_sdk_result("Thanks, that's all.", stop=True, stop_reason="goal_completed")]
    )
    strands = StrandsSimulatedUser(
        None,
        "I want a refund.",
        simulator_cls=lambda **kw: _FakeStrandsActor(
            [_StrandsResult("Thanks, that's all.", True, "goal_completed")]
        ),
    )
    for user in (agentcore, strands):
        assert user.act("Your refund is on its way.") == UserTurn(
            message="Thanks, that's all.", stop=True, stop_reason="goal_completed"
        )


def test_a_strands_simulator_out_of_turns_reports_max_turns() -> None:
    def actor(**kw: Any) -> _FakeStrandsActor:
        assert kw["max_turns"] == 1
        return _FakeStrandsActor([_StrandsResult("Can you make an exception?", False, None)])

    user = StrandsSimulatedUser(None, "I want a refund.", max_turns=1, simulator_cls=actor)
    assert user.act("No.").stop is False
    assert user.act("Still no.") == UserTurn(message=None, stop=True, stop_reason="max_turns")
    # A simulator that stops before its limit chose to stop.
    early = StrandsSimulatedUser(
        None, "I want a refund.", max_turns=5, simulator_cls=lambda **_kw: _FakeStrandsActor([])
    )
    assert early.act("No.") == UserTurn(message=None, stop=True, stop_reason="actor_stopped")


def test_the_agentcore_user_gets_the_sdks_own_prompt_and_turn_limit() -> None:
    built: list[_FakeAgentCoreActor] = []

    def factory(**kw: Any) -> _FakeAgentCoreActor:
        built.append(_FakeAgentCoreActor([], **kw))
        return built[-1]

    AgentCoreSimulatedUser(
        {"traits": {}, "context": "ctx", "actor_goal": "get a refund"},
        "I want a refund.",
        max_turns=4,
        scenario_description="refund outside the window",
        simulator_cls=factory,
    )
    [actor] = built
    assert actor.kwargs["initial_query"] == "I want a refund."
    assert actor.kwargs["max_turns"] == 4
    assert "## User Simulation" in actor.kwargs["system_prompt"]
    assert "get a refund" in actor.kwargs["system_prompt"]


def test_simulator_adapters_match_the_installed_sdks() -> None:
    from bedrock_agentcore.evaluation.runner import AgentCoreActorSimulator, SimulatorResult
    from strands_evals import ActorSimulator

    agentcore = inspect.signature(AgentCoreActorSimulator.__init__).parameters
    assert {
        "actor_profile",
        "initial_query",
        "system_prompt",
        "model",
        "max_turns",
        "tools",
    } <= set(agentcore)
    assert set(inspect.signature(AgentCoreActorSimulator.act).parameters) >= {"agent_message"}
    assert {"message", "stop", "stop_reason"} <= set(SimulatorResult.__dataclass_fields__)
    strands = inspect.signature(ActorSimulator.__init__).parameters
    assert {"actor_profile", "initial_query", "model", "max_turns"} <= set(strands)
    assert hasattr(ActorSimulator, "has_next")


def test_a_failing_simulator_raises_a_typed_error() -> None:
    with pytest.raises(AgentCoreError, match="AgentCore actor simulator failed") as agentcore:
        _agentcore_user([RuntimeError("model throttled")]).act("hello")
    assert isinstance(agentcore.value.__cause__, RuntimeError)

    class Exploding:
        def has_next(self) -> bool:
            return True

        def act(self, agent_message: str, **_kwargs: Any) -> Any:
            raise ValueError("bad structured output")

    with pytest.raises(SynthError, match="input simulator failed"):
        StrandsSimulatedUser(None, "hi", simulator_cls=lambda **kw: Exploding()).act("hello")


def test_the_strands_simulator_is_the_default() -> None:
    assert conversation_route.simulator_from(argparse.Namespace()) == "strands"
    assert conversation_route.simulator_from(argparse.Namespace(simulator=None)) == "strands"
    with pytest.raises(CliError, match="--simulator must be one of"):
        conversation_route.simulator_from(argparse.Namespace(simulator="other"))
    # With no --simulator, the factory builds the strands user, never the AgentCore one.
    factory = conversation_route.user_factory(
        "strands", max_turns=3, simulator_cls=lambda **kw: _FakeStrandsActor([])
    )
    assert isinstance(factory("hi", None, ""), StrandsSimulatedUser)


def test_a_missing_extra_is_refused_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from calibrated_agent_evaluation.cli import config

    real = config.probe_extra

    def without(*hidden: str) -> Any:
        def probe(module: str) -> Any:
            if module in hidden:
                return config.Degradation(capability=module, available=False, install_hint="x")
            return real(module)

        return probe

    monkeypatch.setattr(config, "probe_extra", without("strands_evals"))
    with pytest.raises(CliError, match="--simulator strands needs strands-agents-evals"):
        conversation_route.require_simulator_extras("strands")
    monkeypatch.setattr(config, "probe_extra", without("bedrock_agentcore"))
    with pytest.raises(CliError, match=r"agentcore extra \(bedrock_agentcore missing\)"):
        conversation_route.require_simulator_extras("agentcore")
    monkeypatch.setattr(config, "probe_extra", without("jinja2"))
    with pytest.raises(CliError, match=r"jinja2 missing"):
        conversation_route.require_simulator_extras("agentcore")
    monkeypatch.setattr(config, "probe_extra", real)
    conversation_route.require_simulator_extras("strands")
    conversation_route.require_simulator_extras("agentcore")


# --- failures mid-conversation against the Runtime executor ----------------------------------


@pytest.fixture
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


class _UnreachableOnCall:
    """A call handler that fails the ``n``-th ``InvokeAgentRuntime`` as an unreachable endpoint."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.seen = 0

    def __call__(self, **kwargs: Any) -> None:
        if kwargs["model"].name == "InvokeAgentRuntime":
            self.seen += 1
            if self.seen == self.n:
                raise EndpointConnectionError(endpoint_url="https://bedrock-agentcore.example")


def _converse_on_runtime(clients: StubbedAwsClients, tmp_path: Path, user: Any) -> Any:
    target = RuntimeTarget(runtime_arn=RUNTIME_ARN, payload_format="agentcore_sse")
    executor = AgentCoreRuntimeExecutor(
        target,
        span_reader=FakeRunSpanReader(clients),
        clients=clients,
        log_dir=tmp_path / "logs",
        sleep=lambda _s: None,
    )

    def agent_turn(turn: ConversationTurn, history: list[dict[str, Any]]) -> Any:
        prompt = history[-1]["content"][0]["text"]
        [trace] = executor.run(AgentCoreRuntimeTask(prompt=prompt, conversation=turn))
        return trace

    with executor:
        return run_simulated_conversation(
            user,
            agent_turn,
            conversation_id="ORD-1001-sim",
            opening="I want a refund.",
            max_turns=4,
            on_end=lambda: executor.end_conversation("ORD-1001-sim"),
        )


@pytest.mark.usefixtures("_no_ambient_aws")
def test_an_unreachable_runtime_mid_conversation_is_a_typed_error(tmp_path: Path) -> None:
    with StubbedAwsClients(handlers=[_UnreachableOnCall(2)]) as clients:
        add_preflight_responses(clients)
        add_invocation_response(clients)
        add_stop_response(clients)
        user = _agentcore_user([_sdk_result("Please?", stop=False, stop_reason=None)])
        with pytest.raises(ConversationError) as failed:
            _converse_on_runtime(clients, tmp_path, user)
        assert (failed.value.side, failed.value.turn) == ("agent", 2)
        [session] = {c.params["runtimeSessionId"] for c in clients.calls_to("InvokeAgentRuntime")}
        assert [c.params["runtimeSessionId"] for c in clients.calls_to("StopRuntimeSession")] == [
            session
        ]


@pytest.mark.usefixtures("_no_ambient_aws")
def test_a_failing_goal_completion_tool_is_a_typed_error(tmp_path: Path) -> None:
    with StubbedAwsClients() as clients:
        add_preflight_responses(clients)
        add_invocation_response(clients)
        add_stop_response(clients)
        # The goal-completion tool runs inside the actor's act; its own Bedrock call failing
        # surfaces from act.
        user = _agentcore_user([RuntimeError("goal-completion tool: Bedrock throttled")])
        with pytest.raises(ConversationError, match="goal-completion tool") as failed:
            _converse_on_runtime(clients, tmp_path, user)
        assert (failed.value.side, failed.value.turn) == ("user", 2)
        assert len(clients.calls_to("StopRuntimeSession")) == 1
