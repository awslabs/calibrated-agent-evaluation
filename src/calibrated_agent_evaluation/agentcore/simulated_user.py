# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentCore's actor simulator behind the library's vendor-free ``SimulatedUser`` port.

``bedrock_agentcore``'s ``AgentCoreActorSimulator`` plays the customer the way AgentCore's own
simulated-scenario runner does: the same persona model (``strands_evals``'s ``ActorProfile``),
the same built-in system prompt (``structured_user_simulator.j2``, rendered with the SDK's
template renderer) and the same goal-completion tool. Calibrated Agent Evaluation runs it inside its
own conversation loop instead of ``SimulatedScenarioExecutor``, so the transcript is kept, failures
are typed and the session ids are the library's.

Every SDK, ``strands`` and template failure is raised as
:class:`~calibrated_agent_evaluation.agentcore.errors.AgentCoreError`, chained. The SDK is imported
when the user is built, so importing this module needs no extra.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, get_args

from calibrated_agent_evaluation._conversation import StopReason, UserTurn
from calibrated_agent_evaluation.agentcore.errors import AgentCoreError, AgentCoreExtraMissing

__all__ = ["AgentCoreSimulatedUser"]

_TEMPLATE = "structured_user_simulator.j2"
_STOP_REASONS: tuple[StopReason, ...] = get_args(StopReason)


class AgentCoreSimulatedUser:
    """The customer in a simulated conversation, played by AgentCore's actor simulator."""

    def __init__(
        self,
        actor_profile: Any,
        opening: str,
        *,
        max_turns: int,
        scenario_description: str = "",
        model: str | None = None,
        simulator_cls: Callable[..., Any] | None = None,
    ) -> None:
        """Build the simulator.

        Args:
            actor_profile: A ``strands_evals`` ``ActorProfile`` (traits, context, actor_goal).
            opening: The first customer message.
            max_turns: The simulator's own turn limit; the conversation loop also bounds turns.
            scenario_description: The scenario's description, rendered into the actor's prompt.
            model: The actor's model id; ``None`` keeps the SDK's default.
            simulator_cls: A factory mirroring ``AgentCoreActorSimulator`` (a test fake);
                ``None`` loads the real one.

        Raises:
            AgentCoreExtraMissing: ``bedrock-agentcore``, ``strands-agents-evals`` or ``jinja2``
                is not installed.
            AgentCoreError: The prompt cannot be rendered or the simulator cannot be built.
        """
        factory, tools = (simulator_cls, []) if simulator_cls is not None else _load_simulator()
        prompt = _system_prompt(actor_profile, scenario_description)
        try:
            self._simulator = factory(
                actor_profile=actor_profile,
                initial_query=opening,
                system_prompt=prompt,
                model=model,
                max_turns=max_turns,
                tools=tools,
            )
        except Exception as exc:
            raise AgentCoreError(
                f"the AgentCore actor simulator could not be built: {type(exc).__name__}: {exc}"
            ) from exc

    def act(self, agent_message: str) -> UserTurn:
        """The next customer move in answer to ``agent_message``.

        Raises:
            AgentCoreError: The simulator or its goal-completion tool failed.
        """
        try:
            result = self._simulator.act(agent_message)
        except Exception as exc:
            raise AgentCoreError(
                f"the AgentCore actor simulator failed: {type(exc).__name__}: {exc}"
            ) from exc
        message = getattr(result, "message", None)
        text = message if isinstance(message, str) and message.strip() else None
        stop = bool(getattr(result, "stop", False)) or text is None
        reason = getattr(result, "stop_reason", None)
        return UserTurn(
            message=text,
            stop=stop,
            stop_reason=(reason if reason in _STOP_REASONS else "actor_stopped") if stop else None,
        )


def _load_simulator() -> tuple[Any, list[Any]]:
    """The SDK's ``AgentCoreActorSimulator`` and the goal-completion tool its runner gives it."""
    try:
        from bedrock_agentcore.evaluation.runner import AgentCoreActorSimulator
        from strands_evals.simulation.tools.goal_completion import (
            get_conversation_goal_completion,
        )
    except ImportError as exc:
        raise AgentCoreExtraMissing("the AgentCore simulated user") from exc
    return AgentCoreActorSimulator, [get_conversation_goal_completion]


def _system_prompt(actor_profile: Any, scenario_description: str) -> str:
    """The SDK's built-in actor prompt, rendered for this persona and scenario."""
    try:
        from bedrock_agentcore.evaluation.runner.prompts import render_template_file
    except ImportError as exc:
        raise AgentCoreExtraMissing("the AgentCore simulated user") from exc
    try:
        profile = (
            actor_profile.model_dump() if hasattr(actor_profile, "model_dump") else actor_profile
        )
        return render_template_file(
            _TEMPLATE,
            actor_profile=profile,
            scenario_description=scenario_description,
            output_schema=None,
        )
    except ImportError as exc:
        # jinja2 is imported when the first template is rendered; any other import failure is not
        # a missing extra.
        if exc.name == "jinja2":
            raise AgentCoreExtraMissing("the AgentCore simulated user") from exc
        raise AgentCoreError(
            f"the AgentCore actor prompt could not be rendered: ImportError: {exc}"
        ) from exc
    except Exception as exc:
        raise AgentCoreError(
            f"the AgentCore actor prompt could not be rendered: {type(exc).__name__}: {exc}"
        ) from exc
