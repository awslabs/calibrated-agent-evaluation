# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-turn persona simulation via the borrowed ``strands_evals`` ``ActorSimulator``.

This is the ONE vendor-touching module under ``calibrated_agent_evaluation.synth`` (confined here by
the boundary test's ``_SYNTH_ROOTS`` scan). ``strands_evals`` is imported **lazily** inside the
function so ``import calibrated_agent_evaluation.synth`` succeeds without the ``[synth]`` extra, and
every vendor exception is caught at this boundary and re-raised as a typed :class:`SynthError`
no raw ``strands_evals`` exception escapes the ``synth/`` public surface.

The route-through contract: the simulator produces each **user** turn; the
caller-supplied ``agent_turn`` callable (the real agent, via the executor) produces each **agent**
turn. This module never authors an agent turn.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast, get_args

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.synth.conversation import (
    StopReason,
    UserTurn,
    drive_conversation,
    message_text,
)
from calibrated_agent_evaluation.synth.errors import SynthError

__all__ = [
    "StrandsSimulatedUser",
    "actor_profile_from_cell",
    "actor_profile_from_dict",
    "run_conversation",
]

_PROFILE_FIELDS = ("traits", "context", "actor_goal")


def _load_actor_simulator() -> Any:
    """Lazy-import ``strands_evals.ActorSimulator``; raise ``SynthError`` if absent."""
    try:
        from strands_evals import ActorSimulator
    except ImportError as exc:
        raise SynthError(
            f"multi-turn synthesis needs strands-agents-evals. Install via: {install_hint('synth')}"
        ) from exc
    return ActorSimulator


def _load_actor_profile() -> Any:
    """Lazy-import ``strands_evals``'s ``ActorProfile``; raise ``SynthError`` if absent."""
    try:
        from strands_evals.types.simulation.actor import ActorProfile
    except ImportError as exc:
        raise SynthError(
            f"multi-turn synthesis needs strands-agents-evals. Install via: {install_hint('synth')}"
        ) from exc
    return ActorProfile


def actor_profile_from_dict(doc: Mapping[str, Any]) -> Any:
    """Build the simulator's persona from a plain mapping (a ``--persona`` file's contents).

    ``traits`` is a mapping of persona attributes, ``context`` the background and ``actor_goal``
    what the simulated person is trying to achieve. A missing or wrong-typed field is a typed
    :class:`SynthError`, never a vendor validation error.
    """
    missing = [field for field in _PROFILE_FIELDS if field not in doc]
    if missing:
        raise SynthError(
            f"a persona needs {', '.join(_PROFILE_FIELDS)}; {', '.join(missing)} missing"
        )
    profile_cls = _load_actor_profile()
    try:
        return profile_cls(
            traits=dict(doc["traits"]),
            context=str(doc["context"]),
            actor_goal=str(doc["actor_goal"]),
        )
    except Exception as exc:
        raise SynthError(f"the persona is not usable: {type(exc).__name__}: {exc}") from exc


def actor_profile_from_cell(cell: Mapping[str, str]) -> Any:
    """The default persona for a taxonomy cell: the cell is the persona's traits.

    The frame's cell is the only thing known about the person behind a synthesized input, so it is
    what the simulated one is given; a caller with a real persona passes it with ``--persona``.
    """
    described = ", ".join(f"{key}={value}" for key, value in sorted(cell.items())) or "no dimension"
    return actor_profile_from_dict(
        {
            "traits": dict(cell),
            "context": f"A customer whose situation exercises {described}.",
            "actor_goal": "Get the request in the first message resolved, or hear a clear refusal.",
        }
    )


class StrandsSimulatedUser:
    """The ``strands_evals`` ``ActorSimulator`` behind the vendor-free ``SimulatedUser`` port.

    Built once per conversation from the persona and the opening message. ``act`` answers the
    agent's latest reply with the next customer move; the simulator's own stop (``has_next()``
    false, or a structured response with ``stop`` set) becomes a stopping :class:`UserTurn` that
    keeps the simulator's stop reason and any last message it wrote. ``has_next()`` false after
    the simulator has answered ``max_turns`` times is ``max_turns``; otherwise it is
    ``actor_stopped``. Every vendor failure is re-raised as a
    typed :class:`SynthError`, chained.
    """

    def __init__(
        self,
        actor_profile: Any,
        opening: str,
        *,
        max_turns: int | None = None,
        model: str | None = None,
        simulator_cls: Callable[..., Any] | None = None,
    ) -> None:
        """Build the simulator.

        Args:
            actor_profile: A ``strands_evals`` ``ActorProfile``.
            opening: The first customer message.
            max_turns: Passed to the simulator when given; the conversation loop bounds turns
                itself, so ``None`` keeps the simulator's own default (and a simulator that runs
                out of turns then reports ``actor_stopped``).
            model: The simulator's model id when given; ``None`` keeps its default.
            simulator_cls: A factory mirroring ``ActorSimulator`` (a test fake); ``None`` loads
                the real one.

        Raises:
            SynthError: ``strands-agents-evals`` is not installed, or the simulator cannot be built.
        """
        factory = simulator_cls or _load_actor_simulator()
        self._max_turns = max_turns
        self._acts = 0
        options: dict[str, Any] = {}
        if max_turns is not None:
            options["max_turns"] = max_turns
        if model is not None:
            options["model"] = model
        try:
            self._simulator = factory(actor_profile=actor_profile, initial_query=opening, **options)
        except Exception as exc:
            raise SynthError(f"input simulator failed: {type(exc).__name__}: {exc}") from exc

    def act(self, agent_message: str) -> UserTurn:
        """The next customer move in answer to ``agent_message``.

        Raises:
            SynthError: The simulator failed.
        """
        try:
            if not self._simulator.has_next():
                return UserTurn(message=None, stop=True, stop_reason=self._exhausted_reason())
            result = self._simulator.act(agent_message)
        except Exception as exc:
            raise SynthError(f"input simulator failed: {type(exc).__name__}: {exc}") from exc
        self._acts += 1
        text = _extract_text(result)
        structured = getattr(result, "structured_output", None)
        stopped = bool(getattr(structured, "stop", False)) or not text
        return UserTurn(
            message=text or None,
            stop=stopped,
            stop_reason=_stop_reason(getattr(structured, "stop_reason", None)) if stopped else None,
        )

    def _exhausted_reason(self) -> StopReason:
        """Why a simulator that has no next turn stopped: its own turn limit, or its own choice."""
        if self._max_turns is not None and self._acts >= self._max_turns:
            return "max_turns"
        return "actor_stopped"


def run_conversation(
    actor_profile: Any,
    initial_query: str,
    agent_turn: Callable[[str], str],
    *,
    max_turns: int = 6,
    simulator_cls: Callable[..., Any] | None = None,
) -> list[dict[str, str]]:
    """Drive a multi-turn conversation between the simulated persona and the real agent.

    ``actor_profile`` is a ``strands_evals`` ``ActorProfile`` (the persona). ``initial_query`` is
    the first user turn (rendered from the taxonomy cell). ``agent_turn`` is the caller's real
    agent; given the latest user message, it returns the agent's reply. Returns the ordered turn
    list ``[{"role": "user"|"agent", "text": ...}, ...]``; user turns come from the simulator,
    agent turns from ``agent_turn`` (this function never authors an agent turn).

    ``simulator_cls`` injects the simulator factory (a test fake mirroring ``ActorSimulator``); when
    ``None`` the real ``strands_evals.ActorSimulator`` is lazily loaded, raising :class:`SynthError`
    if the ``[synth]`` extra is absent. Any failure is raised as a :class:`SynthError` (a
    ``ConversationError`` once the conversation has started), chained, so no raw vendor exception
    escapes.
    """
    user = StrandsSimulatedUser(
        actor_profile, initial_query, max_turns=max_turns, simulator_cls=simulator_cls
    )
    history, _reason = drive_conversation(
        user,
        lambda _turn, so_far: agent_turn(message_text(so_far[-1])),
        conversation_id="simulated-conversation",
        opening=initial_query,
        max_turns=max_turns,
    )
    return [
        {"role": "agent" if m["role"] == "assistant" else "user", "text": message_text(m)}
        for m in history
    ]


def _stop_reason(raw: Any) -> StopReason:
    """The simulator's stop reason in the engine's vocabulary.

    An unknown or missing reason is ``actor_stopped``.
    """
    if raw in get_args(StopReason):
        return cast("StopReason", raw)
    return "actor_stopped"


def _extract_text(result: Any) -> str:
    """Pull the user-turn text out of an ``ActorSimulator.act`` result (a ``strands.AgentResult``).

    The simulator asks its model for structured output, so the next user message is
    ``result.structured_output.message`` and is ``None`` when the simulator has decided to stop
    (verified against ``strands-agents-evals==1.6.0``: ``ActorResponse.message`` is "Provide when
    stop=false; set to null when stop=true"). ``str(result)`` is not that message: an
    ``AgentResult`` with structured output stringifies to the whole JSON document, reasoning field
    and all, which once went into the transcript as what the customer said.

    A plain ``str`` (a test fake) passes through unchanged; anything else falls back to
    ``str(result)`` rather than dropping a turn.
    """
    if isinstance(result, str):
        return result
    structured = getattr(result, "structured_output", None)
    if structured is not None:
        message = getattr(structured, "message", None)
        return "" if message is None else str(message)
    return str(result)
