# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""strands-agents-evals still offers the simulator surface the synthesis borrow calls."""

from __future__ import annotations

import inspect
from typing import Any, ClassVar

import pytest

_PROFILE_FIELDS = ("traits", "context", "actor_goal")


_RESPONSE_FIELDS = ("message", "stop")


def _surface_problems(simulator: type, profile: type, response: type | None = None) -> list[str]:
    """Every way ``simulator``/``profile`` differ from what ``synth/borrow/simulator.py`` calls."""
    problems: list[str] = []
    params = inspect.signature(simulator.__init__).parameters
    for keyword in ("actor_profile", "initial_query"):
        param = params.get(keyword)
        if param is None or param.kind is inspect.Parameter.POSITIONAL_ONLY:
            problems.append(f"ActorSimulator does not accept {keyword}= as a keyword")
    for method in ("has_next", "act"):
        attr = getattr(simulator, method, None)
        if attr is None:
            problems.append(f"ActorSimulator.{method} is missing")
        elif inspect.iscoroutinefunction(attr):
            problems.append(f"ActorSimulator.{method} is now a coroutine function")
    act = getattr(simulator, "act", None)
    if act is not None and len(inspect.signature(act).parameters) < 2:
        problems.append("ActorSimulator.act no longer takes the agent message")
    fields: dict[str, Any] = getattr(profile, "model_fields", {})
    problems.extend(f"ActorProfile.{f} is missing" for f in _PROFILE_FIELDS if f not in fields)
    if response is not None:
        # The borrow reads the next user turn off `act(...).structured_output.message`, so a
        # rename there would silently put the simulator's whole document into the transcript.
        reply: dict[str, Any] = getattr(response, "model_fields", {})
        problems.extend(f"ActorResponse.{f} is missing" for f in _RESPONSE_FIELDS if f not in reply)
    return problems


def test_the_installed_simulator_matches_the_borrowed_surface() -> None:
    strands_evals = pytest.importorskip("strands_evals")
    from strands_evals.types.simulation import ActorResponse
    from strands_evals.types.simulation.actor import ActorProfile

    assert _surface_problems(strands_evals.ActorSimulator, ActorProfile, ActorResponse) == []


class _Profile:
    model_fields: ClassVar[dict[str, Any]] = {"traits": None, "context": None, "actor_goal": None}


def test_a_simulator_without_act_is_reported() -> None:
    class NoAct:
        def __init__(self, actor_profile: object, initial_query: str) -> None: ...

        def has_next(self) -> bool:
            return False

    assert _surface_problems(NoAct, _Profile) == ["ActorSimulator.act is missing"]


def test_an_async_act_and_a_renamed_profile_field_are_reported() -> None:
    class AsyncAct:
        def __init__(self, actor_profile: object, initial_query: str) -> None: ...

        def has_next(self) -> bool:
            return False

        async def act(self, agent_message: str) -> str:
            return agent_message

    class RenamedProfile:
        model_fields: ClassVar[dict[str, Any]] = {"traits": None, "context": None, "goal": None}

    assert _surface_problems(AsyncAct, RenamedProfile) == [
        "ActorSimulator.act is now a coroutine function",
        "ActorProfile.actor_goal is missing",
    ]


def test_a_renamed_structured_reply_field_is_reported() -> None:
    class NoMessage:
        model_fields: ClassVar[dict[str, Any]] = {"text": None, "stop": None}

    class Fine:
        def __init__(self, actor_profile: object, initial_query: str) -> None: ...

        def has_next(self) -> bool:
            return False

        def act(self, agent_message: str) -> str:
            return agent_message

    assert _surface_problems(Fine, _Profile, NoMessage) == ["ActorResponse.message is missing"]


def test_a_positional_only_initial_query_is_reported() -> None:
    class PositionalOnly:
        def __init__(self, actor_profile: object, initial_query: str, /) -> None: ...

        def has_next(self) -> bool:
            return False

        def act(self, agent_message: str) -> str:
            return agent_message

    assert _surface_problems(PositionalOnly, _Profile) == [
        "ActorSimulator does not accept actor_profile= as a keyword",
        "ActorSimulator does not accept initial_query= as a keyword",
    ]
