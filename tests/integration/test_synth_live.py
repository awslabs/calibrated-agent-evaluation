# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live multi-turn against the real strands_evals ActorSimulator.

Marked `integration` (not run in the default hermetic suite): it drives the REAL
``strands_evals.ActorSimulator`` (not the FakeActorSimulator) with a stub in-process agent, to
keep the borrow adapter honest against SDK drift — proving the ``[synth]`` surface the unit tests
stub actually behaves as ``run_conversation`` expects. No network/LLM: the "agent" is a local
callable and the ActorSimulator is constructed with a minimal ActorProfile. Run with
`just test-integration -k synth` (needs the `[synth]` extra).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_real_actor_simulator_drives_a_multi_turn_conversation() -> None:
    # Import the real SDK surface (skips cleanly if the [synth] extra is absent).
    strands_evals = pytest.importorskip("strands_evals")
    from strands_evals.types.simulation.actor import ActorProfile

    from calibrated_agent_evaluation.synth.borrow.simulator import run_conversation

    profile = ActorProfile(
        traits={"mood": "frustrated", "role": "customer"},
        context="A customer whose order arrived damaged.",
        actor_goal="Get a full refund for the damaged order.",
    )

    def agent_turn(user_message: str) -> str:
        # The real agent would run here; a deterministic in-process stub keeps the test offline.
        return f"I understand your concern about: {user_message[:40]}. Let me help."

    turns = run_conversation(
        actor_profile=profile,
        initial_query="My order arrived shattered and I want a refund.",
        agent_turn=agent_turn,
        max_turns=2,
        simulator_cls=strands_evals.ActorSimulator,
    )

    # The first turn is the rendered user query; agent turns come from our stub (route-through),
    # never authored by the simulator.
    assert turns[0] == {"role": "user", "text": "My order arrived shattered and I want a refund."}
    agent_turns = [t for t in turns if t["role"] == "agent"]
    assert agent_turns, "the real agent (stub) must contribute at least one turn"
    assert all(t["text"].startswith("I understand your concern") for t in agent_turns)
