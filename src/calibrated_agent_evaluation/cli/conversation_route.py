# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What ``generate --multi-turn`` and ``dataset run`` share: who plays the customer.

``--simulator strands`` (the default) is the ``strands_evals`` actor ``generate --multi-turn``
has always used; ``--simulator agentcore`` is AgentCore's actor simulator. Both need an extra,
which is refused by name before anything runs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any, cast, get_args

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.cli.errors import CliError, UsageError
from calibrated_agent_evaluation.synth.conversation import SimulatedUser, Simulator

__all__ = [
    "SIMULATORS",
    "UserFactory",
    "require_simulator_extras",
    "simulator_from",
    "user_factory",
]

SIMULATORS: tuple[Simulator, ...] = get_args(Simulator)

UserFactory = Callable[[str, Any, str], SimulatedUser]
"""Builds a simulated user from the opening message, the persona and the scenario description."""


def simulator_from(args: argparse.Namespace) -> Simulator:
    """The ``--simulator`` choice, ``strands`` when it is not given."""
    chosen = getattr(args, "simulator", None) or "strands"
    if chosen not in SIMULATORS:
        raise UsageError(f"--simulator must be one of {', '.join(SIMULATORS)}, got {chosen!r}")
    return cast("Simulator", chosen)


def require_simulator_extras(simulator: str) -> None:
    """Refuse a simulator whose extra is not installed, naming the extra, before any call.

    Raises:
        CliError: ``strands-agents-evals`` (for either simulator: both use its persona model) or,
            for ``agentcore``, ``bedrock-agentcore`` or ``jinja2`` is missing.
    """
    from calibrated_agent_evaluation.cli.config import probe_extra

    if not probe_extra("strands_evals").available:
        extra = "agentcore" if simulator == "agentcore" else "synth"
        raise CliError(
            f"--simulator {simulator} needs strands-agents-evals for the simulated user's persona: "
            f"{install_hint(extra)}"
        )
    if simulator == "agentcore":
        missing = [m for m in ("bedrock_agentcore", "jinja2") if not probe_extra(m).available]
        if missing:
            raise CliError(
                f"--simulator agentcore needs the agentcore extra ({', '.join(missing)} missing): "
                f"{install_hint('agentcore')}"
            )


def user_factory(
    simulator: str,
    *,
    max_turns: int,
    model: str | None = None,
    simulator_cls: Callable[..., Any] | None = None,
) -> UserFactory:
    """A factory for the chosen simulated user.

    Args:
        simulator: ``strands`` or ``agentcore``.
        max_turns: The simulator's own turn limit (the conversation loop also bounds turns).
        model: The simulated user's model id, or ``None`` for the simulator's default.
        simulator_cls: A factory mirroring the vendor simulator (a test fake).

    Returns:
        ``(opening, persona, scenario_description) -> SimulatedUser``.
    """
    if simulator == "agentcore":
        from calibrated_agent_evaluation.agentcore.simulated_user import AgentCoreSimulatedUser

        def agentcore_user(opening: str, profile: Any, description: str) -> SimulatedUser:
            return AgentCoreSimulatedUser(
                profile,
                opening,
                max_turns=max_turns,
                scenario_description=description,
                model=model,
                simulator_cls=simulator_cls,
            )

        return agentcore_user

    from calibrated_agent_evaluation.synth.borrow.simulator import StrandsSimulatedUser

    def strands_user(opening: str, profile: Any, _description: str) -> SimulatedUser:
        return StrandsSimulatedUser(
            profile, opening, max_turns=max_turns, model=model, simulator_cls=simulator_cls
        )

    return strands_user
