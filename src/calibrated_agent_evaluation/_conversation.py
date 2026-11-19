# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The value types both sides of a simulated conversation share, kept outside the ``synth`` plane.

The AgentCore adapters need these types at run time (the Runtime executor checks a task's
:class:`ConversationTurn`, the AgentCore simulated user returns a :class:`UserTurn`), and the
``[agentcore]`` extra does not install the sampling stack that importing anything under
``calibrated_agent_evaluation.synth`` loads. So the types live here, and
:mod:`calibrated_agent_evaluation.synth.conversation` re-exports them under its own names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel

__all__ = ["ConversationTurn", "SimulatedUser", "StopReason", "UserTurn"]

StopReason = Literal["goal_completed", "actor_stopped", "max_turns"]


class UserTurn(CalibratedAgentEvaluationModel):
    """The simulated user's next move: a message, or a stop with its reason.

    A stop may carry a last message (a customer who says "Thanks, that's all" and leaves); the
    conversation keeps it, and no agent turn follows it. A turn with ``stop=False`` but no message
    is read as a stop (``actor_stopped``): there is no customer message to send, and inventing one
    would put a turn in the transcript nobody took.
    """

    message: str | None
    stop: bool
    stop_reason: StopReason | None = None


class SimulatedUser(Protocol):
    """The port a simulated-user adapter implements."""

    def act(self, agent_message: str) -> UserTurn:
        """Answer the agent's latest reply with the next customer move."""
        ...


@dataclass(frozen=True)
class ConversationTurn:
    """Which conversation and which agent turn (1-based) a task belongs to."""

    conversation_id: str
    turn: int
