# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error + warning family for the input-synthesis engine.

Every synthesis failure is a typed, explained refusal (no silent failure).
Hard failures raise :class:`SynthError` (a data-plane error); directional / short-fall
conditions emit a typed :class:`UserWarning` per the enforcement tiering (a warning is
warn-and-allow; a refusal is a hard stop). ``SynthError`` subclasses
:class:`~calibrated_agent_evaluation.schema.errors.CalibratedAgentEvaluationDataError` so callers
can catch the whole data-plane family at the boundary.
"""

from __future__ import annotations

from typing import Literal

from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError


class SynthError(CalibratedAgentEvaluationDataError):
    """A hard synthesis refusal.

    Raised when: outputs are requested without a real executor (outputs are
    never fabricated); the executor run fails; a borrowed ``strands_evals`` call fails;
    multi-turn is requested without the ``[synth]`` extra; or a rendered input
    is empty. The message explains the cause; the chained ``__cause__`` preserves
    the original traceback.
    """


class ConversationError(SynthError):
    """A simulated conversation failed part way: the simulated user or the agent raised.

    Carries the scenario (or conversation) id, the 1-based agent-turn number and the side that
    failed (``"user"`` for the simulated user, ``"agent"`` for the agent under test), so the
    message and a caller can name exactly where the conversation stopped. The cause is chained.
    No trace of the run is stored.
    """

    def __init__(
        self, message: str, *, scenario_id: str, turn: int, side: Literal["user", "agent"]
    ) -> None:
        super().__init__(message)
        self.scenario_id = scenario_id
        self.turn = turn
        self.side = side


class ConversationReplayDiverged(ConversationError):
    """A replayed conversation asked for a turn the recording does not hold.

    Raised when the cassette has no recording for a simulated-user or agent turn, or when the
    agent's reply differs from the one the simulated user's recorded turn answered, so replay would
    have to invent a conversation that never happened.
    """


class RiskZoneWarning(UserWarning):
    """Non-fatal high-stakes-domain warning.

    Emitted (warn-and-allow) when a taxonomy value names a risk-zone domain. Synthetic *inputs*
    are allowed everywhere; the "block synthetic labels" half is structural; the
    engine has no label-writing path.
    """


class DiversityFloorWarning(UserWarning):
    """Non-fatal per-cell diversity short-fall warning.

    Emitted when a cell cannot meet its distinct-input floor after the configured regeneration
    budget; surfaced explicitly rather than silently under-delivering (conventions: "no silent
    truncation").
    """


class RepresentativenessWarning(UserWarning):
    """Non-fatal: the guards removed every input of a production-representative cell.

    Dedup or the diversity floor left a cell of a production-representative frame with no input,
    so no weighting can restore the declared distribution. The inputs are retagged ``discovery``
    (never read as headline rates) and this warning names the emptied cells.
    """
