# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Types tests — SynthInput frame/cell propagation."""

from __future__ import annotations

from calibrated_agent_evaluation.sampling.types import FrameRow
from calibrated_agent_evaluation.synth.types import SynthInput


class TestSynthInputPropagation:
    """A SynthInput carries its originating FrameRow's cell/frame/weight unchanged."""

    def test_propagates_cell_frame_weight_from_frame_row(self) -> None:
        row = FrameRow(
            cell={"feature": "returns", "persona": "angry"}, frame="discovery", weight=2.0
        )
        synth = SynthInput(
            text="a rendered input", cell=row.cell, frame=row.frame, weight=row.weight
        )
        assert synth.cell == row.cell
        assert synth.frame == row.frame
        assert synth.weight == row.weight
        assert synth.frame in ("discovery", "production-representative")

    def test_single_shot_has_no_turns(self) -> None:
        synth = SynthInput(text="x", cell={"a": "1"}, frame="discovery")
        assert synth.turns is None

    def test_frozen_cannot_be_mutated_after_creation(self) -> None:
        synth = SynthInput(text="x", cell={"a": "1"}, frame="discovery")
        import dataclasses

        try:
            synth.text = "y"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("SynthInput must be frozen (immutable frame tagging)")
