# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedup + diversity-floor guard tests."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.synth import DiversityFloorWarning, dedup, enforce_diversity_floor
from calibrated_agent_evaluation.synth.types import SynthInput
from test_synth_helpers import DuplicateRenderer, FakeEmbedder, FakeRenderer


def _inputs(texts: list[str], cell: dict[str, str] | None = None) -> list[SynthInput]:
    cell = cell or {"feature": "returns"}
    return [SynthInput(text=t, cell=cell, frame="discovery") for t in texts]


class TestDedup:
    """Near-duplicate inputs collapse to one representative; distinct survive."""

    def test_identical_inputs_collapse_to_one(self) -> None:
        out = dedup(_inputs(["same", "same", "same"]), FakeEmbedder(), threshold=0.05)
        assert len(out) == 1
        assert out[0].text == "same"  # first occurrence kept

    def test_distinct_inputs_all_survive(self) -> None:
        out = dedup(_inputs(["alpha", "beta", "gamma"]), FakeEmbedder(), threshold=0.05)
        assert len(out) == 3


class TestDedupEdges:
    """Edges: threshold=0 keeps everything; empty input is empty."""

    def test_threshold_zero_keeps_all_even_identical(self) -> None:
        out = dedup(_inputs(["same", "same", "same"]), FakeEmbedder(), threshold=0.0)
        assert len(out) == 3

    def test_empty_inputs_return_empty(self) -> None:
        assert dedup([], FakeEmbedder(), threshold=0.05) == []


class TestDiversityFloor:
    """Per-cell distinct minimum, with bounded regeneration."""

    def test_met_floor_no_warning(self, recwarn: pytest.WarningsRecorder) -> None:
        # Three already-distinct inputs, floor 3 → met immediately, no regen, no warning.
        out = enforce_diversity_floor(
            _inputs(["a", "b", "c"]), FakeEmbedder(), FakeRenderer(), floor=3, max_regen=5
        )
        assert len(out) == 3
        assert not [w for w in recwarn.list if issubclass(w.category, DiversityFloorWarning)]

    def test_floor_one_is_noop(self) -> None:
        inputs = _inputs(["a", "b"])
        assert enforce_diversity_floor(inputs, FakeEmbedder(), FakeRenderer(), floor=1) is inputs


class TestDiversityFloorShortfall:
    """An unmeetable floor warns explicitly, never silently under-delivers."""

    def test_unmeetable_floor_warns_naming_cell(self) -> None:
        # DuplicateRenderer only ever emits one distinct text, so regeneration can't add variety.
        with pytest.warns(DiversityFloorWarning, match="returns"):
            out = enforce_diversity_floor(
                _inputs(["identical input every time"]),
                FakeEmbedder(),
                DuplicateRenderer(),
                floor=3,
                max_regen=3,
            )
        # It surfaced the shortfall rather than fabricating distinct inputs.
        assert len(out) < 3
