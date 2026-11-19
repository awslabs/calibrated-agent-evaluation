# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-shot generation tests."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.sampling.types import FrameResult, FrameRow
from calibrated_agent_evaluation.synth import SynthError, generate, generate_and_run
from test_synth_helpers import BlankRenderer, FakeRenderer, SpyExecutor


def _frame(cells: list[dict[str, str]], frame: str = "discovery") -> FrameResult:
    rows = [FrameRow(cell=c, frame=frame) for c in cells]  # type: ignore[arg-type]
    return FrameResult(rows=rows, frame=frame)  # type: ignore[arg-type]


class TestGenerateSingleShot:
    """One SynthInput per FrameRow, non-empty, frame/cell matching."""

    def test_one_input_per_row_frame_tagged(self) -> None:
        frame = _frame([{"feature": "returns"}, {"feature": "billing"}, {"feature": "shipping"}])
        out = generate(frame, FakeRenderer())
        assert len(out) == 3
        for synth, row in zip(out, frame.rows, strict=True):
            assert synth.text.strip()
            assert synth.cell == row.cell
            assert synth.frame == row.frame == "discovery"

    def test_two_step_renders_from_the_cell(self) -> None:
        renderer = FakeRenderer()
        frame = _frame([{"feature": "returns", "persona": "angry"}])
        out = generate(frame, renderer)
        # Step (ii) rendered from the cell the row carried (step (i)).
        assert renderer.calls == [{"feature": "returns", "persona": "angry"}]
        assert "returns" in out[0].text


class TestGenerateRefusals:
    """Negative and edge paths."""

    def test_empty_render_refused_naming_cell(self) -> None:
        frame = _frame([{"feature": "returns"}])
        with pytest.raises(SynthError, match="returns"):
            generate(frame, BlankRenderer())

    def test_empty_frame_yields_empty_not_fabricated(self) -> None:
        frame = _frame([])
        assert generate(frame, FakeRenderer()) == []

    def test_output_cells_are_exactly_the_frame_cells(self) -> None:
        cells = [{"a": "1"}, {"a": "2"}]
        out = generate(_frame(cells), FakeRenderer())
        assert [s.cell for s in out] == cells


class TestGenerateOnlyTierDoesNotRoute:
    """Generate performs zero agent invocations, even with an executor in scope."""

    def test_spy_executor_never_invoked_by_generate(self) -> None:
        spy = SpyExecutor()
        frame = _frame([{"feature": "returns"}])
        generate(frame, FakeRenderer())
        assert spy.called is False

    def test_generate_only_returns_inputs_not_traces(self) -> None:
        frame = _frame([{"feature": "returns"}])
        out = generate(frame, FakeRenderer())
        # The generate-only tier yields SynthInputs (no executor touched); route-through is a
        # separate call (generate_and_run) that requires an executor.
        assert all(hasattr(s, "text") and hasattr(s, "cell") for s in out)
        with pytest.raises(SynthError):
            generate_and_run(frame, FakeRenderer(), None, None)
