# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The guards keep a production-representative set representative, or say it no longer is.

Dedup and the diversity floor drop or add inputs per cell. The inputs used to keep their frame tag
and original weight, so a production-representative set whose cells were thinned unevenly stopped
matching the declared distribution while still passing ``assert_production_frame``.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.sampling.types import FrameResult, FrameRow
from calibrated_agent_evaluation.synth import RepresentativenessWarning, generate
from test_synth_helpers import DuplicateRenderer, FakeEmbedder


class _PerCellRenderer:
    """The same text for every row of a cell: dedup keeps one input per cell."""

    def render(self, cell: dict[str, str]) -> str:
        return f"the only phrasing for {cell['intent']}"


def _production(counts: dict[str, int]) -> FrameResult:
    rows = [
        FrameRow(cell={"intent": intent}, frame="production-representative")
        for intent, n in counts.items()
        for _ in range(n)
    ]
    return FrameResult(rows=rows, frame="production-representative")


def _shares(inputs: list) -> dict[str, float]:  # type: ignore[type-arg]
    total = sum(i.weight for i in inputs)
    out: dict[str, float] = {}
    for i in inputs:
        out[i.cell["intent"]] = out.get(i.cell["intent"], 0.0) + i.weight / total
    return out


def test_dedup_reweights_each_cell_back_to_its_declared_share() -> None:
    out = generate(
        _production({"refund": 3, "order": 1}), _PerCellRenderer(), embedder=FakeEmbedder()
    )
    assert len(out) == 2
    assert {i.frame for i in out} == {"production-representative"}
    assert _shares(out) == pytest.approx({"refund": 0.75, "order": 0.25})
    assert {i.cell["intent"]: i.weight for i in out} == {"refund": 3.0, "order": 1.0}


def test_a_cell_emptied_by_dedup_drops_the_production_tag() -> None:
    frame = _production({"refund": 2, "order": 1})
    with pytest.warns(RepresentativenessWarning, match="retagged 'discovery'"):
        out = generate(frame, DuplicateRenderer(), embedder=FakeEmbedder())
    assert len(out) == 1
    assert out[0].frame == "discovery"


def test_without_the_guards_the_frame_passes_through_unchanged() -> None:
    frame = _production({"refund": 3, "order": 1})
    out = generate(frame, _PerCellRenderer())
    assert [i.weight for i in out] == [1.0] * 4
    assert {i.frame for i in out} == {"production-representative"}


def test_a_discovery_frame_keeps_its_weights() -> None:
    rows = [FrameRow(cell={"intent": "refund"}, frame="discovery", weight=1.0) for _ in range(3)]
    out = generate(
        FrameResult(rows=rows, frame="discovery"), _PerCellRenderer(), embedder=FakeEmbedder()
    )
    assert [(i.frame, i.weight) for i in out] == [("discovery", 1.0)]
