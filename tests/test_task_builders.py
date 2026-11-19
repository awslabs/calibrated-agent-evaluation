# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete TaskBuilder tests.

Proves the synthesis-to-executor composition contract: the Strands builder threads the synthesized
``taxonomy_cell`` into the ``StrandsTask`` so the produced Trace stays taxonomy-attributable, and
``synth._verify_frame_propagation`` rejects a mismatch. Empty inputs are refused. Tests assert on
the built task's observable fields, not on internal calls.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.execute.strands_executor import StrandsTask
from calibrated_agent_evaluation.execute.task_builders import InspectTaskBuilder, StrandsTaskBuilder
from calibrated_agent_evaluation.synth.errors import SynthError
from calibrated_agent_evaluation.synth.types import SynthInput


def _input(cell: dict[str, str], text: str = "a synthesized input") -> SynthInput:
    return SynthInput(text=text, cell=cell, frame="discovery")


def test_strands_builder_threads_taxonomy_cell():
    """The built StrandsTask carries the synthesized input's taxonomy_cell (the
    contract)."""
    cell = {"feature": "refund", "persona": "consumer"}
    builder = StrandsTaskBuilder(build_topology=lambda _model: lambda _inp: None)

    task = builder.build([_input(cell)])

    assert isinstance(task, StrandsTask)
    assert task.taxonomy_cell == cell
    assert task.input == "a synthesized input"


def test_cell_mismatch_is_rejected_by_verify_frame_propagation():
    """Negative: a Trace whose cell isn't among the inputs is rejected (the drop-in guard).

    Exercises the exact rejection synth relies on — a builder/executor that produced a Trace with a
    foreign cell is refused, so a silently-altered population cannot slip through.
    """
    from calibrated_agent_evaluation.schema.types import Trace
    from calibrated_agent_evaluation.synth.generate import _verify_frame_propagation

    synth_input = _input({"feature": "refund"})
    foreign = Trace(
        trace_id="t1",
        frame="discovery",
        ownership="owned",
        input={"text": "in"},
        output={"text": "out"},
        taxonomy_cell={"feature": "SOMETHING_ELSE"},
        spans=[],
    )

    with pytest.raises(SynthError, match="taxonomy propagation violated"):
        _verify_frame_propagation([foreign], [synth_input])


def test_strands_builder_refuses_empty_inputs():
    """Negative/edge: building from [] raises SynthError, not a degenerate task."""
    builder = StrandsTaskBuilder(build_topology=lambda _model: None)
    with pytest.raises(SynthError, match="received no inputs"):
        builder.build([])


def test_inspect_builder_refuses_empty_inputs():
    """Negative/edge: the Inspect builder also refuses an empty inputs list."""
    builder = InspectTaskBuilder(solver=object())
    with pytest.raises(SynthError, match="received no inputs"):
        builder.build([])
