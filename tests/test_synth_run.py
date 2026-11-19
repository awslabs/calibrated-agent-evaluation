# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route-through tests."""

from __future__ import annotations

from typing import Any

import pytest

from calibrated_agent_evaluation.sampling.types import FrameResult, FrameRow
from calibrated_agent_evaluation.schema.types import Trace
from calibrated_agent_evaluation.synth import SynthError, generate_and_run
from calibrated_agent_evaluation.synth.types import SynthInput
from test_synth_helpers import (
    EXECUTOR_SENTINEL,
    FakeRenderer,
    MislabelingExecutor,
    RaisingExecutor,
    StubExecutor,
    StubTaskBuilder,
)


def _frame(cells: list[dict[str, str]], frame: str = "discovery") -> FrameResult:
    rows = [FrameRow(cell=c, frame=frame) for c in cells]  # type: ignore[arg-type]
    return FrameResult(rows=rows, frame=frame)  # type: ignore[arg-type]


class TestRouteThrough:
    """Render → build task → executor.run → frame-tagged validated Traces."""

    def test_returns_frame_tagged_traces(self) -> None:
        frame = _frame([{"feature": "returns"}, {"feature": "billing"}])
        traces = generate_and_run(frame, FakeRenderer(), StubExecutor(), StubTaskBuilder())
        assert len(traces) == 2
        for trace, row in zip(traces, frame.rows, strict=True):
            assert trace.frame == row.frame == "discovery"
            assert trace.taxonomy_cell == row.cell

    def test_task_builder_receives_the_synth_inputs(self) -> None:
        builder = StubTaskBuilder()
        generate_and_run(_frame([{"a": "1"}]), FakeRenderer(), StubExecutor(), builder)
        assert len(builder.built) == 1
        assert builder.built[0][0].cell == {"a": "1"}


class TestNeverFabricatesOutput:
    """The routed output is the executor's, never the renderer's."""

    def test_output_is_executor_sentinel_not_renderer_text(self) -> None:
        renderer = FakeRenderer(prefix="RENDERED")
        traces = generate_and_run(_frame([{"a": "1"}]), renderer, StubExecutor(), StubTaskBuilder())
        assert traces[0].output == EXECUTOR_SENTINEL
        # The renderer text appears nowhere in the output — proves no fabrication.
        assert "RENDERED" not in str(traces[0].output)


class TestMissingExecutorRefuses:
    """Outputs require a real executor — never fabricated."""

    def test_none_executor_raises_saying_never_fabricates(self) -> None:
        with pytest.raises(SynthError, match="never fabricates outputs"):
            generate_and_run(_frame([{"a": "1"}]), FakeRenderer(), None, StubTaskBuilder())


class TestExecutorFailureWrapped:
    """An executor exception is wrapped in SynthError (chained), no partial trace."""

    def test_executor_raise_wrapped_and_chained(self) -> None:
        with pytest.raises(SynthError, match="agent run failed") as exc_info:
            generate_and_run(
                _frame([{"a": "1"}]), FakeRenderer(), RaisingExecutor(), StubTaskBuilder()
            )
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)


class TestFramePropagationInvariant:
    """Negative: a trace whose frame/cell don't match the input is rejected, not accepted."""

    def test_mislabeled_trace_rejected(self) -> None:
        # The frame requests production-representative cells; the executor returns a discovery
        # trace with an alien cell → propagation invariant violated → refuse, emit nothing.
        frame = _frame([{"feature": "returns"}], frame="production-representative")
        with pytest.raises(SynthError, match="propagation"):
            generate_and_run(frame, FakeRenderer(), MislabelingExecutor(), StubTaskBuilder())


# --- every synthesized input must reach the agent ----------------------------


class _OneTracePerTaskExecutor:
    """An executor that returns exactly one trace per task it is handed.

    Which is what a real one-input-per-task framework does, and what makes the collapse visible:
    the trace count equals the executor-call count, so N inputs collapsing to 1 task yields 1 trace.
    """

    def __init__(self) -> None:
        self.tasks: list[object] = []

    def run(self, task: object) -> list[Trace]:
        self.tasks.append(task)
        cell = dict(getattr(task, "taxonomy_cell", {}) or {})
        return [
            Trace(
                trace_id=f"SYN-{len(self.tasks):04d}",
                frame="discovery",
                ownership="probed",
                input={"query": str(getattr(task, "input", ""))},
                output=dict(EXECUTOR_SENTINEL),
                taxonomy_cell=cell,
                endpoint_fingerprint="one-per-task",
            )
        ]


def test_the_strands_builder_routes_every_input_not_just_the_first() -> None:
    """The ticket's case, with the REAL `StrandsTaskBuilder`.

    `StrandsTask` carries exactly one `input` string, so `build` could only ever represent
    `inputs[0]` — and `generate_and_run` called it once and ran the single task it got back. So a
    3-row frame was evaluated at 1/3 of its data, and nothing caught it: the one returned trace
    had a valid frame and a valid cell, so `_verify_frame_propagation` passed.
    """
    from calibrated_agent_evaluation.execute.task_builders import StrandsTaskBuilder

    frame = _frame([{"feature": "returns"}, {"feature": "billing"}, {"feature": "shipping"}])
    executor = _OneTracePerTaskExecutor()
    builder = StrandsTaskBuilder(build_topology=lambda _model: object())

    traces = generate_and_run(frame, FakeRenderer(), executor, builder)

    assert len(executor.tasks) == 3, "one task per input, not one task for the batch"
    assert len(traces) == 3
    assert sorted(t.taxonomy_cell["feature"] for t in traces) == [
        "billing",
        "returns",
        "shipping",
    ], "every cell must reach the agent, not just the first"


def test_the_strands_builder_refuses_a_multi_input_build() -> None:
    """`build` no longer silently keeps `inputs[0]`: a caller bypassing `build_all` is refused.

    Belt and braces with `build_all`. The collapse was possible because a lossy call looked
    successful, so the lossy call now raises.
    """
    from calibrated_agent_evaluation.execute.task_builders import StrandsTaskBuilder
    from calibrated_agent_evaluation.synth import SynthInput

    builder = StrandsTaskBuilder(build_topology=lambda _model: object())
    two = [
        SynthInput(text="a", cell={"feature": "returns"}, frame="discovery"),
        SynthInput(text="b", cell={"feature": "billing"}, frame="discovery"),
    ]
    with pytest.raises(SynthError, match="carries exactly one"):
        builder.build(two)


def test_fewer_traces_than_inputs_is_refused_structurally() -> None:
    """The guard that catches any future builder/executor pair that drops inputs.

    Every returned trace here is individually valid — right frame, right cell — which is precisely
    how the Strands collapse evaded the existing checks. The count is the only thing that sees it.
    """

    class _DropsAllButFirst:
        def build_all(self, inputs: list[SynthInput]) -> list[Any]:
            return [inputs[0]]

        def build(self, inputs: list[SynthInput]) -> Any:
            return inputs[0]

    frame = _frame([{"feature": "returns"}, {"feature": "billing"}])
    with pytest.raises(SynthError, match="dropped inputs"):
        generate_and_run(frame, FakeRenderer(), StubExecutor(), _DropsAllButFirst())


def test_a_task_yielding_several_traces_is_not_refused() -> None:
    """The refusal is one-sided on purpose: MORE traces than inputs is not a drop.

    One task can legitimately produce several traces (the multi-turn path does), so an `==` check
    would refuse correct behaviour. Without this, the count guard would be a new defect.
    """

    class _TwoTracesPerInput:
        def build_all(self, inputs: list[SynthInput]) -> list[Any]:
            return [inputs, inputs]  # two executor calls over the same inputs

        def build(self, inputs: list[SynthInput]) -> Any:
            return inputs

    frame = _frame([{"feature": "returns"}])
    traces = generate_and_run(frame, FakeRenderer(), StubExecutor(), _TwoTracesPerInput())
    assert len(traces) == 2


def test_the_inspect_builder_still_batches_into_one_task() -> None:
    """Inspect must not be turned into N evals: its Task already holds one Sample per input.

    `build_all` exists so the batching decision belongs to the builder. If it had been fixed by
    looping in `generate_and_run` instead, Inspect would run N separate evals and scatter N logs.
    """
    pytest.importorskip("inspect_ai")
    from calibrated_agent_evaluation.execute.task_builders import InspectTaskBuilder
    from calibrated_agent_evaluation.synth import SynthInput

    builder = InspectTaskBuilder(solver=object())
    inputs = [
        SynthInput(text="a", cell={"feature": "returns"}, frame="discovery"),
        SynthInput(text="b", cell={"feature": "billing"}, frame="discovery"),
    ]
    tasks = builder.build_all(inputs)
    assert len(tasks) == 1, "one batched Task, not one per input"
    assert len(tasks[0].dataset) == 2, "carrying a Sample per input"
