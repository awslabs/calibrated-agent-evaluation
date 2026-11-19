# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AgentCore task builder makes one runtime task per synthesized input, and drops none."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("boto3")

from _agentcore_runtime_fakes import RUNTIME_ARN, FakeRunSpanReader, StubbedAwsClients
from _agentcore_runtime_queue import hide_ambient_aws, queue_preflight, queue_run
from calibrated_agent_evaluation.agentcore import (
    AgentCoreRuntimeExecutor,
    AgentCoreRuntimeTask,
    AgentCoreTaskBuilder,
    RuntimeTarget,
)
from calibrated_agent_evaluation.sampling.types import FrameResult, FrameRow
from calibrated_agent_evaluation.synth import SynthError, SynthInput, generate_and_run

_CELLS = (
    {"feature": "claim_adjudication", "persona": "P2_homeowner_absent"},
    {"feature": "claim_adjudication", "persona": "P1_consumer"},
    {"feature": "fraud_review", "persona": "P1_consumer"},
)


@pytest.fixture(autouse=True)
def _no_ambient_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hide_ambient_aws(monkeypatch, tmp_path)


def _input(cell: dict[str, str], text: str) -> SynthInput:
    return SynthInput(text=text, cell=cell, frame="discovery")


def test_build_all_makes_one_task_per_input_with_its_text_and_cell() -> None:
    inputs = [_input(cell, f"question {n}") for n, cell in enumerate(_CELLS)]

    tasks = AgentCoreTaskBuilder().build_all(inputs)

    assert tasks == [
        AgentCoreRuntimeTask(prompt=f"question {n}", taxonomy_cell=cell)
        for n, cell in enumerate(_CELLS)
    ]
    # Each task holds its own copy of the cell.
    assert all(
        task.taxonomy_cell is not item.cell for task, item in zip(tasks, inputs, strict=True)
    )


class _CellRenderer:
    def render(self, cell: dict[str, str]) -> str:
        return f"A customer request exercising {', '.join(f'{k}={v}' for k, v in cell.items())}."


def test_generate_and_run_returns_one_trace_per_input_through_the_agentcore_executor(
    tmp_path: Path,
) -> None:
    frame = FrameResult(
        rows=[FrameRow(cell=dict(cell), frame="discovery") for cell in _CELLS], frame="discovery"
    )
    clients = StubbedAwsClients()
    queue_preflight(clients)
    for n in range(len(_CELLS)):
        queue_run(clients, f"answer {n}")
    executor = AgentCoreRuntimeExecutor(
        RuntimeTarget(runtime_arn=RUNTIME_ARN, payload_format="agentcore_sse"),
        span_reader=FakeRunSpanReader(clients),
        clients=clients,
        frame="discovery",
        log_dir=tmp_path,
    )

    with executor:
        traces = generate_and_run(frame, _CellRenderer(), executor, AgentCoreTaskBuilder())

    assert len(clients.calls_to("InvokeAgentRuntime")) == len(_CELLS)
    assert [trace.taxonomy_cell for trace in traces] == list(_CELLS)
    assert {trace.frame for trace in traces} == {"discovery"}
    assert [trace.output for trace in traces] == [{"text": f"answer {n}"} for n in range(3)]
    assert len({trace.trace_id for trace in traces}) == len(_CELLS)
    clients.assert_no_pending_responses()


def test_build_all_refuses_no_inputs() -> None:
    with pytest.raises(SynthError, match="no inputs"):
        AgentCoreTaskBuilder().build_all([])


def test_build_refuses_more_than_one_input_naming_build_all() -> None:
    with pytest.raises(SynthError, match="build_all"):
        AgentCoreTaskBuilder().build([_input(_CELLS[0], "first"), _input(_CELLS[1], "second")])
