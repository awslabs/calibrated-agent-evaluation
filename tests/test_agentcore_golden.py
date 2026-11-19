# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden regression: a frozen real AgentCore trace converts + normalizes to owned, with content.

Against a live capture under ``fixtures/raw/agentcore/online/`` (a real 14-span Strands OrderBot
run on a deployed runtime, with its ``otel-rt-logs`` content records). Runs offline in the default
suite: ``adot_to_otlp`` is vendor-free and ``normalize`` is core. This is the binding proof that the
ADOT converter's content re-home reproduces an analyzable trace from a real capture, not a
hand-built fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

from _agentcore_capture import CAPTURED_QUESTION, captured_run
from calibrated_agent_evaluation.agentcore.adot import adot_to_otlp
from calibrated_agent_evaluation.normalize import StrandsSession, normalize


def _convert_and_normalize(tmp_path: Path):
    run = captured_run()
    otlp = adot_to_otlp(run["span_records"], run["log_records"])
    path = tmp_path / f"{run['trace_id']}.otlp.json"
    path.write_text(json.dumps(otlp), encoding="utf-8")
    return normalize(StrandsSession(path, trace_id=run["trace_id"]))


def test_frozen_agentcore_trace_normalizes_to_an_owned_single_agent_tree(tmp_path: Path) -> None:
    trace = _convert_and_normalize(tmp_path)
    assert trace.ownership == "owned"
    assert [s.agent for s in trace.spans] == ["orderbot"]


def test_frozen_agentcore_trace_recovers_every_tool_call_in_order(tmp_path: Path) -> None:
    trace = _convert_and_normalize(tmp_path)
    tool_names = [tc.name for s in trace.spans for tc in s.tool_calls]
    assert tool_names == ["lookup_order", "check_refund_policy", "issue_refund"]


def test_frozen_agentcore_trace_recovers_agent_io_content(tmp_path: Path) -> None:
    trace = _convert_and_normalize(tmp_path)
    # Every agent recovered both a prompt and a response (the content re-home worked end to end).
    for span in trace.spans:
        assert span.input, f"{span.agent} has no recovered input"
        assert span.output, f"{span.agent} has no recovered output"
    # The original user request reached the agent's input.
    assert CAPTURED_QUESTION in " ".join(str(s.input) for s in trace.spans)


def test_frozen_agentcore_trace_recovers_gold_bearing_tool_args_and_result(tmp_path: Path) -> None:
    trace = _convert_and_normalize(tmp_path)
    calls = {tc.name: tc for s in trace.spans for tc in s.tool_calls}
    # Args carry the real order id; the results carry the policy verdict and the refund issued.
    assert calls["lookup_order"].args == {"order_id": "Q-30187"}
    assert "within_window" in str(calls["check_refund_policy"].result)
    assert "RF-30187" in str(calls["issue_refund"].result)
