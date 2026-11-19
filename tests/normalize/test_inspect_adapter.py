# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect adapter tests.

Against the real frozen ``fixtures/raw/inspect/*.eval`` corpus: a single-agent ``.eval``
normalizes to a one-span owned ``Trace``, tool calls flatten onto the root span
with the gold-bearing content preserved, attachments are resolved, a zero-tool
run still yields a valid one-span tree, and a leaf ToolEvent is never promoted to a Span.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.normalize import InspectEvalLog, normalize
from calibrated_agent_evaluation.schema.validate import validate


def _eval(raw_dir: Path, name: str) -> InspectEvalLog:
    return InspectEvalLog(raw_dir / "inspect" / name)


def test_orderbot_eval_normalizes_to_owned_one_span_tree(raw_dir: Path, oracle: Any) -> None:
    trace = normalize(_eval(raw_dir, "ORD-1001.eval"))
    validate(trace)

    assert trace.ownership == "owned"
    assert trace.trace_id == "ORD-1001"  # read from sample.metadata
    # one-span tree, root parent=null, seq=0.
    assert len(trace.spans) == 1
    root = trace.spans[0]
    assert root.parent is None
    assert root.seq == 0

    # the three business tool calls flatten onto the root span, in order, with args+result;
    # the react `submit` control tool is NOT a tool_call.
    names = [tc.name for tc in root.tool_calls]
    assert names == ["lookup_order", "check_refund_policy", "issue_refund"]
    assert "submit" not in names

    # The gold-bearing failure content is preserved EXACTLY (the trusted_broken_policy_tool bug):
    crp = next(tc for tc in root.tool_calls if tc.name == "check_refund_policy")
    assert crp.result["within_window"] is True
    assert crp.result["days_since_delivery"] == 51
    assert crp.result["policy_verdict"] == "WITHIN_WINDOW_REFUND_ALLOWED"

    # Structural/semantic equality with the oracle (agent, tree shape, tool names) — the
    # normalizer emits the real free-text terminal answer, never a synthesized output.action.
    gold = oracle("ORD-1001")
    assert root.agent == gold["spans"][0]["agent"]
    assert [tc.name for tc in root.tool_calls] == [
        tc["name"] for tc in gold["spans"][0]["tool_calls"]
    ]
    assert "action" not in trace.output  # never a synthesized classification
    assert trace.output.get("answer")  # the real free-text answer is carried


def test_attachments_are_resolved_so_content_is_materialized(raw_dir: Path) -> None:
    # reading with resolve_attachments="full" means tool/message content is real, not an
    # `attachment://` placeholder. Assert the recovered content carries no attachment sentinel.
    trace = normalize(_eval(raw_dir, "ORD-1001.eval"))
    root = trace.spans[0]
    blob = repr([tc.args for tc in root.tool_calls] + [tc.result for tc in root.tool_calls])
    assert "attachment://" not in blob
    # and the content is genuinely non-empty (materialized).
    assert all(tc.result for tc in root.tool_calls)


def test_claims_eval_is_the_single_agent_path_four_tools(raw_dir: Path) -> None:
    # The CLM-4471 .eval is one `claims` react agent driving four tools (Inspect's react is
    # single-agent by construction) — a multi-tool instance of the single-agent path, one span.
    trace = normalize(_eval(raw_dir, "CLM-4471.eval"))
    validate(trace)
    assert trace.ownership == "owned"
    assert len(trace.spans) == 1
    names = [tc.name for tc in trace.spans[0].tool_calls]
    assert "extract_claim_fields" in names
    assert "submit" not in names


def test_zero_tool_call_single_agent_still_yields_valid_one_span_tree(
    raw_dir: Path, tmp_path: Path
) -> None:
    # Edge (derived by mutating a real capture): slice a real .eval's events to remove every
    # ToolEvent, leaving a single-agent run with zero tool calls. It must still normalize to a
    # valid one-span tree whose root mirrors the trace I/O.
    from inspect_ai.log import read_eval_log, write_eval_log

    log = read_eval_log(str(raw_dir / "inspect" / "ORD-1001.eval"))
    assert log.samples is not None
    sample = log.samples[0]
    sample.events = [e for e in sample.events if getattr(e, "event", None) != "tool"]
    sliced = tmp_path / "zero_tools.eval"
    write_eval_log(log, str(sliced))

    trace = normalize(InspectEvalLog(sliced))
    validate(trace)
    assert trace.ownership == "owned"
    assert len(trace.spans) == 1
    root = trace.spans[0]
    assert root.tool_calls == []
    assert root.input == trace.input
    assert root.output == trace.output


def test_leaf_tool_is_not_promoted_to_a_span_and_agent_is_a_span(raw_dir: Path) -> None:
    # The tool-vs-span policy applied in ONE shape: leaf ToolEvents land in tool_calls (not their
    # own Span), and the single agent boundary is the one Span. A single-agent .eval therefore has
    # exactly one span carrying N tool_calls — never N+1 spans.
    trace = normalize(_eval(raw_dir, "ORD-1001.eval"))
    assert len(trace.spans) == 1
    assert len(trace.spans[0].tool_calls) == 3
    # None of the tool names leaked in as a span agent (a tool was not promoted to a Span).
    span_agents = {s.agent for s in trace.spans}
    assert "lookup_order" not in span_agents
    assert "check_refund_policy" not in span_agents
