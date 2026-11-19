# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strands adapter tests.

Against the real ``fixtures/raw/strands/*.otlp.json`` corpus: the ClaimsCrew Graph → a 5-span
supervisor-rooted crew tree with the gold ``occupancy_status: null``, an unsynthesizable
handoff recorded as an empty object (not dropped), a dangling parent raising a typed error, and
the three multi-agent patterns (Graph/Swarm/Agent-as-Tools) all normalizing through the one
pattern-agnostic adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.normalize import NormalizationError, StrandsSession, normalize
from calibrated_agent_evaluation.schema.validate import validate


def _strands(raw_dir: Path, name: str, **over: object) -> StrandsSession:
    return StrandsSession(raw_dir / "strands" / name, **over)  # type: ignore[arg-type]


def test_claimscrew_graph_normalizes_to_owned_five_span_supervisor_tree(
    raw_dir: Path, oracle: Any
) -> None:
    trace = normalize(_strands(raw_dir, "CLM-4471.otlp.json", trace_id="CLM-4471"))
    validate(trace)

    assert trace.ownership == "owned"
    assert len(trace.spans) == 5
    by_agent = {s.agent: s for s in trace.spans}
    assert set(by_agent) == {"supervisor", "intake", "policy-checker", "fraud-scorer", "payout"}

    # Supervisor is the root at seq 0; the other four are its children.
    supervisor = by_agent["supervisor"]
    assert supervisor.parent is None
    assert supervisor.seq == 0
    for name in ("intake", "policy-checker", "fraud-scorer", "payout"):
        assert by_agent[name].parent == supervisor.span_id

    # seq is unique and contiguous from 0.
    seqs = sorted(s.seq for s in trace.spans)
    assert seqs == [0, 1, 2, 3, 4]

    # The gold-bearing intake extraction preserved: occupancy_status is genuinely null.
    intake = by_agent["intake"]
    extract = next(tc for tc in intake.tool_calls if tc.name == "extract_claim_fields")
    assert extract.result["occupancy_status"] is None

    # Faithful to the real capture: payout made NO tool call; policy-checker called twice.
    assert by_agent["payout"].tool_calls == []
    assert [tc.name for tc in by_agent["policy-checker"].tool_calls] == [
        "lookup_policy_exclusions",
        "lookup_policy_exclusions",
    ]

    # Structural equality with the oracle (agent set, parent tree, tool names).
    gold = oracle("CLM-4471")
    gold_agents = {s["agent"] for s in gold["spans"]}
    assert set(by_agent) == gold_agents


def test_each_non_root_span_handoff_input_output_populated(raw_dir: Path) -> None:
    # Every non-root crew span carries a populated (non-empty) handoff input/output object — the
    # full tree + handoff payloads captured, not only the terminal decision.
    trace = normalize(_strands(raw_dir, "CLM-4471.otlp.json", trace_id="CLM-4471"))
    for span in trace.spans:
        if span.parent is not None:
            assert isinstance(span.input, dict)
            assert span.input, f"{span.agent} handoff input empty"
            assert isinstance(span.output, dict)


def test_unsynthesizable_handoff_is_empty_object_not_dropped(raw_dir: Path, tmp_path: Path) -> None:
    # Edge (derived by mutating the real crew export): strip a crew agent's content events so its
    # handoff input/output cannot be synthesized. It must be recorded with an EMPTY object (not
    # dropped, not None) and the trace must still validate.
    otlp = json.loads((raw_dir / "strands" / "CLM-4471.otlp.json").read_text(encoding="utf-8"))
    for rs in otlp["resourceSpans"]:
        for ss in rs["scopeSpans"]:
            for span in ss["spans"]:
                if span["name"] == "invoke_agent fraud-scorer":
                    span["events"] = []  # no user.message / choice → nothing to synthesize
    mutated = tmp_path / "no_handoff.otlp.json"
    mutated.write_text(json.dumps(otlp), encoding="utf-8")

    trace = normalize(StrandsSession(mutated, trace_id="CLM-4471"))
    validate(trace)  # still valid
    fraud = next(s for s in trace.spans if s.agent == "fraud-scorer")
    assert fraud.input == {} or fraud.output == {}
    assert fraud.input is not None
    assert fraud.output is not None


def test_dangling_parent_raises_normalization_error(raw_dir: Path, tmp_path: Path) -> None:
    # Negative (derived by mutating the real crew export): point a crew agent's parentSpanId at a
    # span id that does not exist. The broken tree must surface a NormalizationError (via validate's
    # parent-resolution), never a Trace with a dangling edge.
    otlp = json.loads((raw_dir / "strands" / "CLM-4471.otlp.json").read_text(encoding="utf-8"))
    for rs in otlp["resourceSpans"]:
        for ss in rs["scopeSpans"]:
            for span in ss["spans"]:
                if span["name"] == "invoke_agent intake":
                    span["parentSpanId"] = "ZZZZdanglingZZZZ=="
    mutated = tmp_path / "dangling.otlp.json"
    mutated.write_text(json.dumps(otlp), encoding="utf-8")

    with pytest.raises(NormalizationError):
        normalize(StrandsSession(mutated, trace_id="CLM-4471"))


def test_strands_graph_owned_with_reconstructed_agents_and_tools(raw_dir: Path) -> None:
    # the frozen Strands Graph export → owned Trace with reconstructed agents + tool calls.
    trace = normalize(_strands(raw_dir, "CLM-4471.otlp.json", trace_id="CLM-4471"))
    assert trace.ownership == "owned"
    assert any(tc.name == "extract_claim_fields" for s in trace.spans for tc in s.tool_calls)
    assert any(tc.name == "score_fraud_risk" for s in trace.spans for tc in s.tool_calls)


def test_swarm_handoff_is_a_tool_call_not_a_phantom_agent_span(raw_dir: Path) -> None:
    # (pattern parity): a Swarm's handoff_to_agent lands as an execute_tool-derived ToolCall
    # (its agent_name/message args present), NOT as a phantom agent Span.
    trace = normalize(_strands(raw_dir, "CLM-7720-swarm.otlp.json", trace_id="CLM-7720"))
    validate(trace)
    span_agents = {s.agent for s in trace.spans}
    assert "handoff_to_agent" not in span_agents  # not a phantom agent span
    handoff_calls = [
        tc for s in trace.spans for tc in s.tool_calls if tc.name == "handoff_to_agent"
    ]
    assert handoff_calls, "handoff_to_agent not landed as a ToolCall"
    # its args carry the receiver name / message (recovered from the tool.message event).
    assert handoff_calls[0].args


def test_agent_as_tools_subagent_is_a_child_span_not_a_tool_call(raw_dir: Path) -> None:
    # (pattern parity): an Agent-as-Tools sub-agent surfaces as a child Span (agent
    # boundary), NOT flattened into a ToolCall; a plain leaf tool stays a ToolCall.
    trace = normalize(_strands(raw_dir, "ORD-3310-as-tool.otlp.json", trace_id="ORD-3310"))
    validate(trace)
    by_agent = {s.agent: s for s in trace.spans}
    # policy_advisor sub-agent is a child Span...
    assert "policy_advisor" in by_agent
    assert by_agent["policy_advisor"].parent is not None
    # ...and was NOT flattened into a tool call on the parent.
    parent = by_agent["orderbot"]
    assert "policy_advisor" not in [tc.name for tc in parent.tool_calls]
    # the plain leaf tool stays a ToolCall.
    assert any(tc.name == "issue_refund" for tc in parent.tool_calls)
