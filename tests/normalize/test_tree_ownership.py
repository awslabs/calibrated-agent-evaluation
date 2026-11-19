# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foundational smoke tests for the vendor-free tree builder + ownership gate.

Drives ``tree.py`` + ``ownership.py`` directly on a synthetic vendor-free intermediate — so the
foundational layer is validated in isolation, not only transitively via the Phase-3 adapters.
Covers the single-agent collapse (a one-span tree mirroring trace I/O) and the degrade path
(a content-less intermediate degrades to ``probed`` with ``spans == []``).
"""

from __future__ import annotations

import logging

import pytest

from calibrated_agent_evaluation.normalize.ownership import resolve
from calibrated_agent_evaluation.normalize.tree import AgentNode, Intermediate, ToolInvocation
from calibrated_agent_evaluation.schema.validate import validate


def _intermediate(nodes: list[AgentNode], **over: object) -> Intermediate:
    base: dict[str, object] = {
        "trace_id": "T-1",
        "frame": "discovery",
        "taxonomy_cell": {"feature": "refund"},
        "trace_input": {"message": "hi"},
        "trace_output": {"answer": "done"},
        "nodes": nodes,
    }
    base.update(over)
    return Intermediate(**base)  # type: ignore[arg-type]


def test_single_agent_intermediate_collapses_to_one_span_tree_mirroring_trace_io() -> None:
    node = AgentNode(
        agent="orderbot",
        role="specialist",
        parent=None,
        tool_calls=[ToolInvocation("lookup", {"id": "A"}, {"ok": True})],
        order=0,
    )
    trace, ownership = resolve(_intermediate([node]))

    assert ownership == "owned"
    assert len(trace.spans) == 1
    root = trace.spans[0]
    assert root.parent is None
    assert root.seq == 0
    # the one-span root mirrors the trace-level input/output.
    assert root.input == trace.input
    assert root.output == trace.output
    assert [tc.name for tc in root.tool_calls] == ["lookup"]
    validate(trace)  # the foundational output is schema-valid on its own


def test_root_not_earliest_starting_gets_unique_seq() -> None:
    # A supervisor-fanout crew whose ROOT is not the earliest-starting node (clock skew /
    # near-simultaneous starts — the crew topology exercises): the root must still carry a
    # unique seq 0, not collide with whatever child sorted to order 0. Ranking every node then
    # force-overriding the root to 0 would produce two seq==0 spans and refuse a valid trace.
    root = AgentNode(
        agent="supervisor", role="supervisor", parent=None, output={"answer": "ok"}, order=5
    )
    children = [
        AgentNode(
            agent=name, role="specialist", parent="supervisor", output={"answer": name}, order=i
        )
        for i, name in enumerate(("intake", "policy-checker", "payout"))  # orders 0,1,2 < root's 5
    ]
    trace, ownership = resolve(_intermediate([root, *children]))

    assert ownership == "owned"
    seqs = [s.seq for s in trace.spans]
    assert len(set(seqs)) == len(seqs), f"duplicate seq: {seqs}"
    assert sorted(seqs) == [0, 1, 2, 3]  # unique + contiguous from 0
    assert next(s.seq for s in trace.spans if s.parent is None) == 0  # root is always 0
    validate(trace)  # would raise "duplicate seq" under the old collision bug


def test_content_less_intermediate_degrades_to_probed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A run whose only span would carry no recovered per-step signal (no tool calls, and no
    # per-step content beyond the empty terminal I/O) must degrade to probed (spans == []), not
    # be dressed up as an owned tree with an empty step.
    empty = AgentNode(agent="orderbot", role="specialist", parent=None, order=0)
    inter = _intermediate([empty], trace_input={}, trace_output={})
    with caplog.at_level(logging.WARNING, logger="calibrated_agent_evaluation.normalize"):
        trace, ownership = resolve(inter)

    assert ownership == "probed"
    assert trace.spans == []
    # the (here empty) terminal input/output is preserved on the probed trace...
    assert trace.input == {}
    assert trace.output == {}
    # ...and the degradation reason is surfaced as a WARNING.
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)
    validate(trace)


def test_explicit_degrade_reason_is_honored_over_a_populated_tree(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # When an adapter sets degrade_reason (it recovered only outcome data), the gate builds probed
    # regardless of any node content — the adapter's honest signal wins.
    node = AgentNode(
        agent="orderbot",
        role="specialist",
        parent=None,
        output={"answer": "done"},
        order=0,
    )
    with caplog.at_level(logging.WARNING, logger="calibrated_agent_evaluation.normalize"):
        trace, ownership = resolve(_intermediate([node], degrade_reason="black-box source"))

    assert ownership == "probed"
    assert trace.spans == []
    assert "black-box source" in caplog.text
