# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in golden-file contract test for the normalizer (companion deliverable #4).

The binding proof for the top risk: for each real ``fixtures/raw/`` input, ``normalize(...)``
must reproduce the hand-authored ``fixtures/normalized/*.json`` oracle — authored
independently of the normalizer, so matching it is a real contract test, not a tautology. It runs
in the default suite: it reads committed files and needs no network. The
Inspect case needs the ``[inspect]`` extra and skips, saying so, only when that extra is absent;
the OTLP-JSON cases need no extra.

Equality is **structural/semantic, modulo timestamps and free-text terminal answers** (the real
agents emit free-text finals; ``fixtures/README.md`` fixes the golden test to the structural
fields): the agent set + tree shape, each span's tool-call name sequence, and the gold-bearing
tool-result content (ORD-1001's over-refund contradiction; CLM-4471's ``occupancy_status: null``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.normalize import (
    InspectEvalLog,
    OTelSpans,
    StrandsSession,
    normalize,
)
from calibrated_agent_evaluation.schema.serde import to_dict
from calibrated_agent_evaluation.schema.types import Trace

_FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures"
_RAW = _FIXTURES / "raw"


def _oracle(trace_id: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / "normalized" / f"{trace_id}.json").read_text(encoding="utf-8"))


def _tree_shape(spans: list[dict[str, Any]]) -> set[tuple[str, str | None]]:
    """The (agent, parent-agent) edge set — the tree shape independent of span_id spelling."""
    by_id = {s["span_id"]: s["agent"] for s in spans}
    return {(s["agent"], by_id.get(s["parent"]) if s["parent"] else None) for s in spans}


def _tool_names(spans: list[dict[str, Any]], agent: str) -> list[str]:
    span = next(s for s in spans if s["agent"] == agent)
    return [tc["name"] for tc in span["tool_calls"]]


def _assert_structural_match(produced: Trace, gold: dict[str, Any]) -> None:
    got = to_dict(produced)
    # Same ownership + trace id.
    assert got["ownership"] == gold["ownership"]
    assert got["trace_id"] == gold["trace_id"]
    # Same agent set and same parent-tree shape (span_id spelling aside).
    assert {s["agent"] for s in got["spans"]} == {s["agent"] for s in gold["spans"]}
    assert _tree_shape(got["spans"]) == _tree_shape(gold["spans"])
    # Same per-agent tool-call name sequence.
    for gspan in gold["spans"]:
        assert _tool_names(got["spans"], gspan["agent"]) == [
            tc["name"] for tc in gspan["tool_calls"]
        ], f"tool sequence mismatch for agent {gspan['agent']}"


def test_golden_orderbot_inspect_matches_oracle_structurally() -> None:
    pytest.importorskip(
        "inspect_ai", reason="the [inspect] extra is not installed; it reads the Inspect .eval log"
    )
    produced = normalize(InspectEvalLog(_RAW / "inspect" / "ORD-1001.eval"))
    gold = _oracle("ORD-1001")
    _assert_structural_match(produced, gold)
    # Gold-bearing tool content preserved field-for-field (the over-refund contradiction).
    got = to_dict(produced)
    crp = next(tc for tc in got["spans"][0]["tool_calls"] if tc["name"] == "check_refund_policy")
    gold_crp = next(
        tc for tc in gold["spans"][0]["tool_calls"] if tc["name"] == "check_refund_policy"
    )
    assert crp["result"]["within_window"] == gold_crp["result"]["within_window"] is True
    assert crp["result"]["days_since_delivery"] == gold_crp["result"]["days_since_delivery"] == 51


def test_golden_claimscrew_strands_graph_matches_oracle_structurally() -> None:
    produced = normalize(
        StrandsSession(_RAW / "strands" / "CLM-4471.otlp.json", trace_id="CLM-4471")
    )
    gold = _oracle("CLM-4471")
    _assert_structural_match(produced, gold)
    # The gold-bearing intake extraction preserved: occupancy_status genuinely null.
    got = to_dict(produced)
    intake = next(s for s in got["spans"] if s["agent"] == "intake")
    extract = next(tc for tc in intake["tool_calls"] if tc["name"] == "extract_claim_fields")
    assert extract["result"]["occupancy_status"] is None


def test_golden_orderbot_two_builds_cross_format_equivalent() -> None:
    # Cross-format equivalence (companion deliverable #4): build #1 (OTel) and build #2 (Anthropic+
    # MCP) reduce to the same tool sequence + same gold failure content — one gold applies to both.
    b1 = normalize(OTelSpans(_RAW / "otel" / "ORD-1001.events.otlp.json", trace_id="ORD-1001"))
    b2 = normalize(OTelSpans(_RAW / "otel" / "ORD-1001.build2.otlp.json", trace_id="ORD-1001"))
    seq1 = [tc.name for s in b1.spans for tc in s.tool_calls]
    seq2 = [tc.name for s in b2.spans for tc in s.tool_calls]
    assert seq1 == seq2

    def crp(t: Trace) -> dict[str, Any]:
        return next(
            tc.result for s in t.spans for tc in s.tool_calls if tc.name == "check_refund_policy"
        )

    assert crp(b1) == crp(b2)
    assert crp(b1)["within_window"] is True
    assert crp(b1)["days_since_delivery"] == 51
