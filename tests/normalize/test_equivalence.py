# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-agnosticism tests.

Two builds of the same OrderBot behaviour — build #1 (hand-coded OTel) and build #2 (Anthropic-
SDK + MCP, a different emitter) — must normalize to *equivalent* Traces: same reconstructed
tool-call sequence and same gold failure-mode content, so a single gold Label applies to both
. And tool calls live ONLY on Span, never on Trace.
"""

from __future__ import annotations

from pathlib import Path

from calibrated_agent_evaluation.normalize import OTelSpans, normalize
from calibrated_agent_evaluation.schema.types import Trace
from calibrated_agent_evaluation.schema.validate import validate


def _tool_sequence(trace: Trace) -> list[str]:
    return [tc.name for span in trace.spans for tc in span.tool_calls]


def _check_refund_policy_result(trace: Trace) -> dict:
    return next(
        tc.result
        for span in trace.spans
        for tc in span.tool_calls
        if tc.name == "check_refund_policy"
    )


def test_two_builds_normalize_to_equivalent_traces(raw_dir: Path) -> None:
    b1_src = OTelSpans(raw_dir / "otel" / "ORD-1001.events.otlp.json", trace_id="ORD-1001")
    b2_src = OTelSpans(raw_dir / "otel" / "ORD-1001.build2.otlp.json", trace_id="ORD-1001b")
    build1 = normalize(b1_src)
    build2 = normalize(b2_src)
    validate(build1)
    validate(build2)

    assert build1.ownership == "owned"
    assert build2.ownership == "owned"

    # Same reconstructed tool-call name sequence (target-agnostic — analysis depends on what the
    # agent DID, not how it was built).
    assert _tool_sequence(build1) == _tool_sequence(build2)
    assert _tool_sequence(build1) == ["lookup_order", "check_refund_policy", "issue_refund"]

    # Same gold failure-mode-relevant content: the trusted_broken_policy_tool over-refund
    # (within_window=true despite days_since_delivery=51) — so ONE gold Label applies to both.
    crp1 = _check_refund_policy_result(build1)
    crp2 = _check_refund_policy_result(build2)
    assert crp1["within_window"] is True
    assert crp1["days_since_delivery"] == 51
    assert crp2["within_window"] is True
    assert crp2["days_since_delivery"] == 51
    assert crp1 == crp2

    # Equivalence is on the structured tool-call content — NOT the builds' (differing) free-text
    # final answers, and NEVER a synthesized output.action classification.
    assert "action" not in build1.output
    assert "action" not in build2.output


def test_tool_calls_live_only_on_span_never_on_trace(raw_dir: Path) -> None:
    # Negative: tool calls are on Span.tool_calls only; Trace has no tool_calls field/attribute
    # (the single read path). A fixture that tried to place tool calls at trace level is
    # ignored — the Trace dataclass has no such attribute to populate.
    src = OTelSpans(raw_dir / "otel" / "ORD-1001.events.otlp.json", trace_id="ORD-1001")
    build1 = normalize(src)
    assert not hasattr(build1, "tool_calls")
    # every reconstructed tool call is reachable only via a span.
    assert _tool_sequence(build1)
    assert all(hasattr(span, "tool_calls") for span in build1.spans)
