# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reply→Label mapping + verdict-schema validation."""

from __future__ import annotations

import pytest

from _judge_helpers import judge
from calibrated_agent_evaluation.execute import StructuredReply
from calibrated_agent_evaluation.judges.errors import JudgeRuntimeError
from calibrated_agent_evaluation.judges.verdict import reply_to_label


def test_maps_structured_reply_to_llm_judge_label() -> None:
    """A valid structured reply becomes a kind='llm-judge' Label with the mapped outcome."""
    reply = StructuredReply(fields={"critique": "wrong refund", "verdict": "fail"}, via="tool_call")
    label = reply_to_label(reply, judge(), trace_id="ORD-1001", span_id=None)
    assert label.kind == "llm-judge"
    assert label.outcome == "FAIL"
    assert label.critique == "wrong refund"
    assert label.trace_id == "ORD-1001"


def test_empty_critique_rejected() -> None:
    """An empty/whitespace critique raises JudgeRuntimeError — no Label built."""
    reply = StructuredReply(fields={"critique": "   ", "verdict": "pass"}, via="response_schema")
    with pytest.raises(JudgeRuntimeError, match="critique"):
        reply_to_label(reply, judge(), trace_id="ORD-1001", span_id=None)


def test_non_binary_verdict_rejected() -> None:
    """A non-binary verdict value raises JudgeRuntimeError — no fake outcome."""
    reply = StructuredReply(fields={"critique": "ok", "verdict": "maybe"}, via="response_schema")
    with pytest.raises(JudgeRuntimeError, match="verdict"):
        reply_to_label(reply, judge(), trace_id="ORD-1001", span_id=None)


def test_missing_verdict_field_rejected() -> None:
    """A payload missing the verdict field is rejected, not defaulted."""
    reply = StructuredReply(fields={"critique": "ok"}, via="tool_call")
    with pytest.raises(JudgeRuntimeError):
        reply_to_label(reply, judge(), trace_id="ORD-1001", span_id=None)
