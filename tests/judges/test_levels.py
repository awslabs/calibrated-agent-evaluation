# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge-level model + calibrate-at-the-level guards.

Every test embeds its FR and asserts behavior (the level a definition carries, or the typed refusal
of a level misuse), never mock calls.
"""

from __future__ import annotations

import pytest

from _helpers import make_label
from calibrated_agent_evaluation.judges import (
    JudgeDefinition,
    JudgeLevel,
    JudgeLevelError,
    SpanGoldError,
    assert_not_workflow_for_agent_rate,
    assert_span_gold_human,
)


def _judge(level: JudgeLevel) -> JudgeDefinition:
    return JudgeDefinition(
        judge_id=f"intake-{level}", level=level, model="m-1", prompt_version="sha-1"
    )


# --- a JudgeDefinition carries its level; gold is keyed at that level ---


def test_span_judge_gold_is_per_span() -> None:
    span_judge = _judge("span")
    assert span_judge.level == "span"
    # Span gold is per-span human gold (span_id set): the guard accepts it.
    span_gold = [make_label(trace_id="CLM-4471", span_id="CLM-4471/intake", kind="human")]
    assert_span_gold_human(span_gold)  # does not raise


def test_workflow_judge_gold_is_per_trace() -> None:
    workflow_judge = _judge("workflow")
    assert workflow_judge.level == "workflow"
    # A workflow judge attributes over the whole trace, so its gold is trace-level (span_id=None);
    # that is a valid workflow verdict but must NOT be used to compute a per-agent rate .
    assert_not_workflow_for_agent_rate(_judge("span"))  # a span judge is fine


# --- (negative) span gold derived from the trace outcome → SpanGoldError ---


def test_outcome_derived_span_gold_refused() -> None:
    outcome_gold = [make_label(trace_id="CLM-4471", span_id=None, kind="human")]
    with pytest.raises(SpanGoldError, match="span_id=None"):
        assert_span_gold_human(outcome_gold)


def test_non_human_span_gold_refused() -> None:
    llm_gold = [make_label(trace_id="CLM-4471", span_id="CLM-4471/intake", kind="llm-judge")]
    with pytest.raises(SpanGoldError, match="must be human"):
        assert_span_gold_human(llm_gold)


# --- (negative) a workflow judge used for a per-agent rate → JudgeLevelError ---


def test_workflow_judge_for_agent_rate_refused() -> None:
    with pytest.raises(JudgeLevelError, match="cannot compute a per-agent rate"):
        assert_not_workflow_for_agent_rate(_judge("workflow"))
