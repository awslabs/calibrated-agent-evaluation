# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for custom scorer implementation."""

from __future__ import annotations

from typing import Any

from calibrated_agent_evaluation.schema.types import Label, Span, Trace
from calibrated_agent_evaluation.scorers.base import Scorer, ScorerLane
from calibrated_agent_evaluation.scorers.batch import score_traces


class StringLengthScorer:
    """A minimal custom scorer: PASS if output has a 'text' key with length > threshold."""

    lane = ScorerLane.CI_GATE

    def __init__(self, min_length: int = 5) -> None:
        self._min_length = min_length

    def score(
        self,
        output: dict[str, Any],
        expected: dict[str, Any] | None,
        *,
        trace_id: str,
    ) -> Label:
        text = str(output.get("text", ""))
        outcome = "PASS" if len(text) >= self._min_length else "FAIL"
        op = "≥" if outcome == "PASS" else "<"
        critique = f"StringLength: len={len(text)} {op} {self._min_length}"
        return Label(
            trace_id=trace_id, span_id=None, kind="code", outcome=outcome, critique=critique
        )


class TestCustomScorer:
    """Custom scorers implementing the Scorer protocol work with score_traces."""

    def test_custom_scorer_satisfies_protocol(self) -> None:
        s = StringLengthScorer(min_length=3)
        assert isinstance(s, Scorer)

    def test_custom_scorer_returns_valid_label(self) -> None:
        s = StringLengthScorer(min_length=3)
        label = s.score(output={"text": "hello"}, expected=None, trace_id="custom1")
        assert label.outcome == "PASS"
        assert label.kind == "code"
        assert label.critique

    def test_custom_scorer_works_with_score_traces(self) -> None:
        trace = Trace(
            trace_id="ct1",
            frame="discovery",
            ownership="owned",
            input={},
            output={"text": "hi"},
            taxonomy_cell={},
            spans=[
                Span(
                    span_id="ct1/a",
                    trace_id="ct1",
                    agent="a",
                    role="specialist",
                    parent=None,
                    seq=0,
                    status="completed",
                    input={},
                    output={"text": "hi"},
                )
            ],
        )
        labels = score_traces([trace], [StringLengthScorer(min_length=10)], {"ct1": None})
        assert len(labels) == 1
        assert labels[0].outcome == "FAIL"
        assert labels[0].trace_id == "ct1"
