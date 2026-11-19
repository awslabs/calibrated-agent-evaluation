# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for calibrated_agent_evaluation.scorers.batch — score_traces."""

from __future__ import annotations

from typing import Any

import pytest

from calibrated_agent_evaluation.schema.types import Label, Span, Trace
from calibrated_agent_evaluation.scorers.base import ScorerLane
from calibrated_agent_evaluation.scorers.batch import score_traces
from calibrated_agent_evaluation.scorers.errors import ScorerError, ScorerLaneError
from calibrated_agent_evaluation.scorers.heuristic import ExactMatchScorer


def _make_trace(trace_id: str, output: dict[str, Any]) -> Trace:
    return Trace(
        trace_id=trace_id,
        frame="discovery",
        ownership="owned",
        input={"message": "test"},
        output=output,
        taxonomy_cell={},
        spans=[
            Span(
                span_id=f"{trace_id}/agent",
                trace_id=trace_id,
                agent="agent",
                role="specialist",
                parent=None,
                seq=0,
                status="completed",
                input={},
                output=output,
            )
        ],
    )


class TestScoreTraces:
    """Batch scoring returns one Label per (trace, scorer) pair."""

    def test_returns_labels_for_each_trace(self) -> None:
        traces = [
            _make_trace("t1", {"answer": "yes"}),
            _make_trace("t2", {"answer": "no"}),
        ]
        expected_map = {"t1": {"answer": "yes"}, "t2": {"answer": "yes"}}
        labels = score_traces(traces, [ExactMatchScorer()], expected_map)

        assert len(labels) == 2
        assert labels[0].trace_id == "t1"
        assert labels[0].outcome == "PASS"
        assert labels[1].trace_id == "t2"
        assert labels[1].outcome == "FAIL"

    def test_empty_traces_returns_empty_list(self) -> None:
        labels = score_traces([], [ExactMatchScorer()], {})
        assert labels == []

    def test_scorer_error_propagates_not_swallowed(self) -> None:
        traces = [_make_trace("t1", {"answer": "x"})]
        scorer = ExactMatchScorer()
        with pytest.raises(ScorerError, match="requires an expected value"):
            score_traces(traces, [scorer], {})


class TestCIGateEnforcement:
    """ci_gate_only enforcement in score_traces."""

    def test_ci_gate_only_with_exploration_scorer_raises(self) -> None:
        class _ExplorationScorer:
            lane = ScorerLane.EXPLORATION

            def score(self, output: Any, expected: Any, *, trace_id: str) -> Label:
                return Label(
                    trace_id=trace_id, span_id=None, kind="code", outcome="PASS", critique="ok"
                )

        traces = [_make_trace("t1", {"x": "y"})]
        with pytest.raises(ScorerLaneError):
            score_traces(traces, [_ExplorationScorer()], {"t1": {"x": "y"}}, ci_gate_only=True)

    def test_ci_gate_only_with_ci_gate_scorer_succeeds(self) -> None:
        traces = [_make_trace("t1", {"answer": "yes"})]
        labels = score_traces(
            traces, [ExactMatchScorer()], {"t1": {"answer": "yes"}}, ci_gate_only=True
        )
        assert len(labels) == 1
        assert labels[0].outcome == "PASS"
