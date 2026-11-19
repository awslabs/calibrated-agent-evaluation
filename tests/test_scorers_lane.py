# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for calibrated_agent_evaluation.scorers.lane — two-lane enforcement."""

from __future__ import annotations

from typing import Any

import pytest

from calibrated_agent_evaluation.schema.types import Label
from calibrated_agent_evaluation.scorers.base import ScorerLane
from calibrated_agent_evaluation.scorers.errors import ScorerLaneError
from calibrated_agent_evaluation.scorers.lane import enforce_ci_gate


class _CIGateScorer:
    lane = ScorerLane.CI_GATE

    def score(
        self, output: dict[str, Any], expected: dict[str, Any] | None, *, trace_id: str
    ) -> Label:
        return Label(trace_id=trace_id, span_id=None, kind="code", outcome="PASS", critique="ok")


class _ExplorationScorer:
    lane = ScorerLane.EXPLORATION

    def score(
        self, output: dict[str, Any], expected: dict[str, Any] | None, *, trace_id: str
    ) -> Label:
        return Label(trace_id=trace_id, span_id=None, kind="code", outcome="PASS", critique="ok")


class TestEnforceCIGate:
    """enforce_ci_gate raises for EXPLORATION, passes for CI_GATE."""

    def test_ci_gate_scorer_passes_silently(self) -> None:
        enforce_ci_gate(_CIGateScorer())

    def test_exploration_scorer_raises_scorer_lane_error(self) -> None:
        with pytest.raises(ScorerLaneError):
            enforce_ci_gate(_ExplorationScorer())

    def test_error_message_names_scorer_and_the_lane_rule(self) -> None:
        with pytest.raises(ScorerLaneError, match="_ExplorationScorer") as exc_info:
            enforce_ci_gate(_ExplorationScorer())
        msg = str(exc_info.value)
        assert "exploration-only" in msg
        assert "EXPLORATION" in msg or "exploration" in msg
