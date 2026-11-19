# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch scoring helper: score_traces().

Iterates traces x scorers, calls ``score()`` per pair, optionally enforces CI-gate lane.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from calibrated_agent_evaluation.schema.types import Label, Trace
from calibrated_agent_evaluation.scorers.base import Scorer
from calibrated_agent_evaluation.scorers.lane import enforce_ci_gate


def score_traces(
    traces: list[Trace],
    scorers: list[Scorer],
    expected_map: Mapping[str, dict[str, Any] | None],
    *,
    ci_gate_only: bool = False,
) -> list[Label]:
    """Score a batch of traces against expected values and return Labels.

    Args:
        traces: the traces to score.
        scorers: the scorers to apply to each trace.
        expected_map: mapping from trace_id → expected dict for scoring.
        ci_gate_only: if True, enforce that all scorers have lane=CI_GATE.

    Returns:
        One Label per (trace, scorer) pair.

    Raises:
        ScorerLaneError: if ci_gate_only=True and a scorer has lane=EXPLORATION.
        ScorerError: if a scorer fails on a trace (propagates, not swallowed).
    """
    if ci_gate_only:
        for scorer in scorers:
            enforce_ci_gate(scorer)

    labels: list[Label] = []
    for trace in traces:
        expected = expected_map.get(trace.trace_id)
        for scorer in scorers:
            label = scorer.score(output=trace.output, expected=expected, trace_id=trace.trace_id)
            labels.append(label)
    return labels
