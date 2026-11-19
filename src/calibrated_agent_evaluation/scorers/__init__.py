# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured-target scorers: deterministic evaluation of agent outputs.

The public surface: ``Scorer`` (the protocol), concrete scorer classes (ExactMatch, Levenshtein,
etc.), ``ScorerLane`` (CI_GATE vs EXPLORATION), ``score_traces()`` (batch helper), and the typed
error family (``ScorerError``, ``ScorerLaneError``). The ``autoevals`` dependency is lazy-imported
so ``import calibrated_agent_evaluation.scorers`` succeeds without the ``[scorers]`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from calibrated_agent_evaluation.scorers.base import Scorer, ScorerLane
from calibrated_agent_evaluation.scorers.errors import ScorerError, ScorerLaneError
from calibrated_agent_evaluation.scorers.lane import enforce_ci_gate

if TYPE_CHECKING:
    from calibrated_agent_evaluation.scorers.batch import score_traces as score_traces
    from calibrated_agent_evaluation.scorers.execution import (
        CodeExecutionScorer as CodeExecutionScorer,
    )
    from calibrated_agent_evaluation.scorers.execution import (
        SqlExecutionScorer as SqlExecutionScorer,
    )
    from calibrated_agent_evaluation.scorers.heuristic import (
        ExactMatchScorer as ExactMatchScorer,
    )
    from calibrated_agent_evaluation.scorers.heuristic import (
        JSONDiffScorer as JSONDiffScorer,
    )
    from calibrated_agent_evaluation.scorers.heuristic import (
        LevenshteinScorer as LevenshteinScorer,
    )
    from calibrated_agent_evaluation.scorers.heuristic import (
        NumericDiffScorer as NumericDiffScorer,
    )
    from calibrated_agent_evaluation.scorers.heuristic import (
        ValidJSONScorer as ValidJSONScorer,
    )
    from calibrated_agent_evaluation.scorers.passk import pass_at_k as pass_at_k

__all__ = [
    "CodeExecutionScorer",
    "ExactMatchScorer",
    "JSONDiffScorer",
    "LevenshteinScorer",
    "NumericDiffScorer",
    "Scorer",
    "ScorerError",
    "ScorerLane",
    "ScorerLaneError",
    "SqlExecutionScorer",
    "ValidJSONScorer",
    "enforce_ci_gate",
    "pass_at_k",
    "score_traces",
]


def __getattr__(name: str) -> object:
    if name in (
        "ExactMatchScorer",
        "LevenshteinScorer",
        "NumericDiffScorer",
        "JSONDiffScorer",
        "ValidJSONScorer",
    ):
        from calibrated_agent_evaluation.scorers import heuristic

        return getattr(heuristic, name)
    if name in ("CodeExecutionScorer", "SqlExecutionScorer"):
        from calibrated_agent_evaluation.scorers import execution

        return getattr(execution, name)
    if name == "pass_at_k":
        from calibrated_agent_evaluation.scorers.passk import pass_at_k

        return pass_at_k
    if name == "score_traces":
        from calibrated_agent_evaluation.scorers.batch import score_traces

        return score_traces
    raise AttributeError(f"module 'calibrated_agent_evaluation.scorers' has no attribute {name!r}")
