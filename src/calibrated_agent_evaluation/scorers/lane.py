# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-lane enforcement: CI_GATE vs EXPLORATION.

Generic generation-quality metrics (ROUGE, BERTScore, "helpfulness") are banned from CI gates.
Only structured-target scorers with ``lane=CI_GATE`` may gate deployments. This module provides
the enforcement predicate.
"""

from __future__ import annotations

from calibrated_agent_evaluation.scorers.base import Scorer, ScorerLane
from calibrated_agent_evaluation.scorers.errors import ScorerLaneError


def enforce_ci_gate(scorer: Scorer) -> None:
    """Raise ``ScorerLaneError`` if ``scorer`` is not permitted as a CI gate.

    Call this before running a scorer in CI mode. Scorers with ``lane=EXPLORATION`` are
    structurally banned from CI.
    """
    if scorer.lane != ScorerLane.CI_GATE:
        raise ScorerLaneError(
            f"scorer '{type(scorer).__name__}' has lane={scorer.lane.value} and cannot be "
            f"used as a CI gate. Only scorers with lane=CI_GATE are permitted in CI mode "
            f"(generic generation-quality metrics are exploration-only)."
        )
