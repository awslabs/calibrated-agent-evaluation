# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-bare-score invariant test.

The CLI's judge/scoring path routes through the shipped ``enforce_score_gate`` — an
uncertified judge is refused with ``UncalibratedScoreError`` before any score is computed. This
proves the invariant the CLI relies on (no ``calibrated-agent-evaluation`` path emits a score
without a calibrated judge).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from _helpers import TEST_STAMP
from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord
from calibrated_agent_evaluation.enforce import enforce_score_gate
from calibrated_agent_evaluation.enforce.errors import UncalibratedScoreError


@dataclass(frozen=True)
class _Judge:
    judge_id: str
    model: str
    prompt_version: str


def _record(result: str, *, model: str, prompt_version: str) -> CalibrationRecord:
    return CalibrationRecord(
        provenance=TEST_STAMP,
        calib_id="c1",
        judge_id="j1",
        model=model,
        prompt_version=prompt_version,
        hunted_set_id="h",
        n_pos=80,
        n_neg=80,
        tpr=Bound(point=0.95, lb95=0.90, n=80),
        tnr=Bound(point=0.95, lb95=0.90, n=80),
        youden_j=0.90,
        result=result,  # type: ignore[arg-type]
    )


def test_uncalibrated_judge_is_refused():
    """Negative: a REFUSED calibration record → UncalibratedScoreError (no bare score)."""
    judge = _Judge("j1", "bedrock/claude-sonnet-4-6", "v1")
    uncertified = _record("REFUSED", model="bedrock/claude-sonnet-4-6", prompt_version="v1")

    with pytest.raises(UncalibratedScoreError):
        enforce_score_gate(uncertified, judge)


def test_prompt_version_drift_is_refused():
    """Negative: a CERTIFIED record whose pinned prompt_version drifted no longer
    certifies."""
    judge = _Judge("j1", "bedrock/claude-sonnet-4-6", "v2")  # judge moved to v2
    stale = _record("CERTIFIED", model="bedrock/claude-sonnet-4-6", prompt_version="v1")

    with pytest.raises(UncalibratedScoreError):
        enforce_score_gate(stale, judge)


def test_certified_matching_judge_passes():
    """Positive: a CERTIFIED record whose pins match the judge admits scoring (gate
    returns)."""
    judge = _Judge("j1", "bedrock/claude-sonnet-4-6", "v1")
    certified = _record("CERTIFIED", model="bedrock/claude-sonnet-4-6", prompt_version="v1")

    # Returns cleanly (no raise) — the only path that admits a score.
    enforce_score_gate(certified, judge)
