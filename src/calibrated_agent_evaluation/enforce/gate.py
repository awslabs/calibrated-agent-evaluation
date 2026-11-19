# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The score gate + the error-analysis-first guard.

No certifying calibration -> no score. The calibration verdict is *read* from
the calibration record (`certifies`), never recomputed (single-source-of-truth).
Scoring a target with no recorded human labels warns that error analysis precedes
scoring -- a directional nudge, not a refusal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from calibrated_agent_evaluation.enforce.decision import EnforcementDecision
from calibrated_agent_evaluation.enforce.errors import UncalibratedScoreError

if TYPE_CHECKING:
    from calibrated_agent_evaluation.calibrate import CalibrationRecord
    from calibrated_agent_evaluation.calibrate.engine import Judge


def enforce_score_gate(record: CalibrationRecord, judge: Judge) -> None:
    """Refuse a score for an uncertified judge.

    Returns cleanly when ``certifies(record, judge)`` holds (a certified record whose pinned
    ``(model, prompt_version)`` still matches the judge). Otherwise records a ``hard_refusal``
    :class:`EnforcementDecision` and raises :class:`UncalibratedScoreError` carrying it -- no score
    is computed or displayed.
    """
    # Imported here: the calibration engine needs the [stats] extra, and the warning below must not.
    from calibrated_agent_evaluation.calibrate import certifies
    from calibrated_agent_evaluation.calibrate.engine import pin_mismatch

    if certifies(record, judge):
        return
    decision = EnforcementDecision(
        rule="ENF-1",
        tier="hard_refusal",
        inputs={
            "judge_id": judge.judge_id,
            "model": judge.model,
            "prompt_version": judge.prompt_version,
            "record_result": record.result,
            "record_model": record.model,
            "record_prompt_version": record.prompt_version,
        },
        outcome="refused",
        message=(
            f"judge {judge.judge_id!r} has no certifying calibration, so no score is computed: "
            + (
                pin_mismatch(record, judge)
                or f"its calibration record {record.calib_id!r} is {record.result}; calibrate the "
                "judge against gold labels first"
            )
        ),
    )
    raise UncalibratedScoreError(decision.message, decision=decision)


def warn_error_analysis_first(judge_id: str, human_label_count: int) -> EnforcementDecision | None:
    """Warn that error analysis precedes scoring when no human labels exist yet.

    Returns a ``warning`` decision when ``human_label_count == 0`` (observe -> open-code -> cluster
    -> evaluate comes before pick-metric -> score); returns ``None`` when labels already exist.
    """
    if human_label_count > 0:
        return None
    return EnforcementDecision(
        rule="PRIN-3",
        tier="warning",
        inputs={"judge_id": judge_id, "human_label_count": str(human_label_count)},
        outcome="warned",
        message=(
            f"scoring judge {judge_id!r} before any human labels exist: error analysis "
            "(observe -> open-code -> cluster -> evaluate) precedes scoring."
        ),
        nudge="Run error analysis and record human labels before requesting a score.",
    )
