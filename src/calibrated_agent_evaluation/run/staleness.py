# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The passive staleness warning.

A judge is *stale* -- needs re-calibration -- when either of the two re-validation triggers fire:
the judge's pinned ``(model, prompt_version)`` no longer matches its
calibration record, OR input-distribution drift for its suite has crossed its threshold. When either
holds, :func:`check_staleness` emits a ``warning``-tier
:class:`~calibrated_agent_evaluation.run.report.RunDecision`.

Crucially this is **passive**: it is surfaced in the run report and NEVER hard-blocks
CI -- a staleness decision attached to a
:class:`~calibrated_agent_evaluation.run.report.RunGateDecision` never flips ``passed`` (the gate
reads only the per-mode lower bounds). There is no scheduler, no calendar-age trigger, and no
Slack/email -- those belong to a later active-cadence feature. The trigger is event-based only
(version drift + drift-threshold crossing), which keeps it deterministic and scheduler-free.
"""

from __future__ import annotations

from calibrated_agent_evaluation.calibrate import certifies
from calibrated_agent_evaluation.calibrate.engine import Judge, pin_mismatch
from calibrated_agent_evaluation.calibrate.record import CalibrationRecord
from calibrated_agent_evaluation.run.report import InputDriftReport, RunDecision


def check_staleness(
    judge: Judge,
    record: CalibrationRecord,
    input_drift: InputDriftReport | None = None,
) -> RunDecision | None:
    """Return a ``warning``-tier staleness :class:`RunDecision`, or ``None``.

    Warns when EITHER trigger fires:

      * **version drift** -- the record no longer certifies ``judge`` (its ``(model,
        prompt_version)`` changed, or it is not CERTIFIED), read via ``certifies``; or
      * **input drift** -- ``input_drift`` is supplied and any of its metrics crossed its threshold.

    Returns ``None`` when the version matches AND drift is sub-threshold (or absent). The decision
    is always ``warning`` tier -- staleness never hard-refuses (it is passive; a caller must not let
    flip a gate). Never raises.
    """
    version_drifted = not certifies(record, judge)
    drift_exceeded = input_drift is not None and input_drift.any_exceeded
    if not (version_drifted or drift_exceeded):
        return None

    reasons: list[str] = []
    if version_drifted:
        reasons.append(
            f"judge {judge.judge_id!r} ({judge.model}@{judge.prompt_version}) no longer matches "
            f"its calibration record ({record.model}@{record.prompt_version}, "
            f"result={record.result}): "
            + (pin_mismatch(record, judge) or "the record is not CERTIFIED")
        )
    if drift_exceeded:
        assert input_drift is not None  # narrowed by drift_exceeded  # noqa: S101
        crossed = [m.name for m in input_drift.metrics if m.exceeded]
        reasons.append(
            f"input-distribution drift on {input_drift.dimension!r} crossed threshold "
            f"({', '.join(crossed)})"
        )

    return RunDecision(
        rule="RUN-4",
        tier="warning",
        inputs={
            "judge_id": judge.judge_id,
            "version_drifted": str(version_drifted),
            "drift_exceeded": str(drift_exceeded),
        },
        outcome="warned",
        message=(
            f"suite for judge {judge.judge_id!r} may be stale and need re-calibration: "
            + "; ".join(reasons)
            + ". This warning never changes the gate; "
            "`calibrated-agent-evaluation run --fail-on-drift` gives it its own exit code."
        ),
    )
