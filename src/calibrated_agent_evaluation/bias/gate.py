# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bias promotion gate + the calibration-record link.

The bias gate owns its own refusal end-to-end -- it does not reopen the enforcement gate.
``enforce_bias_gate`` refuses promotion when the bias battery is absent, stale, or ``BIASED``,
recording a ``hard_refusal``
:class:`~calibrated_agent_evaluation.enforce.decision.EnforcementDecision` (reused from enforcement)
for the audit trail. ``link_battery_to_calibration`` fills the ``bias_battery_ref`` slot left on the
``CalibrationRecord``.
"""

from __future__ import annotations

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.bias.errors import (
    BiasedJudgeError,
    MissingBiasBatteryError,
    StaleBiasBatteryError,
)
from calibrated_agent_evaluation.bias.record import BiasBatteryRecord
from calibrated_agent_evaluation.calibrate import CalibrationRecord
from calibrated_agent_evaluation.calibrate.engine import Judge, pin_mismatch
from calibrated_agent_evaluation.enforce.decision import EnforcementDecision


def link_battery_to_calibration(
    calibration_record: CalibrationRecord, battery: BiasBatteryRecord
) -> CalibrationRecord:
    """Return a copy of ``calibration_record`` with ``bias_battery_ref`` set to it.

    Frozen dataclass -> ``dataclasses.replace``. Set for both CLEAR and BIASED batteries so a run
    is always auditable via the calibration record.
    """
    return replace(calibration_record, bias_battery_ref=battery.battery_id)


def enforce_bias_gate(
    calibration_record: CalibrationRecord, judge: Judge, battery: BiasBatteryRecord
) -> None:
    """Refuse promotion unless the judge has a fresh, CLEAR bias battery.

    Refuses when (a) ``bias_battery_ref`` is null -> :class:`MissingBiasBatteryError`;
    (b) the referenced battery's ``(model, prompt_version)`` differs from ``judge`` or its id does
    not match the record's ref -> :class:`StaleBiasBatteryError` (treated as absent); (c) the
    battery verdict is ``BIASED`` -> :class:`BiasedJudgeError`. Returns cleanly otherwise. Each
    refusal carries a recorded ``hard_refusal`` decision.

    Args:
        calibration_record: The judge's calibration record, which references its battery.
        judge: The judge being promoted.
        battery: The bias battery record.

    Raises:
        MissingBiasBatteryError: The calibration record references no battery.
        StaleBiasBatteryError: The battery is for another model, prompt version or record.
        BiasedJudgeError: The battery's verdict is BIASED.
    """
    if calibration_record.bias_battery_ref is None:
        raise MissingBiasBatteryError(
            f"judge {judge.judge_id!r} has no bias battery (bias_battery_ref is null); "
            "refusing to promote on unproven bias. Run the bias battery first.",
            decision=_refusal("BIAS-2", "no bias battery run", judge, battery),
        )
    mismatch = pin_mismatch(battery, judge)
    if mismatch is not None or calibration_record.bias_battery_ref != battery.battery_id:
        why = mismatch or (
            f"the calibration record references battery {calibration_record.bias_battery_ref!r}, "
            f"not {battery.battery_id!r}"
        )
        raise StaleBiasBatteryError(
            f"the bias battery for judge {judge.judge_id!r} is stale and treated as absent: {why}. "
            "Run the bias battery again for the current judge.",
            decision=_refusal("BIAS-3", "stale bias battery", judge, battery),
        )
    if battery.verdict == "BIASED":
        raise BiasedJudgeError(
            f"judge {judge.judge_id!r} failed the bias gate on must-pass bias(es) "
            f"{list(battery.failed_biases)}; refusing promotion despite calibration. Fix the "
            "judge prompt and run the bias battery again.",
            decision=_refusal("BIAS-1", "biased judge", judge, battery),
        )


def _refusal(
    rule: str, reason: str, judge: Judge, battery: BiasBatteryRecord
) -> EnforcementDecision:
    """Build the hard_refusal audit record for a bias-gate refusal."""
    return EnforcementDecision(
        rule=rule,
        tier="hard_refusal",
        inputs={
            "judge_id": judge.judge_id,
            "model": judge.model,
            "prompt_version": judge.prompt_version,
            "battery_id": battery.battery_id,
            "battery_verdict": battery.verdict,
            "failed_biases": ",".join(battery.failed_biases),
        },
        outcome="refused",
        message=f"bias gate refused promotion of {judge.judge_id!r}: {reason}",
    )
