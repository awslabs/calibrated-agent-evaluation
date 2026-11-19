# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The three-gate judge-trust chain: whether a judge may write a score.

A judge may write a Score only after it clears **all three** gates, in order, on one
``CalibrationRecord``:

  1. ``certifies`` -- the 95% lower bound on both TPR and TNR cleared the 0.80 floor.
  2. ``enforce_bias_gate`` -- the CALM bias battery verdict is ``CLEAR`` (not ``BIASED``),
     the battery matches the judge's ``(model, prompt_version)``, and it is linked.
  3. ``enforce_score_gate`` -- the enforcement layer admits the certified record.

:func:`admit_judge` runs them in order and returns a :class:`GateDecisionRef` -- ``admitted`` only
when all three pass, else a refused ref carrying the first failure's reason (and the caller writes
**no** Score from a refused judge). ``calibrated-agent-evaluation judge``,
``calibrated-agent-evaluation run`` and the flagship proof all admit judges through it.
``run_judge`` then re-checks ``enforce_score_gate`` at write-back, so the gate is enforced twice --
here as an admission decision, and again at the model call.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import ConfigDict

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.bias import (
    BiasBatteryRecord,
    BiasError,
    enforce_bias_gate,
    link_battery_to_calibration,
)
from calibrated_agent_evaluation.calibrate import CalibrationRecord, certifies
from calibrated_agent_evaluation.calibrate.engine import Judge, pin_mismatch
from calibrated_agent_evaluation.enforce.errors import EnforcementError
from calibrated_agent_evaluation.enforce.gate import enforce_score_gate

__all__ = ["BiasVerdict", "GateDecisionRef", "admit_judge"]

BiasVerdict = Literal["CLEAR", "BIASED", "REFUSED"]


class GateDecisionRef(CalibratedAgentEvaluationModel):
    """The three-gate judge-trust outcome for one judge.

    A judge is admitted to score only when all three gates pass -- ``certify`` then
    ``enforce_bias_gate`` then ``enforce_score_gate``. A refused ref carries the
    ``refusal_reason`` and means no Score was written from that judge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = EnforcementError

    judge_id: str
    certified: bool
    bias_verdict: BiasVerdict
    score_gate_admitted: bool
    refusal_reason: str | None = None

    @property
    def admitted(self) -> bool:
        """True iff every gate passed -- the only state in which a Score may be written."""
        return self.certified and self.bias_verdict == "CLEAR" and self.score_gate_admitted


def admit_judge(
    judge: Judge, calib: CalibrationRecord, battery: BiasBatteryRecord
) -> GateDecisionRef:
    """Run the three-gate judge-trust chain and return the decision.

    Returns an *admitted* :class:`GateDecisionRef` iff ``certifies`` then
    ``enforce_bias_gate`` then ``enforce_score_gate`` all pass; otherwise a
    *refused* ref carrying the first failing gate's reason. A refused ref means no Score may be
    written from
    this judge -- the caller checks ``ref.admitted`` before calling ``run_judge`` with a store.
    """
    # Gate 1 -- calibration certification. A record certifies this judge only if it is
    # CERTIFIED and still pinned to the judge's current (model, prompt_version).
    certified = certifies(calib, judge)
    if not certified:
        return GateDecisionRef(
            judge_id=judge.judge_id,
            certified=False,
            bias_verdict="REFUSED",
            score_gate_admitted=False,
            refusal_reason=(
                f"calibration gate refused: record {calib.calib_id!r} does not certify judge "
                f"{judge.judge_id!r} at {judge.model}@{judge.prompt_version}: "
                + (pin_mismatch(calib, judge) or f"its result is {calib.result}")
            ),
        )

    # Gate 2 -- the CALM bias battery. enforce_bias_gate raises on a BIASED / stale /
    # missing battery; we link it to the calibration record first (the bias-promotion shape).
    bias_verdict: BiasVerdict = "CLEAR"
    try:
        linked = link_battery_to_calibration(calib, battery)
        enforce_bias_gate(linked, judge, battery)
    except BiasError as exc:
        bias_verdict = "BIASED" if battery.verdict == "BIASED" else "REFUSED"
        return GateDecisionRef(
            judge_id=judge.judge_id,
            certified=True,
            bias_verdict=bias_verdict,
            score_gate_admitted=False,
            refusal_reason=f"bias gate refused: {type(exc).__name__}: {exc}",
        )

    # Gate 3 -- the enforcement score gate. Returns cleanly iff the certified record admits
    # a score; raises UncalibratedScoreError otherwise.
    try:
        enforce_score_gate(calib, judge)
    except EnforcementError as exc:
        return GateDecisionRef(
            judge_id=judge.judge_id,
            certified=True,
            bias_verdict=bias_verdict,
            score_gate_admitted=False,
            refusal_reason=f"score gate refused: {type(exc).__name__}: {exc}",
        )

    return GateDecisionRef(
        judge_id=judge.judge_id,
        certified=True,
        bias_verdict="CLEAR",
        score_gate_admitted=True,
        refusal_reason=None,
    )
