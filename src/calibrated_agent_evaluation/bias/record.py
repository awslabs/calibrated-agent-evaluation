# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bias-battery data types -- the gate evidence and its inputs.

Pure, frozen data holders (the "typed functional core" house style): the battery orchestration
and gate logic live in :mod:`calibrated_agent_evaluation.bias.battery` /
:mod:`calibrated_agent_evaluation.bias.gate`, never here. No vendor SDK type appears.

  * :class:`ProbeResult` -- one bias probe's outcome: the measured magnitude, the one-sided 95%
    upper bound on its excess over the placebo (the *gated* number), the model-relative threshold
    applied, and pass/fail.
  * :class:`BiasBatteryRecord` -- the reproducible gate evidence, pinned to ``(model,
    prompt_version)``, carrying the per-probe results and the overall CLEAR/BIASED verdict.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import ConfigDict, Field

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._provenance import ProvenanceStamp
from calibrated_agent_evaluation.bias.errors import BiasError
from calibrated_agent_evaluation.bias.version import BIAS_SCHEMA_VERSION

# The CALM completeness baseline, split by gating strength.
BiasName = Literal[
    "length",
    "self_preference",
    "sycophancy",  # must-pass (hard gate)
    "distraction",
    "bandwagon",
    "compassion_fade",  # advisory (recorded, non-gating in v1)
    "positional",  # pairwise-only
]
MUST_PASS: frozenset[BiasName] = frozenset({"length", "self_preference", "sycophancy"})
ADVISORY: frozenset[BiasName] = frozenset({"distraction", "bandwagon", "compassion_fade"})

BiasVerdict = Literal["CLEAR", "BIASED"]
# not_applicable = positional probe on a single-verdict judge path.
ProbeStatus = Literal["run", "not_applicable"]


class ProbeResult(CalibratedAgentEvaluationModel):
    """One bias probe's outcome.

    The gate reads ``magnitude_ub95``, a one-sided 95% **upper** bound on how much more often the
    judge's verdicts disagree across the biasing dimension than across a placebo that varies
    nothing, never the point ``magnitude``: a bias is cleared only when we are confident it is
    small. The placebo nets out the judge's own sampling noise, which is not bias.

    It read the *lower* bound until 2026-09-28, rationalized as "a judge is declared biased only
    when we are confident the bias is real". That is the right conservatism for a rate that must
    clear a floor and the wrong one for a rate that must stay under a ceiling. Bias magnitude is a
    badness measure, so gating its lower bound made the gate maximally permissive -- it meant
    "promote unless we are confident the judge IS biased", inverting the burden of proof, and it
    contradicted the bias gate's own rule, which gates the *measured* magnitude.

    A probe whose inputs declare a direction also carries the directional test: how much more
    often the perturbed input moves the verdict toward the bias-favored label than the placebo
    repeat does (``direction_magnitude``), its one-sided 95% upper bound (``direction_ub95``) and
    the bound it must stay under (``direction_threshold``). ``direction_passed`` is ``None`` when
    the probe declares no direction.

    ``passed`` is ``magnitude_ub95 <= threshold`` and, when the probe has a direction, the
    directional test passing too. It is ``None`` when the probe did not run (``not_applicable``)
    or no threshold applied (an unconfigured advisory bias). Only a probe with ``is_must_pass``
    can drive the overall verdict to ``BIASED``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = BiasError

    bias: BiasName
    status: ProbeStatus
    n: int  # probe trials evaluated (0 when not_applicable)
    biased_count: int  # trials whose verdicts disagreed across the biasing dimension
    magnitude: float  # trial disagreement rate in excess of the placebos', in [0, 1]
    magnitude_ub95: float  # one-sided 95% UPPER bound on that excess -- the gated number
    threshold: float | None  # model-relative threshold applied; None when not gated
    passed: bool | None  # magnitude_ub95 <= threshold; None when not gated
    is_must_pass: bool  # only a must-pass probe can drive the verdict
    tie_count: int = 0  # positional order-swap-and-tie disagreements
    detail: str = ""  # explanation (e.g. the not_applicable reason)
    placebo_count: int = 0  # placebos (nothing varied) whose verdicts disagreed
    both_count: int = 0  # trials where the trial and its placebo both disagreed
    direction_n: int = 0  # trials whose perturbed inputs declare a direction
    direction_count: int = 0  # trials whose perturbed input moved toward the favored verdict
    direction_placebo_count: int = 0  # trials whose placebo repeat made the same move
    direction_both_count: int = 0  # trials where both moved
    direction_magnitude: float | None = None  # trial moves in excess of placebo moves
    direction_ub95: float | None = None  # one-sided 95% upper bound on that excess
    direction_threshold: float | None = None  # the bound direction_ub95 must not exceed
    direction_passed: bool | None = None  # direction_ub95 <= threshold; None: no direction


class BiasBatteryRecord(CalibratedAgentEvaluationModel):
    """The reproducible bias-gate evidence.

    Pins the judge's exact ``(model, prompt_version)`` -- a change to either makes this record
    "absent" for the changed judge, the same version-invalidation discipline as the
    ``CalibrationRecord``. Carries every probe's inputs/magnitude/threshold/verdict so a promotion
    decision is reconstructable from stored artifacts alone. Referenced by
    ``CalibrationRecord.bias_battery_ref`` (a judge-bound artifact referenced by id).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = BiasError

    battery_id: str
    judge_id: str
    model: str  # pinned judge model      ┐ a change to either makes
    prompt_version: str  # pinned judge prompt sha  ┘ this record absent for the judge
    is_pairwise: bool  # whether the judge uses a pairwise path
    results: tuple[ProbeResult, ...]
    verdict: BiasVerdict  # BIASED iff any must-pass ProbeResult.passed is False
    schema_version: str = BIAS_SCHEMA_VERSION
    failed_biases: tuple[BiasName, ...] = Field(default_factory=tuple)  # the must-pass exceedances
    judge_fingerprint: str | None = None  # see CalibrationRecord.judge_fingerprint
    # The per-bias thresholds the gate applied. The battery draws no random numbers.
    provenance: ProvenanceStamp
