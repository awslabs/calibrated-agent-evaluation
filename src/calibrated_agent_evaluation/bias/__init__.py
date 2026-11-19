# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bias-check promotion gate -- the third judge-trust gate.

Runs the CALM bias battery (must-pass: length/self-preference/sycophancy; advisory: distraction/
bandwagon/compassion-fade; positional pairwise-only) over controlled probe items. Each probe is
run with a placebo repeat, and its magnitude is the flip rate beyond the placebo's, bounded above
by Newcombe's paired-difference interval. It refuses promotion of a judge that passed calibration
but fails a must-pass bias -- filling the ``bias_battery_ref`` slot left on the calibration record.

Pure methodology plane: no vendor SDK type appears here. The judge is run through an injected
:class:`~calibrated_agent_evaluation.bias.probe.ProbeRunner` port. The shipped concrete runner is
:class:`calibrated_agent_evaluation.judges.JudgeProbeRunner`, which reaches the model through
``execute/``'s ``JudgeModelBackend`` seam, so ``bias/`` itself imports no vendor SDK.
"""

from __future__ import annotations

from calibrated_agent_evaluation.bias.battery import run_battery
from calibrated_agent_evaluation.bias.errors import (
    BiasedJudgeError,
    BiasError,
    BiasStoreError,
    EmptyProbeError,
    IncompleteBatteryError,
    MissingBiasBatteryError,
    MissingBiasThresholdError,
    ProbeExecutionError,
    StaleBiasBatteryError,
)
from calibrated_agent_evaluation.bias.gate import enforce_bias_gate, link_battery_to_calibration
from calibrated_agent_evaluation.bias.loader import load_probe_set, load_thresholds
from calibrated_agent_evaluation.bias.mitigate import order_swap_and_tie
from calibrated_agent_evaluation.bias.probe import BiasProbe, ProbeInput, ProbeRunner, ProbeVerdict
from calibrated_agent_evaluation.bias.record import (
    ADVISORY,
    MUST_PASS,
    BiasBatteryRecord,
    BiasName,
    BiasVerdict,
    ProbeResult,
    ProbeStatus,
)
from calibrated_agent_evaluation.bias.store import dump_battery, load_battery
from calibrated_agent_evaluation.bias.thresholds import BiasThresholds
from calibrated_agent_evaluation.bias.version import BIAS_SCHEMA_VERSION

__all__ = [
    "ADVISORY",
    "BIAS_SCHEMA_VERSION",
    "MUST_PASS",
    "BiasBatteryRecord",
    "BiasError",
    "BiasName",
    "BiasProbe",
    "BiasStoreError",
    "BiasThresholds",
    "BiasVerdict",
    "BiasedJudgeError",
    "EmptyProbeError",
    "IncompleteBatteryError",
    "MissingBiasBatteryError",
    "MissingBiasThresholdError",
    "ProbeExecutionError",
    "ProbeInput",
    "ProbeResult",
    "ProbeRunner",
    "ProbeStatus",
    "ProbeVerdict",
    "StaleBiasBatteryError",
    "dump_battery",
    "enforce_bias_gate",
    "link_battery_to_calibration",
    "load_battery",
    "load_probe_set",
    "load_thresholds",
    "order_swap_and_tie",
    "run_battery",
]
