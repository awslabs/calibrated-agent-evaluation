# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete a bias battery under test with must-pass probes that always clear.

A battery refuses a CLEAR verdict unless every must-pass bias ran a must-pass probe. Tests that
exercise one probe's arithmetic pad the battery with a clean 40-trial probe per missing must-pass
bias, gated at the loosest threshold (1.0) and answered PASS throughout by the wrapping runner, so
the padding never reaches the test's own runner and never decides the verdict.
"""

from __future__ import annotations

from collections.abc import Sequence

from calibrated_agent_evaluation.bias import (
    MUST_PASS,
    BiasBatteryRecord,
    BiasProbe,
    BiasThresholds,
    ProbeInput,
    ProbeRunner,
    ProbeVerdict,
    run_battery,
)
from calibrated_agent_evaluation.bias.record import BiasName
from calibrated_agent_evaluation.calibrate.engine import Judge

PAD_PREFIX = "pad:"
_PAD_TRIALS = 40


class _PadAnsweringRunner:
    def __init__(self, inner: ProbeRunner) -> None:
        self._inner = inner

    def run(self, judge_model: str, prompt_version: str, probe_input: ProbeInput) -> ProbeVerdict:
        if probe_input.probe_id.startswith(PAD_PREFIX):
            return ProbeVerdict(label="PASS")
        return self._inner.run(judge_model, prompt_version, probe_input)


def complete_battery(
    probes: Sequence[BiasProbe], thresholds: BiasThresholds, *, model: str
) -> tuple[list[BiasProbe], BiasThresholds]:
    """``probes`` plus a clearing must-pass probe for each must-pass bias they leave out."""
    covered = {p.bias for p in probes if p.is_must_pass}
    padding = [
        BiasProbe(
            bias=bias,
            is_must_pass=True,
            trials=tuple(
                (
                    ProbeInput(bias, f"{PAD_PREFIX}{bias}-{i}-a", {}),
                    ProbeInput(bias, f"{PAD_PREFIX}{bias}-{i}-b", {}),
                )
                for i in range(_PAD_TRIALS)
            ),
        )
        for bias in sorted(MUST_PASS - covered)
    ]
    by_model = {m: dict(t) for m, t in thresholds.by_model.items()}
    row: dict[BiasName, float] = by_model.setdefault(model, {})
    for probe in padding:
        row.setdefault(probe.bias, 1.0)
    return [*probes, *padding], BiasThresholds(by_model=by_model)


def run_complete_battery(
    judge: Judge,
    runner: ProbeRunner,
    thresholds: BiasThresholds,
    probes: Sequence[BiasProbe],
    **kwargs: object,
) -> BiasBatteryRecord:
    """``run_battery`` with the must-pass biases ``probes`` leave out padded by clearing probes."""
    full, padded = complete_battery(probes, thresholds, model=judge.model)
    return run_battery(judge, _PadAnsweringRunner(runner), padded, full, **kwargs)  # type: ignore[arg-type]
