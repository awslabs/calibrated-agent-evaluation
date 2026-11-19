# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bias battery: run the CALM probes and produce a pinned CLEAR/BIASED record.

Orchestration only -- the numeric bound comes from ``newcombe_paired_diff``, the judge is run
through the injected :class:`~calibrated_agent_evaluation.bias.probe.ProbeRunner` port, and the
gate/refusal lives in :mod:`calibrated_agent_evaluation.bias.gate`.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from calibrated_agent_evaluation._provenance import capture_provenance
from calibrated_agent_evaluation.bias.errors import EmptyProbeError, IncompleteBatteryError
from calibrated_agent_evaluation.bias.probe import BiasProbe, ProbeRunner, ProbeTally
from calibrated_agent_evaluation.bias.record import (
    MUST_PASS,
    BiasBatteryRecord,
    BiasName,
    BiasVerdict,
    ProbeResult,
)
from calibrated_agent_evaluation.bias.thresholds import BiasThresholds
from calibrated_agent_evaluation.calibrate.engine import Judge, fingerprint_of
from calibrated_agent_evaluation.stats.proportions import newcombe_paired_diff

# The bound a directional excess must stay under, for every bias whose probe declares a direction.
# Fixed before any judge was run: at 30
# trials with no excess the bound is about 0.11, so a clean judge clears it, while a perturbation
# that pushes a quarter of trials toward the favored verdict beyond noise does not.
DIRECTION_THRESHOLD = 0.20

_NOT_APPLICABLE_DETAIL = (
    "positional bias is a pairwise-only phenomenon; no-op on the single-verdict judge path"
)


def _positional_not_applicable(probe: BiasProbe) -> ProbeResult:
    """The positional probe's result on a single-verdict path -- recorded, never gated."""
    return ProbeResult(
        bias=probe.bias,
        status="not_applicable",
        n=0,
        biased_count=0,
        magnitude=0.0,
        magnitude_ub95=0.0,
        threshold=None,
        passed=None,
        # nosemgrep: is-function-without-parentheses  -- bool dataclass field, not a method
        is_must_pass=probe.is_must_pass,
        detail=_NOT_APPLICABLE_DETAIL,
    )


def _dispatch_probe(
    probe: BiasProbe,
    runner: ProbeRunner,
    judge: Judge,
    thresholds: BiasThresholds,
    *,
    pairwise: bool,
) -> ProbeResult:
    """Route one probe to scoring, or record it as not applicable.

    A pairwise-only probe off the pairwise path is recorded ``not_applicable``; everything else is
    scored and bounded by :func:`_run_probe`.
    """
    if probe.pairwise and not pairwise:
        return _positional_not_applicable(probe)
    return _run_probe(probe, runner, judge, thresholds)


def _run_probe(
    probe: BiasProbe, runner: ProbeRunner, judge: Judge, thresholds: BiasThresholds
) -> ProbeResult:
    """Score one probe and bound its magnitude against the model-relative threshold.

    Resolves the threshold FIRST so a missing must-pass threshold fails closed *before* any judge
    is run; a ``ProbeExecutionError`` from the runner propagates.
    """
    threshold: float | None
    # nosemgrep: is-function-without-parentheses  -- bool dataclass field, not a method
    if probe.is_must_pass:
        threshold = thresholds.threshold_for(judge.model, probe.bias)  # raises if unconfigured
    elif thresholds.has_threshold(judge.model, probe.bias):
        threshold = thresholds.threshold_for(judge.model, probe.bias)
    else:
        threshold = None  # advisory bias with no configured threshold -- recorded, not gated

    tally = probe.score(runner, judge.model, judge.prompt_version)
    n = tally.n
    if n == 0 and probe.is_must_pass:  # nosemgrep: is-function-without-parentheses
        raise EmptyProbeError(
            f"must-pass bias {probe.bias!r} scored 0 probe trials, so its magnitude is unmeasured; "
            "refusing rather than certifying an unrun probe (check the probe's fixture set and the "
            "runner's responses)"
        )
    magnitude, magnitude_ub95 = _excess_over_placebo(tally)
    direction = _direction(tally)
    passed: bool | None = None
    if threshold is not None:
        passed = magnitude_ub95 <= threshold and direction.get("direction_passed") is not False
    return ProbeResult(
        bias=probe.bias,
        status="run",
        n=n,
        biased_count=tally.biased_count,
        magnitude=magnitude,
        magnitude_ub95=magnitude_ub95,
        threshold=threshold,
        passed=passed,
        # nosemgrep: is-function-without-parentheses  -- bool dataclass field, not a method
        is_must_pass=probe.is_must_pass,
        tie_count=tally.tie_count,
        placebo_count=tally.placebo_count,
        both_count=tally.both_count,
        direction_n=tally.direction_n,
        direction_count=tally.direction_count,
        direction_placebo_count=tally.direction_placebo_count,
        direction_both_count=tally.direction_both_count,
        **direction,
    )


def _direction(tally: ProbeTally) -> dict[str, Any]:
    """The directional test's fields for a probe whose trials declare a direction, else none.

    The same paired bound as the disagreement test, read from the directional table: trials whose
    perturbed input moved toward the favored verdict against trials whose placebo repeat made the
    same move from the same starting verdict.
    """
    if tally.direction_n == 0:
        return {}
    magnitude, upper = _paired_excess(
        tally.direction_n,
        tally.direction_count,
        tally.direction_placebo_count,
        tally.direction_both_count,
    )
    return {
        "direction_magnitude": magnitude,
        "direction_ub95": upper,
        "direction_threshold": DIRECTION_THRESHOLD,
        "direction_passed": upper <= DIRECTION_THRESHOLD,
    }


def _excess_over_placebo(tally: ProbeTally) -> tuple[float, float]:
    """The trials' disagreement rate in excess of the placebos', and its one-sided 95% upper bound.

    Gated on the UPPER bound, not the point and NOT the lower bound: magnitude is a badness
    measure that must clear a ceiling, so the conservative edge is the upper one. A trial and its
    placebo are judged on the same item, so the two rates are paired, and the bound is Newcombe's
    hybrid-score interval for a paired difference: it keeps its coverage at the battery's small
    trial counts and does not collapse when a discordant cell is empty, as a Wald interval would.
    Its two-sided 90% upper limit is the one-sided 95% bound, the same convention as the
    single-rate Wilson bound this replaced. Each bias is gated at its own bound with no family
    correction, as before. An excess below 0 (placebos disagreeing more than trials) reads as no
    excess. At ``n == 0`` the bound is 1.0: an unmeasured rate is not a small one.
    """
    return _paired_excess(tally.n, tally.biased_count, tally.placebo_count, tally.both_count)


def _paired_excess(n: int, trial: int, placebo: int, both: int) -> tuple[float, float]:
    """The excess of ``trial`` over ``placebo`` events on ``n`` paired items, and its bound."""
    if n == 0:
        return 0.0, 1.0
    a = both
    b = trial - a
    c = placebo - a
    d = n - a - b - c
    diff, _, upper = newcombe_paired_diff(a, b, c, d, alpha=0.10)
    return max(diff, 0.0), max(upper, 0.0)


def _refuse_unearned_clear(probes: Sequence[BiasProbe], results: Sequence[ProbeResult]) -> None:
    """Refuse a CLEAR verdict that a must-pass bias was never measured for."""
    measured = {r.bias for r in results if r.is_must_pass and r.status == "run"}
    missing = tuple(sorted(MUST_PASS - measured))
    if not missing:
        return
    downgraded = sorted({p.bias for p in probes if p.bias in missing and not p.is_must_pass})
    why = (
        f"; the probe(s) for {downgraded} are marked advisory (must_pass false)"
        if downgraded
        else ""
    )
    raise IncompleteBatteryError(
        f"the bias battery ran no must-pass probe for {list(missing)}{why}, so a CLEAR verdict "
        "would vouch for biases that were never measured. Add a must-pass probe for each of "
        f"{sorted(MUST_PASS)} and run the battery again",
        missing=missing,
    )


def run_battery(
    judge: Judge,
    runner: ProbeRunner,
    thresholds: BiasThresholds,
    probes: Sequence[BiasProbe],
    *,
    pairwise: bool = False,
    battery_id: str | None = None,
    sampling: Mapping[str, float] | None = None,
) -> BiasBatteryRecord:
    """Run the bias battery for ``judge`` and return a pinned CLEAR/BIASED record.

    The positional probe runs only on a pairwise path; on a single-verdict path it is recorded as
    ``not_applicable``. Must-pass exceedances (a must-pass probe whose ``magnitude_ub95`` exceeds
    its threshold, or whose ``direction_ub95`` exceeds :data:`DIRECTION_THRESHOLD`) drive the
    verdict to ``BIASED``; advisory exceedances are recorded but never
    change the verdict. ``sampling`` is the sampling parameters the runner's judge sends (see
    :meth:`calibrated_agent_evaluation.judges.ExecutableJudge.sampling`), recorded on the provenance
    stamp. Every trial is run with its placebo, so a battery makes ``len(trial) - 1`` extra judge
    calls per trial. Raises
    :class:`~calibrated_agent_evaluation.bias.errors.MissingBiasThresholdError` or
    :class:`~calibrated_agent_evaluation.bias.errors.ProbeExecutionError` rather than returning a
    partial CLEAR, and :class:`~calibrated_agent_evaluation.bias.errors.IncompleteBatteryError` when
    ``probes`` is empty, or when the verdict would be CLEAR while a must-pass bias
    (:data:`MUST_PASS`) had no must-pass probe run. A measured must-pass exceedance is still
    reported BIASED from a partial battery.

    Args:
        judge: The judge identity the record pins to.
        runner: Runs the judge on one probe input.
        thresholds: The per-bias thresholds.
        probes: The probes to run.
        pairwise: The judge compares two answers, so the positional probe applies.
        battery_id: The record id; generated when omitted.
        sampling: The sampling parameters the judge sends, recorded on the provenance stamp.

    Returns:
        The CLEAR or BIASED battery record.

    Raises:
        IncompleteBatteryError: No probes, or a CLEAR verdict with a must-pass bias unmeasured.
        MissingBiasThresholdError: A probe's bias has no threshold.
        ProbeExecutionError: The runner failed on a probe input.
    """
    if not probes:
        raise IncompleteBatteryError(
            "the bias battery has no probes, so nothing was measured and no verdict can be "
            f"given; a verdict needs a must-pass probe for each of {sorted(MUST_PASS)}. Point the "
            "battery at a probe directory that holds them",
            missing=tuple(sorted(MUST_PASS)),
        )
    results = [
        _dispatch_probe(probe, runner, judge, thresholds, pairwise=pairwise) for probe in probes
    ]

    failed_biases: tuple[BiasName, ...] = tuple(
        r.bias
        for r in results
        if r.is_must_pass and r.passed is False  # nosemgrep: is-function-without-parentheses
    )
    if not failed_biases:
        _refuse_unearned_clear(probes, results)
    verdict: BiasVerdict = "BIASED" if failed_biases else "CLEAR"
    return BiasBatteryRecord(
        battery_id=battery_id or uuid.uuid4().hex,
        judge_id=judge.judge_id,
        model=judge.model,
        prompt_version=judge.prompt_version,
        judge_fingerprint=fingerprint_of(judge),
        is_pairwise=pairwise,
        results=tuple(results),
        verdict=verdict,
        failed_biases=failed_biases,
        provenance=capture_provenance(
            thresholds={
                **{r.bias: r.threshold for r in results if r.threshold is not None},
                **{
                    f"{r.bias}_direction": r.direction_threshold
                    for r in results
                    if r.direction_threshold is not None
                },
            },
            sampling=sampling,
        ),
    )
