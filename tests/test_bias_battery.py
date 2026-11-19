# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bias battery behavior.

The battery is exercised through the injected ``ProbeRunner`` port with a stub that returns canned
verdicts (a blessed test double per test-rubric rule 2 — the port is the contract). No vendor code
is touched; the gate/verdict logic is proven in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from _bias_helpers import run_complete_battery
from calibrated_agent_evaluation.bias import (
    MUST_PASS,
    BiasProbe,
    BiasThresholds,
    EmptyProbeError,
    IncompleteBatteryError,
    MissingBiasThresholdError,
    ProbeExecutionError,
    ProbeInput,
    ProbeVerdict,
    run_battery,
)
from calibrated_agent_evaluation.bias.record import BiasName


@dataclass(frozen=True)
class FakeJudge:
    judge_id: str = "orderbot-refund-judge"
    model: str = "test-model"
    prompt_version: str = "v1"


class StubRunner:
    """A canned ``ProbeRunner``: maps ``probe_id`` -> label; may raise on one id."""

    def __init__(self, labels: dict[str, str], *, raise_on: str | None = None) -> None:
        self._labels = labels
        self._raise_on = raise_on

    def run(self, judge_model: str, prompt_version: str, probe_input: ProbeInput) -> ProbeVerdict:
        if self._raise_on is not None and probe_input.probe_id == self._raise_on:
            raise ProbeExecutionError(f"stub execution failure on {probe_input.probe_id}")
        return ProbeVerdict(label=self._labels.get(probe_input.probe_id, "PASS"))


def _probe(bias: BiasName, *, must_pass: bool, n_trials: int, pairwise: bool = False) -> BiasProbe:
    trials = tuple(
        (ProbeInput(bias, f"{bias}-{i}-a", {}), ProbeInput(bias, f"{bias}-{i}-b", {}))
        for i in range(n_trials)
    )
    return BiasProbe(bias=bias, is_must_pass=must_pass, trials=trials, pairwise=pairwise)


def _labels(bias: BiasName, biased: set[int], n_trials: int) -> dict[str, str]:
    """Label map where trial ``i`` is 'biased' (verdicts disagree) iff ``i in biased``."""
    out: dict[str, str] = {}
    for i in range(n_trials):
        out[f"{bias}-{i}-a"] = "PASS"
        out[f"{bias}-{i}-b"] = "FAIL" if i in biased else "PASS"
    return out


def _thresholds(**by_bias: float) -> BiasThresholds:
    thresholds = cast("dict[BiasName, float]", dict(by_bias))
    return BiasThresholds(by_model={"test-model": thresholds})


def test_records_magnitude_upper_bound_and_pins_the_judge() -> None:
    """The gate reads the one-sided Wilson UPPER bound, and it is conservative in the right
    direction.

    2 of 4 length trials biased -> point magnitude 0.5, upper bound 0.818, lower bound 0.182. A 0.3
    threshold sits BELOW the point and BELOW the upper bound, so the probe must fail.

    This test previously asserted the opposite and called it the contract: the same inputs gave
    `passed is True` because the gate read the *lower* bound (0.182 <= 0.3). Bias magnitude must
    stay under a ceiling, so the lower bound is the anti-conservative edge -- gating it means
    "promote unless we are confident the judge IS biased", which inverts the burden of proof, and
    contradicts the battery's rule of gating on the measured magnitude. A judge biased on half
    of all probe trials cleared a 30% must-pass bar.
    """
    probe = _probe("length", must_pass=True, n_trials=4)
    runner = StubRunner(_labels("length", {0, 1}, 4))
    record = run_complete_battery(FakeJudge(), runner, _thresholds(length=0.3), [probe])

    result = record.results[0]
    assert result.magnitude == pytest.approx(0.5)
    assert result.magnitude_ub95 == pytest.approx(0.818, abs=0.001)
    assert result.magnitude_ub95 > result.magnitude, "the gated bound is ABOVE the point estimate"
    assert result.passed is False, "half the trials biased must not clear a 30% bar"
    assert record.model == "test-model"
    assert record.prompt_version == "v1"
    assert record.verdict == "BIASED"


def test_the_gated_bound_is_the_upper_not_the_lower_edge() -> None:
    """The inversion, asserted as its own fact so a future refactor cannot quietly flip it back.

    The two numbers the ticket verified by hand: `wilson_upper(5, 10) == 0.731` against a 0.30
    threshold, and `wilson_upper(3, 4) == 0.942` against 0.40. Both cleared under the lower bound
    (0.269 and 0.356 respectively) and both must now refuse.
    """
    ten = run_complete_battery(
        FakeJudge(),
        StubRunner(_labels("length", {0, 1, 2, 3, 4}, 10)),
        _thresholds(length=0.30),
        [_probe("length", must_pass=True, n_trials=10)],
    )
    assert ten.results[0].magnitude_ub95 == pytest.approx(0.731, abs=0.001)
    assert ten.results[0].passed is False

    four = run_complete_battery(
        FakeJudge(),
        StubRunner(_labels("sycophancy", {0, 1, 2}, 4)),
        _thresholds(sycophancy=0.40),
        [_probe("sycophancy", must_pass=True, n_trials=4)],
    )
    assert four.results[0].magnitude_ub95 == pytest.approx(0.942, abs=0.001)
    assert four.results[0].passed is False


def test_a_must_pass_probe_that_scored_nothing_refuses() -> None:
    """Zero trials must never clear a must-pass gate (fail-closed).

    `n == 0` used to give `magnitude_lb95 = 0.0` and therefore `passed = True` unconditionally, so
    an unrun must-pass probe certified the judge -- absence of evidence read as evidence of absence.
    """
    probe = _probe("length", must_pass=True, n_trials=0)
    with pytest.raises(EmptyProbeError, match="0 probe trials"):
        run_complete_battery(FakeJudge(), StubRunner({}), _thresholds(length=0.3), [probe])


def test_a_clean_judge_needs_enough_trials_to_clear_a_tight_threshold() -> None:
    """The sample size an upper-bound gate implies, made explicit.

    A perfectly clean probe (0 biased trials) has point magnitude 0.0 at any `n`, so under the old
    lower-bound gate 4 trials certified a 10% threshold. The upper bound at `k=0, n=4` is 0.403:
    four trials cannot establish that a rate is below 10%, and now the gate says so. 40 trials
    (upper 0.063) can.

    This is the honest cost of the fix, asserted rather than discovered later by whoever wonders why
    a clean judge stopped passing.
    """
    few = run_complete_battery(
        FakeJudge(),
        StubRunner(_labels("length", set(), 4)),
        _thresholds(length=0.1),
        [_probe("length", must_pass=True, n_trials=4)],
    )
    assert few.results[0].magnitude == pytest.approx(0.0)
    assert few.results[0].magnitude_ub95 == pytest.approx(0.403, abs=0.001)
    assert few.verdict == "BIASED", "4 trials cannot certify a 10% ceiling"

    enough = run_complete_battery(
        FakeJudge(),
        StubRunner(_labels("length", set(), 40)),
        _thresholds(length=0.1),
        [_probe("length", must_pass=True, n_trials=40)],
    )
    assert enough.results[0].magnitude_ub95 == pytest.approx(0.063, abs=0.001)
    assert enough.verdict == "CLEAR"


def test_clear_when_all_must_pass_within_threshold_biased_otherwise() -> None:
    # 40 trials rather than 4: an upper-bound gate needs enough evidence to clear a 0.1 ceiling at
    # all (see test_a_clean_judge_needs_enough_trials_to_clear_a_tight_threshold).
    probe = _probe("length", must_pass=True, n_trials=40)
    clean = run_complete_battery(
        FakeJudge(), StubRunner(_labels("length", set(), 40)), _thresholds(length=0.1), [probe]
    )
    assert clean.verdict == "CLEAR"

    # all 40 trials biased -> ub95 1.0, over the 0.1 threshold -> not CLEAR.
    biased = run_complete_battery(
        FakeJudge(),
        StubRunner(_labels("length", set(range(40)), 40)),
        _thresholds(length=0.1),
        [probe],
    )
    assert biased.verdict == "BIASED"
    assert biased.results[0].passed is False


def test_only_a_must_pass_bias_drives_the_verdict() -> None:
    # A must-pass exceedance flips the verdict...
    mp = _probe("sycophancy", must_pass=True, n_trials=4)
    runner = StubRunner(_labels("sycophancy", {0, 1, 2, 3}, 4))
    biased = run_complete_battery(FakeJudge(), runner, _thresholds(sycophancy=0.1), [mp])
    assert biased.verdict == "BIASED"
    assert biased.failed_biases == ("sycophancy",)

    # ...but an advisory exceedance (same magnitudes) does not.
    adv = _probe("bandwagon", must_pass=False, n_trials=4)
    adv_runner = StubRunner(_labels("bandwagon", {0, 1, 2, 3}, 4))
    clear = run_complete_battery(FakeJudge(), adv_runner, _thresholds(bandwagon=0.1), [adv])
    assert clear.verdict == "CLEAR"
    assert clear.results[0].passed is False  # recorded as failing its threshold, but non-gating
    assert clear.failed_biases == ()


def test_advisory_exceedance_is_flagged_but_keeps_clear() -> None:
    mp = _probe("length", must_pass=True, n_trials=40)
    adv = _probe("compassion_fade", must_pass=False, n_trials=4)
    runner = StubRunner(
        {**_labels("length", set(), 40), **_labels("compassion_fade", {0, 1, 2, 3}, 4)}
    )
    record = run_complete_battery(
        FakeJudge(), runner, _thresholds(length=0.1, compassion_fade=0.1), [mp, adv]
    )
    advisory = next(r for r in record.results if r.bias == "compassion_fade")
    assert advisory.passed is False  # the exceedance is recorded/flagged
    assert record.verdict == "CLEAR"  # ...but does not gate in v1


def test_missing_must_pass_threshold_fails_closed() -> None:
    probe = _probe("length", must_pass=True, n_trials=4)
    runner = StubRunner(_labels("length", set(), 4))
    with pytest.raises(MissingBiasThresholdError):
        run_complete_battery(
            FakeJudge(), runner, _thresholds(), [probe]
        )  # no length threshold for model


def test_positional_not_applicable_on_single_verdict_path() -> None:
    positional = _probe("positional", must_pass=False, n_trials=2, pairwise=True)
    runner = StubRunner({})  # never called for a not_applicable probe
    record = run_complete_battery(FakeJudge(), runner, _thresholds(), [positional], pairwise=False)
    result = record.results[0]
    assert result.status == "not_applicable"
    assert result.n == 0
    assert "pairwise-only" in result.detail
    assert record.verdict == "CLEAR"  # a not_applicable probe never affects the verdict


def test_positional_included_on_pairwise_path_excluded_otherwise() -> None:
    positional = _probe("positional", must_pass=False, n_trials=2, pairwise=True)
    # On the pairwise path the probe runs (order disagreement -> tie), and is recorded as run.
    pairwise_labels = {
        "positional-0-a": "A",
        "positional-0-b": "B",
        "positional-1-a": "A",
        "positional-1-b": "A",
    }
    run = run_complete_battery(
        FakeJudge(), StubRunner(pairwise_labels), _thresholds(), [positional], pairwise=True
    )
    positional_result = next(r for r in run.results if r.bias == "positional")
    assert positional_result.status == "run"
    assert positional_result.tie_count == 1  # trial 0 flipped with order -> tie

    off = run_complete_battery(
        FakeJudge(), StubRunner({}), _thresholds(), [positional], pairwise=False
    )
    assert next(r for r in off.results if r.bias == "positional").status == "not_applicable"


def test_probe_execution_failure_propagates_with_no_clear() -> None:
    probe = _probe("length", must_pass=True, n_trials=4)
    runner = StubRunner(_labels("length", set(), 4), raise_on="length-2-a")
    with pytest.raises(ProbeExecutionError):
        run_complete_battery(FakeJudge(), runner, _thresholds(length=0.2), [probe])


def test_an_empty_battery_is_refused_rather_than_clear() -> None:
    with pytest.raises(IncompleteBatteryError, match="no probes") as excinfo:
        run_battery(FakeJudge(), StubRunner({}), _thresholds(), [])
    assert set(excinfo.value.missing) == set(MUST_PASS)


def test_a_battery_missing_a_must_pass_bias_cannot_come_back_clear() -> None:
    probes = [
        _probe("length", must_pass=True, n_trials=40),
        _probe("self_preference", must_pass=True, n_trials=40),
    ]
    with pytest.raises(IncompleteBatteryError, match="sycophancy") as excinfo:
        run_battery(
            FakeJudge(), StubRunner({}), _thresholds(length=0.1, self_preference=0.1), probes
        )
    assert excinfo.value.missing == ("sycophancy",)


def test_a_must_pass_bias_probed_as_advisory_does_not_count() -> None:
    probes = [
        _probe("length", must_pass=True, n_trials=40),
        _probe("self_preference", must_pass=True, n_trials=40),
        _probe("sycophancy", must_pass=False, n_trials=40),
    ]
    thresholds = _thresholds(length=0.1, self_preference=0.1, sycophancy=0.1)
    with pytest.raises(IncompleteBatteryError, match="marked advisory"):
        run_battery(FakeJudge(), StubRunner({}), thresholds, probes)


def test_every_must_pass_bias_measured_and_clean_is_clear() -> None:
    probes = [_probe(b, must_pass=True, n_trials=40) for b in sorted(MUST_PASS)]
    thresholds = _thresholds(**dict.fromkeys(MUST_PASS, 0.1))
    record = run_battery(FakeJudge(), StubRunner({}), thresholds, probes)
    assert record.verdict == "CLEAR"
    assert {r.bias for r in record.results if r.status == "run"} == set(MUST_PASS)


def test_a_partial_battery_still_reports_a_measured_bias() -> None:
    probe = _probe("length", must_pass=True, n_trials=4)
    runner = StubRunner(_labels("length", {0, 1, 2, 3}, 4))
    record = run_battery(FakeJudge(), runner, _thresholds(length=0.1), [probe])
    assert record.verdict == "BIASED"
    assert record.failed_biases == ("length",)
