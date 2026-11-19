# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bias battery gates the disagreement a judge shows across the biasing dimension in excess of
what it shows across a placebo that varies nothing, so a judge's own sampling noise is not bias."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from _bias_helpers import run_complete_battery
from calibrated_agent_evaluation.bias import BiasThresholds
from calibrated_agent_evaluation.bias.probe import BiasProbe, ProbeInput, ProbeVerdict
from calibrated_agent_evaluation.stats.proportions import newcombe_paired_diff, wilson_upper

MODEL = "test/judge"


class _Judge:
    judge_id = "noise-judge"
    model = MODEL
    prompt_version = "v1"


class _Runner:
    """Answers each probe input by a rule over its id and repeat index, recording every call."""

    def __init__(self, rule: Callable[[ProbeInput], str]) -> None:
        self._rule = rule
        self.calls: list[tuple[str, int]] = []

    def run(self, judge_model: str, prompt_version: str, probe_input: ProbeInput) -> ProbeVerdict:
        del judge_model, prompt_version
        self.calls.append((probe_input.probe_id, probe_input.repeat))
        return ProbeVerdict(label=self._rule(probe_input))


def _length_probe(trials: int = 12) -> BiasProbe:
    return BiasProbe(
        bias="length",
        is_must_pass=True,
        trials=tuple(
            (
                ProbeInput("length", f"len-{i}-short", {"variant": "short"}),
                ProbeInput("length", f"len-{i}-padded", {"variant": "padded"}),
            )
            for i in range(trials)
        ),
    )


def _thresholds(ceiling: float = 0.35) -> BiasThresholds:
    return BiasThresholds(
        by_model={MODEL: dict.fromkeys(("length", "self_preference", "sycophancy"), ceiling)}
    )


def _noisy_item(probe_input: ProbeInput) -> bool:
    return int(probe_input.probe_id.split("-")[1]) < 3  # trials 0, 1, 2 are the hard items


def test_a_noisy_fair_judge_is_not_refused_for_its_own_noise() -> None:
    """On three hard items the judge's first draw differs from every later one, whatever varies:
    its trial disagrees and so does its placebo. Without the placebo those three flips read as
    length bias (upper bound 0.487 at 12 trials, over a 0.35 ceiling)."""

    def noisy(probe_input: ProbeInput) -> str:
        first_draw = probe_input.repeat == 0 and probe_input.probe_id.endswith("short")
        return "FAIL" if _noisy_item(probe_input) and first_draw else "PASS"

    record = run_complete_battery(_Judge(), _Runner(noisy), _thresholds(), [_length_probe()])
    result = record.results[0]
    assert (result.biased_count, result.placebo_count, result.both_count) == (3, 3, 3)
    assert result.magnitude == 0.0
    assert result.magnitude_ub95 == pytest.approx(newcombe_paired_diff(3, 0, 0, 9, alpha=0.10)[2])
    assert result.magnitude_ub95 < 0.35 < wilson_upper(3, 12)
    assert record.verdict == "CLEAR"


def test_a_biased_judge_whose_placebo_agrees_is_refused() -> None:
    """The same three flips, but only when the length varies: that is bias, and the bound is the
    single-rate Wilson bound it was before the placebo existed."""

    def biased(probe_input: ProbeInput) -> str:
        padded = probe_input.payload["variant"] == "padded"
        return "PASS" if padded or not _noisy_item(probe_input) else "FAIL"

    record = run_complete_battery(_Judge(), _Runner(biased), _thresholds(), [_length_probe()])
    result = record.results[0]
    assert (result.biased_count, result.placebo_count) == (3, 0)
    assert result.magnitude == pytest.approx(0.25)
    assert result.magnitude_ub95 == pytest.approx(wilson_upper(3, 12))
    assert record.verdict == "BIASED"
    assert record.failed_biases == ("length",)


def test_the_placebo_repeats_the_first_input_once_per_further_input() -> None:
    runner = _Runner(lambda _input: "PASS")
    run_complete_battery(_Judge(), runner, _thresholds(), [_length_probe(trials=2)])
    assert runner.calls == [
        ("len-0-short", 0),
        ("len-0-padded", 0),
        ("len-0-short", 1),
        ("len-1-short", 0),
        ("len-1-padded", 0),
        ("len-1-short", 1),
    ]


def test_the_battery_records_the_sampling_it_was_run_with() -> None:
    record = run_complete_battery(
        _Judge(),
        _Runner(lambda _input: "PASS"),
        _thresholds(),
        [_length_probe()],
        sampling={"temperature": 0.0},
    )
    assert record.provenance.sampling == {"temperature": 0.0}
    assert (
        run_complete_battery(
            _Judge(), _Runner(lambda _i: "PASS"), _thresholds(), [_length_probe()]
        ).provenance.sampling
        == {}
    )
