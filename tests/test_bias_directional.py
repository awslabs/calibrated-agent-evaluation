# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bias battery also gates the direction of a judge's verdict changes, so a noisy judge whose
perturbed verdicts all move the same way cannot hide behind its own placebo disagreements."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from _bias_helpers import run_complete_battery
from calibrated_agent_evaluation.bias import BiasThresholds
from calibrated_agent_evaluation.bias.battery import DIRECTION_THRESHOLD
from calibrated_agent_evaluation.bias.errors import BiasStoreError, ProbeExecutionError
from calibrated_agent_evaluation.bias.loader import load_probe_set
from calibrated_agent_evaluation.bias.probe import BiasProbe, ProbeInput, ProbeVerdict
from calibrated_agent_evaluation.bias.store import dump_battery, load_battery

MODEL = "test/judge"
TRIALS = 200


class _Judge:
    judge_id = "direction-judge"
    model = MODEL
    prompt_version = "v1"


class _Runner:
    def __init__(self, rule: Callable[[ProbeInput], str]) -> None:
        self._rule = rule

    def run(self, judge_model: str, prompt_version: str, probe_input: ProbeInput) -> ProbeVerdict:
        del judge_model, prompt_version
        return ProbeVerdict(label=self._rule(probe_input))


def _probe(*, toward: str | None = "PASS", trials: int = TRIALS) -> BiasProbe:
    return BiasProbe(
        bias="length",
        is_must_pass=True,
        trials=tuple(
            (
                ProbeInput("length", f"len-{i}-short", {"variant": "short"}),
                ProbeInput("length", f"len-{i}-padded", {"variant": "padded"}, toward=toward),
            )
            for i in range(trials)
        ),
    )


def _thresholds() -> BiasThresholds:
    return BiasThresholds(
        by_model={MODEL: dict.fromkeys(("length", "self_preference", "sycophancy"), 0.35)}
    )


def _coin(probe_input: ProbeInput) -> str:
    """A fair coin per call, fixed by the input's id and repeat so the test is deterministic."""
    key = f"{probe_input.probe_id}/{probe_input.repeat}".encode()
    return "PASS" if hashlib.sha256(key).digest()[0] % 2 == 0 else "FAIL"


def _always_lenient_on_padded(probe_input: ProbeInput) -> str:
    if probe_input.payload["variant"] == "padded":
        return "PASS"
    return _coin(probe_input)


def test_a_noisy_judge_that_always_passes_the_padded_answer_fails_on_direction() -> None:
    record = run_complete_battery(
        _Judge(), _Runner(_always_lenient_on_padded), _thresholds(), [_probe()]
    )
    result = record.results[0]
    assert result.magnitude_ub95 <= 0.35  # the disagreement test alone lets this judge through
    assert result.direction_ub95 is not None
    assert result.direction_ub95 > DIRECTION_THRESHOLD
    assert result.direction_passed is False
    assert result.passed is False
    assert record.verdict == "BIASED"
    assert record.failed_biases == ("length",)


def test_a_noisy_fair_judge_passes_the_directional_test() -> None:
    record = run_complete_battery(_Judge(), _Runner(_coin), _thresholds(), [_probe()])
    result = record.results[0]
    assert result.direction_ub95 is not None
    assert result.direction_passed is True
    assert record.verdict == "CLEAR"


def test_a_probe_without_a_direction_keeps_only_the_disagreement_test() -> None:
    record = run_complete_battery(
        _Judge(), _Runner(_always_lenient_on_padded), _thresholds(), [_probe(toward=None)]
    )
    result = record.results[0]
    assert (result.direction_n, result.direction_ub95, result.direction_passed) == (0, None, None)
    assert record.verdict == "CLEAR"
    assert "length_direction" not in record.provenance.thresholds


def test_the_directional_bound_is_recorded_on_the_provenance_stamp() -> None:
    record = run_complete_battery(_Judge(), _Runner(_coin), _thresholds(), [_probe()])
    assert record.provenance.thresholds["length_direction"] == DIRECTION_THRESHOLD


def _write_probe(tmp_path: Path, inputs: list[dict[str, object]]) -> Path:
    doc = {
        "bias": "length",
        "must_pass": True,
        "trials": [{"trial_id": "t0", "question": "q", "inputs": inputs}],
    }
    (tmp_path / "length.json").write_text(json.dumps(doc), encoding="utf-8")
    return tmp_path


def test_a_probe_file_may_declare_the_direction_of_a_perturbed_input(tmp_path: Path) -> None:
    root = _write_probe(
        tmp_path,
        [
            {"probe_id": "a", "payload": {"answer": "x"}},
            {"probe_id": "b", "payload": {"answer": "x y"}, "toward": "PASS"},
        ],
    )
    (probe,) = load_probe_set(root)
    assert [i.toward for i in probe.trials[0]] == [None, "PASS"]
    assert all("toward" not in i.payload for i in probe.trials[0])


def test_the_unperturbed_first_input_cannot_declare_a_direction(tmp_path: Path) -> None:
    root = _write_probe(
        tmp_path,
        [
            {"probe_id": "a", "payload": {"answer": "x"}, "toward": "PASS"},
            {"probe_id": "b", "payload": {"answer": "x y"}},
        ],
    )
    with pytest.raises(ProbeExecutionError, match="toward"):
        load_probe_set(root)


def test_a_record_from_before_the_directional_test_is_refused(tmp_path: Path) -> None:
    record = run_complete_battery(_Judge(), _Runner(_coin), _thresholds(), [_probe(trials=4)])
    dump_battery([record], tmp_path, fmt="json")
    path = tmp_path / "bias_battery_record.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"4"', '"3"'), encoding="utf-8")
    with pytest.raises(BiasStoreError, match="calibrated-agent-evaluation bias"):
        load_battery(tmp_path, fmt="json")


def test_a_probe_too_small_to_rule_out_direction_says_so_in_plain_words() -> None:
    from calibrated_agent_evaluation.cli import plain

    record = run_complete_battery(
        _Judge(), _Runner(lambda _i: "PASS"), _thresholds(), [_probe(trials=6)]
    )
    assert record.verdict == "BIASED"
    (head,) = plain.bias(record, judge_path="judge.json")
    assert "too few length trials to rule the bias out" in head
    assert "changes its verdict" not in head
