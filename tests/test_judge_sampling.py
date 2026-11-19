# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A judge's temperature is opt-in and recorded; a repeated request is a fresh draw."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrated_agent_evaluation.cli.errors import CliError
from calibrated_agent_evaluation.cli.judgefile import load_judge_file
from calibrated_agent_evaluation.execute import Cassette, CassetteJudgeModel, Mode
from calibrated_agent_evaluation.execute.cassette import fingerprint
from calibrated_agent_evaluation.execute.errors import JudgeConfigurationError
from calibrated_agent_evaluation.execute.judge_model import InspectJudgeModel
from calibrated_agent_evaluation.judges import ExecutableJudge

_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string"}}}
_JUDGE = {
    "judge_id": "j",
    "model": "bedrock/x",
    "prompt_version": "v1",
    "level": "workflow",
    "prompt_template": "grade",
    "failure_mode": "m",
}


def _write(tmp_path: Path, **extra: object) -> Path:
    path = tmp_path / "judge.json"
    path.write_text(json.dumps(_JUDGE | extra), encoding="utf-8")
    return path


def test_a_judge_file_sends_no_temperature_unless_it_names_one(tmp_path: Path) -> None:
    assert load_judge_file(_write(tmp_path)).judge.sampling() == {}
    assert load_judge_file(_write(tmp_path, temperature=0)).judge.sampling() == {"temperature": 0.0}


@pytest.mark.parametrize("bad", [-0.1, "0", True])
def test_a_judge_file_refuses_a_temperature_that_is_not_a_non_negative_number(
    tmp_path: Path, bad: object
) -> None:
    with pytest.raises(CliError, match="temperature"):
        load_judge_file(_write(tmp_path, temperature=bad))


def test_an_executable_judge_refuses_a_negative_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        ExecutableJudge("j", "m", "v1", "workflow", "grade", temperature=-1.0)


def _request(**extra: object) -> dict[str, object]:
    return {"system": "s", "user": "u", "schema": _SCHEMA, "via": "response_schema", **extra}


def test_a_first_draw_keeps_the_fingerprint_it_had_before_repeats_existed(tmp_path: Path) -> None:
    cassette = Cassette(tmp_path / "c.json", mode=Mode.AUTO)
    reply = {"fields": {"verdict": "pass"}, "via": "response_schema"}
    cassette.record(fingerprint(_request()), _request(), reply)
    model = CassetteJudgeModel(Cassette(tmp_path / "c.json", mode=Mode.REPLAY))
    assert model.generate_structured(system="s", user="u", schema=_SCHEMA).fields == {
        "verdict": "pass"
    }


def test_a_repeat_replays_its_own_draw_and_misses_without_one(tmp_path: Path) -> None:
    from calibrated_agent_evaluation.execute.cassette import CassetteMiss

    cassette = Cassette(tmp_path / "c.json", mode=Mode.AUTO)
    cassette.record(
        fingerprint(_request()), _request(), {"fields": {"verdict": "pass"}, "via": "x"}
    )
    cassette.record(
        fingerprint(_request(sample=1)),
        _request(sample=1),
        {"fields": {"verdict": "fail"}, "via": "x"},
    )
    model = CassetteJudgeModel(Cassette(tmp_path / "c.json", mode=Mode.REPLAY))
    assert model.generate_structured(system="s", user="u", schema=_SCHEMA, sample=1).fields == {
        "verdict": "fail"
    }
    with pytest.raises(CassetteMiss):
        model.generate_structured(system="s", user="u", schema=_SCHEMA, sample=2)


def test_a_set_temperature_is_part_of_the_cassette_fingerprint(tmp_path: Path) -> None:
    from calibrated_agent_evaluation.execute.cassette import CassetteMiss

    cassette = Cassette(tmp_path / "c.json", mode=Mode.AUTO)
    cassette.record(
        fingerprint(_request()), _request(), {"fields": {"verdict": "pass"}, "via": "x"}
    )
    model = CassetteJudgeModel(Cassette(tmp_path / "c.json", mode=Mode.REPLAY), temperature=0.0)
    with pytest.raises(CassetteMiss):
        model.generate_structured(system="s", user="u", schema=_SCHEMA)


def test_a_temperature_the_provider_did_not_send_is_refused() -> None:
    """Inspect's mock model records a request with no sampling parameter in it, the same as a
    provider that drops one; the judge refuses rather than run at a temperature it did not ask for.
    """
    pytest.importorskip("inspect_ai")
    model = InspectJudgeModel("mockllm/model", temperature=0.0)
    with pytest.raises(JudgeConfigurationError, match="temperature"):
        model.generate_structured(system="s", user="u", schema=_SCHEMA)
