# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cassette writers merge instead of clobbering, refuse corrupt entries typed, and re-record."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.execute import judge_model
from calibrated_agent_evaluation.execute.cassette import Cassette, CassetteError, Mode, fingerprint
from calibrated_agent_evaluation.execute.errors import ExecutionError, JudgeConfigurationError
from calibrated_agent_evaluation.execute.judge_model import CassetteJudgeModel, StructuredReply

_SCHEMA: dict[str, Any] = {"type": "object"}


def test_two_recorders_on_one_cassette_keep_both_paid_entries(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    first, second = Cassette(path, Mode.AUTO), Cassette(path, Mode.AUTO)
    first.play({"q": 1}, lambda: {"paid": "call 1"})
    second.play({"q": 2}, lambda: {"paid": "call 2"})

    entries = json.loads(path.read_text())["entries"]
    assert sorted(e["response"]["paid"] for e in entries) == ["call 1", "call 2"]
    replay = Cassette(path, Mode.REPLAY)
    assert replay.play({"q": 1}, lambda: None) == {"paid": "call 1"}


def test_an_entry_missing_its_response_is_a_typed_cassette_error(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"version": 1, "entries": [{"fingerprint": "f", "request": {}}]}))
    with pytest.raises(CassetteError, match="entry 0"):
        Cassette(path)


def _record_verdict(path: Path, verdict: str) -> None:
    request = {"system": "s", "user": "u", "schema": _SCHEMA, "via": "response_schema"}
    Cassette(path).record(
        fingerprint(request), request, {"fields": {"verdict": verdict}, "via": "response_schema"}
    )


class _LiveJudge:
    """Stands in for the live Bedrock call behind the Inspect judge model."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def generate_structured(self, **_kwargs: Any) -> StructuredReply:
        return StructuredReply(fields={"verdict": "FAIL"}, via="response_schema")


def test_record_mode_re_records_instead_of_returning_the_old_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "judge.json"
    _record_verdict(path, "PASS")
    monkeypatch.setattr(judge_model, "InspectJudgeModel", _LiveJudge)

    model = CassetteJudgeModel(Cassette(path, Mode.RECORD), record_model="bedrock/m")
    reply = model.generate_structured(system="s", user="u", schema=_SCHEMA)

    assert reply.fields == {"verdict": "FAIL"}
    replay = CassetteJudgeModel(Cassette(path, Mode.REPLAY))
    assert replay.generate_structured(system="s", user="u", schema=_SCHEMA).fields == {
        "verdict": "FAIL"
    }


def test_a_verdict_that_cannot_be_recorded_is_a_permanent_execution_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(judge_model, "InspectJudgeModel", _LiveJudge)
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    model = CassetteJudgeModel(Cassette(blocker / "judge.json", Mode.AUTO), record_model="m")

    with pytest.raises(JudgeConfigurationError, match="could not be recorded") as excinfo:
        model.generate_structured(system="s", user="u", schema=_SCHEMA)
    assert isinstance(excinfo.value, ExecutionError)
