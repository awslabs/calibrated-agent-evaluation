# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A cassette entry remembers the model that produced it, and refuses to replay for another.

Entry keys hash request content only, so a recording replayed under a different model id or
decoding config used to hand that model the old model's answers. Entries recorded before they
carried a model still replay, so existing cassettes keep working.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.execute import (
    Cassette,
    CassetteJudgeModel,
    CassetteModelMismatch,
    Mode,
    model_stamp,
)
from calibrated_agent_evaluation.execute.cassette import fingerprint

_REQUEST = {"system": "s", "user": "u"}


def test_a_hit_recorded_under_another_model_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    sonnet = model_stamp("bedrock/claude-sonnet", {"temperature": 0.0})
    Cassette(path, mode=Mode.AUTO).record("fp", _REQUEST, {"x": 1}, model=sonnet)

    reloaded = Cassette(path, mode=Mode.REPLAY)
    assert reloaded.lookup("fp", model=sonnet) == {"x": 1}
    with pytest.raises(CassetteModelMismatch, match="claude-haiku"):
        reloaded.lookup("fp", model=model_stamp("bedrock/claude-haiku", {"temperature": 0.0}))
    with pytest.raises(CassetteModelMismatch, match="temperature=1.0"):
        reloaded.lookup("fp", model=model_stamp("bedrock/claude-sonnet", {"temperature": 1.0}))


def test_an_entry_recorded_without_a_model_still_replays(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {"version": 1, "entries": [{"fingerprint": "fp", "request": {}, "response": {"x": 2}}]}
        ),
        encoding="utf-8",
    )
    cassette = Cassette(path, mode=Mode.REPLAY)
    assert cassette.lookup("fp", model=model_stamp("any/model")) == {"x": 2}


def test_recording_into_an_old_cassette_leaves_its_entries_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    old = {"fingerprint": "fp", "request": {}, "response": {"x": 2}}
    path.write_text(json.dumps({"version": 1, "entries": [old]}), encoding="utf-8")

    Cassette(path, mode=Mode.AUTO).record("fp2", {}, {"y": 3}, model=model_stamp("m"))

    entries = {e["fingerprint"]: e for e in json.loads(path.read_text())["entries"]}
    assert entries["fp"] == old
    assert entries["fp2"]["model"] == {"id": "m", "config": {}}


def test_the_judge_cassette_refuses_a_verdict_recorded_by_another_model(tmp_path: Path) -> None:
    path = tmp_path / "judge.json"
    schema = {"type": "object"}
    request = {"system": "sys", "user": "usr", "schema": schema, "via": "response_schema"}
    Cassette(path, mode=Mode.AUTO).record(
        fingerprint(request),
        request,
        {"fields": {"verdict": "PASS"}, "via": "response_schema"},
        model=model_stamp("bedrock/haiku"),
    )

    same = CassetteJudgeModel(Cassette(path, mode=Mode.REPLAY), model_id="bedrock/haiku")
    reply = same.generate_structured(system="sys", user="usr", schema=schema)
    assert reply.fields == {"verdict": "PASS"}

    other = CassetteJudgeModel(Cassette(path, mode=Mode.REPLAY), model_id="bedrock/sonnet")
    with pytest.raises(CassetteModelMismatch):
        other.generate_structured(system="sys", user="usr", schema=schema)


def test_the_strands_cassette_refuses_a_stream_recorded_by_another_model(tmp_path: Path) -> None:
    pytest.importorskip("strands")
    import asyncio

    from calibrated_agent_evaluation.execute.strands_cassette_model import (
        CassetteModel,
        DeferredModel,
    )

    def never() -> Any:
        raise AssertionError("a replay must not build the live model")

    path = tmp_path / "strands.json"
    request = {"messages": [], "tool_specs": None, "system_prompt": "be brief"}
    Cassette(path, mode=Mode.AUTO).record(
        fingerprint(request),
        request,
        {"events": [{"messageStop": {"stopReason": "end_turn"}}]},
        model=model_stamp("model-a", {"temperature": 0.0}),
    )

    async def drain(model_id: str) -> list[Any]:
        inner = DeferredModel(never, {"model_id": model_id, "temperature": 0.0})
        model = CassetteModel(inner, Cassette(path, mode=Mode.REPLAY))
        return [event async for event in model.stream([], None, "be brief")]

    assert asyncio.run(drain("model-a")) == [{"messageStop": {"stopReason": "end_turn"}}]
    with pytest.raises(CassetteModelMismatch, match="model-b"):
        asyncio.run(drain("model-b"))
