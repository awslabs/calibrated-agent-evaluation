# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`--cassette PATH --cassette-mode record`: a judge run records its model replies for replay.

The judge runs through the real CLI, trust chain, runtime and Inspect model path. The only stand-in
is the LLM itself: Inspect's offline `mockllm/model` with scripted tool-call replies, so a record
run is exercised without the network. A replay of the recorded cassette then runs with every
model call forbidden.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
from cli_store_helpers import (
    certified_calibration,
    clear_battery,
    judge_fingerprint,
    seed_verdicts,
    store_battery,
    store_calibration,
    write_judge_file,
)

pytest.importorskip("inspect_ai")

_MODEL = "mockllm/model"


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    root = tmp_path / "store"
    seed_verdicts(root, 2, 1, rater_id=None, kind="human")
    fingerprint = judge_fingerprint(model=_MODEL)
    store_calibration(root, certified_calibration(model=_MODEL, fingerprint=fingerprint))
    store_battery(root, clear_battery(model=_MODEL, fingerprint=fingerprint))
    return root


@pytest.fixture
def judge_file(tmp_path) -> Path:
    return write_judge_file(tmp_path / "judge.json", model=_MODEL)


def _script_the_llm(monkeypatch: pytest.MonkeyPatch, verdicts: list[str]) -> list[int]:
    """Make Inspect's mock LLM answer each judge call with the next scripted verdict."""
    import inspect_ai.model as inspect_model
    from inspect_ai.model import ModelOutput

    real_get_model = inspect_model.get_model
    replies = iter(verdicts)
    calls = [0]

    def reply(_messages: Any, tools: Any, _choice: Any, _config: Any) -> ModelOutput:
        calls[0] += 1
        verdict = next(replies)
        return ModelOutput.for_tool_call(
            _MODEL, tools[0].name, {"critique": f"looked; {verdict}", "verdict": verdict}
        )

    def get_model(model: Any = None, **kwargs: Any) -> Any:
        return real_get_model(model, memoize=False, custom_outputs=reply, **kwargs)

    monkeypatch.setattr(inspect_model, "get_model", get_model)
    return calls


def _forbid_the_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect_ai.model as inspect_model

    def get_model(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a replay must not call the model")

    monkeypatch.setattr(inspect_model, "get_model", get_model)


def _judge(root: Path, judge_file: Path, *argv: str) -> int:
    return main(["--root", str(root), "judge", "--judge", str(judge_file), *argv])


def _verdicts(root: Path) -> list[str]:
    with LocalBackend(root=root) as backend:
        return sorted(
            f"{t.trace_id}:{lb.outcome}"
            for t in backend.query()
            for lb in backend.get_scores(t.trace_id)
            if lb.kind == "llm-judge"
        )


def test_record_mode_writes_every_reply_and_a_replay_reproduces_the_run(
    root, judge_file, tmp_path, monkeypatch, capsys
):
    cassette = tmp_path / "cassettes" / "judge.cassette.json"
    calls = _script_the_llm(monkeypatch, ["pass", "fail", "pass"])
    args = ("--cassette", str(cassette), "--cassette-mode", "record")
    assert _judge(root, judge_file, *args) == EXIT_OK
    assert calls[0] == 3
    entries = json.loads(cassette.read_text(encoding="utf-8"))["entries"]
    assert len(entries) == 3
    assert {e["model"]["id"] for e in entries} == {_MODEL}
    recorded = _verdicts(root)

    replay_root = tmp_path / "replay"
    seed_verdicts(replay_root, 2, 1, rater_id=None, kind="human")
    fingerprint = judge_fingerprint(model=_MODEL)
    store_calibration(replay_root, certified_calibration(model=_MODEL, fingerprint=fingerprint))
    store_battery(replay_root, clear_battery(model=_MODEL, fingerprint=fingerprint))
    _forbid_the_llm(monkeypatch)
    assert _judge(replay_root, judge_file, "--cassette", str(cassette)) == EXIT_OK
    assert _verdicts(replay_root) == recorded


def test_record_mode_calls_the_model_again_and_overwrites_what_the_cassette_holds(
    root, judge_file, tmp_path, monkeypatch, capsys
):
    cassette = tmp_path / "judge.cassette.json"
    args = ("--cassette", str(cassette), "--cassette-mode", "record")
    _script_the_llm(monkeypatch, ["pass", "pass", "pass"])
    assert _judge(root, judge_file, *args) == EXIT_OK
    before = cassette.read_text(encoding="utf-8")

    monkeypatch.undo()
    _script_the_llm(monkeypatch, ["fail", "fail", "fail"])
    assert _judge(root, judge_file, *args) == EXIT_OK
    after = cassette.read_text(encoding="utf-8")
    assert after != before
    assert len(json.loads(after)["entries"]) == len(json.loads(before)["entries"])


def test_a_failed_record_run_keeps_the_replies_it_paid_for_and_writes_no_verdict(
    root, judge_file, tmp_path, monkeypatch, capsys
):
    cassette = tmp_path / "judge.cassette.json"
    _script_the_llm(monkeypatch, ["pass", "fail"])  # the third call has no reply
    args = ("--cassette", str(cassette), "--cassette-mode", "record")
    assert _judge(root, judge_file, *args) == EXIT_DOMAIN_ERROR
    assert len(json.loads(cassette.read_text(encoding="utf-8"))["entries"]) == 2
    assert _verdicts(root) == []


def test_record_mode_without_a_cassette_path_is_refused(root, judge_file, capsys):
    assert _judge(root, judge_file, "--cassette-mode", "record") == EXIT_DOMAIN_ERROR
    assert "--cassette-mode record needs --cassette PATH" in capsys.readouterr().err


def test_replaying_a_missing_cassette_points_at_record_mode(root, judge_file, tmp_path, capsys):
    missing = tmp_path / "absent.cassette.json"
    assert _judge(root, judge_file, "--cassette", str(missing)) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "is not a file" in err
    assert "--cassette-mode record" in err


@pytest.mark.parametrize("stage", ["judge", "calibrate", "bias"])
def test_every_judge_stage_takes_the_cassette_mode(stage, capsys):
    with pytest.raises(SystemExit):
        main([stage, "--help"])
    out = capsys.readouterr().out
    assert "--cassette-mode {replay,record}" in out
