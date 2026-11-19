# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The detector's model stack caps live calls, records and replays them, and builds Bedrock lazily.

Every call is made the way the detector and the namer make it: the capturing wrapper is armed with
the one prompt it may send, and the call is driven to completion from a worker thread. A cassette
replay is not a live call, so it neither counts against ``max_live_calls`` nor needs a Bedrock
client, AWS credentials or a network connection. A live call through the Bedrock model is one
request on the wire, because each retry the SDK made on its own would be another billed request.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("strands")

from strands.models.bedrock import BedrockModel
from strands.models.model import Model
from strands.types.exceptions import ModelThrottledException

from _agentcore_runtime_fakes import SendCounter, fake_session, zero_retry_backoff
from _detector_fakes import ScriptedModel, spy_connections, unset_aws_env
from calibrated_agent_evaluation.agentcore.detectors import models
from calibrated_agent_evaluation.agentcore.errors import (
    DetectorCallLimitExceeded,
    DetectorConfigError,
    DetectorError,
)
from calibrated_agent_evaluation.execute import CassetteMiss, CassetteModelMismatch, Mode

PROMPT = "Find the failures in this session."
OTHER_PROMPT = "Find the failures in another session."


def _call(model: Any, prompt: str = PROMPT) -> str:
    """Make one call through the wrapped stack, as the detector does, and return its text."""
    model.arm(prompt)
    return models.run_stream(model, [{"role": "user", "content": [{"text": prompt}]}])


def _wrap(
    model: Model,
    *,
    model_id: str = "scripted-model",
    cassette_path: Path | None = None,
    cassette_mode: Mode | None = None,
    max_live_calls: int = 0,
) -> Any:
    return models.wrap_model(
        model,
        model_id=model_id,
        cassette_path=cassette_path,
        cassette_mode=cassette_mode,
        max_live_calls=max_live_calls,
    )


def _entries(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["entries"]


class _RecordingBedrock:
    """Stands in for ``BedrockModel``: records how it was built and answers with a script."""

    def __init__(self, *responses: str) -> None:
        self.builds: list[dict[str, Any]] = []
        self._responses = responses

    def __call__(self, **kwargs: Any) -> ScriptedModel:
        self.builds.append(kwargs)
        return ScriptedModel(self._responses, config={})


@pytest.fixture
def bedrock(monkeypatch: pytest.MonkeyPatch) -> _RecordingBedrock:
    stand_in = _RecordingBedrock("the Bedrock answer")
    monkeypatch.setattr(models, "BedrockModel", stand_in)
    return stand_in


def test_one_live_call_is_allowed_under_a_cap_of_one() -> None:
    scripted = ScriptedModel(["the only answer"])
    wrapped = _wrap(scripted, max_live_calls=1)

    assert _call(wrapped) == "the only answer"
    assert len(scripted.calls) == 1


@pytest.mark.parametrize("replay_mode", [Mode.REPLAY, Mode.AUTO])
def test_cassette_replays_do_not_count_against_the_cap(tmp_path: Path, replay_mode: Mode) -> None:
    path = tmp_path / "detector.cassette.json"
    recorder = _wrap(
        ScriptedModel(["the recorded answer"]),
        cassette_path=path,
        cassette_mode=Mode.RECORD,
        max_live_calls=1,
    )
    assert _call(recorder) == "the recorded answer"

    unscripted = ScriptedModel([])
    replayer = _wrap(unscripted, cassette_path=path, cassette_mode=replay_mode, max_live_calls=0)

    assert [_call(replayer) for _ in range(3)] == ["the recorded answer"] * 3
    assert unscripted.calls == []


def test_a_live_call_past_the_cap_is_refused_before_it_is_sent() -> None:
    nothing_allowed = ScriptedModel(["never sent"])
    with pytest.raises(DetectorCallLimitExceeded, match="max_live_calls=0") as refused:
        _call(_wrap(nothing_allowed))
    assert refused.value.limit == 0
    assert nothing_allowed.calls == []

    one_allowed = ScriptedModel(["first answer", "never sent"])
    wrapped = _wrap(one_allowed, max_live_calls=1)
    assert _call(wrapped) == "first answer"
    with pytest.raises(DetectorCallLimitExceeded, match="max_live_calls=1") as refused:
        _call(wrapped, OTHER_PROMPT)
    assert refused.value.limit == 1
    assert len(one_allowed.calls) == 1


def test_the_bedrock_model_is_built_with_explicit_timeouts(bedrock: _RecordingBedrock) -> None:
    default = _wrap(
        models.detector_bedrock_model("m-bedrock"), model_id="m-bedrock", max_live_calls=1
    )
    custom = _wrap(
        models.detector_bedrock_model("m-bedrock", region="us-west-2", timeout_seconds=7.5),
        model_id="m-bedrock",
        max_live_calls=1,
    )
    assert bedrock.builds == []

    _call(default)
    _call(custom)

    built_default, built_custom = bedrock.builds
    assert built_default["model_id"] == "m-bedrock"
    assert built_default["boto_client_config"].connect_timeout == 120.0
    assert built_default["boto_client_config"].read_timeout == 120.0
    assert built_custom["region_name"] == "us-west-2"
    assert built_custom["boto_client_config"].connect_timeout == 7.5
    assert built_custom["boto_client_config"].read_timeout == 7.5


def test_a_throttled_live_call_is_sent_once_whatever_retries_the_environment_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The environment asks the SDK for four attempts; each would be another billed request.
    unset_aws_env(monkeypatch)
    monkeypatch.setenv("AWS_RETRY_MODE", "standard")
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "4")
    zero_retry_backoff(monkeypatch)
    connections = spy_connections(monkeypatch)
    throttled = SendCounter.error("ThrottlingException", 429)

    def counted_bedrock(**kwargs: Any) -> BedrockModel:
        # The real model and client over fake credentials; every HTTP attempt is answered here.
        built = BedrockModel(boto_session=fake_session(), **kwargs)
        built.client.meta.events.register("before-send.*.*", throttled)
        return built

    monkeypatch.setattr(models, "BedrockModel", counted_bedrock)
    wrapped = _wrap(
        models.detector_bedrock_model("m-bedrock"), model_id="m-bedrock", max_live_calls=1
    )

    with pytest.raises(DetectorError) as refused:
        _call(wrapped)
    assert isinstance(refused.value.__cause__, ModelThrottledException)
    assert throttled.count("ConverseStream") == 1
    assert connections == []


@pytest.mark.parametrize("timeout_seconds", [0, -1.0])
def test_a_non_positive_timeout_is_refused(timeout_seconds: float) -> None:
    with pytest.raises(DetectorConfigError, match="timeout_seconds"):
        models.detector_bedrock_model("m-bedrock", timeout_seconds=timeout_seconds)


def test_a_replay_builds_no_bedrock_model_and_opens_no_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bedrock: _RecordingBedrock
) -> None:
    path = tmp_path / "detector.cassette.json"
    deferred = models.detector_bedrock_model("m-bedrock")
    recorder = _wrap(
        ScriptedModel(["the recorded answer"], config=deferred.get_config()),
        model_id="m-bedrock",
        cassette_path=path,
        cassette_mode=Mode.RECORD,
        max_live_calls=1,
    )
    _call(recorder)

    unset_aws_env(monkeypatch)
    connections = spy_connections(monkeypatch)
    replayer = _wrap(deferred, model_id="m-bedrock", cassette_path=path, cassette_mode=Mode.REPLAY)

    assert _call(replayer) == "the recorded answer"
    assert connections == []
    assert bedrock.builds == []


def test_record_mode_builds_the_bedrock_model_on_the_first_call(
    tmp_path: Path, bedrock: _RecordingBedrock
) -> None:
    path = tmp_path / "detector.cassette.json"
    recorder = _wrap(
        models.detector_bedrock_model("m-bedrock"),
        model_id="m-bedrock",
        cassette_path=path,
        cassette_mode=Mode.RECORD,
        max_live_calls=1,
    )
    assert bedrock.builds == []

    assert _call(recorder) == "the Bedrock answer"
    assert len(bedrock.builds) == 1
    assert len(_entries(path)) == 1


def test_a_recorded_call_replays_with_its_model_stamp(tmp_path: Path) -> None:
    path = tmp_path / "detector.cassette.json"
    config = {"model_id": "m-1", "temperature": 0.0}
    recorder = _wrap(
        ScriptedModel(["the recorded answer"], config=config),
        model_id="m-1",
        cassette_path=path,
        cassette_mode=Mode.RECORD,
        max_live_calls=1,
    )
    _call(recorder)

    (entry,) = _entries(path)
    assert entry["model"] == {"id": "m-1", "config": {"temperature": 0.0}}
    assert entry["response"]["events"] == [
        {"contentBlockDelta": {"delta": {"text": "the recor"}}},
        {"contentBlockDelta": {"delta": {"text": "ded answer"}}},
    ]

    replayer = _wrap(
        ScriptedModel([], config=config),
        model_id="m-1",
        cassette_path=path,
        cassette_mode=Mode.REPLAY,
    )
    assert _call(replayer) == "the recorded answer"


def test_a_passed_model_id_stamps_entries_for_a_config_without_one(tmp_path: Path) -> None:
    path = tmp_path / "detector.cassette.json"
    recorder = _wrap(
        ScriptedModel(["an answer from m-1"], config={}),
        model_id="m-1",
        cassette_path=path,
        cassette_mode=Mode.RECORD,
        max_live_calls=1,
    )
    _call(recorder)

    (entry,) = _entries(path)
    assert entry["model"] == {"id": "m-1", "config": {}}

    same_model = _wrap(
        ScriptedModel([], config={}), model_id="m-1", cassette_path=path, cassette_mode=Mode.REPLAY
    )
    assert _call(same_model) == "an answer from m-1"

    other_model = ScriptedModel([], config={})
    replayer = _wrap(other_model, model_id="m-2", cassette_path=path, cassette_mode=Mode.REPLAY)
    with pytest.raises(DetectorError) as refused:
        _call(replayer)
    assert isinstance(refused.value.__cause__, CassetteModelMismatch)
    assert other_model.calls == []


def test_an_unrecorded_call_in_replay_mode_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "detector.cassette.json"
    recorder = _wrap(
        ScriptedModel(["the recorded answer"]),
        cassette_path=path,
        cassette_mode=Mode.RECORD,
        max_live_calls=1,
    )
    _call(recorder)

    unscripted = ScriptedModel(["never sent"])
    replayer = _wrap(unscripted, cassette_path=path, cassette_mode=Mode.REPLAY, max_live_calls=1)
    with pytest.raises(DetectorError) as refused:
        _call(replayer, OTHER_PROMPT)
    assert isinstance(refused.value.__cause__, CassetteMiss)
    assert unscripted.calls == []


class _ConfiguredAs(ScriptedModel):
    """A scripted model whose ``get_config()`` returns what it is given, or raises it."""

    def __init__(self, config: object) -> None:
        super().__init__([])
        self._given = config

    def get_config(self) -> Any:
        if isinstance(self._given, BaseException):
            raise self._given
        return self._given


def _unreadable_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models.prepare_model(_ConfiguredAs(RuntimeError("the configuration store is offline")))


def _config_that_is_not_a_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models.prepare_model(_ConfiguredAs(["model_id", "m-1"]))


def _configured_id_that_is_not_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models.prepare_model(_ConfiguredAs({"model_id": 7}))


def _passed_id_that_is_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models.prepare_model(ScriptedModel(config={}), model_id="")


def _config_with_a_key_json_cannot_hold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models.prepare_model(_ConfiguredAs({"model_id": "m-1", "stop": {("a", "b"): 1}}))


class _Unsearchable(type(Path())):
    """A path whose existence cannot be checked, as when a directory on it cannot be searched."""

    def exists(self, *args: Any, **kwargs: Any) -> bool:
        raise PermissionError(13, "Permission denied", str(self))


def _cassette_path_that_cannot_be_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "Path", _Unsearchable)
    models.prepare_model(
        ScriptedModel(), cassette_path=tmp_path / "c.cassette.json", cassette_mode=Mode.REPLAY
    )


def _cassette_that_cannot_be_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unreadable(path: Path, **_kwargs: Any) -> Any:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(models, "Cassette", unreadable)
    models.prepare_model(
        ScriptedModel(), cassette_path=tmp_path / "c.cassette.json", cassette_mode=Mode.AUTO
    )


@pytest.mark.parametrize(
    ("prepare", "problem", "next_step"),
    [
        (
            _unreadable_config,
            "the model's configuration cannot be read (RuntimeError)",
            "fix the model's get_config(), whose error is chained as the cause, or pass "
            "another model",
        ),
        (
            _config_that_is_not_a_mapping,
            "get_config() returned list, not a mapping",
            "pass a model whose get_config() returns a mapping, as strands models do",
        ),
        (
            _configured_id_that_is_not_text,
            "the model's config holds model_id=7, which is not a model id",
            "configure the model with its id as a non-empty string",
        ),
        (
            _passed_id_that_is_empty,
            "model_id='' is not a model id",
            "pass the model's id as a non-empty string, or leave model_id out when the model's "
            "config has one",
        ),
        (
            _config_with_a_key_json_cannot_hold,
            "the model's configuration cannot be recorded as JSON (TypeError",
            "configure the model with plain JSON data under string keys",
        ),
        (
            _cassette_path_that_cannot_be_checked,
            "cannot be checked (PermissionError",
            "check that the path and the directories above it can be read, or pass another "
            "cassette_path",
        ),
        (
            _cassette_that_cannot_be_loaded,
            "cannot be loaded: [Errno 13] Permission denied",
            "check that the file and its directory can be read, or pass another cassette_path",
        ),
    ],
    ids=[
        "an unreadable config",
        "a config that is not a mapping",
        "a configured model id that is not text",
        "an empty passed model id",
        "a config JSON cannot hold",
        "a cassette path that cannot be checked",
        "a cassette that cannot be loaded",
    ],
)
def test_each_configuration_refusal_names_the_problem_and_what_to_do_next(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepare: Callable[[Path, pytest.MonkeyPatch], None],
    problem: str,
    next_step: str,
) -> None:
    with pytest.raises(DetectorConfigError) as refused:
        prepare(tmp_path, monkeypatch)
    assert problem in str(refused.value)
    assert next_step in str(refused.value)
