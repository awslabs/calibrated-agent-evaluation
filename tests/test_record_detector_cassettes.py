# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The recording script replays under the model its cassettes were recorded with.

``scripts/record_detector_cassettes.py`` replays the committed detector and namer cassettes to
write their oracles: on its own with ``--oracles-only``, and after a ``--yes`` recording. The
replay runs under the model id stamped on the cassette entries, not the one this shell's
``CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID`` names, so an exported id cannot break it. Entries
that name no model, or more than one, are refused before anything is replayed or written. When a
step stops, its report gives a Calibrated Agent Evaluation error's message and only the type of any
other error, whose message can echo a credential. Every test here runs with no AWS configuration and
opens no connection; the one Bedrock model a test builds is a scripted stand-in.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("strands_evals")

from _detector_fakes import ScriptedModel, spy_connections, unset_aws_env
from _helpers import FIXTURES
from calibrated_agent_evaluation.agentcore.detectors import models
from calibrated_agent_evaluation.agentcore.detectors.models import detector_bedrock_model

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "record_detector_cassettes.py"
RECORDED_MODEL_ID = "global.anthropic.claude-sonnet-4-6"
OTHER_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
# A credential-shaped string a provider might echo into an error message.
TOKEN = "AKIAIOSFODNN7EXAMPLE"
CASSETTES = FIXTURES / "raw" / "cassettes"
DETECTOR_CASSETTE = CASSETTES / "detector_fixture.cassette.json"
NAMER_CASSETTE = CASSETTES / "namer_gold_fails.cassette.json"
ORACLES = FIXTURES / "expected" / "detectors"
ORACLE_NAMES = (
    "ORD-1001.detection.json",
    "ORD-1002.detection.json",
    "CLM-4471.detection.json",
    "namer_gold_fails.json",
)
# A response chunk at least this long is specific enough that finding it in a message means the
# message quoted the recorded answer.
_QUOTED_CHUNK = 15


def _load_recorder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("record_detector_cassettes", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs, so its dataclass can resolve the module it is defined in.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


recorder = _load_recorder()


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """No AWS configuration, no Bedrock model built, and every connection refused and recorded."""
    unset_aws_env(monkeypatch)

    def never_built(**_kwargs: Any) -> None:
        raise AssertionError("a replay built a Bedrock model")

    monkeypatch.setattr(models, "BedrockModel", never_built)
    return spy_connections(monkeypatch)


def _outputs(
    root: Path, *, detector: Path = DETECTOR_CASSETTE, namer: Path = NAMER_CASSETTE
) -> Any:
    """Read the given cassettes and write every oracle under ``root / "oracles"``."""
    return recorder.Outputs(
        detector_cassette=detector,
        namer_cassette=namer,
        namer_oracle=root / "oracles" / "namer_gold_fails.json",
        oracle_dir=root / "oracles",
    )


def _copied_cassettes(root: Path) -> tuple[Path, Path]:
    """Copies of the committed cassettes under ``root / "cassettes"``, for a test to alter."""
    (root / "cassettes").mkdir(parents=True)
    detector = root / "cassettes" / DETECTOR_CASSETTE.name
    namer = root / "cassettes" / NAMER_CASSETTE.name
    shutil.copyfile(DETECTOR_CASSETTE, detector)
    shutil.copyfile(NAMER_CASSETTE, namer)
    return detector, namer


def _rewrite(path: Path, change: Callable[[list[dict[str, Any]]], None]) -> None:
    """Apply ``change`` to the entries of the cassette at ``path`` and write it back."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload["entries"])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _assert_oracles_equal_the_committed_ones(root: Path) -> None:
    for name in ORACLE_NAMES:
        assert (root / "oracles" / name).read_bytes() == (ORACLES / name).read_bytes(), name


def _response_chunks(*paths: Path) -> list[str]:
    """The longer text deltas of every recorded response in the cassettes at ``paths``."""
    chunks = []
    for path in paths:
        for entry in json.loads(path.read_text(encoding="utf-8"))["entries"]:
            for event in entry["response"]["events"]:
                text = event.get("contentBlockDelta", {}).get("delta", {}).get("text", "")
                if len(text) >= _QUOTED_CHUNK:
                    chunks.append(text)
    assert chunks, "the cassettes hold no response text to look for"
    return chunks


@pytest.mark.parametrize(
    "exported",
    [
        pytest.param(OTHER_MODEL_ID, id="another-model"),
        pytest.param(f"bedrock/{RECORDED_MODEL_ID}", id="an-id-a-live-run-refuses"),
    ],
)
def test_oracles_only_replays_under_the_stamped_model_whatever_id_is_exported(
    exported: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline: list[Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID", exported)
    out = _outputs(tmp_path)
    monkeypatch.setattr(recorder, "Outputs", lambda: out)

    code = recorder.main(["--oracles-only"])

    printed = capsys.readouterr()
    assert code == 0, printed.err
    assert offline == []
    _assert_oracles_equal_the_committed_ones(tmp_path)
    assert RECORDED_MODEL_ID in printed.out


def test_the_replay_after_a_recording_runs_under_the_model_id_stamped_on_the_cassettes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline: list[Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AWS_PROFILE", "calibrated-agent-evaluation-test-no-such-profile")
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID", OTHER_MODEL_ID)
    out = _outputs(
        tmp_path,
        detector=tmp_path / "cassettes" / DETECTOR_CASSETTE.name,
        namer=tmp_path / "cassettes" / NAMER_CASSETTE.name,
    )

    def recording(_model_factory: Any, into: Any) -> None:
        # Stands in for the billed calls: it leaves cassettes stamped with the model that answered.
        into.detector_cassette.parent.mkdir(parents=True)
        shutil.copyfile(DETECTOR_CASSETTE, into.detector_cassette)
        shutil.copyfile(NAMER_CASSETTE, into.namer_cassette)

    monkeypatch.setattr(recorder, "record", recording)
    monkeypatch.setattr(recorder, "Outputs", lambda: out)

    code = recorder.main(["--yes", "--region", "us-west-2"])

    printed = capsys.readouterr()
    assert code == 0, printed.err
    assert offline == []
    _assert_oracles_equal_the_committed_ones(tmp_path)


def _namer_under_another_model(_detector: Path, namer: Path) -> None:
    def restamp(entries: list[dict[str, Any]]) -> None:
        entries[0]["model"]["id"] = OTHER_MODEL_ID

    _rewrite(namer, restamp)


def _one_detection_under_another_model(detector: Path, _namer: Path) -> None:
    def restamp(entries: list[dict[str, Any]]) -> None:
        entries[-1]["model"]["id"] = OTHER_MODEL_ID

    _rewrite(detector, restamp)


@pytest.mark.parametrize(
    "restamp",
    [
        pytest.param(_namer_under_another_model, id="across-the-cassettes"),
        pytest.param(_one_detection_under_another_model, id="within-one-cassette"),
    ],
)
def test_cassettes_stamped_with_more_than_one_model_id_are_refused(
    restamp: Callable[[Path, Path], None],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline: list[Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    detector, namer = _copied_cassettes(tmp_path)
    restamp(detector, namer)
    out = _outputs(tmp_path, detector=detector, namer=namer)
    monkeypatch.setattr(recorder, "Outputs", lambda: out)

    with pytest.raises(recorder.UnusableModelStamps) as refused:
        recorder.replay(detector_bedrock_model, out)
    code = recorder.main(["--oracles-only"])

    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("REFUSED: ")
    for message in (str(refused.value), err):
        assert RECORDED_MODEL_ID in message
        assert OTHER_MODEL_ID in message
        # Each cassette is named with the ids found in it, so the odd one out can be found.
        assert detector.name in message
        assert namer.name in message
        assert not [c for c in _response_chunks(detector, namer) if c in message]
    assert not (tmp_path / "oracles").exists()
    assert offline == []


def _entry_without_a_stamp(entries: list[dict[str, Any]]) -> None:
    del entries[1]["model"]


def _stamp_without_an_id(entries: list[dict[str, Any]]) -> None:
    del entries[1]["model"]["id"]


def _no_entry(entries: list[dict[str, Any]]) -> None:
    entries.clear()


@pytest.mark.parametrize(
    "unstamp",
    [
        pytest.param(_entry_without_a_stamp, id="an-entry-with-no-stamp"),
        pytest.param(_stamp_without_an_id, id="a-stamp-with-no-model-id"),
        pytest.param(_no_entry, id="a-cassette-with-no-entry"),
    ],
)
def test_a_cassette_with_no_stamped_model_id_is_refused(
    unstamp: Callable[[list[dict[str, Any]]], None],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline: list[Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    detector, namer = _copied_cassettes(tmp_path)
    chunks = _response_chunks(detector, namer)
    _rewrite(detector, unstamp)
    out = _outputs(tmp_path, detector=detector, namer=namer)
    monkeypatch.setattr(recorder, "Outputs", lambda: out)

    with pytest.raises(recorder.UnusableModelStamps) as refused:
        recorder.replay(detector_bedrock_model, out)
    code = recorder.main(["--oracles-only"])

    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("REFUSED: ")
    for message in (str(refused.value), err):
        assert detector.name in message
        assert namer.name not in message
        assert not [c for c in chunks if c in message]
    assert not (tmp_path / "oracles").exists()
    assert offline == []


def test_a_failed_live_call_reports_the_vendor_errors_type_but_never_its_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline: list[Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AWS_PROFILE", "calibrated-agent-evaluation-test-no-such-profile")
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID", RECORDED_MODEL_ID)
    out = _outputs(
        tmp_path,
        detector=tmp_path / "cassettes" / DETECTOR_CASSETTE.name,
        namer=tmp_path / "cassettes" / NAMER_CASSETTE.name,
    )
    monkeypatch.setattr(recorder, "Outputs", lambda: out)
    echoed = RuntimeError(f"the provider rejected credential {TOKEN} for this request")

    def rejecting_bedrock(**_kwargs: Any) -> ScriptedModel:
        # Stands in for the billed call, which the provider refuses with the credential echoed.
        return ScriptedModel([echoed], config={})

    monkeypatch.setattr(models, "BedrockModel", rejecting_bedrock)

    code = recorder.main(["--yes", "--region", "us-west-2"])

    printed = capsys.readouterr()
    assert code == 1
    assert "STOPPED: call 1 (detect ORD-1001) failed with DetectorCallError: " in printed.err
    assert "\n  caused by RuntimeError (its message is not shown" in printed.err
    assert TOKEN not in printed.err + printed.out
    assert not (tmp_path / "oracles").exists()
    assert offline == []


def _first_detection_unrecorded(entries: list[dict[str, Any]]) -> None:
    del entries[0]


def test_a_failed_replay_reports_the_calibrated_agent_evaluation_error_it_was_caused_by_in_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline: list[Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    detector, namer = _copied_cassettes(tmp_path)
    chunks = _response_chunks(detector, namer)
    _rewrite(detector, _first_detection_unrecorded)
    out = _outputs(tmp_path, detector=detector, namer=namer)
    monkeypatch.setattr(recorder, "Outputs", lambda: out)

    code = recorder.main(["--oracles-only"])

    err = capsys.readouterr().err
    assert code == 1
    assert "failed with DetectorCallError: " in err
    # A cassette miss is a Calibrated Agent Evaluation error: its message names the cassette and the
    # fingerprint.
    assert f"\n  caused by CassetteMiss: cassette {detector.name}: no recorded response" in err
    assert not [c for c in chunks if c in err]
    assert not (tmp_path / "oracles").exists()
    assert offline == []
