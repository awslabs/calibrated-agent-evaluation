# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation generate --run --multi-turn``: the conversation wiring and its
refusals.

The conversation itself is driven library-side (``tests/test_synth_multiturn.py`` runs it against a
stub executor that mirrors the real ``StrandsExecutor`` protocol). What these tests cover is the
CLI: what it refuses before touching a model, and that the pieces it injects (the per-turn task and
the persona) are the real ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.synth.types import SynthInput

TAXONOMY = {"feature": ["refund", "exchange"], "persona": ["new", "returning"]}


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    return tmp_path / "store"


@pytest.fixture
def taxonomy(tmp_path: Path) -> Path:
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(TAXONOMY), encoding="utf-8")
    return path


def _argv(root: Path, taxonomy: Path, *extra: str) -> list[str]:
    return [
        "--root",
        str(root),
        "generate",
        "--multi-turn",
        "--taxonomy",
        str(taxonomy),
        *extra,
    ]


class TestRefusals:
    def test_multi_turn_without_run_is_refused(
        self, root: Path, taxonomy: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(_argv(root, taxonomy)) == EXIT_USAGE
        err = capsys.readouterr().err
        assert "--multi-turn" in err
        assert "--run" in err

    def test_multi_turn_on_the_inspect_executor_is_refused(
        self, root: Path, taxonomy: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = _argv(
            root,
            taxonomy,
            "--run",
            "--executor",
            "inspect",
            "--agent",
            "cli_agents:echo",
            "--agent-model",
            "mockllm/model",
        )
        assert main(argv) == EXIT_USAGE
        err = capsys.readouterr().err
        assert "--executor strands" in err
        assert "no turns" in err

    def test_a_persona_file_that_is_not_an_object_is_refused(
        self, root: Path, taxonomy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        persona = tmp_path / "persona.json"
        persona.write_text('["not", "an", "object"]', encoding="utf-8")
        argv = _argv(
            root,
            taxonomy,
            "--run",
            "--executor",
            "strands",
            "--agent",
            "cli_agents:echo",
            "--persona",
            str(persona),
        )
        assert main(argv) == EXIT_DOMAIN_ERROR
        assert "traits, context and actor_goal" in capsys.readouterr().err

    def test_a_persona_file_missing_a_field_is_refused(
        self, root: Path, taxonomy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        persona = tmp_path / "persona.json"
        persona.write_text(json.dumps({"traits": {}, "context": "x"}), encoding="utf-8")
        argv = _argv(
            root,
            taxonomy,
            "--run",
            "--executor",
            "strands",
            "--agent",
            "cli_agents:echo",
            "--persona",
            str(persona),
        )
        assert main(argv) == EXIT_DOMAIN_ERROR
        assert "actor_goal missing" in capsys.readouterr().err

    def test_a_missing_persona_file_is_refused(
        self, root: Path, taxonomy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = _argv(
            root,
            taxonomy,
            "--run",
            "--executor",
            "strands",
            "--agent",
            "cli_agents:echo",
            "--persona",
            str(tmp_path / "nope.json"),
        )
        assert main(argv) == EXIT_DOMAIN_ERROR
        assert "cannot read the persona file" in capsys.readouterr().err

    def test_zero_max_turns_is_refused(
        self, root: Path, taxonomy: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = _argv(
            root,
            taxonomy,
            "--run",
            "--executor",
            "strands",
            "--agent",
            "cli_agents:echo",
            "--max-turns",
            "0",
        )
        assert main(argv) == EXIT_USAGE
        assert "at least one turn" in capsys.readouterr().err

    def test_a_strands_replay_without_a_simulated_user_recording_is_refused(
        self,
        root: Path,
        taxonomy: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", "replay")
        argv = _argv(
            root,
            taxonomy,
            "--run",
            "--executor",
            "strands",
            "--agent",
            "cli_agents:support_agent",
            "--cassette",
            str(tmp_path / "runs.cassette.json"),
        )
        assert main(argv) == EXIT_DOMAIN_ERROR
        err = capsys.readouterr().err
        assert "simulated-user recording" in err
        assert "runs.actor.json" in err


class TestInjections:
    """What the CLI hands the engine: the real per-turn task builder and a real persona."""

    def _kwargs(self, args: argparse.Namespace) -> dict[str, Any]:
        from calibrated_agent_evaluation.cli.stages.generate import _conversation_kwargs
        from calibrated_agent_evaluation.execute.task_builders import StrandsTaskBuilder

        return _conversation_kwargs(args, StrandsTaskBuilder(build_topology=lambda _m: None))

    def test_the_default_persona_comes_from_the_inputs_cell(self) -> None:
        pytest.importorskip("strands_evals")
        kwargs = self._kwargs(argparse.Namespace(persona=None))
        item = SynthInput(text="hi", cell={"feature": "refund"}, frame="discovery")
        profile = kwargs["actor_profile_for"](item)  # type: ignore[operator]
        assert profile.traits == {"feature": "refund"}
        assert "feature=refund" in profile.context
        assert profile.actor_goal

    def test_a_persona_file_is_used_for_every_input(self, tmp_path: Path) -> None:
        pytest.importorskip("strands_evals")
        persona = tmp_path / "persona.json"
        persona.write_text(
            json.dumps(
                {
                    "traits": {"tone": "impatient"},
                    "context": "A customer on their third call.",
                    "actor_goal": "Get the refund today.",
                }
            ),
            encoding="utf-8",
        )
        kwargs = self._kwargs(argparse.Namespace(persona=str(persona)))
        first = kwargs["actor_profile_for"](  # type: ignore[operator]
            SynthInput(text="a", cell={"feature": "refund"}, frame="discovery")
        )
        second = kwargs["actor_profile_for"](  # type: ignore[operator]
            SynthInput(text="b", cell={"feature": "exchange"}, frame="discovery")
        )
        assert first is second
        assert first.traits == {"tone": "impatient"}

    def test_the_turn_task_is_the_executors_own_task_carrying_the_history(self) -> None:
        pytest.importorskip("strands_evals")
        from calibrated_agent_evaluation.execute.strands_executor import StrandsTask

        kwargs = self._kwargs(argparse.Namespace(persona=None))
        history = [
            {"role": "user", "content": [{"text": "my refund never arrived"}]},
            {"role": "assistant", "content": [{"text": "which order?"}]},
            {"role": "user", "content": [{"text": "A-7781"}]},
        ]
        task = kwargs["turn_task"](  # type: ignore[operator]
            SynthInput(
                text="my refund never arrived", cell={"feature": "refund"}, frame="discovery"
            ),
            history,
        )
        assert isinstance(task, StrandsTask)
        assert task.input == history
        assert task.taxonomy_cell == {"feature": "refund"}

    def test_a_turn_with_no_history_is_refused(self) -> None:
        from calibrated_agent_evaluation.execute.task_builders import StrandsTaskBuilder
        from calibrated_agent_evaluation.synth.errors import SynthError

        builder = StrandsTaskBuilder(build_topology=lambda _m: None)
        with pytest.raises(SynthError, match="history"):
            builder.build_conversation_turn(
                SynthInput(text="hi", cell={"feature": "refund"}, frame="discovery"), []
            )


def test_the_help_names_the_conversation_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["generate", "--help"])
    assert exit_info.value.code == EXIT_OK
    out = capsys.readouterr().out
    assert "--multi-turn" in out
    assert "--max-turns" in out
    assert "--persona" in out
