# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each stage's help says what it does, the docs page keeps pipeline order,
and `--trace` exists only where a stage reads stored traces.
"""

from __future__ import annotations

import argparse

import pytest

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.parser import PIPELINE_STAGES, STAGE_HELP, build_parser


def _subparsers() -> dict[str, argparse.ArgumentParser]:
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise AssertionError("no subcommands")


@pytest.mark.parametrize("stage", PIPELINE_STAGES)
def test_each_stage_help_is_specific(stage, capsys):
    with pytest.raises(SystemExit):
        main([stage, "--help"])
    out = capsys.readouterr().out
    assert "stage of the evaluation pipeline" not in out
    sub = _subparsers()[stage]
    assert sub.description
    assert " ".join(sub.description.split())[:40] in " ".join(out.split())


def test_the_one_line_helps_are_distinct():
    assert len(set(STAGE_HELP.values())) == len(STAGE_HELP)


def test_subcommands_are_declared_in_pipeline_order():
    """gen_cli_pages renders in declaration order, so this is the docs page's order too."""
    assert list(_subparsers()) == [
        "quickstart",
        "doctor",
        "ingest",
        "capture",
        "dataset",
        "audit",
        "online",
        "probe",
        "define",
        "generate",
        "label",
        "cluster",
        "calibrate",
        "bias",
        "judge",
        "score",
        "run",
        "prefer",
        "improve",
        "dump",
        "load",
        "plan",
        "results",
    ]


def test_the_docs_generator_does_not_sort():
    from pathlib import Path

    source = (Path(__file__).parent.parent / "docs" / "site" / "gen_cli_pages.py").read_text()
    assert "sorted(_subparser_choices" not in source


@pytest.mark.parametrize("stage", ["probe", "improve"])
def test_stages_that_read_no_stored_trace_reject_trace(stage, capsys):
    with pytest.raises(SystemExit) as exc:
        main([stage, "--trace", "X"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --trace" in capsys.readouterr().err


@pytest.mark.parametrize("stage", sorted(_subparsers()))
def test_no_help_points_at_a_repository_path_the_wheel_does_not_ship(stage):
    text = _subparsers()[stage].format_help()
    for path in ("fixtures/", "scripts/", "docs/"):
        assert path not in text, f"`calibrated-agent-evaluation {stage} --help` points at {path}"
