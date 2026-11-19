# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the ``calibrated-agent-evaluation`` CLI package: it imports and wires up.

Behavioural tests per stage live in test_cli_surface / test_cli_dispatch / test_cli_generate / etc.
This file keeps the package-level smoke tests (version, help, parser registration). The CLI
replaced the scaffold: stages now dispatch to shipped features or raise a typed error — a bare
``calibrated-agent-evaluation <stage>`` no longer universally exits 2, so that scaffold assertion is
gone.
"""

from __future__ import annotations

import pytest

import calibrated_agent_evaluation
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.parser import STAGES, build_parser


def test_version_is_exposed() -> None:
    assert isinstance(calibrated_agent_evaluation.__version__, str)
    assert calibrated_agent_evaluation.__version__


def test_help_runs_without_a_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "calibrated-agent-evaluation" in capsys.readouterr().out


def test_parser_registers_a_known_stage() -> None:
    parser = build_parser()
    args = parser.parse_args(["probe"])
    assert args.command == "probe"


@pytest.mark.parametrize("stage", STAGES)
def test_parser_registers_every_pipeline_stage(stage: str) -> None:
    """Every canonical stage is a registered subcommand."""
    parser = build_parser()
    # `dump`/`load` take a positional directory; the pipeline stages don't require one.
    argv = [stage, "/tmp/x"] if stage in ("dump", "load") else [stage]
    if stage == "score":
        argv += ["--scorer", "exact", "--mode", "m"]
    args = parser.parse_args(argv)
    assert args.command == stage
