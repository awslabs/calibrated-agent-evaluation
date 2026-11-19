# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI surface tests.

Asserts the ten-subcommand surface, the error-analysis-first ordering, and that a
missing/invalid argument is a usage error (exit 2) that persists nothing.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.parser import (
    PIPELINE_STAGES,
    STAGES,
    build_parser,
    scoring_stage_indices,
)


def test_help_lists_every_stage(capsys):
    """`calibrated-agent-evaluation --help` lists every pipeline stage + dump/load."""
    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for stage in STAGES:
        assert stage in out
    assert len(STAGES) == len(set(STAGES))


def test_every_stage_is_registered():
    """Each stage is a registered subcommand."""
    parser = build_parser()
    for stage in STAGES:
        argv = [stage, "/tmp/x"] if stage in ("dump", "load") else [stage]
        if stage == "score":
            argv += ["--scorer", "exact", "--mode", "m"]
        assert parser.parse_args(argv).command == stage


def test_error_analysis_stages_precede_scoring():
    """The discovery/error-analysis stages are ordered before `judge`.

    Structural check on the declared pipeline order — no scoring-first entry point precedes the
    observe → open-code → cluster path.
    """
    _first_discovery, first_scoring = scoring_stage_indices()
    # `judge` is the first scoring stage; everything before it is error analysis.
    assert STAGES[first_scoring] == "judge"
    discovery = STAGES[:first_scoring]
    assert discovery == ("probe", "define", "generate", "label", "cluster")
    # None of the discovery stages is a scoring stage.
    assert "judge" not in discovery
    assert "run" not in discovery


def test_no_scoring_stage_registered_before_discovery():
    """Negative: no scoring stage appears ahead of the discovery stages in the surface."""
    # The pipeline stage tuple IS the surface order; assert judge/run come after the discovery set.
    assert PIPELINE_STAGES.index("judge") > PIPELINE_STAGES.index("cluster")
    assert PIPELINE_STAGES.index("run") > PIPELINE_STAGES.index("judge")


def test_missing_required_arg_is_usage_error(capsys):
    """Negative: `calibrated-agent-evaluation dump` with no directory → argparse usage error, exit
    2, nothing done."""
    with pytest.raises(SystemExit) as ei:
        main(["dump"])  # missing the required `directory` positional
    assert ei.value.code == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_unknown_stage_is_usage_error(capsys):
    """Negative: an unknown stage is an argparse usage error (exit 2)."""
    with pytest.raises(SystemExit) as ei:
        main(["not-a-stage"])
    assert ei.value.code == 2
