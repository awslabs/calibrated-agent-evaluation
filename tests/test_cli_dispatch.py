# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Headless-stage dispatch tests.

Each headless stage dispatched against the seeded backend exits 0 and reuses its shipped feature
module (asserted via the observable it produces / the count it reports), not by checking a wrapper
was called.
"""

from __future__ import annotations

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_OK


def test_every_command_the_parser_accepts_has_a_handler():
    """The registry's final refusal is unreachable: each parsed command maps to a stage module."""
    from calibrated_agent_evaluation.cli.dispatch import _COMMANDS
    from calibrated_agent_evaluation.cli.handlers import for_stage

    for command in _COMMANDS:
        assert callable(for_stage(command)), command


def test_generate_dispatches_and_reports_inputs(seeded_local_root, capsys):
    """`calibrated-agent-evaluation generate` (offline) synthesizes inputs from the seeded taxonomy,
    exit 0."""
    rc = main(["generate"])
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "synthesized" in out
    assert "discovery frame" in out
