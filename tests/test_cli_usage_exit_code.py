# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A malformed command line exits 2, the code ``--help`` documents for a usage error.

Argument checks the parser cannot express (a value out of range, a flag another flag needs) used
to raise a plain ``CliError`` and exit 1, the code for a refusal of valid input or a failed gate,
so CI could not tell "the job is misconfigured" from "the agent failed the gate".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE


@pytest.mark.parametrize(
    "argv",
    [
        ("run", "--verdicts", "human", "--threshold", "1.5"),
        ("judge",),
        ("plan", "--gate", "--expected-pass", "0"),
        ("score", "--scorer", "exact", "--mode", " "),
        ("ingest", "--seed", "7", "x.otlp.json"),
        ("improve", "--register", "m"),
        ("audit", "abtest", "--region", "us-east-1"),
    ],
    ids=lambda argv: " ".join(argv[:2]),
)
def test_a_malformed_command_line_exits_2(
    seeded_local_root: Path, capsys: pytest.CaptureFixture[str], argv: tuple[str, ...]
) -> None:
    assert main(list(argv)) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert err.startswith(f"calibrated-agent-evaluation {argv[0]}: ")


def test_a_parser_usage_error_exits_2_too() -> None:
    with pytest.raises(SystemExit) as raised:
        main(["calibrate"])
    assert raised.value.code == EXIT_USAGE


def test_a_refusal_of_valid_input_still_exits_1(
    seeded_local_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["load", str(tmp_path / "no-such-dump")]) == EXIT_DOMAIN_ERROR


def test_a_failed_gate_under_ci_still_exits_1(seeded_local_root: Path) -> None:
    assert main(["run", "--verdicts", "human"]) == EXIT_OK
    assert main(["run", "--verdicts", "human", "--ci", "--threshold", "0.99"]) == EXIT_DOMAIN_ERROR


def test_the_run_help_documents_the_codes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "2 a usage error" in text
