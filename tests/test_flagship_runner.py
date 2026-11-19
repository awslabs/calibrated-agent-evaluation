# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reference proof runner: the flagship record it writes, and its refusal of a credential-less
live run.

A live run must re-execute against a real model and refuse when none is configured, never replay
the cassette and report a live pass. The live re-execution itself is exercised in
tests/integration/test_flagship_live.py (opt-in, needs credentials).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from _flagship_helpers import certified_record, clear_battery, corpus, local_backend_factory
from calibrated_agent_evaluation.flagship import (
    FlagshipError,
    load_flagship_reports,
    run_reference_proof,
)
from calibrated_agent_evaluation.flagship.runner import (
    format_report,
    live_bedrock_model_id,
    require_live_credentials,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "flagship" / "run.py"


def test_a_live_run_without_credentials_is_refused() -> None:
    with pytest.raises(FlagshipError, match="AWS credentials"):
        require_live_credentials({})
    require_live_credentials({"AWS_PROFILE": "my-profile"})


def test_the_live_model_is_a_bare_id_from_the_one_env_var() -> None:
    assert live_bedrock_model_id({}) == "global.anthropic.claude-sonnet-4-6"
    assert (
        live_bedrock_model_id({"CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID": " us.x.y "})
        == "us.x.y"
    )


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({"CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL": "bedrock/a.b"}, "no longer read"),
        ({"CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID": "bedrock/a.b"}, "bare Bedrock model id"),
        ({"CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID": "  "}, "bare Bedrock model id"),
    ],
)
def test_a_retired_or_prefixed_model_setting_is_refused(
    environ: dict[str, str], message: str
) -> None:
    with pytest.raises(FlagshipError, match=message):
        live_bedrock_model_id(environ)


def test_the_reference_proof_writes_a_flagship_record_that_loads_back(tmp_path: Path) -> None:
    reference = corpus()
    judge = reference.orderbot_judge()
    report = run_reference_proof(
        reference,
        agents=("orderbot",),
        worlds=("local",),
        model_plane="cassette",
        backend_factory=local_backend_factory(tmp_path / "store"),
        calibration=certified_record(judge),
        battery=clear_battery(judge),
        out_dir=tmp_path / "out",
    )
    assert report.verdict == "proven", report.reasons
    assert load_flagship_reports(tmp_path / "out", fmt="json") == [report]
    assert "VERDICT: PROVEN" in format_report(report)


def test_an_agent_with_no_reference_plan_is_refused(tmp_path: Path) -> None:
    reference = corpus()
    judge = reference.orderbot_judge()
    with pytest.raises(FlagshipError, match="no reference plan"):
        run_reference_proof(
            reference,
            agents=("thirdbot",),
            worlds=("local",),
            model_plane="cassette",
            backend_factory=local_backend_factory(tmp_path),
            calibration=certified_record(judge),
            battery=clear_battery(judge),
        )


def test_a_directory_without_the_corpus_is_refused(tmp_path: Path) -> None:
    from calibrated_agent_evaluation.flagship import ReferenceCorpus

    with pytest.raises(FlagshipError, match="not the reference corpus"):
        ReferenceCorpus(tmp_path)


def _script(*argv: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in ("AWS_PROFILE", "AWS_ACCESS_KEY_ID")}
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *argv],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )


def test_the_script_refuses_live_without_credentials_rather_than_replaying() -> None:
    result = _script("--agent", "orderbot", "--local-only", "--live")
    assert result.returncode == 2, result.stderr
    assert "AWS credentials" in result.stderr


def test_the_script_proves_orderbot_offline(tmp_path: Path) -> None:
    result = _script("--agent", "orderbot", "--local-only", "--out", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VERDICT: PROVEN" in result.stdout
    assert (tmp_path / "flagship_report.json").exists()
