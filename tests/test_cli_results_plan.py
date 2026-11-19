# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation results --out FILE` tabulates every stored record;
`calibrated-agent-evaluation plan --gate` sizes the gate.

The plan is checked against the gate itself: at the planned size a suite at the expected pass
rate clears the real run gate's bound, at every larger size up to 2000 too, and one size below
it does not.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from calibrated_agent_evaluation.calibrate import gate_set_size
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.records import append_records
from calibrated_agent_evaluation.run import ModeCounts, ModeKey, gate_mode_counts
from cli_store_helpers import certified_calibration, store_calibration
from test_agentcore_online_record import online_record, session_log
from test_judge_check import _record as judge_check_record


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)


def test_results_writes_one_row_per_estimate_across_kinds(tmp_path, capsys):
    root = tmp_path / "root"
    store_calibration(root, certified_calibration())
    append_records(root, "judge-check", [judge_check_record()])
    out = tmp_path / "out" / "results.csv"
    assert main(["--root", str(root), "results", "--out", str(out)]) == EXIT_OK
    assert "Wrote 6 estimate(s) from 2 record(s) (1 calibration, 1 judge-check)" in (
        capsys.readouterr().out
    )
    with out.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["estimand"] for r in rows] == [
        "tpr",
        "tnr",
        "check_tpr",
        "check_tnr",
        "delta_tpr",
        "delta_tnr",
    ]
    assert all(r["level"] and r["sided"] and r["method"] for r in rows)


def test_results_reads_past_the_agentcore_online_records_which_report_no_estimate(tmp_path, capsys):
    root = tmp_path / "root"
    store_calibration(root, certified_calibration())
    append_records(root, "agentcore-online", [online_record()])
    append_records(root, "agentcore-session", [session_log()])
    out = tmp_path / "results.csv"

    assert main(["--root", str(root), "results", "--out", str(out)]) == EXIT_OK
    assert "from 1 record(s) (1 calibration)" in capsys.readouterr().out


def test_results_over_only_online_records_says_they_report_no_estimate(tmp_path, capsys):
    root = tmp_path / "root"
    append_records(root, "agentcore-online", [online_record()])
    append_records(root, "agentcore-session", [session_log()])
    out = tmp_path / "results.csv"

    assert main(["--root", str(root), "results", "--out", str(out)]) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "1 agentcore-online, 1 agentcore-session" in err
    assert "report no estimate" in err
    assert "no stored records" not in err
    assert not out.exists()


def test_results_with_no_records_is_refused(tmp_path, capsys):
    out = tmp_path / "results.csv"
    assert main(["--root", str(tmp_path / "root"), "results", "--out", str(out)]) == (
        EXIT_DOMAIN_ERROR
    )
    assert "no stored records" in capsys.readouterr().err
    assert not out.exists()


def test_results_into_a_directory_is_refused(tmp_path, capsys):
    root = tmp_path / "root"
    store_calibration(root, certified_calibration())
    assert main(["--root", str(root), "results", "--out", str(tmp_path)]) == EXIT_DOMAIN_ERROR
    assert "is a directory" in capsys.readouterr().err


def test_results_to_dash_is_a_usage_error_not_a_file_named_dash(tmp_path, capsys, monkeypatch):
    """`--out -` reads as stdout to a shell user; it used to write a file literally named `-`."""
    root = tmp_path / "root"
    store_calibration(root, certified_calibration())
    monkeypatch.chdir(tmp_path)
    assert main(["--root", str(root), "results", "--out", "-"]) == EXIT_USAGE
    assert "--out -" in capsys.readouterr().err
    assert not (tmp_path / "-").exists()


def test_results_needs_out():
    with pytest.raises(SystemExit) as info:
        main(["results"])
    assert info.value.code == 2


def _gate_passes(n: int, expected: float, threshold: float) -> bool:
    fails = n - math.floor(expected * n + 1e-9)
    mode = ModeKey(failure_type="m")
    gate = gate_mode_counts(
        {mode: ModeCounts(mode=mode, n_total=n, n_judged_fail=fails)}, threshold=threshold
    )
    return gate.passed


@pytest.mark.parametrize(("expected", "threshold"), [(0.95, 0.80), (0.90, 0.80), (0.99, 0.95)])
def test_the_gate_plan_is_the_smallest_size_the_real_gate_keeps_passing(expected, threshold):
    n = gate_set_size(expected, threshold)
    assert _gate_passes(n, expected, threshold)
    assert not _gate_passes(n - 1, expected, threshold)
    assert all(_gate_passes(m, expected, threshold) for m in range(n, 2001))


def test_plan_gate_prints_the_size_and_the_fails_it_tolerates(capsys):
    assert main(["plan", "--gate", "--threshold", "0.8", "--expected-pass", "0.95"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "verdicts on 28 production-representative traces" in out
    assert "tolerates up to 2 FAIL verdict(s) per mode" in out
    assert "the bound first clears at 20, but some sizes between 20 and 28 fall short" in out


def test_plan_without_gate_is_refused(capsys):
    assert main(["plan", "--expected-pass", "0.95"]) == EXIT_USAGE
    assert "calibrated-agent-evaluation plan needs --gate" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "message", "code"),
    [
        (
            ["--expected-pass", "0.8", "--threshold", "0.8"],
            "cannot clear a gate",
            EXIT_DOMAIN_ERROR,
        ),
        (["--expected-pass", "1.5"], "--expected-pass must be in (0, 1]", EXIT_USAGE),
        (
            ["--expected-pass", "0.9", "--threshold", "1"],
            "--threshold must be in (0, 1)",
            EXIT_USAGE,
        ),
    ],
)
def test_plan_refuses_an_impossible_gate(capsys, argv, message, code):
    """A value out of range is a usage error; valid values that no sample size can meet are not."""
    assert main(["plan", "--gate", *argv]) == code
    assert message in capsys.readouterr().err


def test_plan_writes_nothing(tmp_path: Path):
    root = tmp_path / "root"
    assert main(["--root", str(root), "plan", "--gate", "--expected-pass", "0.95"]) == EXIT_OK
    assert not root.exists()
