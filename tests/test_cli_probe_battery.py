# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation probe --battery``: what it refuses, and what it records.

Running a published benchmark needs its dataset and its grader model, so the end-to-end run is an
opt-in live test (``tests/integration/test_probe_battery_live.py``) and the bridge is covered
offline against a loopback endpoint in ``tests/test_probe_battery_bridge.py``. Here: every refusal
the CLI makes before it touches the network, and the record the run writes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _helpers import TEST_STAMP
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.dumpload.results import results_table
from calibrated_agent_evaluation.probe.framing import LimitsFraming
from calibrated_agent_evaluation.probe.record import ProbeBatteryRecord, ProbeFinding
from calibrated_agent_evaluation.probe.store import dump_probes, load_probes

_ENDPOINT = "http://127.0.0.1:9/chat"  # discard port: nothing listens, nothing is sent


def _argv(root: Path, *extra: str) -> list[str]:
    return [
        "--root",
        str(root),
        "probe",
        "--endpoint",
        _ENDPOINT,
        "--authorized",
        *extra,
    ]


class TestRefusals:
    def test_a_battery_without_trials_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(_argv(tmp_path, "--battery")) == EXIT_USAGE
        err = capsys.readouterr().err
        assert "--trials" in err
        assert "one benchmark sample is one trial" in err

    def test_zero_trials_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(_argv(tmp_path, "--battery", "--trials", "0")) == EXIT_USAGE
        assert "at least one trial" in capsys.readouterr().err

    @pytest.mark.parametrize("benchmark", ["agentdojo", "agent_threat_bench", "ipi_coding_agent"])
    def test_a_tool_environment_benchmark_is_refused_with_the_reason(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], benchmark: str
    ) -> None:
        argv = _argv(tmp_path, "--battery", benchmark, "--trials", "2")
        assert main(argv) == EXIT_DOMAIN_ERROR
        err = capsys.readouterr().err
        assert "tool environment" in err
        assert "strong_reject or xstest" in err

    def test_agentharm_needs_the_use_restriction_acknowledged_first(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = _argv(tmp_path, "--battery", "agentharm", "--trials", "2")
        assert main(argv) == EXIT_DOMAIN_ERROR
        assert "--allow-use-restricted" in capsys.readouterr().err

    def test_agentharm_is_then_refused_for_needing_a_tool_environment(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = _argv(tmp_path, "--battery", "agentharm", "--trials", "2", "--allow-use-restricted")
        assert main(argv) == EXIT_DOMAIN_ERROR
        assert "tool environment" in capsys.readouterr().err

    def test_an_unknown_benchmark_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = _argv(tmp_path, "--battery", "no_such_bench", "--trials", "1")
        assert main(argv) == EXIT_DOMAIN_ERROR
        assert "unknown benchmark" in capsys.readouterr().err

    def test_a_chat_safety_default_is_still_refused_as_an_agent_security_default(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = _argv(tmp_path, "--battery", "harmbench", "--trials", "1")
        assert main(argv) == EXIT_DOMAIN_ERROR
        assert "MUST NOT be used as the agent-security default" in capsys.readouterr().err

    def test_a_battery_without_authorization_sends_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = [
            "--root",
            str(tmp_path),
            "probe",
            "--endpoint",
            _ENDPOINT,
            "--battery",
            "--trials",
            "1",
        ]
        assert main(argv) == EXIT_DOMAIN_ERROR
        assert "authorized" in capsys.readouterr().err.lower()
        assert load_records_of(tmp_path, "probe") == []


def test_the_help_names_the_battery_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["probe", "--help"])
    assert exit_info.value.code == EXIT_OK
    out = capsys.readouterr().out
    for flag in ("--battery", "--trials", "--allow-use-restricted", "--grader-model"):
        assert flag in out


def _record() -> ProbeBatteryRecord:
    finding = ProbeFinding(
        kind="benchmark",
        captured_at="2026-10-26T00:00:00+00:00",
        endpoint_fingerprint="ep-1",
        framing=LimitsFraming(),
        payload={
            "benchmark": "strong_reject",
            "layer": "a_chat_safety",
            "k": "20",
            "breaks": "1",
            "break_rate_ci95": "[0.0013, 0.2487]",
            "break_rate_ub95": "0.2165",
            "max_break_rate": "0.0500",
            "vulnerable": "true",
        },
    )
    return ProbeBatteryRecord(
        probe_id="probe-1",
        endpoint_fingerprint="ep-1",
        findings=(finding,),
        provenance=TEST_STAMP,
    )


class TestRecord:
    def test_a_battery_record_round_trips_through_its_store(self, tmp_path: Path) -> None:
        dump_probes([_record()], tmp_path, fmt="json")
        assert load_probes(tmp_path, fmt="json") == [_record()]

    def test_the_results_table_reports_the_bound_and_refuses_a_point(self) -> None:
        [row] = results_table([_record()])
        assert row.estimand == "break_rate"
        assert row.point is None
        assert row.hi == pytest.approx(0.2165)
        assert row.sided == "upper"
        assert row.n == 20
        assert "do not own" in str(row.refusal_reason)

    def test_a_finding_without_a_fingerprint_is_refused(self) -> None:
        from calibrated_agent_evaluation.probe.errors import ProbeError

        with pytest.raises(ProbeError, match="endpoint_fingerprint"):
            ProbeFinding(
                kind="benchmark",
                captured_at="2026-10-26T00:00:00+00:00",
                endpoint_fingerprint="",
                framing=LimitsFraming(),
            )
