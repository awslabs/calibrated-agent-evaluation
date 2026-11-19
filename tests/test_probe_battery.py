# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the benchmark battery orchestration."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.probe import (
    AuthorizationError,
    ConnectionConfig,
    ProbeError,
    run_battery,
)


def _config() -> ConnectionConfig:
    return ConnectionConfig(endpoint="https://a")


def test_battery_refuses_without_authorization() -> None:
    # Negative: no authorization → no run.
    with pytest.raises(AuthorizationError):
        run_battery(_config(), lambda _id: False, authorized=False)


def test_default_battery_excludes_agentharm() -> None:
    findings = run_battery(_config(), lambda _id: False, authorized=True)
    ran = {f.payload["benchmark"] for f in findings}
    assert {"agentdojo", "strong_reject", "xstest"} <= ran
    assert "agentharm" not in ran  # use-restricted, excluded from the default install


def test_vulnerable_if_any_trial_breaks() -> None:
    # One benchmark "breaks" on any trial → flagged vulnerable (not averaged).
    def run_trial(identifier: str) -> bool:
        return identifier == "agentdojo"  # agentdojo trials break

    findings = run_battery(_config(), run_trial, authorized=True, k=5)
    by_bench = {f.payload["benchmark"]: f.payload["vulnerable"] for f in findings}
    assert by_bench["agentdojo"] == "true"
    # 0 of 5 broke: the break rate is bounded only at 0.45, so it is not ruled out.
    assert by_bench["strong_reject"] == "not_ruled_out"


def test_requested_absent_benchmark_surfaces_substitution() -> None:
    # Negative/edge: requesting JailbreakBench (absent) runs a substitute and records the note.
    findings = run_battery(
        _config(), lambda _id: False, authorized=True, requested=["jailbreakbench"]
    )
    assert findings
    assert all(f.substitutions for f in findings)  # every finding carries the substitution note
    assert any("jailbreakbench" in note for f in findings for note in f.substitutions)


def test_no_break_in_too_few_trials_is_not_ruled_out_rather_than_safe() -> None:
    """One unbroken trial used to read vulnerable: "false" with no bound beside it."""
    (finding,) = run_battery(_config(), lambda _id: False, authorized=True, requested=["xstest"])
    assert finding.payload["k"] == "1"
    assert finding.payload["breaks"] == "0"
    assert finding.payload["break_rate_ub95"] == "0.9500"
    assert finding.payload["vulnerable"] == "not_ruled_out"


def test_enough_unbroken_trials_bound_the_break_rate_under_the_tolerance() -> None:
    (few,) = run_battery(_config(), lambda _id: False, authorized=True, requested=["xstest"], k=58)
    (enough,) = run_battery(
        _config(), lambda _id: False, authorized=True, requested=["xstest"], k=59
    )
    assert float(few.payload["break_rate_ub95"]) > 0.05
    assert few.payload["vulnerable"] == "not_ruled_out"
    assert float(enough.payload["break_rate_ub95"]) <= 0.05
    assert enough.payload["vulnerable"] == "false"
    assert enough.payload["break_rate_ci95"].startswith("[0.0000, ")


def test_the_tolerated_break_rate_is_validated() -> None:
    with pytest.raises(ProbeError, match="max_break_rate"):
        run_battery(_config(), lambda _id: False, authorized=True, max_break_rate=0.0)
    with pytest.raises(ProbeError, match="k must be at least 1"):
        run_battery(_config(), lambda _id: False, authorized=True, k=0)
