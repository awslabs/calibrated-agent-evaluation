# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the three-layer benchmark policy."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.probe import (
    DEFAULT_BATTERY,
    ProbeError,
    installable_battery,
    resolve,
)


def test_default_battery_is_the_four_agent_appropriate_benchmarks() -> None:
    assert set(DEFAULT_BATTERY) == {"agentdojo", "agentharm", "strong_reject", "xstest"}


def test_harmbench_advbench_rejected_as_agent_security_default() -> None:
    # Negative: single-turn chat-safety sets must not be the agent-security default.
    for forbidden in ("harmbench", "advbench"):
        with pytest.raises(ProbeError):
            resolve(forbidden)


def test_absent_named_default_surfaces_explicit_substitution() -> None:
    # Negative/edge: JailbreakBench/InjecAgent are absent → resolve surfaces a substitution, never
    # a silent drop.
    jb = resolve("jailbreakbench")
    assert jb.substitution_note is not None
    assert jb.resolved  # concrete substitutes named
    inj = resolve("injecagent")
    assert "agent_threat_bench" in inj.resolved
    assert inj.substitution_note is not None


def test_present_benchmark_resolves_to_itself_without_substitution() -> None:
    r = resolve("xstest")
    assert r.resolved == ("xstest",)
    assert r.substitution_note is None


def test_default_install_excludes_agentharm_until_opted_in() -> None:
    # Negative/edge: AgentHarm (use-restricted) is excluded from the default install.
    default = installable_battery()
    assert "agentharm" not in default
    assert {"agentdojo", "strong_reject", "xstest"} <= set(default)
    # Opt-in includes it.
    opted = installable_battery(allow_use_restricted=True)
    assert "agentharm" in opted


def test_license_excluded_benchmark_is_never_resolvable() -> None:
    with pytest.raises(ProbeError):
        resolve("cyberseceval_4")
