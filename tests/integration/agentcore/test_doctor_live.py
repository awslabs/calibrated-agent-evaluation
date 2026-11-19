# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The readiness checks run read-only against a real AgentCore account under the live-test role."""

from __future__ import annotations

import os
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.agentcore]


def test_doctor_checks_run_read_only_against_the_live_account(
    live_budget: Any, live_region: str | None, live_identity: str
) -> None:
    from calibrated_agent_evaluation.agentcore import (
        AwsClients,
        DoctorTarget,
        read_only_guard,
        run_checks,
    )
    from calibrated_agent_evaluation.agentcore.doctor import DEFAULT_CHECKS

    guard = read_only_guard(DEFAULT_CHECKS)
    used_before = live_budget.used
    with AwsClients.from_profile(region=live_region, handlers=[guard, live_budget]) as clients:
        assert clients.region is not None
        target = DoctorTarget(
            region=clients.region,
            runtime_name=os.environ.get("CALIBRATED_AGENT_EVALUATION_AGENTCORE_RUNTIME") or None,
            omni_domain=os.environ.get("CALIBRATED_AGENT_EVALUATION_AGENTCORE_OMNI_DOMAIN") or None,
        )
        report = run_checks(DEFAULT_CHECKS, clients, target)

    results = {r.name: r for r in report.results}
    assert report.identity == live_identity
    assert results["region"].status == "pass", results["region"]
    assert all(r.status == "pass" or r.next_step for r in report.results)
    assert live_budget.used == used_before, "doctor must make no billed call"
