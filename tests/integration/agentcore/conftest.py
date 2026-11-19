# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live AgentCore tests: run only when asked, only as the scoped live-test role.

Run with ``just test-agentcore-live`` after setting:

- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE=1`` to opt in (anything else skips every test here);
- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_ROLE_ARN`` to the role deployed from
  ``infra/agentcore-live-test-role.yaml``; a run as any other identity fails before any other call;
- ``AWS_PROFILE`` to a profile that assumes that role, and optionally
  ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_REGION``;
- optionally ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS`` to cap billed calls for the
  run.

The live tests of the runtime executor and the runtime probe adapter (``test_runtime_live.py``)
also read these variables, and are skipped, naming the unset ones, without them. The runtime ARN
is account specific, so it is never written in the repository:

- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_RUNTIME_ARN``: the SigV4 test runtime; its ``DEFAULT``
  endpoint is invoked;
- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_PAYLOAD_FORMAT``: the named payload format that runtime
  reads;
- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_QUESTION``: the question to send it;
- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_EXPECTED_AGENTS``: the agents that question drives,
  comma-separated; the executor's test only.

The online-config test (``test_online_live.py``) also reads the runtime variables above, and:

- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_EVAL_ROLE_ARN``: the role AgentCore runs the online config
  as;
- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_JUDGE``: a certified judge's ``agentcore`` judge
  file, which the test attaches but never deletes (certifying a new one is 324 billed calls);
- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_STORE``: the local store holding that judge's
  calibration record, ``CLEAR`` bias battery and session logs.

The on-demand log test there reads ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_JUDGE`` too, and:

- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_CONFIG``: an active online config with only that
  judge attached;
- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_READ_PROFILE``: optionally, a profile that may search
  the evaluation log groups the live-test role may not (read-only calls); without it, a group the
  role cannot search fails the test before its one billed call.

A test that needs a prerequisite this installation lacks, such as one of the variables above, is
skipped naming it.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

OPT_IN = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE"
ROLE = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_ROLE_ARN"
REGION = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_REGION"


@pytest.fixture(scope="session")
def live_budget() -> Any:
    """The run's billed-call budget, shared by every client the live tests build."""
    _require_opt_in()
    from calibrated_agent_evaluation.agentcore import BilledCallBudget

    return BilledCallBudget.from_env(os.environ)


@pytest.fixture(scope="session")
def live_region() -> str | None:
    return os.environ.get(REGION) or None


@pytest.fixture(scope="session")
def live_identity(live_budget: Any, live_region: str | None) -> str:
    """The caller's ARN, confirmed to be the live-test role before any test touches AgentCore."""
    role = os.environ.get(ROLE, "").strip()
    if not role:
        pytest.fail(f"{OPT_IN}=1 but {ROLE} is unset; set it to the deployed live-test role's ARN")
    from calibrated_agent_evaluation.agentcore import AwsClients, WrongAwsIdentity, require_role

    with AwsClients.from_profile(region=live_region, handlers=[live_budget]) as clients:
        try:
            return require_role(clients, role)
        except WrongAwsIdentity as exc:
            pytest.fail(str(exc))


@pytest.fixture(autouse=True)
def _live_gate(live_identity: str) -> None:
    """Every test here waits on the opt-in and identity checks, even one that asks for neither."""
    del live_identity


def _require_opt_in() -> None:
    if os.environ.get(OPT_IN) != "1":
        pytest.skip(f"live AgentCore test; set {OPT_IN}=1 (and {ROLE}) to run it")
    pytest.importorskip("boto3")
