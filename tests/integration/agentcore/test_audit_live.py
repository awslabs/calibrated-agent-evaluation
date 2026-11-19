# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The audit's reads against a real A/B test and recommendation, through a read-only guard.

Like every test here, it runs only with ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE=1`` and as the
live-test role (see ``conftest.py``), whose audit statement
(``infra/agentcore-live-test-role.yaml``) grants the four ``Get`` reads on the ARNs below and
nothing else. It needs:

- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_AB_TEST_ID``: an A/B test whose arms are configuration
  bundle versions;
- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_RECOMMENDATION_ID``: a recommendation (optional; skipped
  when unset).

It builds no trace source and reads no trace: the role grants none of the ingest's trace reads,
so the live arms read is the acceptance rows' job, under the operator's read-only profile. It
writes nothing to AWS; every record goes to a temporary local root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.agentcore]

_AB_TEST = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_AB_TEST_ID"
_RECOMMENDATION = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_RECOMMENDATION_ID"


def _clients(live_budget: Any, live_region: str | None) -> Any:
    """Clients whose guard allows the four audit reads only, with the run's billed-call budget."""
    from calibrated_agent_evaluation.agentcore import AwsClients
    from calibrated_agent_evaluation.agentcore.guards import ReadOnlyGuard
    from calibrated_agent_evaluation.agentcore.optimization import AUDIT_READ_OPERATIONS

    return AwsClients.from_profile(
        region=live_region, handlers=[ReadOnlyGuard(AUDIT_READ_OPERATIONS), live_budget]
    )


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.skip(f"set {name} to run this live audit test")
    return value


def _bundle_state(record: Any) -> list[tuple[str, str, str]]:
    return [
        (v.bundle.version_id, v.bundle.version_created_at, v.bundle.components_sha256)
        for v in record.variants
        if v.bundle is not None
    ]


def test_an_ab_test_is_read_and_recorded_and_a_second_read_finds_it_unchanged(
    live_budget: Any, live_region: str | None, tmp_path: Path
) -> None:
    from calibrated_agent_evaluation.agentcore.audit_types import (
        dump_abtest_claims,
        load_abtest_claims,
    )
    from calibrated_agent_evaluation.agentcore.optimization import read_ab_test

    test_id = _require(_AB_TEST)
    with _clients(live_budget, live_region) as clients:
        first = read_ab_test(clients, test_id)
        again = read_ab_test(clients, test_id)

    assert first.ab_test_id == test_id
    assert all(v.bundle is not None for v in first.variants), "an arm routes to a target"
    dump_abtest_claims([first], tmp_path)
    assert load_abtest_claims(tmp_path) == [first]
    # Reading changed nothing: the test's state and both bundle versions are as they were.
    assert (again.updated_at, again.status, again.execution_status) == (
        first.updated_at,
        first.status,
        first.execution_status,
    )
    assert _bundle_state(again) == _bundle_state(first)


def test_a_recommendation_is_read_and_recorded_and_a_second_read_finds_it_unchanged(
    live_budget: Any, live_region: str | None, tmp_path: Path
) -> None:
    from calibrated_agent_evaluation.agentcore.audit_types import dump_recommendation_claims
    from calibrated_agent_evaluation.agentcore.optimization import read_recommendation

    rec_id = _require(_RECOMMENDATION)
    with _clients(live_budget, live_region) as clients:
        first = read_recommendation(clients, rec_id)
        again = read_recommendation(clients, rec_id)

    assert first.recommendation_id == rec_id
    dump_recommendation_claims([first], tmp_path)
    assert (again.updated_at, again.status) == (first.updated_at, first.status)
    for side in ("input_bundle", "result_bundle"):
        before, after = getattr(first, side), getattr(again, side)
        if before is not None:
            assert (after.version_created_at, after.components_sha256) == (
                before.version_created_at,
                before.components_sha256,
            )


def test_the_live_guard_refuses_a_list_and_a_write_before_sending_them(
    live_budget: Any, live_region: str | None
) -> None:
    from calibrated_agent_evaluation.agentcore.errors import ReadOnlyViolation

    test_id = _require(_AB_TEST)
    with _clients(live_budget, live_region) as clients:
        data = clients.client("bedrock-agentcore")
        with pytest.raises(ReadOnlyViolation):
            data.list_ab_tests()
        with pytest.raises(ReadOnlyViolation):
            data.delete_ab_test(abTestId=test_id)
