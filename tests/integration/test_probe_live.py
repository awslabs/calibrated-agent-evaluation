# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in LIVE probe against a real authed endpoint.

Skipped by default (marked ``integration``, deselected in the default pytest run). Point it at a
real endpoint you are AUTHORIZED to probe:

    export CALIBRATED_AGENT_EVALUATION_PROBE_ENDPOINT="https://your-agent/invoke"
    export CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN="<bearer token>"   # optional, if the endpoint
    needs auth uv run pytest -m integration -k probe_live

The "does it actually work against a real authed endpoint" gate: exercises the real stdlib
transport + protocol auto-detection + auth headers.
"""

from __future__ import annotations

import os

import pytest

from calibrated_agent_evaluation.probe import BearerAuth, ConnectionConfig, probe

pytestmark = pytest.mark.integration

_ENDPOINT = os.environ.get("CALIBRATED_AGENT_EVALUATION_PROBE_ENDPOINT")


@pytest.mark.skipif(
    not _ENDPOINT, reason="set CALIBRATED_AGENT_EVALUATION_PROBE_ENDPOINT to run the live probe"
)
def test_live_probe_against_real_endpoint() -> None:
    assert _ENDPOINT is not None  # guaranteed by skipif; narrows for the type checker
    auth = (
        BearerAuth("CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN")
        if os.environ.get("CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN")
        else None
    )
    config = ConnectionConfig(endpoint=_ENDPOINT, auth=auth, timeout_seconds=60.0)

    report = probe(config, authorized=True)  # you are attesting authorization by running this

    # The real endpoint answered and was characterized into probed Traces + a capability map.
    assert report.traces
    for trace in report.traces:
        assert trace.ownership == "probed"
        assert trace.endpoint_fingerprint
    # Limits framing is attached (the honesty rail), and the capability map is a lower bound.
    assert report.capability_map.framing.tool_inventory_is_lower_bound is True
