# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in LIVE WebSocket probe against a real authed endpoint.

Skipped by default (marked ``integration``, deselected in the default pytest run). Point it at a
real WS endpoint you are AUTHORIZED to probe -- e.g. TACO:

    export CALIBRATED_AGENT_EVALUATION_PROBE_WS_ENDPOINT="wss://taco.internal/agent"
    export CALIBRATED_AGENT_EVALUATION_PROBE_WS_TOKEN="<bearer token or subprotocol token>"
    export CALIBRATED_AGENT_EVALUATION_PROBE_WS_AUTH="subprotocol"   # or "bearer" (default:
    subprotocol, TACO shape) uv run pytest -m integration -k probe_ws_live

The "does it actually work against a real authed WS endpoint" gate for the addition: exercises
the real ``websockets.sync.client`` transport + subprotocol/bearer auth + circuit-breaker + the
same ``probed`` Trace shape as the HTTP live test.
"""

from __future__ import annotations

import os

import pytest

from calibrated_agent_evaluation.probe import BearerAuth, ConnectionConfig, SubprotocolAuth, probe

pytestmark = pytest.mark.integration

_ENDPOINT = os.environ.get("CALIBRATED_AGENT_EVALUATION_PROBE_WS_ENDPOINT")
_TOKEN_VAR = "CALIBRATED_AGENT_EVALUATION_PROBE_WS_TOKEN"


@pytest.mark.skipif(
    not _ENDPOINT,
    reason="set CALIBRATED_AGENT_EVALUATION_PROBE_WS_ENDPOINT to run the live WS probe",
)
def test_live_ws_probe_against_real_endpoint() -> None:
    assert _ENDPOINT is not None  # narrows for the type checker
    auth_mode = os.environ.get("CALIBRATED_AGENT_EVALUATION_PROBE_WS_AUTH", "subprotocol").lower()
    # A missing token env is a caller error for the live run — but a missing
    # CALIBRATED_AGENT_EVALUATION_PROBE_WS_TOKEN is only *resolved* when a probe is sent, so the
    # config itself constructs fine either way.
    if os.environ.get(_TOKEN_VAR) is None:
        pytest.skip(f"set {_TOKEN_VAR} to run the live WS probe")

    if auth_mode == "bearer":
        auth = BearerAuth(token_env=_TOKEN_VAR)
    elif auth_mode == "subprotocol":
        auth = SubprotocolAuth(token_env=_TOKEN_VAR)  # type: ignore[assignment]
    else:
        pytest.fail(
            f"CALIBRATED_AGENT_EVALUATION_PROBE_WS_AUTH={auth_mode!r}; expected 'bearer' or "
            "'subprotocol'"
        )

    config = ConnectionConfig(endpoint=_ENDPOINT, auth=auth, timeout_seconds=60.0)
    report = probe(config, authorized=True)  # attesting authorization by running this

    assert report.traces
    for trace in report.traces:
        assert trace.ownership == "probed"
        assert trace.endpoint_fingerprint
    assert report.capability_map.framing.tool_inventory_is_lower_bound is True
