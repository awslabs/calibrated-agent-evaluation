# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for capability discovery."""

from __future__ import annotations

from _probe_helpers import sse_transport
from calibrated_agent_evaluation.probe import ConnectionConfig, probe


def test_capability_map_lists_observed_tools_and_a_hypothesis() -> None:
    report = probe(
        ConnectionConfig(endpoint="https://a"),
        authorized=True,
        transport=sse_transport(tools=("lookup_order", "issue_refund")),
    )
    assert set(report.capability_map.observed_tools) == {"lookup_order", "issue_refund"}
    assert len(report.hypotheses) >= 1
    assert report.hypotheses[0].evidence_trace_ids  # traces back to evidence


def test_unused_self_reported_tool_is_not_listed() -> None:
    # Negative/edge: the map lists only tools OBSERVED in use. A response that merely *mentions* a
    # tool in prose (no tool_use event) must not add it to the observed lower-bound inventory.
    def prose_transport(url, headers, body, timeout):
        import json

        return 200, [
            b"data: "
            + json.dumps(
                {"type": "response", "response": "I have a delete_account tool available."}
            ).encode()
        ]

    config = ConnectionConfig(endpoint="https://a")
    report = probe(config, authorized=True, transport=prose_transport)
    assert report.capability_map.observed_tools == ()  # mentioned, never observed used
