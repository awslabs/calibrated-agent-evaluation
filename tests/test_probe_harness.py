# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the probe orchestrator."""

from __future__ import annotations

from _probe_helpers import empty_transport, sse_transport
from calibrated_agent_evaluation.probe import ConnectionConfig, probe


def _config() -> ConnectionConfig:
    return ConnectionConfig(endpoint="https://agent.example/invoke")


def test_probe_returns_probed_traces_and_lower_bound_capability_map() -> None:
    report = probe(_config(), authorized=True, transport=sse_transport(tools=("lookup_order",)))
    assert report.traces  # ≥1 seed trace
    for t in report.traces:
        assert t.ownership == "probed"
        assert t.spans == []
        assert t.endpoint_fingerprint  # stamp
    assert "lookup_order" in report.capability_map.observed_tools
    assert report.capability_map.framing.tool_inventory_is_lower_bound is True
    assert len(report.hypotheses) >= 1


def test_unparseable_response_degrades_to_outcome_only() -> None:
    # Negative/edge: a 200 with no parseable content still yields a probed Trace (no fake spans).
    report = probe(_config(), authorized=True, transport=empty_transport())
    assert report.traces
    for t in report.traces:
        assert t.ownership == "probed"
        assert t.spans == []
    assert report.findings[0].payload["degraded"] == "true"


# --- which payload shape won is reported, not inferred ---------------------------------


def _only_format_transport(accepted: str):
    """A transport that answers 200 to exactly one payload shape and 404 to the others."""
    import json as _json

    def _t(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, list[bytes]]:
        payload = _json.loads(body)
        matched = {
            "generic_message": "message" in payload,
            "openai": "messages" in payload,
            "agentcore_sse": "prompt" in payload,
            "nested_agentcore": "input" in payload,
        }[accepted]
        if not matched:
            return 404, [b""]
        return 200, [_json.dumps({"response": "ok"}).encode()]

    return _t


def test_the_report_names_the_payload_format_detection_settled_on() -> None:
    report = probe(_config(), authorized=True, transport=_only_format_transport("agentcore_sse"))
    assert report.detected_format == "agentcore_sse"
    assert report.findings[0].payload["detected_format"] == "agentcore_sse"


def test_a_first_shape_that_wins_is_visible_even_when_the_answers_look_fine() -> None:
    """The reason this field exists. An endpoint that answers 200 to a payload it could not read —
    coercing a missing prompt to an empty one — makes detection settle on `generic_message`, and
    every reply parses, so nothing else in the report says the prompt was never delivered."""
    report = probe(_config(), authorized=True, transport=_only_format_transport("generic_message"))
    assert report.detected_format == "generic_message"
    # The traces themselves look perfectly healthy, which is exactly the problem.
    assert report.traces
    assert not any(t.output.get("degraded") for t in report.traces)
