# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``probe_with`` runs the probe's seed loop over an adapter the caller passes in: the same
capability map, hypotheses and ``probed`` traces as ``probe``, named by the adapter's trace IDs
when it gives them, with the payload format the caller named, and the adapter's own errors left
as they are."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.probe import (
    DEFAULT_SEEDS,
    AuthorizationError,
    ProbeResponse,
    endpoint_fingerprint,
    probe_with,
)
from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError

_TARGET = (
    "arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/"
    "calibrated_agent_evaluation_fixture-AbCdEf1234@DEFAULT"
)


class _ScriptedAdapter:
    """A probe adapter that answers each prompt with its next scripted response, or raises it.

    It has the one method the probe calls, and keeps every prompt it was asked to send.
    """

    def __init__(self, *answers: ProbeResponse | Exception) -> None:
        self.answers = list(answers)
        self.sent: list[str] = []

    def send(self, prompt: str) -> ProbeResponse:
        self.sent.append(prompt)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class _RuntimeRefused(CalibratedAgentEvaluationDataError):
    """An adapter's own typed error: one of the Calibrated Agent Evaluation error family, but not a
    probe error."""


def _response(text: str, *tools: str, trace_id: str | None = None) -> ProbeResponse:
    events = tuple({"type": "tool_use", "tool_name": tool} for tool in tools)
    return ProbeResponse(
        status=200,
        text=text,
        observed_tools=tools,
        events=(*events, {"response": text}),
        trace_id=trace_id,
    )


def test_probe_with_builds_the_report_from_an_injected_adapter() -> None:
    adapter = _ScriptedAdapter(
        _response("I can look up orders.", "lookup_order"),
        _response("Refunded.", "lookup_order", "refund_order"),
    )
    seeds = ("What can you do?", "Refund order 1002.")

    report = probe_with(
        adapter, target=_TARGET, authorized=True, seeds=seeds, payload_format="agentcore_sse"
    )

    fingerprint = endpoint_fingerprint(_TARGET)
    assert report.capability_map.observed_tools == ("lookup_order", "refund_order")
    assert report.capability_map.framing.tool_inventory_is_lower_bound is True
    assert [trace.input for trace in report.traces] == [{"prompt": seed} for seed in seeds]
    assert [trace.output for trace in report.traces] == [
        {"text": "I can look up orders."},
        {"text": "Refunded."},
    ]
    for trace in report.traces:
        assert trace.ownership == "probed"
        assert trace.spans == []
        assert trace.frame == "discovery"
        assert trace.taxonomy_cell == {}
        assert trace.endpoint_fingerprint == fingerprint
    [hypothesis] = report.hypotheses
    assert hypothesis.evidence_trace_ids == tuple(trace.trace_id for trace in report.traces)
    [finding] = report.findings
    assert finding.kind == "capability_map"
    assert finding.endpoint_fingerprint == fingerprint
    assert finding.payload == {
        "observed_tools": "lookup_order,refund_order",
        "n_seed_traces": "2",
        "degraded": "false",
        "detected_format": "agentcore_sse",
    }
    # Named by the caller, not detected: no detection prompt was sent.
    assert report.detected_format == "agentcore_sse"
    assert adapter.sent == list(seeds)


def test_probe_with_names_traces_by_index_when_the_adapter_gives_no_trace_id() -> None:
    adapter = _ScriptedAdapter(_response("one"), _response("two"))

    # No seeds given: the probe's two built-in prompts are sent.
    report = probe_with(adapter, target=_TARGET, authorized=True)

    fingerprint = endpoint_fingerprint(_TARGET)
    assert [trace.trace_id for trace in report.traces] == [
        f"probe-{fingerprint}-0",
        f"probe-{fingerprint}-1",
    ]
    assert adapter.sent == list(DEFAULT_SEEDS)
    assert len(DEFAULT_SEEDS) == 2
    # No format named, so none is reported.
    assert report.detected_format is None
    assert report.findings[0].payload["detected_format"] == ""


def test_probe_with_uses_the_adapters_trace_id_when_it_gives_one() -> None:
    named = "probe-ep-0123456789abcdef-4bf92f3577b34da6a3ce929d0e0e4736"
    adapter = _ScriptedAdapter(_response("one", trace_id=named), _response("two"))

    report = probe_with(adapter, target=_TARGET, authorized=True, seeds=("a", "b"))

    # Each response decides its own trace's name; one without an ID falls back to its index.
    assert [trace.trace_id for trace in report.traces] == [
        named,
        f"probe-{endpoint_fingerprint(_TARGET)}-1",
    ]
    assert report.hypotheses[0].evidence_trace_ids == (named, report.traces[1].trace_id)


def test_probe_with_lets_the_adapters_error_through_unchanged() -> None:
    error = _RuntimeRefused("session calibrated-agent-evaluation-1 failed with RuntimeClientError")
    adapter = _ScriptedAdapter(error, _response("never asked for"))

    with pytest.raises(_RuntimeRefused) as refused:
        probe_with(adapter, target=_TARGET, authorized=True, seeds=("first", "second"))

    assert refused.value is error
    # The probe stops at the prompt that failed.
    assert adapter.sent == ["first"]


def test_probe_with_refuses_without_authorization_and_sends_nothing() -> None:
    adapter = _ScriptedAdapter(_response("never asked for"))

    with pytest.raises(AuthorizationError) as refused:
        probe_with(adapter, target=_TARGET, authorized=False, seeds=("a",))

    assert _TARGET in str(refused.value)
    assert "authorization" in str(refused.value)
    assert adapter.sent == []
