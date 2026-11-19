# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ADOT-to-OTLP converter: envelope assembly, content re-home, X-Ray id normalization."""

from __future__ import annotations

import json

import pytest

from calibrated_agent_evaluation.agentcore.adot import adot_to_otlp, xray_trace_id_to_otel
from calibrated_agent_evaluation.agentcore.errors import AgentCoreContentMissing, AgentCoreError
from calibrated_agent_evaluation.normalize import otlp_json


def _span(
    span_id: str,
    op: str | None = None,
    *,
    parent: str | None = None,
    trace_id: str = "t1",
    **attrs: str,
) -> dict:
    a: dict = dict(attrs)
    if op is not None:
        a["gen_ai.operation.name"] = op
    span: dict = {"spanId": span_id, "traceId": trace_id, "name": op or "span", "attributes": a}
    if parent is not None:
        span["parentSpanId"] = parent
    return span


def _log(span_id: str, event: str, body: dict) -> dict:
    return {"spanId": span_id, "eventName": event, "body": body}


def _events(span: dict) -> dict[str, str]:
    """Map each span event name to its single string-attribute value, for assertions."""
    out: dict[str, str] = {}
    for ev in span.get("events") or []:
        for kv in ev.get("attributes") or []:
            out[ev["name"]] = kv["value"]["stringValue"]
    return out


def _by_id(otlp: dict) -> dict[str, dict]:
    return {s["spanId"]: s for s in otlp_json.iter_spans(otlp)}


# --- X-Ray trace-id normalization ---------------------------------------------------------


def test_xray_trace_id_is_normalized_to_otel_hex() -> None:
    assert xray_trace_id_to_otel("1-58406520-a006649127e371903a2de979") == (
        "58406520a006649127e371903a2de979"
    )


def test_an_already_hex_trace_id_is_returned_unchanged() -> None:
    hex_id = "58406520a006649127e371903a2de979"
    assert xray_trace_id_to_otel(hex_id.upper()) == hex_id


def test_a_malformed_trace_id_raises_rather_than_returning_a_corrupt_id() -> None:
    with pytest.raises(AgentCoreError):
        xray_trace_id_to_otel("not-a-trace-id")


# --- envelope assembly --------------------------------------------------------------------


def test_every_span_is_assembled_into_the_canonical_envelope() -> None:
    spans = [
        _span("a", "invoke_agent"),
        _span("c", "chat", parent="a"),
        _span("p"),
    ]
    logs = [
        _log("c", "gen_ai.user.message", {"content": [{"text": "hi"}]}),
        _log("c", "gen_ai.choice", {"message": {"content": [{"text": "done"}]}}),
    ]
    otlp = adot_to_otlp(spans, logs)
    assert len(otlp_json.iter_spans(otlp)) == 3


# --- agent content re-home ----------------------------------------------------------------


def test_agent_input_and_output_are_rehomed_from_its_chat_subtree() -> None:
    spans = [_span("a", "invoke_agent"), _span("c", "chat", parent="a")]
    logs = [
        _log("c", "gen_ai.user.message", {"content": [{"text": "the question"}]}),
        _log("c", "gen_ai.choice", {"message": {"content": [{"text": "the answer"}]}}),
    ]
    agent = _by_id(adot_to_otlp(spans, logs))["a"]
    events = _events(agent)
    assert json.loads(events["gen_ai.user.message"]) == [{"text": "the question"}]
    assert events["gen_ai.choice"] == "the answer"


def test_output_is_the_last_non_empty_choice_past_the_tool_use_turns() -> None:
    # Two chat turns: the first choice is empty (a tool-use turn), the second is the final answer.
    spans = [
        _span("a", "invoke_agent"),
        _span("c1", "chat", parent="a"),
        _span("c2", "chat", parent="a"),
    ]
    spans[1]["startTimeUnixNano"] = 1
    spans[2]["startTimeUnixNano"] = 2
    logs = [
        _log("c1", "gen_ai.user.message", {"content": [{"text": "q"}]}),
        _log("c1", "gen_ai.choice", {"message": {"content": []}}),
        _log("c2", "gen_ai.choice", {"message": {"content": [{"text": "final"}]}}),
    ]
    agent = _by_id(adot_to_otlp(spans, logs))["a"]
    assert _events(agent)["gen_ai.choice"] == "final"


def test_a_sub_agents_chat_does_not_leak_into_its_parent_agent() -> None:
    # parent agent -> execute_tool boundary -> sub agent -> its own chat.
    spans = [
        _span("pa", "invoke_agent"),
        _span("pc", "chat", parent="pa"),
        _span("t", "execute_tool", parent="pc"),
        _span("sa", "invoke_agent", parent="t"),
        _span("sc", "chat", parent="sa"),
    ]
    logs = [
        _log("pc", "gen_ai.user.message", {"content": [{"text": "parent q"}]}),
        _log("pc", "gen_ai.choice", {"message": {"content": [{"text": "parent a"}]}}),
        _log("sc", "gen_ai.user.message", {"content": [{"text": "sub q"}]}),
        _log("sc", "gen_ai.choice", {"message": {"content": [{"text": "sub a"}]}}),
    ]
    by_id = _by_id(adot_to_otlp(spans, logs))
    assert json.loads(_events(by_id["pa"])["gen_ai.user.message"]) == [{"text": "parent q"}]
    assert json.loads(_events(by_id["sa"])["gen_ai.user.message"]) == [{"text": "sub q"}]


# --- tool content re-home -----------------------------------------------------------------


def test_tool_args_and_result_are_rehomed_by_tool_call_id() -> None:
    call_id = "tooluse_ABC"
    spans = [
        _span("a", "invoke_agent"),
        _span("c", "chat", parent="a"),
        _span("t", "execute_tool", parent="c", **{"gen_ai.tool.call.id": call_id}),
    ]
    logs = [
        _log("c", "gen_ai.user.message", {"content": [{"text": "q"}]}),
        _log("c", "gen_ai.choice", {"message": {"content": [{"text": "a"}]}}),
        _log(
            "c",
            "gen_ai.assistant.message",
            {"content": [{"toolUse": {"toolUseId": call_id, "input": {"query": "x"}}}]},
        ),
        _log("c", "gen_ai.tool.message", {"id": call_id, "content": [{"text": "result"}]}),
    ]
    tool = _by_id(adot_to_otlp(spans, logs))["t"]
    events = _events(tool)
    assert json.loads(events["gen_ai.tool.message"]) == {"query": "x"}
    assert json.loads(events["gen_ai.choice"]) == [{"text": "result"}]


def test_tool_args_also_match_the_openai_style_tool_calls_block() -> None:
    call_id = "tooluse_XYZ"
    spans = [
        _span("a", "invoke_agent"),
        _span("c", "chat", parent="a"),
        _span("t", "execute_tool", parent="c", **{"gen_ai.tool.call.id": call_id}),
    ]
    logs = [
        _log("c", "gen_ai.user.message", {"content": [{"text": "q"}]}),
        _log("c", "gen_ai.choice", {"message": {"content": [{"text": "a"}]}}),
        _log(
            "c",
            "gen_ai.assistant.message",
            {"tool_calls": [{"id": call_id, "function": {"arguments": {"n": 1}}}]},
        ),
        _log("c", "gen_ai.tool.message", {"id": call_id, "content": [{"text": "r"}]}),
    ]
    tool = _by_id(adot_to_otlp(spans, logs))["t"]
    assert json.loads(_events(tool)["gen_ai.tool.message"]) == {"n": 1}


# --- content-missing refusal + inline content ---------------------------------------------


def test_an_agent_that_recovers_no_handoff_content_is_refused() -> None:
    spans = [_span("a", "invoke_agent")]  # no chat subtree, no inline content
    with pytest.raises(AgentCoreContentMissing):
        adot_to_otlp(spans, [])


def test_a_leaf_tool_with_no_content_degrades_instead_of_failing_the_trace() -> None:
    # A real trace can have a tool call whose args/result were never emitted as a content record;
    # the tool is still structurally recovered (empty args/result), and the trace is not refused.
    call_id = "tooluse_NOCONTENT"
    spans = [
        _span("a", "invoke_agent"),
        _span("c", "chat", parent="a"),
        _span("t", "execute_tool", parent="c", **{"gen_ai.tool.call.id": call_id}),
    ]
    logs = [
        _log("c", "gen_ai.user.message", {"content": [{"text": "q"}]}),
        _log("c", "gen_ai.choice", {"message": {"content": [{"text": "a"}]}}),
    ]
    otlp = adot_to_otlp(spans, logs)  # must not raise
    tool = _by_id(otlp)["t"]
    # No content events were attached to the tool span (nothing matched its call id).
    assert not tool.get("events")


def test_inline_content_already_on_the_span_is_not_treated_as_missing() -> None:
    # Content-extraction-on: the span already carries its events; the converter leaves it alone.
    spans = [_span("a", "invoke_agent")]
    spans[0]["events"] = [
        {
            "name": "gen_ai.user.message",
            "attributes": [{"key": "content", "value": {"stringValue": "hi"}}],
        }
    ]
    otlp = adot_to_otlp(spans, [])  # must not raise
    assert len(otlp_json.iter_spans(otlp)) == 1


def test_legacy_content_attributes_on_the_span_are_not_treated_as_missing() -> None:
    # A pipeline that keeps attributes but drops events leaves the legacy single-blob
    # gen_ai.input.content / gen_ai.output.content as the only content placement. The gate must
    # recognize it (as the normalizer does), not refuse a trace the pipeline could still read.
    spans = [
        _span(
            "a",
            "invoke_agent",
            **{"gen_ai.input.content": "the question", "gen_ai.output.content": "the answer"},
        )
    ]
    otlp = adot_to_otlp(spans, [])  # must not raise AgentCoreContentMissing
    assert len(otlp_json.iter_spans(otlp)) == 1


def test_a_non_content_bearing_span_without_a_record_is_kept() -> None:
    spans = [_span("plain")]  # no gen_ai.operation.name
    otlp = adot_to_otlp(spans, [])
    assert len(otlp_json.iter_spans(otlp)) == 1


# --- dangling parent ----------------------------------------------------------------------


def test_a_dangling_parent_reference_is_cleared_so_the_tree_re_roots() -> None:
    spans = [
        _span("a", "invoke_agent", parent="ghost"),  # parent not in the trace
        _span("c", "chat", parent="a"),
    ]
    logs = [
        _log("c", "gen_ai.user.message", {"content": [{"text": "q"}]}),
        _log("c", "gen_ai.choice", {"message": {"content": [{"text": "a"}]}}),
    ]
    root = _by_id(adot_to_otlp(spans, logs))["a"]
    assert root["parentSpanId"] == ""
