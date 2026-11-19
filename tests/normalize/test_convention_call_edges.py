# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convention and OTLP reader edges: id-less calls, call order, ids, hostile content."""

from __future__ import annotations

import json
from typing import Any

from calibrated_agent_evaluation.normalize import SessionSpans, normalize, otlp_json
from calibrated_agent_evaluation.normalize import content as content_mod


def _kv(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _calls(spans: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    trace = normalize(SessionSpans(spans=spans))
    return [(c.name, c.args) for s in trace.spans for c in s.tool_calls]


def _openinference_span(*attrs: dict[str, Any], span_id: str = "01" * 8) -> dict[str, Any]:
    return {
        "traceId": "ab" * 16,
        "spanId": span_id,
        "startTimeUnixNano": "1",
        "endTimeUnixNano": "2",
        "attributes": [
            _kv("openinference.span.kind", "LLM"),
            _kv("input.value", "refund order 7"),
            _kv("llm.output_messages.0.message.role", "assistant"),
            *attrs,
        ],
    }


_OI_CALL = "llm.output_messages.0.message.tool_calls.{j}.tool_call.function.{field}"


def test_an_openinference_tool_call_without_an_id_is_recovered() -> None:
    span = _openinference_span(
        _kv(_OI_CALL.format(j=0, field="name"), "issue_refund"),
        _kv(_OI_CALL.format(j=0, field="arguments"), '{"amount": 500}'),
    )
    assert _calls([span]) == [("issue_refund", {"amount": 500})]


def test_an_id_less_call_resent_on_a_later_span_is_counted_once() -> None:
    call = (
        _kv(_OI_CALL.format(j=0, field="name"), "issue_refund"),
        _kv(_OI_CALL.format(j=0, field="arguments"), '{"amount": 500}'),
    )
    first = _openinference_span(*call, span_id="01" * 8)
    resent = _openinference_span(*call, span_id="02" * 8)
    resent["startTimeUnixNano"], resent["endTimeUnixNano"] = "3", "4"
    assert _calls([first, resent]) == [("issue_refund", {"amount": 500})]


def test_a_genuine_repeat_of_an_id_less_call_within_one_span_is_kept() -> None:
    span = _openinference_span(
        _kv(_OI_CALL.format(j=0, field="name"), "search"),
        _kv(_OI_CALL.format(j=0, field="arguments"), '{"q": "x"}'),
        _kv(_OI_CALL.format(j=1, field="name"), "search"),
        _kv(_OI_CALL.format(j=1, field="arguments"), '{"q": "x"}'),
    )
    assert _calls([span]) == [("search", {"q": "x"}), ("search", {"q": "x"})]


def _openllmetry_span(*attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "traceId": "ab" * 16,
        "spanId": "03" * 8,
        "attributes": [
            _kv("gen_ai.prompt.0.role", "user"),
            _kv("gen_ai.prompt.0.content", "go"),
            _kv("gen_ai.completion.0.role", "assistant"),
            *attrs,
        ],
    }


def test_a_legacy_openllmetry_tool_call_without_an_id_is_recovered() -> None:
    span = _openllmetry_span(
        _kv("gen_ai.completion.0.tool_calls.0.name", "search_flights"),
        _kv("gen_ai.completion.0.tool_calls.0.arguments", '{"to": "SFO"}'),
    )
    assert _calls([span]) == [("search_flights", {"to": "SFO"})]


def test_twelve_calls_in_one_completion_keep_their_numeric_order() -> None:
    attrs: list[dict[str, Any]] = []
    for j in range(12):
        attrs += [
            _kv(f"gen_ai.completion.0.tool_calls.{j}.id", f"c{j}"),
            _kv(f"gen_ai.completion.0.tool_calls.{j}.name", f"step{j}"),
        ]
    names = [name for name, _ in _calls([_openllmetry_span(*attrs)])]
    assert names == [f"step{j}" for j in range(12)]


def test_openinference_calls_keep_their_numeric_order() -> None:
    attrs: list[dict[str, Any]] = []
    for j in range(11):
        attrs.append(_kv(_OI_CALL.format(j=j, field="name"), f"step{j}"))
    names = [name for name, _ in _calls([_openinference_span(*attrs)])]
    assert names == [f"step{j}" for j in range(11)]


def test_a_base64_trace_id_on_a_convention_path_reads_as_hex() -> None:
    span = _openllmetry_span(_kv("gen_ai.completion.0.content", "done"))
    span["traceId"] = "q6urq6urq6urq6urq6urqw=="  # the 16 bytes 0xab, protobuf-JSON style
    assert normalize(SessionSpans(spans=[span])).trace_id == "ab" * 16


def test_messages_spans_with_mixed_start_time_shapes_sort_numerically() -> None:
    def span(text: str, start: Any) -> dict[str, Any]:
        msgs = json.dumps([{"role": "user", "parts": [{"type": "text", "content": text}]}])
        reply = json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": text}]}])
        out: dict[str, Any] = {
            "spanId": text,
            "attributes": {"gen_ai.input.messages": msgs, "gen_ai.output.messages": reply},
        }
        if start is not None:
            out["startTimeUnixNano"] = start
        return out

    # "10" sorted before "9" as strings; an int beside a string raised TypeError.
    spans = [span("late", "10"), span("early", 9), span("unknown", None)]
    trace = normalize(SessionSpans(spans=spans, semconv="gen_ai/messages"))
    assert trace.input == {"message": "unknown"}
    assert trace.output == {"answer": "late"}


def test_pathologically_nested_content_is_text_wrapped_not_raised() -> None:
    for payload in ("[" * 100_000 + "]" * 100_000, "(" * 100_000 + ")" * 100_000):
        assert content_mod.as_object(payload) == {"answer": payload}


def test_an_out_of_range_timestamp_reads_as_no_timestamp() -> None:
    assert otlp_json.started_at({"startTimeUnixNano": "9" * 30}) is None
