# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serializing a normalized Trace into the AgentCore on-demand Evaluate payload."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

import pytest

from _helpers import FIXTURES, load_corpus_trace, load_quickstart_trace
from calibrated_agent_evaluation.agentcore.errors import AgentCoreContentUnserializable
from calibrated_agent_evaluation.agentcore.evaluate_payload import (
    aws_span_id,
    aws_trace_id,
    trace_to_evaluation_input,
)
from calibrated_agent_evaluation.schema.types import Span, ToolCall, Trace

_SYNTHETIC = "calibrated_agent_evaluation.timestamps.synthetic"


def _span(
    span_id: str = "t-1/agent",
    *,
    trace_id: str = "t-1",
    agent: str = "orderbot",
    role: str = "specialist",
    parent: str | None = None,
    seq: int = 0,
    inp: dict[str, Any] | None = None,
    out: dict[str, Any] | None = None,
    tools: list[ToolCall] | None = None,
    started_at: str | None = "2026-01-01T00:00:00+00:00",
    ended_at: str | None = "2026-01-01T00:00:01+00:00",
) -> Span:
    return Span(
        span_id=span_id,
        trace_id=trace_id,
        agent=agent,
        role=role,  # type: ignore[arg-type]
        parent=parent,
        seq=seq,
        status="completed",
        input=inp if inp is not None else {"message": "refund please"},
        tool_calls=tools or [],
        output=out if out is not None else {"answer": "refunded"},
        started_at=started_at,
        ended_at=ended_at,
    )


def _trace(
    trace_id: str = "t-1",
    *,
    ownership: str = "owned",
    spans: list[Span] | None = None,
) -> Trace:
    if spans is None:
        spans = [_span(trace_id=trace_id, span_id=f"{trace_id}/agent")]
    return Trace(
        trace_id=trace_id,
        frame="production-representative",
        ownership=ownership,  # type: ignore[arg-type]
        input={"message": "refund please"},
        output={"answer": "refunded"},
        taxonomy_cell={},
        spans=spans,
    )


def _span_docs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload["sessionSpans"])


def test_the_payload_carries_the_span_fields_timing_and_gen_ai_content_attributes() -> None:
    tool = ToolCall(name="issue_refund", args={"order_id": "A-7781"}, result={"status": "issued"})
    span = _span(inp={"message": "refund please"}, out={"answer": "refunded"}, tools=[tool])
    trace = _trace(spans=[span])
    payload = trace_to_evaluation_input(trace)

    doc = _span_docs(payload)[0]
    # Ids are conformed to the Evaluate target's required length, derived from our own id.
    assert doc["traceId"] == aws_trace_id("t-1")
    assert doc["spanId"] == aws_span_id("t-1/agent")
    assert doc["name"] == "orderbot"
    # Evaluate requires a start and end time on every span; this span's are its own, real ones.
    assert doc["startTimeUnixNano"] < doc["endTimeUnixNano"]
    assert _SYNTHETIC not in doc["attributes"]

    attrs = doc["attributes"]
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    # One trace scored as its own session: the session id is that same derived id.
    assert attrs["session.id"] == aws_trace_id("t-1")
    # Content rides on the span's own gen_ai attributes (unified shape), not separate log records.
    assert attrs["gen_ai.task.input"] == "refund please"
    assert attrs["gen_ai.task.output"] == "refunded"
    assert attrs["gen_ai.input.messages"] == [
        {"role": "user", "content": [{"text": "refund please"}]}
    ]
    assert attrs["gen_ai.output.messages"] == [
        {"role": "assistant", "content": [{"text": "refunded"}]}
    ]


def test_a_tool_span_is_marked_as_a_tool_operation() -> None:
    trace = _trace(spans=[_span(role="tool")])
    (doc,) = _span_docs(trace_to_evaluation_input(trace))
    assert doc["attributes"]["gen_ai.operation.name"] == "execute_tool"


def test_the_payload_has_no_separate_log_event_records() -> None:
    # Unified telemetry: every entry in sessionSpans is a span document, never a log-event record.
    payload = trace_to_evaluation_input(_trace(spans=[_span()]))
    assert all("eventName" not in doc for doc in payload["sessionSpans"])


def _ingest_raw(relative: str) -> Trace:
    from calibrated_agent_evaluation.ingest import detect_source
    from calibrated_agent_evaluation.normalize import normalize

    return normalize(detect_source(FIXTURES / "raw" / relative))


def _assert_document_shape(trace: Trace, documents: list[dict[str, Any]]) -> None:
    for doc in documents:
        attrs = doc["attributes"]
        assert doc["traceId"] == aws_trace_id(trace.trace_id)
        assert doc["spanId"]
        assert doc["name"]
        assert doc["startTimeUnixNano"] <= doc["endTimeUnixNano"]
        assert attrs["session.id"] == doc["traceId"]
        assert "eventName" not in doc
        if attrs["gen_ai.operation.name"] == "execute_tool":
            assert {
                "gen_ai.tool.name",
                "gen_ai.tool.call.arguments",
                "gen_ai.tool.call.result",
            } <= set(attrs)
        else:
            assert attrs["gen_ai.operation.name"] == "invoke_agent"
            assert attrs["gen_ai.agent.name"]
    (root,) = [d for d in documents if not d.get("parentSpanId")]
    assert {"gen_ai.task.input", "gen_ai.task.output"} <= set(root["attributes"])


def test_one_trace_ingested_from_two_sources_serializes_to_the_same_document_shape() -> None:
    # CLM-4471 captured by Strands (a timed multi-agent tree) and by Inspect (one untimed agent
    # view): the trajectories differ, so the payloads cannot be equal, but the serializer reads only
    # the normalized schema, so both meet one document shape with content on the span.
    pytest.importorskip("inspect_ai")
    strands = _ingest_raw("strands/CLM-4471.otlp.json")
    inspect = _ingest_raw("inspect/CLM-4471.eval")
    strands_docs = _span_docs(trace_to_evaluation_input(strands))
    inspect_docs = _span_docs(trace_to_evaluation_input(inspect))
    _assert_document_shape(strands, strands_docs)
    _assert_document_shape(inspect, inspect_docs)
    # The timed Strands agent spans keep their real times; only their tool spans are synthetic.
    # The untimed Inspect trace is synthetic throughout, and marked so.
    for doc in strands_docs:
        is_tool = doc["attributes"]["gen_ai.operation.name"] == "execute_tool"
        assert (_SYNTHETIC in doc["attributes"]) is is_tool
    assert all(doc["attributes"].get(_SYNTHETIC) for doc in inspect_docs)

    def tools(docs: list[dict[str, Any]]) -> set[str]:
        return {
            d["attributes"]["gen_ai.tool.name"]
            for d in docs
            if d["attributes"]["gen_ai.operation.name"] == "execute_tool"
        }

    shared = {c.name for s in strands.spans for c in s.tool_calls} & {
        c.name for s in inspect.spans for c in s.tool_calls
    }
    assert shared, "the two captures share tool calls"
    assert shared <= tools(strands_docs)
    assert shared <= tools(inspect_docs)


def test_a_single_span_trace_serializes_to_just_that_span() -> None:
    trace = _trace(spans=[_span(span_id="only", trace_id="t-1")])
    docs = _span_docs(trace_to_evaluation_input(trace))
    assert [d["spanId"] for d in docs] == [aws_span_id("only")]


def test_the_target_ids_meet_the_aws_evaluate_length_constraints() -> None:
    # AWS Evaluate requires a 32-char traceId and a 16-char spanId; an arbitrary Calibrated Agent
    # Evaluation id is conformed to that exact length so the inline payload and the evaluation
    # target agree.
    payload = trace_to_evaluation_input(_trace(spans=[_span()]))
    for doc in _span_docs(payload):
        assert len(doc["traceId"]) == 32
        assert len(doc["spanId"]) == 16


def test_a_probed_trace_with_no_content_is_refused() -> None:
    # An outcome-only / probed trace has no spans and no content to serialize; refuse rather than
    # send an empty payload that asks the evaluator to score nothing.
    probed = _trace(ownership="probed", spans=[])
    with pytest.raises(AgentCoreContentUnserializable):
        trace_to_evaluation_input(probed)


def test_an_owned_trace_whose_spans_carry_no_content_is_refused() -> None:
    empty = _trace(spans=[_span(inp={}, out={}, tools=[])])
    with pytest.raises(AgentCoreContentUnserializable):
        trace_to_evaluation_input(empty)


def _windows(payload: dict[str, Any]) -> dict[str, tuple[int, int]]:
    return {
        doc["spanId"]: (doc["startTimeUnixNano"], doc["endTimeUnixNano"])
        for doc in _span_docs(payload)
    }


def _untimed(*spans: Span) -> Trace:
    return _trace(
        spans=[span.model_copy(update={"started_at": None, "ended_at": None}) for span in spans]
    )


def test_synthetic_timing_nests_every_corpus_span_inside_its_parent_with_the_root_ending_last() -> (
    None
):
    trace = load_corpus_trace("CLM-4471")
    before = trace.model_copy(deep=True)
    payload = trace_to_evaluation_input(trace)
    windows = _windows(payload)
    agent_ids = {aws_span_id(span.span_id) for span in trace.spans}
    assert agent_ids <= set(windows)
    root = next(span for span in trace.spans if span.parent is None)
    children = sorted((span for span in trace.spans if span.parent), key=lambda span: span.seq)
    root_start, root_end = windows[aws_span_id(root.span_id)]
    child_windows = [windows[aws_span_id(span.span_id)] for span in children]
    # Each specialist sits inside the supervisor, in seq order, without overlapping the next.
    assert all(root_start < start <= end < root_end for start, end in child_windows)
    assert all(a_end < b_start for (_, a_end), (b_start, _) in pairwise(child_windows))
    # The supervisor ends last, so it is the span AgentCore reads as the trace's answer.
    assert root_end == max(end for _, end in windows.values())
    # Agent spans and tool spans alike carry made-up times here, and every one is marked.
    assert all(doc["attributes"][_SYNTHETIC] is True for doc in _span_docs(payload))
    # The same trace always yields the same request, and the trace itself is left untimed.
    assert trace_to_evaluation_input(trace) == payload
    assert trace == before
    assert all(span.started_at is None and span.ended_at is None for span in trace.spans)


def test_synthetic_timing_lays_two_roots_one_after_the_other_in_seq_order() -> None:
    first = _span("t-1/first", seq=0)
    inner = _span("t-1/inner", parent="t-1/first", seq=1)
    innermost = _span("t-1/innermost", parent="t-1/inner", seq=2)
    second = _span("t-1/second", seq=3)
    trace = _untimed(second, innermost, inner, first)
    windows = _windows(trace_to_evaluation_input(trace))
    first_w, inner_w, innermost_w, second_w = (
        windows[aws_span_id(span_id)]
        for span_id in ("t-1/first", "t-1/inner", "t-1/innermost", "t-1/second")
    )
    # Three levels nest inside one another, and the later root follows the first one entirely.
    assert first_w[0] < inner_w[0] < innermost_w[0] <= innermost_w[1] < inner_w[1] < first_w[1]
    assert first_w[1] < second_w[0]
    assert second_w[1] == max(end for _, end in windows.values())


def test_synthetic_timing_is_not_used_for_a_fully_timed_trace_with_no_tool_calls() -> None:
    trace = _trace()
    (doc,) = _span_docs(trace_to_evaluation_input(trace))
    expected = int(datetime.fromisoformat("2026-01-01T00:00:00+00:00").timestamp() * 1_000_000_000)
    assert doc["startTimeUnixNano"] == expected
    assert _SYNTHETIC not in doc["attributes"]


@pytest.mark.parametrize(
    "spans",
    [
        [_span("t-1/a", seq=0), _span("t-1/b", seq=1, started_at=None, ended_at=None)],
        [_span("t-1/a", seq=0, ended_at=None)],
    ],
    ids=["one-span-untimed", "end-time-missing"],
)
def test_a_partly_timed_trace_is_refused_rather_than_given_synthetic_timing(
    spans: list[Span],
) -> None:
    with pytest.raises(AgentCoreContentUnserializable, match="partial"):
        trace_to_evaluation_input(_trace(spans=spans))


def test_synthetic_timing_refuses_a_span_whose_parent_is_not_in_the_trace() -> None:
    orphan = _untimed(_span("t-1/root", seq=0), _span("t-1/lost", parent="t-1/ghost", seq=1))
    with pytest.raises(AgentCoreContentUnserializable, match="t-1/ghost"):
        trace_to_evaluation_input(orphan)


def test_a_probed_trace_is_refused_for_its_missing_content_not_for_synthetic_timing() -> None:
    with pytest.raises(AgentCoreContentUnserializable, match="no message content"):
        trace_to_evaluation_input(_trace(ownership="probed", spans=[]))


def test_synthetic_timing_refuses_spans_that_name_each_other_as_parent() -> None:
    cycle = _untimed(
        _span("t-1/root", seq=0),
        _span("t-1/a", parent="t-1/b", seq=1),
        _span("t-1/b", parent="t-1/a", seq=2),
    )
    with pytest.raises(AgentCoreContentUnserializable, match="cannot be reached"):
        trace_to_evaluation_input(cycle)


def test_an_untimed_trace_with_no_content_is_refused_for_content_before_synthetic_timing() -> None:
    empty = _untimed(_span(inp={}, out={}, tools=[]))
    with pytest.raises(AgentCoreContentUnserializable, match="no message content"):
        trace_to_evaluation_input(empty)


def _tool_docs(payload: dict[str, Any], parent: str) -> list[dict[str, Any]]:
    return [
        doc
        for doc in _span_docs(payload)
        if doc.get("parentSpanId") == aws_span_id(parent)
        and doc["attributes"]["gen_ai.operation.name"] == "execute_tool"
    ]


def test_tool_spans_carry_each_tool_call_in_order_inside_its_agent_span() -> None:
    trace = load_corpus_trace("ORD-1002")
    (agent,) = trace.spans
    payload = trace_to_evaluation_input(trace)
    tools = _tool_docs(payload, agent.span_id)
    attributes = [doc["attributes"] for doc in tools]
    assert [a["gen_ai.tool.name"] for a in attributes] == [c.name for c in agent.tool_calls]
    assert [a["gen_ai.tool.call.id"] for a in attributes] == [
        f"{agent.span_id}#{index}" for index in range(len(agent.tool_calls))
    ]
    assert [json.loads(a["gen_ai.tool.call.arguments"]) for a in attributes] == [
        c.args for c in agent.tool_calls
    ]
    assert [json.loads(a["gen_ai.tool.call.result"]) for a in attributes] == [
        c.result for c in agent.tool_calls
    ]
    assert all(a[_SYNTHETIC] is True for a in attributes)
    # Each tool span has its own id, derived from the agent span's id and the call's position.
    assert [doc["spanId"] for doc in tools] == [
        aws_span_id(f"{agent.span_id}#tool-{index}") for index in range(len(agent.tool_calls))
    ]
    assert len({doc["spanId"] for doc in _span_docs(payload)}) == len(_span_docs(payload))
    agent_start, agent_end = _windows(payload)[aws_span_id(agent.span_id)]
    starts = [doc["startTimeUnixNano"] for doc in tools]
    assert all(
        agent_start < doc["startTimeUnixNano"] <= doc["endTimeUnixNano"] < agent_end
        for doc in tools
    )
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)
    # The same trace always yields the same tool spans and ids.
    assert trace_to_evaluation_input(trace) == payload


def test_tool_spans_under_a_timed_parent_are_marked_instants_and_the_parent_keeps_its_times() -> (
    None
):
    trace = load_quickstart_trace()
    (agent,) = trace.spans
    assert agent.started_at
    assert agent.ended_at
    assert agent.tool_calls
    payload = trace_to_evaluation_input(trace)
    (agent_doc,) = [
        doc for doc in _span_docs(payload) if doc["spanId"] == aws_span_id(agent.span_id)
    ]
    start = int(datetime.fromisoformat(agent.started_at).timestamp() * 1_000_000_000)
    end = int(datetime.fromisoformat(agent.ended_at).timestamp() * 1_000_000_000)
    assert (agent_doc["startTimeUnixNano"], agent_doc["endTimeUnixNano"]) == (start, end)
    assert _SYNTHETIC not in agent_doc["attributes"]
    tools = _tool_docs(payload, agent.span_id)
    n = len(agent.tool_calls)
    assert [doc["startTimeUnixNano"] for doc in tools] == [
        start + k * (end - start) // (n + 1) for k in range(1, n + 1)
    ]
    assert all(doc["startTimeUnixNano"] == doc["endTimeUnixNano"] < end for doc in tools)
    assert all(doc["attributes"][_SYNTHETIC] is True for doc in tools)


def test_tool_spans_come_before_child_spans_in_an_untimed_trace() -> None:
    lookup = ToolCall(name="lookup", args={"q": "x"}, result={"hit": True})
    root = _span("t-1/root", seq=0, tools=[lookup])
    child = _span("t-1/child", parent="t-1/root", seq=1)
    payload = trace_to_evaluation_input(_untimed(root, child))
    (tool,) = _tool_docs(payload, "t-1/root")
    child_start, _ = _windows(payload)[aws_span_id("t-1/child")]
    assert tool["endTimeUnixNano"] < child_start


def test_a_span_with_no_tool_calls_gets_no_tool_span() -> None:
    payload = trace_to_evaluation_input(_trace())
    assert _tool_docs(payload, "t-1/agent") == []
    assert len(_span_docs(payload)) == 1


@pytest.mark.parametrize(
    "started_at",
    ["not-a-time", "2026-01-01T00:00:00"],  # unreadable, and a time with no UTC offset
)
def test_a_span_time_that_is_not_a_utc_aware_timestamp_is_refused_naming_the_span(
    started_at: str,
) -> None:
    trace = _trace(spans=[_span(started_at=started_at)])
    with pytest.raises(AgentCoreContentUnserializable, match=r"'t-1'.*'t-1/agent'"):
        trace_to_evaluation_input(trace)


@pytest.mark.parametrize("where", ["span input", "tool result"])
def test_content_that_json_cannot_write_is_refused_naming_the_span(where: str) -> None:
    unwritable = {"at": datetime(2026, 1, 1, tzinfo=UTC), "note": "two fields, so not unwrapped"}
    if where == "span input":
        span = _span(inp=unwritable)
    else:
        span = _span(tools=[ToolCall(name="lookup_order", args={}, result=unwritable)])
    with pytest.raises(AgentCoreContentUnserializable, match=r"'t-1'.*'t-1/agent'"):
        trace_to_evaluation_input(_trace(spans=[span]))
