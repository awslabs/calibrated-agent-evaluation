# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A normalized trace converts into the session the failure detector reads.

Each Calibrated Agent Evaluation span becomes an agent-invocation span followed by one
tool-execution span per tool call, with the facts the vendor types cannot hold written into span
metadata. The serialized session is what the detector's prompt and its cassette fingerprint are
built from, so it must not depend on the process, on a backend's key order, or on whether the trace
recorded span times.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

pytest.importorskip("strands_evals")

from strands_evals.types.trace import AgentInvocationSpan, SpanUnion, ToolExecutionSpan

from _helpers import FIXTURES, load_corpus_trace, make_span, make_tool_call, make_trace
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.agentcore import SessionConversionError
from calibrated_agent_evaluation.agentcore.detectors.session import (
    ConvertedSession,
    session_json,
    trace_to_session,
)
from calibrated_agent_evaluation.normalize import OTelSpans, normalize
from calibrated_agent_evaluation.schema.types import Span, ToolCall, Trace

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
OTEL_ORD = FIXTURES / "raw" / "otel" / "ORD-1001.events.otlp.json"
OTEL_DEGRADED = FIXTURES / "raw" / "otel" / "ORD-DEGRADED.otlp.json"
NORMALIZED_IDS = sorted(path.stem for path in (FIXTURES / "normalized").glob("*.json"))
TESTS_DIR = Path(__file__).resolve().parent


def _spans(converted: ConvertedSession) -> list[SpanUnion]:
    (trace,) = converted.session.traces
    return trace.spans


def _agent_spans(converted: ConvertedSession) -> list[AgentInvocationSpan]:
    return [s for s in _spans(converted) if isinstance(s, AgentInvocationSpan)]


def _tool_spans(converted: ConvertedSession) -> list[ToolExecutionSpan]:
    return [s for s in _spans(converted) if isinstance(s, ToolExecutionSpan)]


def _shape(converted: ConvertedSession) -> list[tuple[str, str | None, str | None]]:
    return [
        (s.span_type.value, s.span_info.span_id, s.span_info.parent_span_id)
        for s in _spans(converted)
    ]


def _intervals(converted: ConvertedSession) -> list[tuple[datetime, datetime]]:
    return [(s.span_info.start_time, s.span_info.end_time) for s in _spans(converted)]


def _notes(span: SpanUnion) -> dict[str, Any]:
    assert span.metadata is not None
    return span.metadata["calibrated_agent_evaluation"]


def _placeholder(k: int) -> tuple[datetime, datetime]:
    return EPOCH + timedelta(seconds=2 * k), EPOCH + timedelta(seconds=2 * k + 1)


def _otel_ord() -> Trace:
    return normalize(OTelSpans(path=OTEL_ORD))


def _timed(span: Span) -> Span:
    return replace(
        span,
        started_at=f"2026-10-16T07:00:{2 * span.seq:02d}+00:00",
        ended_at=f"2026-10-16T07:00:{2 * span.seq + 1:02d}+00:00",
    )


def test_fixture_traces_convert_to_one_agent_span_per_span_and_one_tool_span_per_call(
    corpus_ord: Trace, corpus_clm: Trace
) -> None:
    orderbot = trace_to_session(corpus_ord)

    assert _shape(orderbot) == [
        ("invoke_agent", "ORD-1001/orderbot", None),
        ("execute_tool", "ORD-1001/orderbot#tool0", "ORD-1001/orderbot"),
        ("execute_tool", "ORD-1001/orderbot#tool1", "ORD-1001/orderbot"),
        ("execute_tool", "ORD-1001/orderbot#tool2", "ORD-1001/orderbot"),
    ]
    assert [(s.agent_span_id, s.tool_call.name) for s in _tool_spans(orderbot)] == [
        ("ORD-1001/orderbot", "lookup_order"),
        ("ORD-1001/orderbot", "check_refund_policy"),
        ("ORD-1001/orderbot", "issue_refund"),
    ]
    assert dict(orderbot.locations) == {
        "ORD-1001/orderbot": ("ORD-1001/orderbot", None),
        "ORD-1001/orderbot#tool0": ("ORD-1001/orderbot", 0),
        "ORD-1001/orderbot#tool1": ("ORD-1001/orderbot", 1),
        "ORD-1001/orderbot#tool2": ("ORD-1001/orderbot", 2),
    }
    assert list(orderbot.locations) == [s.span_info.span_id for s in _spans(orderbot)]
    assert dict(orderbot.tool_names) == {
        "ORD-1001/orderbot#tool0": "lookup_order",
        "ORD-1001/orderbot#tool1": "check_refund_policy",
        "ORD-1001/orderbot#tool2": "issue_refund",
    }

    crew = trace_to_session(corpus_clm)
    supervisor = "CLM-4471/supervisor"
    assert _shape(crew) == [
        ("invoke_agent", supervisor, None),
        ("invoke_agent", "CLM-4471/intake", supervisor),
        ("execute_tool", "CLM-4471/intake#tool0", "CLM-4471/intake"),
        ("invoke_agent", "CLM-4471/policy-checker", supervisor),
        ("execute_tool", "CLM-4471/policy-checker#tool0", "CLM-4471/policy-checker"),
        ("execute_tool", "CLM-4471/policy-checker#tool1", "CLM-4471/policy-checker"),
        ("invoke_agent", "CLM-4471/fraud-scorer", supervisor),
        ("execute_tool", "CLM-4471/fraud-scorer#tool0", "CLM-4471/fraud-scorer"),
        ("invoke_agent", "CLM-4471/payout", supervisor),
    ]
    assert [s.agent_span_id for s in _tool_spans(crew)] == [
        "CLM-4471/intake",
        "CLM-4471/policy-checker",
        "CLM-4471/policy-checker",
        "CLM-4471/fraud-scorer",
    ]
    assert list(crew.locations) == [s.span_info.span_id for s in _spans(crew)]

    # Spans stored out of order still convert in seq order.
    shuffled = replace(corpus_clm, spans=list(reversed(corpus_clm.spans)))
    assert _shape(trace_to_session(shuffled)) == _shape(crew)

    for trace, converted in ((corpus_ord, orderbot), (corpus_clm, crew)):
        (vendor_trace,) = converted.session.traces
        assert converted.session.session_id == trace.trace_id
        assert (vendor_trace.trace_id, vendor_trace.session_id) == (trace.trace_id, trace.trace_id)
        assert {(s.span_info.trace_id, s.span_info.session_id) for s in vendor_trace.spans} == {
            (trace.trace_id, trace.trace_id)
        }


def test_tool_lists_hold_each_called_tool_once_in_first_call_order(corpus_clm: Trace) -> None:
    crew = trace_to_session(corpus_clm)
    agents = {s.span_info.span_id: s for s in _agent_spans(crew)}
    tools_under = {
        agent_id: [
            (s.span_info.span_id, s.tool_call.name)
            for s in _tool_spans(crew)
            if s.agent_span_id == agent_id
        ]
        for agent_id in agents
    }

    for no_calls in ("CLM-4471/supervisor", "CLM-4471/payout"):
        assert agents[no_calls].available_tools == []
        assert tools_under[no_calls] == []

    policy = "CLM-4471/policy-checker"
    assert [t.name for t in agents[policy].available_tools] == ["lookup_policy_exclusions"]
    assert tools_under[policy] == [
        (f"{policy}#tool0", "lookup_policy_exclusions"),
        (f"{policy}#tool1", "lookup_policy_exclusions"),
    ]

    calls = [make_tool_call(name=name) for name in ("b", "a", "b")]
    (agent,) = _agent_spans(trace_to_session(make_trace(spans=[make_span(tool_calls=calls)])))
    assert [t.name for t in agent.available_tools] == ["b", "a"]


def test_tool_arguments_serialize_with_sorted_keys() -> None:
    def converted_with(args: dict[str, Any], result: dict[str, Any]) -> ConvertedSession:
        call = make_tool_call(args=args, result=result)
        span = make_span(input={"z": 1, "a": 2}, output={"y": "é", "b": 1}, tool_calls=[call])
        return trace_to_session(make_trace(spans=[span]))

    converted = converted_with({"z": 1, "a": 2, "m": {"y": 1, "b": 2}}, {"z": True, "a": "é"})
    reordered = converted_with({"a": 2, "m": {"b": 2, "y": 1}, "z": 1}, {"a": "é", "z": True})

    (agent,) = _agent_spans(converted)
    (tool,) = _tool_spans(converted)
    assert list(tool.tool_call.arguments) == ["a", "m", "z"]
    assert list(tool.tool_call.arguments["m"]) == ["b", "y"]
    assert tool.tool_result.content == '{"a":"é","z":true}'
    assert agent.user_prompt == '{"a":2,"z":1}'
    assert agent.agent_response == '{"b":1,"y":"é"}'
    assert agent.system_prompt is None

    serialized = session_json(converted)
    arguments = json.loads(serialized)[0]["spans"][1]["tool_call"]["arguments"]
    assert list(arguments) == ["a", "m", "z"]
    assert serialized == session_json(reordered)


def test_recorded_times_are_used_only_when_every_span_has_them(corpus_clm: Trace) -> None:
    trace = _otel_ord()
    assert trace.trace_id == "ad66fcf93d899366e10b923c366a42cb"
    (span,) = trace.spans
    assert len(span.tool_calls) == 3
    assert span.started_at is not None
    assert span.ended_at is not None
    recorded = (datetime.fromisoformat(span.started_at), datetime.fromisoformat(span.ended_at))

    converted = trace_to_session(trace)

    assert converted.times == "recorded"
    assert _intervals(converted) == [recorded] * 4
    assert '"start_time": "2026-10-16T07:33:58.788872Z"' in session_json(converted)

    without_end = trace_to_session(replace(trace, spans=[replace(span, ended_at=None)]))
    assert without_end.times == "placeholder"
    assert _intervals(without_end) == [_placeholder(0)] * 4

    # One span short of a time sends every span of the trace to placeholders.
    timed = [_timed(s) for s in corpus_clm.spans]
    all_timed = trace_to_session(replace(corpus_clm, spans=timed))
    assert all_timed.times == "recorded"
    assert [(a.span_info.start_time, a.span_info.end_time) for a in _agent_spans(all_timed)] == [
        (datetime.fromisoformat(s.started_at or ""), datetime.fromisoformat(s.ended_at or ""))
        for s in timed
    ]
    one_short = trace_to_session(
        replace(corpus_clm, spans=[*timed[:-1], replace(timed[-1], started_at=None)])
    )
    assert one_short.times == "placeholder"
    assert [(a.span_info.start_time, a.span_info.end_time) for a in _agent_spans(one_short)] == [
        _placeholder(k) for k in range(5)
    ]


@pytest.mark.parametrize("trace_id", NORMALIZED_IDS)
def test_traces_without_times_get_deterministic_placeholder_times(trace_id: str) -> None:
    trace = load_corpus_trace(trace_id)
    assert all(s.started_at is None and s.ended_at is None for s in trace.spans)

    converted = trace_to_session(trace)

    assert converted.times == "placeholder"
    agents = _agent_spans(converted)
    assert [a.span_info.span_id for a in agents] == [
        s.span_id for s in sorted(trace.spans, key=lambda s: s.seq)
    ]
    assert [(a.span_info.start_time, a.span_info.end_time) for a in agents] == [
        _placeholder(k) for k in range(len(agents))
    ]
    by_id = {a.span_info.span_id: a.span_info for a in agents}
    tools = _tool_spans(converted)
    assert len(tools) == sum(len(s.tool_calls) for s in trace.spans)
    for tool in tools:
        assert tool.agent_span_id is not None
        agent = by_id[tool.agent_span_id]
        assert agent.start_time <= tool.span_info.start_time
        assert tool.span_info.start_time <= tool.span_info.end_time <= agent.end_time
    assert session_json(trace_to_session(trace)) == session_json(converted)


_PRINT_SESSIONS = """
import sys
sys.path.insert(0, sys.argv[1])
from _helpers import load_corpus_trace
from calibrated_agent_evaluation.agentcore.detectors.session import session_json, trace_to_session
for trace_id in ("ORD-1001", "CLM-4471"):
    sys.stdout.buffer.write(session_json(trace_to_session(load_corpus_trace(trace_id))).encode())
    sys.stdout.buffer.write(b"\\n")
"""


def _sessions_printed_under_hash_seed(seed: str) -> bytes:
    done = subprocess.run(
        [sys.executable, "-c", _PRINT_SESSIONS, str(TESTS_DIR)],
        env={**os.environ, "PYTHONHASHSEED": seed},
        capture_output=True,
        check=True,
    )
    return done.stdout


def test_a_converted_session_serializes_the_same_in_every_process() -> None:
    in_process = b"".join(
        session_json(trace_to_session(load_corpus_trace(trace_id))).encode() + b"\n"
        for trace_id in ("ORD-1001", "CLM-4471")
    )

    seed_one = _sessions_printed_under_hash_seed("1")
    seed_two = _sessions_printed_under_hash_seed("2")

    assert seed_one == seed_two
    assert seed_one == in_process


def test_span_metadata_states_what_the_vendor_types_cannot_hold(corpus_clm: Trace) -> None:
    spans = list(corpus_clm.spans)
    spans[2] = replace(spans[2], status="errored", occurrence=2)
    trace = replace(corpus_clm, spans=spans)

    converted = trace_to_session(trace)

    by_id = {s.span_id: s for s in trace.spans}
    for vendor in _spans(converted):
        assert vendor.span_info.span_id is not None
        source = by_id[converted.locations[vendor.span_info.span_id][0]]
        assert vendor.metadata == {
            "calibrated_agent_evaluation": {
                "agent": source.agent,
                "role": source.role,
                "occurrence": source.occurrence,
                "status": source.status,
                "times": "placeholder",
                "available_tools": "only the tools this span called",
                "system_prompt": "not recorded",
            }
        }
    notes = {s.span_info.span_id: _notes(s) for s in _spans(converted)}
    assert {span_id: n["status"] for span_id, n in notes.items()} == {
        "CLM-4471/supervisor": "completed",
        "CLM-4471/intake": "completed",
        "CLM-4471/intake#tool0": "completed",
        "CLM-4471/policy-checker": "errored",
        "CLM-4471/policy-checker#tool0": "errored",
        "CLM-4471/policy-checker#tool1": "errored",
        "CLM-4471/fraud-scorer": "completed",
        "CLM-4471/fraud-scorer#tool0": "completed",
        "CLM-4471/payout": "completed",
    }
    assert [(n["agent"], n["role"], n["occurrence"]) for n in notes.values()] == [
        ("supervisor", "supervisor", 1),
        ("intake", "specialist", 1),
        ("intake", "specialist", 1),
        ("policy-checker", "specialist", 2),
        ("policy-checker", "specialist", 2),
        ("policy-checker", "specialist", 2),
        ("fraud-scorer", "specialist", 1),
        ("fraud-scorer", "specialist", 1),
        ("payout", "specialist", 1),
    ]

    recorded = trace_to_session(_otel_ord())
    assert [_notes(s)["times"] for s in _spans(recorded)] == ["recorded"] * 4
    assert "placeholder" not in session_json(recorded)


def test_placeholder_and_recorded_times_follow_seq_order_whatever_the_stored_order(
    corpus_clm: Trace,
) -> None:
    by_seq = sorted(corpus_clm.spans, key=lambda s: s.seq)
    shuffled = replace(corpus_clm, spans=[by_seq[k] for k in (3, 0, 4, 2, 1)])

    placeholder = trace_to_session(shuffled)
    assert placeholder.times == "placeholder"
    assert {a.span_info.span_id: _intervals_of(a) for a in _agent_spans(placeholder)} == {
        span.span_id: _placeholder(k) for k, span in enumerate(by_seq)
    }

    timed = {span.span_id: _timed(span) for span in by_seq}
    recorded = trace_to_session(replace(shuffled, spans=[timed[s.span_id] for s in shuffled.spans]))
    assert recorded.times == "recorded"
    assert {s.span_info.span_id: _intervals_of(s) for s in _spans(recorded)} == {
        vendor_id: _recorded(timed[span_id])
        for vendor_id, (span_id, _) in recorded.locations.items()
    }


def _intervals_of(span: SpanUnion) -> tuple[datetime, datetime]:
    return span.span_info.start_time, span.span_info.end_time


def _recorded(span: Span) -> tuple[datetime, datetime]:
    assert span.started_at is not None
    assert span.ended_at is not None
    return datetime.fromisoformat(span.started_at), datetime.fromisoformat(span.ended_at)


_NOT_JSON = datetime(2026, 10, 16, 7, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("span", "field"),
    [
        ({"input": {"asked_at": _NOT_JSON}}, "input"),
        ({"output": {"answered_at": _NOT_JSON}}, "output"),
        ({"tool_calls": [make_tool_call(args={"since": _NOT_JSON})]}, "tool call 0 arguments"),
        ({"tool_calls": [make_tool_call(result={"at": {_NOT_JSON}})]}, "tool call 0 result"),
    ],
    ids=["input", "output", "tool arguments", "tool result"],
)
def test_a_span_payload_that_is_not_json_is_refused_with_the_span_named(
    span: dict[str, Any], field: str
) -> None:
    refused = _refusal(_span(**span))

    expected = f"trace run-a: span 'run-a/root' cannot be converted: its {field} is not JSON"
    assert str(refused).startswith(expected)
    assert isinstance(refused.__cause__, TypeError)


def test_a_span_the_detectors_span_types_refuse_is_refused_with_the_span_named() -> None:
    # A trace built in code can skip validation, so a tool name that is not text reaches the
    # detector's span types, which refuse it.
    unnamed = ToolCall.model_construct(name=7, args={}, result={})

    refused = _refusal(_span(tool_calls=[unnamed]))

    assert str(refused) == (
        "trace run-a: span 'run-a/root' cannot be converted: the detector's span types refuse "
        "it (ValidationError on name); check those fields of the span. If they are valid, the "
        "installed strands-agents-evals checks spans differently from the version Calibrated "
        "Agent Evaluation was "
        "checked against: install that version"
    )
    assert isinstance(refused.__cause__, ValidationError)


def test_a_probed_or_spanless_trace_is_refused() -> None:
    probed = normalize(OTelSpans(path=OTEL_DEGRADED))
    assert (probed.trace_id, probed.ownership) == ("bf080971947dc8f6fa1aed373164ce70", "probed")

    with pytest.raises(SessionConversionError) as refused_probed:
        trace_to_session(probed)

    assert str(refused_probed.value) == (
        "trace bf080971947dc8f6fa1aed373164ce70 is probed: "
        "the detector needs an owned trace with spans"
    )

    spanless = make_trace(trace_id="run-empty", ownership="owned", spans=[])
    with pytest.raises(SessionConversionError) as refused_spanless:
        trace_to_session(spanless)

    assert str(refused_spanless.value) == (
        "trace run-empty has no spans: the detector needs an owned trace with spans"
    )


def _span(span_id: str = "run-a/root", **over: object) -> Span:
    return make_span(trace_id="run-a", span_id=span_id, **over)


def _refusal(*spans: Span) -> SessionConversionError:
    with pytest.raises(SessionConversionError) as refused:
        trace_to_session(make_trace(trace_id="run-a", spans=list(spans)))
    return refused.value


def test_bad_span_times_and_colliding_tool_ids_are_refused() -> None:
    start, end = "2026-10-16T07:00:00+00:00", "2026-10-16T07:00:01+00:00"

    unparseable = _refusal(_span(started_at="yesterday at noon", ended_at=end))
    assert str(unparseable) == (
        "trace run-a: span 'run-a/root' has started_at 'yesterday at noon', "
        "which is not an ISO-8601 time; write it as one with a UTC offset, such as "
        "'2026-10-16T07:00:00+00:00'"
    )
    assert isinstance(unparseable.__cause__, ValueError)

    # A bad time is refused even when another span's missing time would send the trace to
    # placeholder times, which would otherwise drop it unread.
    second = _span("run-a/second", seq=1, parent="run-a/root", tool_calls=[])
    beside_untimed = _refusal(_span(ended_at="not-a-time"), second)
    assert str(beside_untimed) == (
        "trace run-a: span 'run-a/root' has ended_at 'not-a-time', which is not an ISO-8601 "
        "time; write it as one with a UTC offset, such as '2026-10-16T07:00:00+00:00'"
    )
    assert isinstance(beside_untimed.__cause__, ValueError)

    naive = _refusal(_span(started_at=start, ended_at="2026-10-16T07:00:01"))
    assert str(naive) == (
        "trace run-a: span 'run-a/root' has ended_at '2026-10-16T07:00:01' with no timezone; "
        "the detector needs times with a UTC offset"
    )

    reversed_span = _refusal(_span(started_at=end, ended_at=start))
    assert str(reversed_span) == (
        "trace run-a: span 'run-a/root' ends at 2026-10-16T07:00:00+00:00, "
        "before it starts at 2026-10-16T07:00:01+00:00; fix its started_at or ended_at"
    )

    colliding = _refusal(_span("X"), _span("X#tool0", seq=1, parent="X", tool_calls=[]))
    assert str(colliding) == (
        "trace run-a: tool call 0 of span 'X' would get the id 'X#tool0', which another span "
        "of the trace already has; give the span 'X#tool0' another id"
    )

    # The same two ids convert when the span named X makes no tool call.
    apart = make_trace(
        trace_id="run-a",
        spans=[_span("X", tool_calls=[]), _span("X#tool0", seq=1, parent="X", tool_calls=[])],
    )
    assert list(trace_to_session(apart).locations) == ["X", "X#tool0"]
