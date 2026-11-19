# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A multi-span OpenLLMetry / OpenInference session keeps every span's content.

Both adapters began with `_merged_attributes(spans)` — one flat map built by `dict.update` over
every span. Both grammars carry content on **indexed** attributes whose numbering restarts per span
(`gen_ai.completion.0.*`, `llm.output_messages.0.*`), so merging made each span overwrite the last:
the session's tool calls collapsed to the final span's, and the trace input became whichever span
merged last rather than the first turn's.

That is the highest-leverage loss in the ingest path, because these two conventions are the only
on-ramp documented for LangGraph, LangChain, CrewAI, AutoGen, the OpenAI Agents SDK and Pydantic AI
(`docs/site/content/integrate/ingest-existing-traces.md`). Every multi-turn session ingested the
documented way lost all but its last span, reported `ownership="owned"`, and exited 0.

The fix concatenates per-span reconstructions. It does **not** build a tree: both conventions carry
no agent-boundary spans, so these sessions stay one collapsed node and crew attribution remains
unavailable on the convention path. The ticket asks that this be said out loud rather than left for
a reader to assume, so `test_the_fix_does_not_add_a_span_tree` asserts it.
"""

from __future__ import annotations

from typing import Any

from calibrated_agent_evaluation.normalize import SessionSpans, normalize
from calibrated_agent_evaluation.schema.validate import validate

# --- OpenLLMetry legacy dotted-index -------------------------------------------------------------


def _openllmetry_span(
    *, tool: str, args: str, result: str, cid: str, user: str, reply: str, end: int
) -> dict[str, Any]:
    """One legacy-era OpenLLMetry span. Every index restarts at 0 — that is the whole defect."""
    return {
        "endTimeUnixNano": str(end),
        "startTimeUnixNano": str(end - 1_000_000),
        "attributes": {
            "gen_ai.system": "openai",
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": user,
            "gen_ai.prompt.1.role": "tool",
            "gen_ai.prompt.1.tool_call_id": cid,
            "gen_ai.prompt.1.content": result,
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.completion.0.content": reply,
            "gen_ai.completion.0.tool_calls.0.id": cid,
            "gen_ai.completion.0.tool_calls.0.name": tool,
            "gen_ai.completion.0.tool_calls.0.arguments": args,
        },
    }


def test_openllmetry_three_span_session_recovers_all_three_tool_calls() -> None:
    """The ticket's case: 3 spans, one distinct tool call each → 3 calls, in call order.

    Before the fix this recovered `['issue_refund']` alone — the last span's — while reporting
    `ownership="owned"`.
    """
    spans = [
        _openllmetry_span(
            tool="lookup_order",
            args='{"order":"A-1"}',
            result='{"status":"delivered"}',
            cid="c1",
            user="where is A-1?",
            reply="looking it up",
            end=1_000,
        ),
        _openllmetry_span(
            tool="check_policy",
            args='{"days":6}',
            result='{"eligible":true}',
            cid="c2",
            user="can I return it?",
            reply="checking policy",
            end=2_000,
        ),
        _openllmetry_span(
            tool="issue_refund",
            args='{"amount":129}',
            result='{"refund":"RF-1"}',
            cid="c3",
            user="do it",
            reply="refund issued",
            end=3_000,
        ),
    ]
    trace = normalize(SessionSpans(spans=spans, semconv="openllmetry", trace_id="OLM-1"))
    validate(trace)

    names = [tc.name for span in trace.spans for tc in span.tool_calls]
    assert names == ["lookup_order", "check_policy", "issue_refund"]


def test_openllmetry_each_recovered_call_keeps_its_own_arguments_and_result() -> None:
    """Concatenating must carry each call's payload, not just its name."""
    spans = [
        _openllmetry_span(
            tool="lookup_order",
            args='{"order":"A-1"}',
            result='{"status":"delivered"}',
            cid="c1",
            user="u1",
            reply="r1",
            end=1_000,
        ),
        _openllmetry_span(
            tool="issue_refund",
            args='{"amount":129}',
            result='{"refund":"RF-1"}',
            cid="c2",
            user="u2",
            reply="r2",
            end=2_000,
        ),
    ]
    trace = normalize(SessionSpans(spans=spans, semconv="openllmetry", trace_id="OLM-2"))
    by_name = {tc.name: tc for span in trace.spans for tc in span.tool_calls}

    assert by_name["lookup_order"].args == {"order": "A-1"}
    assert by_name["lookup_order"].result == {"status": "delivered"}
    assert by_name["issue_refund"].args == {"amount": 129}
    assert by_name["issue_refund"].result == {"refund": "RF-1"}


def test_openllmetry_terminal_io_spans_the_session() -> None:
    """Input is the FIRST turn's request; output is the LAST-ended span's reply.

    The merge gave whichever span happened to land last in `dict.update` order for both.
    """
    spans = [
        _openllmetry_span(
            tool="lookup_order",
            args="{}",
            result="{}",
            cid="c1",
            user="where is A-1?",
            reply="looking it up",
            end=1_000,
        ),
        _openllmetry_span(
            tool="issue_refund",
            args="{}",
            result="{}",
            cid="c2",
            user="do it",
            reply="refund issued",
            end=2_000,
        ),
    ]
    trace = normalize(SessionSpans(spans=spans, semconv="openllmetry", trace_id="OLM-3"))

    assert "where is A-1?" in str(trace.input), "the first turn's request"
    assert "refund issued" in str(trace.output), "the last span's reply"


def test_openllmetry_spans_are_ordered_by_end_time_not_wire_order() -> None:
    """OTLP carries no ordering guarantee, so the wire order is not the session order."""
    late = _openllmetry_span(
        tool="issue_refund",
        args="{}",
        result="{}",
        cid="c2",
        user="do it",
        reply="refund issued",
        end=2_000,
    )
    early = _openllmetry_span(
        tool="lookup_order",
        args="{}",
        result="{}",
        cid="c1",
        user="where is A-1?",
        reply="looking it up",
        end=1_000,
    )
    trace = normalize(SessionSpans(spans=[late, early], semconv="openllmetry", trace_id="OLM-4"))

    names = [tc.name for span in trace.spans for tc in span.tool_calls]
    assert names == ["lookup_order", "issue_refund"], "reordered by end time, not as received"
    assert "where is A-1?" in str(trace.input)
    assert "refund issued" in str(trace.output)


def test_openllmetry_single_span_session_is_unchanged() -> None:
    """Regression guard: the one-span case the adapter was originally written for."""
    span = _openllmetry_span(
        tool="lookup_order",
        args='{"order":"A-1"}',
        result='{"status":"delivered"}',
        cid="c1",
        user="where is A-1?",
        reply="delivered",
        end=1_000,
    )
    trace = normalize(SessionSpans(spans=[span], semconv="openllmetry", trace_id="OLM-5"))
    validate(trace)

    assert len(trace.spans) == 1
    assert [tc.name for tc in trace.spans[0].tool_calls] == ["lookup_order"]
    assert trace.ownership == "owned"


def test_openllmetry_a_call_repeated_across_spans_is_recovered_once() -> None:
    """A resent history reports the same id again; that is one call, not two."""
    first = _openllmetry_span(
        tool="lookup_order",
        args="{}",
        result="{}",
        cid="same-id",
        user="u",
        reply="r1",
        end=1_000,
    )
    resent = _openllmetry_span(
        tool="lookup_order",
        args="{}",
        result="{}",
        cid="same-id",
        user="u",
        reply="r2",
        end=2_000,
    )
    trace = normalize(SessionSpans(spans=[first, resent], semconv="openllmetry", trace_id="OLM-6"))

    names = [tc.name for span in trace.spans for tc in span.tool_calls]
    assert names == ["lookup_order"]


# --- OpenInference indexed messages ---------------------------------------------------------------


def _openinference_span(
    *, agent: str, tool: str, args: str, cid: str, value_in: str, value_out: str, end: int
) -> dict[str, Any]:
    """One OpenInference span; its `llm.output_messages` index also restarts at 0."""
    return {
        "endTimeUnixNano": str(end),
        "startTimeUnixNano": str(end - 1_000_000),
        "attributes": {
            "openinference.span.kind": "LLM",
            "agent.name": agent,
            "input.value": value_in,
            "output.value": value_out,
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.tool_calls.0.tool_call.id": cid,
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": tool,
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": args,
        },
    }


def test_openinference_two_agent_session_recovers_both_tool_calls() -> None:
    """The ticket's second case: 2 spans, one tool call each → 2 calls, and agent 1's input kept.

    Before the fix this gave 1 span with 1 tool call and exit 0: agent 1's `web_search` was gone and
    the trace input was agent 2's.
    """
    spans = [
        _openinference_span(
            agent="researcher",
            tool="web_search",
            args='{"q":"refund policy"}',
            cid="oi-1",
            value_in='{"task":"research refunds"}',
            value_out='{"notes":"found policy"}',
            end=1_000,
        ),
        _openinference_span(
            agent="writer",
            tool="write_file",
            args='{"path":"out.md"}',
            cid="oi-2",
            value_in='{"task":"write it up"}',
            value_out='{"answer":"done"}',
            end=2_000,
        ),
    ]
    trace = normalize(SessionSpans(spans=spans, semconv="openinference", trace_id="OI-1"))
    validate(trace)

    names = [tc.name for span in trace.spans for tc in span.tool_calls]
    assert names == ["web_search", "write_file"]
    assert "research refunds" in str(trace.input), "the first span's input, not the second's"
    assert "done" in str(trace.output), "the last-ended span's output"


def test_openinference_single_span_session_is_unchanged() -> None:
    """Regression guard for the one-span case."""
    span = _openinference_span(
        agent="solo",
        tool="web_search",
        args='{"q":"x"}',
        cid="oi-1",
        value_in='{"task":"t"}',
        value_out='{"answer":"a"}',
        end=1_000,
    )
    trace = normalize(SessionSpans(spans=[span], semconv="openinference", trace_id="OI-2"))
    validate(trace)

    assert len(trace.spans) == 1
    assert [tc.name for tc in trace.spans[0].tool_calls] == ["web_search"]


def test_the_fix_does_not_add_a_span_tree() -> None:
    """Explicitly asserted, because the ticket asks that the limit be stated rather than assumed.

    Both conventions carry no agent-boundary spans, so a multi-span session still normalizes to ONE
    node. Crew attribution stays unavailable on the convention path — the calls are recovered, the
    tree is not built, and anyone reading this fix as "these sessions now support
    attribution" would be wrong.
    """
    spans = [
        _openinference_span(
            agent="researcher",
            tool="web_search",
            args="{}",
            cid="oi-1",
            value_in='{"task":"a"}',
            value_out='{"notes":"b"}',
            end=1_000,
        ),
        _openinference_span(
            agent="writer",
            tool="write_file",
            args="{}",
            cid="oi-2",
            value_in='{"task":"c"}',
            value_out='{"answer":"d"}',
            end=2_000,
        ),
    ]
    trace = normalize(SessionSpans(spans=spans, semconv="openinference", trace_id="OI-3"))

    assert len(trace.spans) == 1, "collapse-first: one node, whatever the span count"
    assert trace.spans[0].parent is None
    assert len(trace.spans[0].tool_calls) == 2, "both calls, on the one node"
