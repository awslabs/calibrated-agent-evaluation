# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Id-less tool calls are deduped by content, not counted once per resend.

`messages_adapter._absorb`'s dedup guard read `if key is None or key not in seen_call_ids`, so the
`key is None` arm appended unconditionally: for a tool call with no `id`, every span that resent the
conversation history counted it again. A harness that resends history each turn — the common pattern
for stateless chat-completions-style harnesses — therefore inflated the tool-call count by the
number of turns. And `id` is often provider-assigned, so a proxying or message-synthesizing harness
routinely has none.

Tool-call counts feed pass/fail rates, per-tool-call scoring, and the ingest gate's own tool-count
check, so the inflation was a wrong number with a clean exit code.

The same session shape drives every case here: `gen_ai/messages` spans whose `parts` carry
`tool_call`/`tool_call_response` entries — the grammar `test_conventions.py` exercises with ids.
"""

from __future__ import annotations

import json
from typing import Any

from calibrated_agent_evaluation.normalize import SessionSpans, normalize
from calibrated_agent_evaluation.schema.validate import validate


def _span(
    *, output_parts: list[dict[str, Any]], input_parts: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.output.messages": json.dumps([{"role": "assistant", "parts": output_parts}]),
            "gen_ai.input.messages": json.dumps([{"role": "tool", "parts": input_parts}]),
        }
    }


def _call(name: str, args: dict[str, Any], *, cid: str | None = None) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "tool_call", "name": name, "arguments": args}
    if cid is not None:
        part["id"] = cid
    return part


def _response(result: dict[str, Any], *, cid: str | None = None) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "tool_call_response", "result": result}
    if cid is not None:
        part["id"] = cid
    return part


def _tool_calls(spans: list[dict[str, Any]], trace_id: str) -> list[Any]:
    trace = normalize(SessionSpans(spans=spans, semconv="gen_ai/messages", trace_id=trace_id))
    validate(trace)
    return [call for span in trace.spans for call in span.tool_calls]


def test_two_idless_calls_resent_across_three_spans_recover_twice() -> None:
    """The ticket's exact case: 2 distinct id-less calls across 3 spans → 2 calls, not 6.

    Before the fix this produced `['lookup', 'refund', 'lookup', 'refund', 'lookup', 'refund']` —
    a 3x inflation matching the span count.
    """
    lookup = _call("lookup", {"order": "A-1"})
    refund = _call("refund", {"order": "A-1", "amount": 89.99})
    spans = [
        _span(output_parts=[lookup, refund], input_parts=[]),
        _span(output_parts=[lookup, refund], input_parts=[]),
        _span(output_parts=[lookup, refund], input_parts=[]),
    ]

    calls = _tool_calls(spans, "RESEND-1")
    assert [c.name for c in calls] == ["lookup", "refund"]


def test_the_recovered_idless_calls_keep_their_arguments() -> None:
    """Dedup must collapse the duplicates, not the content — the args are the call's identity."""
    spans = [
        _span(output_parts=[_call("lookup", {"order": "A-1"})], input_parts=[]),
        _span(output_parts=[_call("lookup", {"order": "A-1"})], input_parts=[]),
    ]
    (call,) = _tool_calls(spans, "RESEND-2")
    assert call.args == {"order": "A-1"}


def test_an_idless_call_recovers_its_result_by_position() -> None:
    """An id-less RESPONSE used to be dropped outright, so an id-less call could never pair.

    `_absorb` only recorded a `tool_call_response` when it carried an id, so every id-less call
    reconstructed with `result={}` even where the source plainly carried the result. A response has
    no name and no arguments, so position within the span is the only linkage there is.
    """
    spans = [
        _span(
            output_parts=[_call("lookup", {"order": "A-1"})],
            input_parts=[_response({"status": "delivered"})],
        ),
    ]
    (call,) = _tool_calls(spans, "RESEND-3")
    assert call.result == {"status": "delivered"}


def test_a_later_resend_supplies_a_result_the_first_span_lacked() -> None:
    """The realistic resend shape: turn 1 issues the call, turn 2's history carries the answer."""
    lookup = _call("lookup", {"order": "A-1"})
    spans = [
        _span(output_parts=[lookup], input_parts=[]),
        _span(output_parts=[lookup], input_parts=[_response({"status": "delivered"})]),
    ]
    (call,) = _tool_calls(spans, "RESEND-4")
    assert call.name == "lookup"
    assert call.result == {"status": "delivered"}


def test_a_genuine_repeat_within_one_span_is_kept() -> None:
    """The boundary dedup must not cross: two real calls to the same tool with the same arguments.

    Occurrences are counted per span and **maxed**, never summed. An agent that really called
    `search` twice in one turn shows it twice within a single span, so max is 2 and both survive.
    Plain content dedup would have erased the second, trading one silent loss for another.
    """
    search = _call("search", {"q": "refund policy"})
    spans = [_span(output_parts=[search, search], input_parts=[])]

    calls = _tool_calls(spans, "REPEAT-1")
    assert [c.name for c in calls] == ["search", "search"]


def test_a_genuine_repeat_survives_being_resent() -> None:
    """Both mechanisms together: 2 real calls in one turn, then the history resent twice.

    Summing would give 6, collapsing would give 1. Maxing per span gives 2 — the number of calls the
    agent actually made.
    """
    search = _call("search", {"q": "refund policy"})
    spans = [
        _span(output_parts=[search, search], input_parts=[]),
        _span(output_parts=[search, search], input_parts=[]),
        _span(output_parts=[search, search], input_parts=[]),
    ]

    calls = _tool_calls(spans, "REPEAT-2")
    assert [c.name for c in calls] == ["search", "search"]


def test_distinct_arguments_are_distinct_calls() -> None:
    """Content identity is `(name, arguments)`: same tool, different arguments, two calls."""
    spans = [
        _span(
            output_parts=[
                _call("lookup", {"order": "A-1"}),
                _call("lookup", {"order": "B-2"}),
            ],
            input_parts=[],
        ),
        _span(
            output_parts=[
                _call("lookup", {"order": "A-1"}),
                _call("lookup", {"order": "B-2"}),
            ],
            input_parts=[],
        ),
    ]

    calls = _tool_calls(spans, "ARGS-1")
    assert [c.args["order"] for c in calls] == ["A-1", "B-2"]


def test_the_existing_id_bearing_dedup_is_unchanged() -> None:
    """Regression guard: the id-keyed path (175 mentions, 38 unique ids) must not move."""
    paid = _call("paid", {"x": 1}, cid="p")
    spans = [
        _span(output_parts=[paid], input_parts=[_response({"ok": True}, cid="p")]),
        _span(output_parts=[paid], input_parts=[_response({"ok": True}, cid="p")]),
        _span(output_parts=[paid], input_parts=[_response({"ok": True}, cid="p")]),
    ]

    calls = _tool_calls(spans, "IDS-1")
    assert [c.name for c in calls] == ["paid"]
    assert calls[0].result == {"ok": True}


def test_id_bearing_and_idless_calls_coexist_in_one_session() -> None:
    """A proxying harness can lose the id on some messages and not others."""
    spans = [
        _span(
            output_parts=[_call("tagged", {"a": 1}, cid="t"), _call("untagged", {"b": 2})],
            input_parts=[_response({"ok": True}, cid="t")],
        ),
        _span(
            output_parts=[_call("tagged", {"a": 1}, cid="t"), _call("untagged", {"b": 2})],
            input_parts=[_response({"ok": True}, cid="t")],
        ),
    ]

    calls = _tool_calls(spans, "MIX-1")
    assert sorted(c.name for c in calls) == ["tagged", "untagged"]
