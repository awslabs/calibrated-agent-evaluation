# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared tool_call ↔ result pairing helper.

Pairing joins a tool call to its result by shared ``id``; an unpaired call must reconstruct with
an empty ``result`` and must NEVER be mis-joined to a different id's result. That invariant is the
whole reason pairing is one shared helper instead of per-adapter ad-hoc zipping.
"""

from __future__ import annotations

from calibrated_agent_evaluation.normalize.conventions.pairing import RawToolCall, pair


def test_pairs_calls_to_results_by_shared_id() -> None:
    calls = [
        RawToolCall(id="a", name="lookup", args={"q": 1}),
        RawToolCall(id="b", name="issue", args={"amt": 5}),
    ]
    results = {"a": {"ok": True}, "b": {"done": 1}}

    out = pair(calls, results)

    assert [t.name for t in out] == ["lookup", "issue"]
    assert out[0].result == {"ok": True}
    assert out[1].result == {"done": 1}


def test_unpaired_call_gets_empty_result_not_a_foreign_one() -> None:
    # 'b' has no result; only 'a' does. The negative invariant: 'b' must NOT inherit 'a''s result.
    calls = [
        RawToolCall(id="a", name="lookup", args={"q": 1}),
        RawToolCall(id="b", name="issue", args={"amt": 5}),
    ]
    results = {"a": {"ok": True}}

    out = pair(calls, results)

    assert out[0].result == {"ok": True}
    assert out[1].result == {}, "an unpaired call must reconstruct with an empty result"


def test_result_for_a_call_without_an_id_is_not_mis_joined() -> None:
    # A call carrying no id must not grab a keyed result meant for another call.
    calls = [RawToolCall(id=None, name="mystery", args={})]
    results = {"a": {"ok": True}}

    out = pair(calls, results)

    assert out[0].result == {}


def test_args_and_result_are_coerced_to_dicts() -> None:
    # Raw args/results may arrive as JSON strings; the normalized schema stores dicts.
    calls = [RawToolCall(id="x", name="t", args='{"location": "Boston"}')]
    results = {"x": '{"temp": 57}'}

    out = pair(calls, results)

    assert out[0].args == {"location": "Boston"}
    assert out[0].result == {"temp": 57}
