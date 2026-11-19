# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge prompts render numbers by value, so 149.0 and 149 read the same.

Langfuse returns a whole-number float as an int. The trace compares equal to the local copy, and
now its judge prompt and cassette fingerprint are equal too (the live half is
tests/integration/test_judges_render_live.py).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from _helpers import FIXTURES, load_corpus_trace
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.execute import CassetteJudgeModel
from calibrated_agent_evaluation.execute.cassette import Cassette, Mode
from calibrated_agent_evaluation.judges import (
    ExecutableJudge,
    build_verdict_schema,
    render_span,
    render_trace,
)
from calibrated_agent_evaluation.schema.types import Trace

_CASSETTE = FIXTURES / "raw" / "cassettes" / "judge_orderbot_resolved.cassette.json"


def _as_langfuse_returns(value: Any) -> Any:
    """``value`` after a JSON round trip that writes a whole-number float as an int."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _as_langfuse_returns(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_langfuse_returns(v) for v in value]
    return value


def _langfuse_copy(trace: Trace) -> Trace:
    spans = [
        replace(
            s,
            input=_as_langfuse_returns(s.input),
            output=_as_langfuse_returns(s.output),
            tool_calls=[
                replace(
                    tc, args=_as_langfuse_returns(tc.args), result=_as_langfuse_returns(tc.result)
                )
                for tc in s.tool_calls
            ],
        )
        for s in trace.spans
    ]
    return replace(
        trace,
        input=_as_langfuse_returns(trace.input),
        output=_as_langfuse_returns(trace.output),
        spans=spans,
    )


@pytest.fixture
def ord_1002() -> Trace:
    return load_corpus_trace("ORD-1002")


def test_the_corpus_trace_holds_a_whole_number_float(ord_1002):
    """The precondition: without it every assertion below passes vacuously."""
    lookup = ord_1002.spans[0].tool_calls[0].result
    assert isinstance(lookup, dict)
    assert repr(lookup["amount_usd"]) == "149.0"
    lf = _langfuse_copy(ord_1002)
    assert repr(lf.spans[0].tool_calls[0].result["amount_usd"]) == "149"  # type: ignore[index]
    assert lf == ord_1002


def test_a_trace_renders_the_same_either_way(ord_1002):
    text = render_trace(ord_1002)
    assert text == render_trace(_langfuse_copy(ord_1002))
    assert "'amount_usd': 149, 'delivered_at': '2026-05-04'" in text
    assert "'amount_usd': 149.0" not in text


def test_a_span_renders_the_same_either_way():
    crew = load_corpus_trace("CLM-4471")
    intake = next(s for s in crew.spans if s.agent == "intake")
    whole = replace(
        intake,
        tool_calls=[replace(tc, result={"amount_usd": 1200.0}) for tc in intake.tool_calls],
    )
    crew = replace(crew, spans=[whole if s is intake else s for s in crew.spans])
    lf = _langfuse_copy(crew)
    assert render_span(crew, whole) == render_span(
        lf, next(s for s in lf.spans if s.agent == "intake")
    )
    assert "-> {'amount_usd': 1200}" in render_span(crew, whole)


def test_values_that_are_not_whole_numbers_keep_their_rendering(ord_1002):
    """89.99 stays 89.99, a string "149.0" stays a string, and a bool is not a number.

    A float past 2**53 keeps its float form: it is not exact as an integer, and Langfuse returns it
    as a float too.
    """
    crew = load_corpus_trace("ORD-1001")
    tc = crew.spans[0].tool_calls[0]
    odd = replace(tc, result={"a": 89.99, "b": "149.0", "c": True, "d": [2.0, -0.5], "e": 1e300})
    trace = replace(crew, spans=[replace(crew.spans[0], tool_calls=[odd])])
    text = render_trace(trace)
    assert "'a': 89.99, 'b': '149.0', 'c': True, 'd': [2, -0.5]" in text
    assert "'e': 1e+300}" in text


def test_nan_and_infinity_are_rendered_as_themselves(ord_1002):
    crew = load_corpus_trace("ORD-1001")
    tc = replace(crew.spans[0].tool_calls[0], result={"x": float("inf"), "y": float("nan")})
    trace = replace(crew, spans=[replace(crew.spans[0], tool_calls=[tc])])
    assert "{'x': inf, 'y': nan}" in render_trace(trace)


def test_the_committed_cassette_replays_a_langfuse_read(ord_1002):
    doc = json.loads(_CASSETTE.read_text(encoding="utf-8"))
    system = doc["entries"][0]["request"]["system"]
    judge = ExecutableJudge(
        judge_id="orderbot-refund-resolved",
        model="bedrock/global.anthropic.claude-sonnet-4-6",
        prompt_version="demo-v1",
        level="workflow",
        prompt_template=system,
    )
    model = CassetteJudgeModel(Cassette(_CASSETTE, mode=Mode.REPLAY), supports_schema=True)
    schema = build_verdict_schema(judge)
    for trace in (ord_1002, _langfuse_copy(ord_1002)):
        reply = model.generate_structured(system=system, user=render_trace(trace), schema=schema)
        assert reply.fields["verdict"] == "pass"
