# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a judge over its locked test to build the set it is certified on.

``run_judge`` refuses a judge without a certifying calibration record, so it cannot produce the
verdicts that calibration needs. ``locked_test_set`` runs the judge for certification only: it
writes nothing, returns no score, and pairs each verdict with its human gold.
"""

from __future__ import annotations

import pytest

from _judge_helpers import FakeJudgeModel, judge
from calibrated_agent_evaluation.calibrate import UnpairedObservationError, UnratifiedGoldError
from calibrated_agent_evaluation.execute import StructuredReply
from calibrated_agent_evaluation.judges import JudgeTarget, SpanGoldError, locked_test_set
from calibrated_agent_evaluation.schema.types import Label


def _gold(trace_id: str, outcome: str, *, span_id: str | None = None, kind: str = "human"):
    return Label(
        trace_id=trace_id,
        span_id=span_id,
        kind=kind,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        critique="a human critique",
    )


def _target(trace_id: str, span_id: str | None = None) -> JudgeTarget:
    return JudgeTarget(trace_id=trace_id, span_id=span_id, content=f"render of {trace_id}")


def test_pairs_each_verdict_with_its_gold_on_the_locked_split() -> None:
    model = FakeJudgeModel()  # always "fail"
    gold = [(_target("T1"), _gold("T1", "FAIL")), (_target("T2"), _gold("T2", "PASS"))]
    locked = locked_test_set(judge(), gold, set_id="gold-1", model_backend=model)
    assert locked.split == "locked_test"
    assert locked.set_id == "gold-1"
    assert [(p.gold.outcome, p.verdict.outcome) for p in locked.pairs] == [
        ("FAIL", "FAIL"),
        ("PASS", "FAIL"),
    ]
    assert all(p.verdict.kind == "llm-judge" for p in locked.pairs)
    assert model.call_count == 2


def test_non_human_gold_is_refused_before_any_model_call() -> None:
    model = FakeJudgeModel()
    gold = [(_target("T1"), _gold("T1", "FAIL", kind="llm-judge"))]
    with pytest.raises(UnratifiedGoldError, match="not 'human'"):
        locked_test_set(judge(), gold, set_id="g", model_backend=model)
    assert model.call_count == 0


def test_a_target_that_is_not_its_gold_is_refused_before_any_model_call() -> None:
    model = FakeJudgeModel()
    gold = [(_target("T1"), _gold("T2", "FAIL"))]
    with pytest.raises(UnpairedObservationError, match="T1"):
        locked_test_set(judge(), gold, set_id="g", model_backend=model)
    assert model.call_count == 0


def test_a_span_judge_needs_per_span_gold() -> None:
    model = FakeJudgeModel()
    span_judge = judge(level="span")
    gold = [(_target("T1"), _gold("T1", "FAIL"))]
    with pytest.raises(SpanGoldError, match="span_id=None"):
        locked_test_set(span_judge, gold, set_id="g", model_backend=model)
    assert model.call_count == 0


def test_a_span_judge_is_paired_on_its_span() -> None:
    fields = {"critique": "dropped vacant", "verdict": "fail"}
    reply = StructuredReply(fields=fields, via="tool_call")
    gold = [(_target("T1", "T1/intake"), _gold("T1", "FAIL", span_id="T1/intake"))]
    locked = locked_test_set(
        judge(level="span"), gold, set_id="g", model_backend=FakeJudgeModel(reply)
    )
    (pair,) = locked.pairs
    assert pair.verdict.span_id == "T1/intake"


def test_an_empty_gold_set_is_refused() -> None:
    with pytest.raises(UnpairedObservationError, match="no gold"):
        locked_test_set(judge(), [], set_id="g", model_backend=FakeJudgeModel())
