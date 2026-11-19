# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mapping AgentCore evaluator results to advisory labels."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreNoPassDirection,
    AgentCoreScoreIncomplete,
    AgentCoreScoreOffScale,
)
from calibrated_agent_evaluation.agentcore.judge import ScaleValue
from calibrated_agent_evaluation.agentcore.scores import read_scores, result_to_label
from calibrated_agent_evaluation.schema.serde import label_identity

# A three-rung scale, used to exercise the scale-keyed outcome path at the mapping boundary.
_CORRECTNESS_SCALE = (
    ScaleValue(0.0, "Incorrect"),
    ScaleValue(0.5, "Partially Correct"),
    ScaleValue(1.0, "Perfectly Correct"),
)


def _result(**over: object) -> dict:
    base = {
        "evaluatorArn": "arn:aws:bedrock-agentcore:us-west-2:111122223333:evaluator/correctness",
        "evaluatorName": "correctness",
        "explanation": "the answer matches the expected SQL result",
        "context": {"spanContext": {"traceId": "t-1", "sessionId": "s-1", "spanId": "sp-1"}},
        "value": 1.0,
        "label": "Perfectly Correct",
    }
    base.update(over)
    return base


def test_a_batch_result_becomes_an_advisory_judge_label() -> None:
    label = result_to_label(_result(), job_id="run-42")
    assert label.kind == "llm-judge"
    assert label.trace_id == "t-1"
    assert label.span_id == "sp-1"
    assert label.outcome == "PASS"
    assert label.critique  # non-empty
    # The raw verdict is preserved verbatim for later calibration.
    assert "value=1.0" in label.critique
    assert "Perfectly Correct" in label.critique


def test_a_result_with_no_explanation_still_gets_a_nonempty_critique() -> None:
    label = result_to_label(_result(explanation=""))
    assert label.critique.strip()  # the schema would reject an empty critique


def test_a_non_numeric_score_maps_without_a_type_error() -> None:
    # A categorical evaluator reports a string score; it must not crash the comparison and, with no
    # pass label, maps to a conservative FAIL rather than raising an untyped error.
    label = result_to_label(_result(value="high", label=None))
    assert label.outcome == "FAIL"
    assert "value=high" in label.critique


def test_a_trace_level_result_has_no_span_id() -> None:
    label = result_to_label(_result(context={"spanContext": {"traceId": "t-1"}}))
    assert label.span_id is None


def test_a_low_score_is_an_advisory_fail() -> None:
    label = result_to_label(_result(value=0.0, label="Incorrect"))
    assert label.outcome == "FAIL"


def test_with_a_scale_the_outcome_follows_the_value_not_the_pass_sounding_label() -> None:
    # The label reads like a pass, but the value sits at the bottom of the scale. Given the scale,
    # the outcome is read from the value (FAIL); without it, the pass-sounding label wins (PASS).
    passy_label = _result(value=0.0, label="Perfectly Correct")
    assert result_to_label(passy_label, rating_scale=_CORRECTNESS_SCALE).outcome == "FAIL"
    assert result_to_label(passy_label).outcome == "PASS"


def test_a_value_off_the_published_scale_is_refused_at_the_mapping_boundary() -> None:
    with pytest.raises(AgentCoreScoreOffScale):
        result_to_label(_result(value=0.42, label="whatever"), rating_scale=_CORRECTNESS_SCALE)


def test_re_reading_the_same_results_yields_a_stable_identity() -> None:
    first = list(read_scores([_result(), _result()]))
    again = list(read_scores([_result(), _result()]))
    assert {label_identity(x) for x in first} == {label_identity(y) for y in again}


def test_a_no_direction_record_in_a_score_stream_is_refused_not_skipped() -> None:
    refusal = _result(
        evaluatorArn="arn:aws:bedrock-agentcore:::evaluator/Builtin.Refusal",
        evaluatorName="Builtin.Refusal",
    )
    stream = read_scores([_result(), refusal])
    assert next(stream).trace_id == "t-1"
    with pytest.raises(AgentCoreNoPassDirection, match=r"Builtin\.Refusal"):
        next(stream)


def test_two_different_results_do_not_collide() -> None:
    a = result_to_label(_result(value=1.0, label="Correct"))
    b = result_to_label(_result(value=0.0, label="Incorrect"))
    assert label_identity(a) != label_identity(b)


def test_a_result_with_no_trace_or_session_id_is_refused() -> None:
    with pytest.raises(AgentCoreScoreIncomplete):
        result_to_label(_result(context={"spanContext": {"spanId": "sp-1"}}))


def test_a_result_with_no_value_or_label_is_refused() -> None:
    with pytest.raises(AgentCoreScoreIncomplete):
        result_to_label(_result(value=None, label=None))


def _managed_result(evaluator_id: str, value: float, label: str) -> dict:
    # The shapes the two read paths carry: the Evaluate result names the evaluator by id and ARN,
    # a result-log record only by name and ARN. Both must resolve to the same evaluator.
    return _result(
        evaluatorArn=f"arn:aws:bedrock-agentcore:::evaluator/{evaluator_id}",
        evaluatorName=evaluator_id,
        value=value,
        label=label,
    )


def test_a_high_score_from_an_evaluator_that_scores_failures_high_is_a_fail() -> None:
    found = _managed_result("Builtin.FailureAnalysis", 1.0, "Failures Found")
    clean = _managed_result("Builtin.FailureAnalysis", 0.0, "No Failures Found")
    assert result_to_label(found).outcome == "FAIL"
    assert result_to_label(clean).outcome == "PASS"
    assert result_to_label(_managed_result("ThirdParty.DeepEval.Toxicity", 0.0, "0.0")).outcome == (
        "PASS"
    )
    assert result_to_label(_managed_result("ThirdParty.DeepEval.Bias", 1.0, "1.0")).outcome == (
        "FAIL"
    )


def test_a_partly_toxic_score_is_not_a_pass() -> None:
    # The default only passes a perfect score; mirrored for an inverted evaluator, only a perfectly
    # clean 0.0 passes, so some toxic opinions are a FAIL, not a lenient PASS.
    partly = _managed_result("ThirdParty.DeepEval.Toxicity", 0.3, "0.3")
    assert result_to_label(partly).outcome == "FAIL"


def test_an_evaluator_whose_high_score_is_good_keeps_high_as_a_pass() -> None:
    harmless = _managed_result("Builtin.Harmfulness", 1.0, "Not Harmful")
    assert result_to_label(harmless).outcome == "PASS"


def test_a_result_from_an_evaluator_with_no_pass_direction_is_refused() -> None:
    with pytest.raises(AgentCoreNoPassDirection):
        result_to_label(_managed_result("Builtin.Refusal", 1.0, "Yes"))
