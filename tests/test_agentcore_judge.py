# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentCore evaluator judge identity, fingerprint and calibratability."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreAdvisoryOnly,
    AgentCoreError,
    AgentCoreEvaluatorLevelUnsupported,
    AgentCoreNoPassDirection,
    AgentCoreRatingScaleMalformed,
    AgentCoreScoreOffScale,
)
from calibrated_agent_evaluation.agentcore.judge import (
    AgentCoreJudge,
    CalibratabilityClass,
    ScaleValue,
    build_agentcore_judge,
    classify_evaluator,
    judge_from_evaluator,
    load_agentcore_judge_file,
    outcome_for_value,
    require_calibratable,
)
from calibrated_agent_evaluation.agentcore.scores import result_to_label
from calibrated_agent_evaluation.calibrate.engine import certifies, pin_mismatch

# The published rating scale AgentCore's Helpfulness evaluator declares: a seven-rung scale from
# 0.0 to 1.0. Used to exercise value-to-outcome mapping against a real multi-rung scale.
_HELPFULNESS_SCALE = (
    ScaleValue(0.0, "Not Helpful At All"),
    ScaleValue(0.17, "Very Unhelpful"),
    ScaleValue(0.33, "Somewhat Unhelpful"),
    ScaleValue(0.5, "Neutral/Mixed"),
    ScaleValue(0.67, "Somewhat Helpful"),
    ScaleValue(0.83, "Very Helpful"),
    ScaleValue(1.0, "Above And Beyond"),
)


def _evaluator(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "evaluatorId": "corr",
        "evaluatorArn": "arn:aws:bedrock-agentcore:us-east-1:111122223333:evaluator/corr",
        "evaluatorType": "Custom",
        "level": "TRACE",
        "updatedAt": datetime(2026, 1, 1, tzinfo=UTC),
        "evaluatorConfig": {
            "llmAsAJudge": {
                "instructions": "judge correctness",
                "ratingScale": {
                    "numerical": [{"value": 0.0, "label": "FAIL"}, {"value": 1.0, "label": "PASS"}]
                },
                "modelConfig": {"bedrockEvaluatorModelConfig": {"modelId": "anthropic.claude-x"}},
            }
        },
    }
    base.update(over)
    return base


def _with_llm(**llm: Any) -> dict[str, Any]:
    """The default Custom evaluator with its ``llmAsAJudge`` fields replaced by ``llm``."""
    config = dict(_evaluator()["evaluatorConfig"]["llmAsAJudge"])
    config.update(llm)
    return _evaluator(evaluatorConfig={"llmAsAJudge": config})


def test_a_custom_evaluator_is_calibratable() -> None:
    assert classify_evaluator(_evaluator()) is CalibratabilityClass.CALIBRATABLE


def test_a_builtin_or_third_party_evaluator_is_advisory_only() -> None:
    for kind in ("Builtin", "ThirdParty"):
        evaluator = _evaluator(evaluatorType=kind, evaluatorConfig={})
        assert classify_evaluator(evaluator) is CalibratabilityClass.ADVISORY_ONLY


def test_a_custom_derived_evaluator_needs_a_pinned_model_to_be_calibratable() -> None:
    with_model = _evaluator(
        evaluatorType="CustomDerived",
        evaluatorConfig={
            "derived": {
                "baseEvaluatorId": "b",
                "modelConfig": {"bedrockEvaluatorModelConfig": {"modelId": "anthropic.claude-x"}},
            }
        },
    )
    without_model = _evaluator(
        evaluatorType="CustomDerived", evaluatorConfig={"derived": {"baseEvaluatorId": "b"}}
    )
    assert classify_evaluator(with_model) is CalibratabilityClass.CALIBRATABLE
    assert classify_evaluator(without_model) is CalibratabilityClass.ADVISORY_ONLY


def test_a_custom_code_evaluator_is_calibratable_only_when_pinned_to_a_lambda_version() -> None:
    fn = "arn:aws:lambda:us-east-1:111122223333:function:scorer"
    qualified = _evaluator(
        evaluatorType="CustomCode",
        evaluatorConfig={"codeBased": {"lambdaConfig": {"lambdaArn": f"{fn}:3"}}},
    )
    unqualified = _evaluator(
        evaluatorType="CustomCode",
        evaluatorConfig={"codeBased": {"lambdaConfig": {"lambdaArn": fn}}},
    )
    latest = _evaluator(
        evaluatorType="CustomCode",
        evaluatorConfig={"codeBased": {"lambdaConfig": {"lambdaArn": f"{fn}:$LATEST"}}},
    )
    assert classify_evaluator(qualified) is CalibratabilityClass.CALIBRATABLE
    assert classify_evaluator(unqualified) is CalibratabilityClass.ADVISORY_ONLY
    assert classify_evaluator(latest) is CalibratabilityClass.ADVISORY_ONLY


def test_an_unrecognised_evaluator_type_defaults_to_advisory_only() -> None:
    assert classify_evaluator(_evaluator(evaluatorType="SomethingNew")) is (
        CalibratabilityClass.ADVISORY_ONLY
    )


def test_the_judge_exposes_the_evaluator_arn_and_a_digest_of_its_configuration_as_its_pin() -> None:
    judge = judge_from_evaluator("corr-judge", _evaluator())
    assert judge.model == "arn:aws:bedrock-agentcore:us-east-1:111122223333:evaluator/corr"
    assert judge.prompt_version == judge.config_digest
    assert len(judge.prompt_version) == 16
    assert judge.prompt_version != judge.updated_at
    # The updatedAt stays on the judge, for display.
    assert judge.updated_at == "2026-01-01T00:00:00+00:00"
    assert judge.calibratability is CalibratabilityClass.CALIBRATABLE


def test_a_matching_record_pins_and_an_evaluator_edit_breaks_the_pin() -> None:
    judge = judge_from_evaluator("corr-judge", _evaluator())
    record = SimpleNamespace(
        judge_id=judge.judge_id,
        model=judge.model,
        prompt_version=judge.prompt_version,
        judge_fingerprint=judge.fingerprint,
    )
    assert pin_mismatch(record, judge) is None

    # The evaluator's instructions are edited in place: the new content (and so a new fingerprint)
    # must break the pin, so the stale calibration no longer certifies the changed evaluator.
    edited = judge_from_evaluator(
        "corr-judge",
        _with_llm(instructions="judge correctness strictly"),
    )
    assert edited.fingerprint != judge.fingerprint
    assert pin_mismatch(record, edited) is not None


def _record_for(judge: AgentCoreJudge) -> SimpleNamespace:
    """A CERTIFIED calibration record pinned to ``judge`` as it is now."""
    return SimpleNamespace(
        result="CERTIFIED",
        judge_id=judge.judge_id,
        model=judge.model,
        prompt_version=judge.prompt_version,
        judge_fingerprint=judge.fingerprint,
    )


def test_locking_an_evaluator_keeps_its_version_and_its_certification() -> None:
    # Attaching an evaluator to an online config locks it and moves its updatedAt (seen live),
    # with nothing that decides its verdicts changed.
    before = judge_from_evaluator("corr-judge", _evaluator())
    locked = judge_from_evaluator(
        "corr-judge",
        _evaluator(updatedAt=datetime(2026, 6, 1, tzinfo=UTC), lockedForModification=True),
    )
    assert (locked.prompt_version, locked.fingerprint) == (
        before.prompt_version,
        before.fingerprint,
    )
    assert certifies(_record_for(before), locked)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "edited",
    [
        _with_llm(instructions="judge correctness strictly"),
        _with_llm(
            ratingScale={
                "numerical": [{"value": 0.0, "label": "BAD"}, {"value": 1.0, "label": "GOOD"}]
            }
        ),
        _with_llm(
            modelConfig={
                "bedrockEvaluatorModelConfig": {
                    "modelId": "anthropic.claude-x",
                    "inferenceConfig": {"temperature": 0.7},
                }
            }
        ),
        _evaluator(description="a new description"),
    ],
    ids=["instructions", "rating-scale", "model-setting", "description"],
)
def test_a_change_to_the_evaluators_content_voids_its_calibration(edited: dict[str, Any]) -> None:
    before = judge_from_evaluator("corr-judge", _evaluator())
    # Same updatedAt on both sides: only the content moved, and that alone voids the record.
    assert edited["updatedAt"] == _evaluator()["updatedAt"]
    after = judge_from_evaluator("corr-judge", edited)
    assert after.prompt_version != before.prompt_version
    assert after.fingerprint != before.fingerprint
    assert not certifies(_record_for(before), after)  # type: ignore[arg-type]


def test_a_change_to_the_evaluators_level_changes_its_version() -> None:
    from calibrated_agent_evaluation.agentcore.judge import config_digest

    # A non-TRACE evaluator is refused before it is built, so the digest itself is compared.
    assert config_digest(_evaluator(level="SESSION")) != config_digest(_evaluator())


@pytest.mark.parametrize(
    "evaluator_id",
    ["Builtin.Helpfulness", "ThirdParty.DeepEval.Toxicity"],
    ids=["builtin", "third-party"],
)
def test_an_aws_owned_evaluators_moved_updatedat_changes_its_version(evaluator_id: str) -> None:
    # AWS owns these evaluators and may change them, so their updatedAt stays in the pin.
    before = judge_from_evaluator("aws", _managed(evaluator_id))
    moved = judge_from_evaluator(
        "aws", {**_managed(evaluator_id), "updatedAt": "2026-06-01T00:00:00+00:00"}
    )
    assert moved.prompt_version != before.prompt_version
    assert moved.fingerprint != before.fingerprint


def test_a_derived_evaluators_own_moved_updatedat_keeps_its_fingerprint() -> None:
    base = _managed("Builtin.Helpfulness")
    before = judge_from_evaluator(
        "my-derived", _derived_from("Builtin.Helpfulness"), base_evaluator=base
    )
    locked = {
        **_derived_from("Builtin.Helpfulness"),
        "updatedAt": "2026-06-01T00:00:00+00:00",
        "lockedForModification": True,
    }
    after = judge_from_evaluator("my-derived", locked, base_evaluator=base)
    assert after.fingerprint == before.fingerprint
    assert certifies(_record_for(before), after)  # type: ignore[arg-type]


@pytest.mark.parametrize("missing", ["instructions", "ratingScale"])
def test_a_custom_evaluator_read_without_its_instructions_or_scale_is_refused(
    missing: str,
) -> None:
    # The shape GetEvaluator returns for includedData=METADATA_ONLY: a version computed without
    # these fields would not change when they do.
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreEvaluatorUnreadable

    document = _evaluator()
    del document["evaluatorConfig"]["llmAsAJudge"][missing]
    with pytest.raises(AgentCoreEvaluatorUnreadable, match=missing) as raised:
        judge_from_evaluator("corr-judge", document)
    assert "corr" in str(raised.value)


def test_an_advisory_only_evaluator_is_refused_at_certification() -> None:
    advisory = judge_from_evaluator(
        "built-in-judge", _evaluator(evaluatorType="Builtin", evaluatorConfig={})
    )
    with pytest.raises(AgentCoreAdvisoryOnly, match="'Builtin'"):
        require_calibratable(advisory)


def test_an_agentcore_judge_with_no_certifying_record_is_refused_by_the_score_gate() -> None:
    from _helpers import TEST_STAMP
    from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord
    from calibrated_agent_evaluation.enforce import UncalibratedScoreError, enforce_score_gate

    advisory = judge_from_evaluator(
        "built-in-judge", _evaluator(evaluatorType="Builtin", evaluatorConfig={})
    )
    # Pinned to this exact judge, but not certifying: nothing in this feature can certify an
    # AgentCore judge, so the existing gate refuses it like any uncalibrated judge.
    record = CalibrationRecord(
        provenance=TEST_STAMP,
        calib_id="c",
        judge_id=advisory.judge_id,
        model=advisory.model,
        prompt_version=advisory.prompt_version,
        hunted_set_id="h",
        n_pos=80,
        n_neg=80,
        tpr=Bound(point=0.6, lb95=0.5, n=80),
        tnr=Bound(point=0.6, lb95=0.5, n=80),
        youden_j=0.2,
        result="REFUSED",
    )
    with pytest.raises(UncalibratedScoreError, match="no certifying calibration"):
        enforce_score_gate(record, advisory)
    # A calibratable judge is no different at the gate until something certifies it.
    calibratable = judge_from_evaluator("corr-judge", _evaluator())
    pinned = record.model_copy(
        update={
            "judge_id": calibratable.judge_id,
            "model": calibratable.model,
            "prompt_version": calibratable.prompt_version,
        }
    )
    with pytest.raises(UncalibratedScoreError, match="no certifying calibration"):
        enforce_score_gate(pinned, calibratable)


def test_a_calibratable_evaluator_is_allowed_to_certify() -> None:
    calibratable = judge_from_evaluator("corr-judge", _evaluator())
    require_calibratable(calibratable)  # does not raise


def test_a_value_above_the_middle_rung_passes_and_one_at_or_below_it_fails() -> None:
    # Seven rungs put the middle at the fourth (0.5, "Neutral/Mixed"); only the rungs above it pass.
    assert outcome_for_value(0.83, _HELPFULNESS_SCALE) == "PASS"
    assert outcome_for_value(0.67, _HELPFULNESS_SCALE) == "PASS"
    assert outcome_for_value(0.5, _HELPFULNESS_SCALE) == "FAIL"
    assert outcome_for_value(0.0, _HELPFULNESS_SCALE) == "FAIL"


def test_a_two_rung_scale_passes_only_its_top_value() -> None:
    binary = (ScaleValue(0.0, "Incorrect"), ScaleValue(1.0, "Perfectly Correct"))
    assert outcome_for_value(1.0, binary) == "PASS"
    assert outcome_for_value(0.0, binary) == "FAIL"


def test_a_value_not_on_the_published_scale_is_refused() -> None:
    # Off the top of the range, and inside the range but not a declared rung: both are unplaceable.
    with pytest.raises(AgentCoreScoreOffScale):
        outcome_for_value(1.5, _HELPFULNESS_SCALE)
    with pytest.raises(AgentCoreScoreOffScale):
        outcome_for_value(0.6, _HELPFULNESS_SCALE)


def test_a_value_cannot_be_mapped_when_the_evaluator_publishes_no_scale() -> None:
    with pytest.raises(AgentCoreScoreOffScale):
        outcome_for_value(0.83, ())


def test_a_value_a_float_error_from_the_middle_rung_gets_the_middle_rungs_outcome() -> None:
    correctness = (
        ScaleValue(0.0, "Incorrect"),
        ScaleValue(0.5, "Partially correct"),
        ScaleValue(1.0, "Correct"),
    )
    # Both sides of 0.5 within the rung tolerance are the "Partially correct" rung, so both fail.
    assert outcome_for_value(0.5 + 5e-10, correctness) == "FAIL"
    assert outcome_for_value(0.5 - 5e-10, correctness) == "FAIL"
    assert outcome_for_value(1.0 - 5e-10, correctness) == "PASS"


def test_a_non_uniform_scale_splits_at_its_middle_rung_not_its_value_midpoint() -> None:
    # The value midpoint of 0..10 is 5, which would fail every rung but the top. By rung position
    # the split sits between the second and third rungs.
    skewed = (ScaleValue(0, "a"), ScaleValue(1, "b"), ScaleValue(2, "c"), ScaleValue(10, "d"))
    assert [outcome_for_value(v, skewed) for v in (0, 1, 2, 10)] == ["FAIL", "FAIL", "PASS", "PASS"]


def test_rungs_given_out_of_order_are_ranked_by_value() -> None:
    shuffled = (ScaleValue(1.0, "Correct"), ScaleValue(0.0, "Incorrect"), ScaleValue(0.5, "Part"))
    assert outcome_for_value(1.0, shuffled) == "PASS"
    assert outcome_for_value(0.5, shuffled) == "FAIL"


def test_on_an_inverted_scale_the_low_rungs_pass_and_the_middle_still_fails() -> None:
    failure_analysis = (ScaleValue(0.0, "No Failures Found"), ScaleValue(1.0, "Failures Found"))
    assert outcome_for_value(0.0, failure_analysis, inverted=True) == "PASS"
    assert outcome_for_value(1.0, failure_analysis, inverted=True) == "FAIL"
    three = (ScaleValue(0.0, "clean"), ScaleValue(0.5, "some"), ScaleValue(1.0, "severe"))
    assert outcome_for_value(0.5, three, inverted=True) == "FAIL"


def _managed(evaluator_id: str, **config: Any) -> dict[str, Any]:
    return _evaluator(
        evaluatorId=evaluator_id,
        evaluatorArn=f"arn:aws:bedrock-agentcore:::evaluator/{evaluator_id}",
        evaluatorType="Builtin" if evaluator_id.startswith("Builtin.") else "ThirdParty",
        evaluatorConfig=config,
    )


def _derived_from(base: str) -> dict[str, Any]:
    return _evaluator(
        evaluatorId="my-derived",
        evaluatorType="CustomDerived",
        evaluatorConfig={
            "derived": {
                "baseEvaluatorId": base,
                "modelConfig": {"bedrockEvaluatorModelConfig": {"modelId": "anthropic.claude-x"}},
            }
        },
    )


_FAILURE_ANALYSIS_SCALE = {
    "llmAsAJudge": {
        "ratingScale": {
            "numerical": [
                {"value": 0.0, "label": "No Failures Found"},
                {"value": 1.0, "label": "Failures Found"},
            ]
        }
    }
}


def _derived_judge(base: str, **base_config: Any) -> AgentCoreJudge:
    return judge_from_evaluator(
        "my-derived", _derived_from(base), base_evaluator=_managed(base, **base_config)
    )


def _outcome(judge: AgentCoreJudge, value: float) -> str:
    result = {
        "evaluatorArn": f"arn:aws:bedrock-agentcore:::evaluator/{judge.evaluator_id}",
        "context": {"spanContext": {"traceId": "t-1", "sessionId": "t-1"}},
        "value": value,
        "label": str(value),
    }
    return result_to_label(result, rating_scale=judge.rating_scale, inverted=judge.inverted).outcome


def test_a_judge_on_an_evaluator_that_scores_failures_high_fails_a_high_score() -> None:
    toxicity = judge_from_evaluator("tox", _managed("ThirdParty.DeepEval.Toxicity"))
    assert [_outcome(toxicity, v) for v in (1.0, 0.3, 0.0)] == ["FAIL", "FAIL", "PASS"]
    assert judge_from_evaluator("bias", _managed("ThirdParty.DeepEval.Bias")).inverted
    assert not judge_from_evaluator("harm", _managed("Builtin.Harmfulness")).inverted
    assert not judge_from_evaluator("corr-judge", _evaluator()).inverted


def test_a_derived_judge_takes_its_bases_direction_and_scale() -> None:
    # A derived evaluator's own document carries neither; the base supplies the scoring.
    derived_toxicity = _derived_judge("ThirdParty.DeepEval.Toxicity")
    assert [_outcome(derived_toxicity, v) for v in (1.0, 0.0)] == ["FAIL", "PASS"]
    helpful = _derived_judge(
        "Builtin.Helpfulness",
        llmAsAJudge={
            "ratingScale": {
                "numerical": [{"value": r.value, "label": r.label} for r in _HELPFULNESS_SCALE]
            }
        },
    )
    assert helpful.rating_scale == _HELPFULNESS_SCALE
    assert outcome_for_value(0.83, helpful.rating_scale) == "PASS"


def test_a_derived_judge_without_its_bases_document_is_refused() -> None:
    # Without the base, the derived judge has no scale to read its value against.
    with pytest.raises(ValueError, match="base"):
        judge_from_evaluator("my-derived", _derived_from("Builtin.Helpfulness"))
    with pytest.raises(ValueError, match="base"):
        judge_from_evaluator(
            "my-derived",
            _derived_from("Builtin.Helpfulness"),
            base_evaluator=_managed("Builtin.Correctness"),
        )


def test_a_judge_on_an_evaluator_with_no_pass_direction_is_refused() -> None:
    with pytest.raises(AgentCoreNoPassDirection):
        judge_from_evaluator("refusal", _managed("Builtin.Refusal"))
    with pytest.raises(AgentCoreNoPassDirection):
        judge_from_evaluator("humor", _managed("ThirdParty.AutoEval.Humor"))
    with pytest.raises(AgentCoreNoPassDirection):
        _derived_judge("Builtin.Refusal")


def test_flipping_a_judges_direction_or_scale_changes_its_fingerprint() -> None:
    # A change to the direction table or the scale changes outcomes, so it must void a calibration.
    judge = judge_from_evaluator("corr-judge", _evaluator())
    assert replace(judge, inverted=True).fingerprint != judge.fingerprint
    assert replace(judge, rating_scale=_HELPFULNESS_SCALE).fingerprint != judge.fingerprint


def test_editing_a_derived_judges_base_changes_its_fingerprint() -> None:
    # The base supplies the prompt and the scoring, so editing it changes the derived judge's
    # verdicts even when the derived evaluator itself is untouched.
    before = judge_from_evaluator(
        "my-derived",
        _derived_from("Builtin.Helpfulness"),
        base_evaluator=_managed("Builtin.Helpfulness"),
    )
    edited_base = {**_managed("Builtin.Helpfulness"), "updatedAt": "2026-06-01T00:00:00+00:00"}
    after = judge_from_evaluator(
        "my-derived", _derived_from("Builtin.Helpfulness"), base_evaluator=edited_base
    )
    assert after.fingerprint != before.fingerprint


def _scaled(numerical: Any) -> dict[str, Any]:
    return _with_llm(ratingScale={"numerical": numerical})


@pytest.mark.parametrize(
    "numerical",
    [
        [{"value": 0.0, "label": "Incorrect"}, {"value": "high", "label": "Correct"}],
        [{"value": 0.0, "label": "Incorrect"}, {"value": True, "label": "Correct"}],
        [{"value": 0.0, "label": "Incorrect"}, {"label": "Correct"}],
        [{"value": 0.0, "label": "Incorrect"}, "Correct"],
        [{"value": 0.0, "label": "Incorrect"}, {"value": 0.0, "label": "Also incorrect"}],
        [{"value": 0.0, "label": "Incorrect"}, {"value": float("nan"), "label": "Correct"}],
        [{"value": 0.0, "label": "Incorrect"}, {"value": float("inf"), "label": "Correct"}],
        [{"value": 1.0, "label": "Correct"}],
        3,
    ],
    ids=[
        "non-numeric",
        "boolean",
        "missing-value",
        "non-object",
        "duplicate",
        "nan",
        "infinite",
        "single-rung",
        "not-a-list",
    ],
)
def test_a_malformed_published_scale_is_refused_when_the_judge_is_built(numerical: Any) -> None:
    with pytest.raises(AgentCoreRatingScaleMalformed):
        judge_from_evaluator("corr-judge", _scaled(numerical))


def test_the_judge_carries_the_evaluators_published_rating_scale() -> None:
    scaled = _with_llm(
        ratingScale={
            "numerical": [
                {"value": 0.0, "label": "Incorrect"},
                {"value": 0.5, "label": "Partially Correct"},
                {"value": 1.0, "label": "Perfectly Correct"},
            ]
        }
    )
    judge = judge_from_evaluator("corr-judge", scaled)
    assert judge.rating_scale == (
        ScaleValue(0.0, "Incorrect"),
        ScaleValue(0.5, "Partially Correct"),
        ScaleValue(1.0, "Perfectly Correct"),
    )


def test_an_evaluator_without_a_numerical_scale_carries_an_empty_scale() -> None:
    codebased = _evaluator(
        evaluatorType="CustomCode",
        evaluatorConfig={"codeBased": {"lambdaConfig": {"lambdaArn": "arn:…:function:f:3"}}},
    )
    assert judge_from_evaluator("code-judge", codebased).rating_scale == ()


def test_a_new_lambda_version_changes_the_fingerprint_and_breaks_the_pin() -> None:
    function = "arn:aws:lambda:us-east-1:111122223333:function:scorer"

    def code_judge(version: int) -> AgentCoreJudge:
        config = {"codeBased": {"lambdaConfig": {"lambdaArn": f"{function}:{version}"}}}
        return judge_from_evaluator(
            "code-judge", _evaluator(evaluatorType="CustomCode", evaluatorConfig=config)
        )

    certified, redeployed = code_judge(3), code_judge(4)
    record = SimpleNamespace(
        judge_id=certified.judge_id,
        model=certified.model,
        prompt_version=certified.prompt_version,
        judge_fingerprint=certified.fingerprint,
    )
    # Same evaluator ARN and updatedAt: only the Lambda version moved, which changes the
    # configuration digest and so voids the pin.
    assert redeployed.model == certified.model
    assert redeployed.prompt_version != certified.prompt_version
    assert redeployed.fingerprint != certified.fingerprint
    assert pin_mismatch(record, certified) is None
    assert pin_mismatch(record, redeployed) is not None


def _classes_named(hint: object) -> set[type]:
    """Every class a type hint names, through unions and generics."""
    import typing

    found = {hint} if isinstance(hint, type) else set()
    for arg in typing.get_args(hint):
        found |= _classes_named(arg)
    return found


def test_the_judge_surface_names_no_aws_sdk_type() -> None:
    import typing

    from calibrated_agent_evaluation.calibrate.engine import Judge

    hints = dict(typing.get_type_hints(AgentCoreJudge))
    properties = [n for n, v in vars(AgentCoreJudge).items() if isinstance(v, property)]
    for name in properties:
        hints[name] = typing.get_type_hints(getattr(AgentCoreJudge, name).fget)["return"]
    named = set().union(*(_classes_named(hint) for hint in hints.values()))
    vendor = ("boto3", "botocore", "bedrock_agentcore")
    assert not [cls for cls in named if cls.__module__.split(".")[0] in vendor]
    # Every member the calibration gate's Judge protocol reads is on the judge, as plain text.
    members = [n for n in vars(Judge) if not n.startswith("_")]
    assert members
    judge = judge_from_evaluator("corr-judge", _evaluator())
    assert all(isinstance(getattr(judge, member), str) for member in members)


def test_the_fingerprint_is_stable_for_an_unchanged_evaluator() -> None:
    first = judge_from_evaluator("corr-judge", _evaluator())
    again = judge_from_evaluator("corr-judge", _evaluator())
    assert first.fingerprint == again.fingerprint
    # A changed pinned model is a different judge, even at the same updatedAt.
    remodelled = AgentCoreJudge(
        judge_id=first.judge_id,
        evaluator_id=first.evaluator_id,
        evaluator_arn=first.evaluator_arn,
        evaluator_type=first.evaluator_type,
        updated_at=first.updated_at,
        config_digest=first.config_digest,
        pinned_model="anthropic.claude-y",
    )
    assert remodelled.fingerprint != first.fingerprint


@pytest.mark.parametrize(
    ("level", "named"),
    [("TOOL_CALL", "TOOL_CALL"), ("SESSION", "SESSION"), (None, "no level")],
    ids=["tool-call", "session", "missing"],
)
def test_an_evaluator_that_is_not_trace_level_is_refused_when_the_judge_is_built(
    level: str | None, named: str
) -> None:
    document = _evaluator()
    if level is None:
        del document["level"]
    else:
        document["level"] = level
    with pytest.raises(AgentCoreEvaluatorLevelUnsupported, match=named):
        judge_from_evaluator("corr-judge", document)


def test_a_session_level_evaluator_that_scores_failures_high_is_refused_for_its_level() -> None:
    # FailureAnalysis is inverted, but it is a SESSION evaluator: its level is what refuses it.
    failure_analysis = _managed("Builtin.FailureAnalysis", **_FAILURE_ANALYSIS_SCALE)
    failure_analysis["level"] = "SESSION"
    with pytest.raises(AgentCoreEvaluatorLevelUnsupported, match="SESSION"):
        judge_from_evaluator("fa", failure_analysis)


def test_a_derived_evaluators_own_level_decides_not_its_bases() -> None:
    session_derived = _derived_from("Builtin.Helpfulness")
    session_derived["level"] = "SESSION"
    with pytest.raises(AgentCoreEvaluatorLevelUnsupported, match="SESSION"):
        judge_from_evaluator("d", session_derived, base_evaluator=_managed("Builtin.Helpfulness"))
    session_base = _managed("Builtin.GoalSuccessRate")
    session_base["level"] = "SESSION"
    built = judge_from_evaluator(
        "d", _derived_from("Builtin.GoalSuccessRate"), base_evaluator=session_base
    )
    assert built.evaluator_level == "TRACE"


_JUDGE_FILE = (
    Path(__file__).resolve().parents[1] / "fixtures/judge-files/agentcore-correctness.json"
)


def _get_evaluator_response(document: dict[str, Any]) -> dict[str, Any]:
    """A ``GetEvaluator`` response, as the service model requires it, for ``document``."""
    return {
        "evaluatorName": document["evaluatorId"],
        "status": "ACTIVE",
        "createdAt": datetime(2026, 1, 1, tzinfo=UTC),
        **document,
    }


# A rating scale in the shape GetEvaluator returns it: every rung has a definition, and the judge
# config carries its instructions and model.
_SERVICE_SCALE = {
    "llmAsAJudge": {
        "instructions": "judge the response",
        "modelConfig": {"bedrockEvaluatorModelConfig": {"modelId": "anthropic.claude-x"}},
        "ratingScale": {
            "numerical": [
                {"value": 0.0, "label": "No Failures Found", "definition": "nothing failed"},
                {"value": 1.0, "label": "Failures Found", "definition": "something failed"},
            ]
        },
    }
}


def _control_clients() -> Any:
    import boto3

    from calibrated_agent_evaluation.agentcore.session import AwsClients

    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    return AwsClients(session)


def test_the_committed_agentcore_judge_file_loads_with_no_aws_call() -> None:
    judge_file = load_agentcore_judge_file(_JUDGE_FILE)
    assert (judge_file.judge_id, judge_file.evaluator_id, judge_file.failure_mode) == (
        "agentcore-correctness",
        "Builtin.Correctness",
        "incorrect_answer",
    )
    assert (judge_file.agent, judge_file.level) == (None, "workflow")


def test_importing_the_cli_and_loading_an_agentcore_judge_file_loads_no_aws_sdk() -> None:
    import subprocess
    import sys

    probe = (
        "import sys, calibrated_agent_evaluation.cli\n"
        f"assert calibrated_agent_evaluation.cli.main(['judge', '--check', {str(_JUDGE_FILE)!r}]) "
        "== 0\n"
        "assert not {'boto3', 'botocore'} & set(sys.modules), 'an AWS SDK was imported'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_an_agentcore_judge_file_builds_its_judge_from_get_evaluator() -> None:
    from botocore.stub import Stubber

    clients = _control_clients()
    builtin = _managed("Builtin.Correctness", **_SERVICE_SCALE)
    stub = Stubber(clients.client("bedrock-agentcore-control"))
    stub.add_response(
        "get_evaluator", _get_evaluator_response(builtin), {"evaluatorId": "Builtin.Correctness"}
    )
    with stub:
        judge = build_agentcore_judge(load_agentcore_judge_file(_JUDGE_FILE), clients=clients)
    assert judge.judge_id == "agentcore-correctness"
    assert judge.evaluator_arn == builtin["evaluatorArn"]
    assert judge.evaluator_level == "TRACE"
    assert [rung.value for rung in judge.rating_scale] == [0.0, 1.0]


def test_a_derived_agentcore_judge_file_builds_from_its_bases_document(tmp_path: Path) -> None:
    from botocore.stub import Stubber

    derived = _derived_from("ThirdParty.DeepEval.Toxicity")
    base = _managed("ThirdParty.DeepEval.Toxicity", **_SERVICE_SCALE)
    path = tmp_path / "derived.json"
    path.write_text(
        json.dumps(
            {
                "backend": "agentcore",
                "judge_id": "my-derived-judge",
                "evaluator_id": "my-derived",
                "failure_mode": "toxic",
            }
        ),
        encoding="utf-8",
    )
    clients = _control_clients()
    stub = Stubber(clients.client("bedrock-agentcore-control"))
    stub.add_response(
        "get_evaluator", _get_evaluator_response(derived), {"evaluatorId": "my-derived"}
    )
    stub.add_response(
        "get_evaluator",
        _get_evaluator_response(base),
        {"evaluatorId": "ThirdParty.DeepEval.Toxicity"},
    )
    with stub:
        judge = build_agentcore_judge(load_agentcore_judge_file(path), clients=clients)
    # The base's scale, and its inverted direction: a high toxicity value is the failure.
    assert [rung.value for rung in judge.rating_scale] == [0.0, 1.0]
    assert judge.inverted is True
    stub.assert_no_pending_responses()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"evaluator_id": None}, "evaluator_id"),
        ({"rubric": "x"}, "unknown field"),
        ({"level": "span"}, "level"),
    ],
    ids=["missing-evaluator", "unknown-field", "span-level"],
)
def test_a_malformed_agentcore_judge_file_is_refused(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    document = json.loads(_JUDGE_FILE.read_text(encoding="utf-8"))
    document.update(change)
    document = {key: value for key, value in document.items() if value is not None}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(AgentCoreError, match=message):
        load_agentcore_judge_file(path)


def test_a_judge_file_naming_an_unknown_evaluator_is_refused_naming_it() -> None:
    from botocore.stub import Stubber

    clients = _control_clients()
    stub = Stubber(clients.client("bedrock-agentcore-control"))
    stub.add_client_error("get_evaluator", service_error_code="ResourceNotFoundException")
    with stub, pytest.raises(AgentCoreError, match="Builtin.Correctness"):
        build_agentcore_judge(load_agentcore_judge_file(_JUDGE_FILE), clients=clients)


def test_a_derived_judge_file_whose_base_is_missing_is_refused_naming_the_base(
    tmp_path: Path,
) -> None:
    from botocore.stub import Stubber

    path = tmp_path / "derived.json"
    path.write_text(
        json.dumps(
            {
                "backend": "agentcore",
                "judge_id": "j",
                "evaluator_id": "my-derived",
                "failure_mode": "toxic",
            }
        ),
        encoding="utf-8",
    )
    clients = _control_clients()
    stub = Stubber(clients.client("bedrock-agentcore-control"))
    derived = _derived_from("Builtin.Gone")
    stub.add_response(
        "get_evaluator", _get_evaluator_response(derived), {"evaluatorId": "my-derived"}
    )
    stub.add_client_error("get_evaluator", service_error_code="ResourceNotFoundException")
    with stub, pytest.raises(AgentCoreError, match="Builtin.Gone"):
        build_agentcore_judge(load_agentcore_judge_file(path), clients=clients)
