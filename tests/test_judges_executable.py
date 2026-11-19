# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ExecutableJudge + verdict-schema builder."""

from __future__ import annotations

import pytest

from _helpers import TEST_STAMP
from calibrated_agent_evaluation.calibrate.engine import certifies
from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord
from calibrated_agent_evaluation.judges.errors import JudgeRuntimeError
from calibrated_agent_evaluation.judges.executable import ExecutableJudge
from calibrated_agent_evaluation.judges.verdict_schema import build_verdict_schema


def _judge(**over: object) -> ExecutableJudge:
    base: dict[str, object] = {
        "judge_id": "orderbot-refund-resolved",
        "model": "bedrock/global.anthropic.claude-sonnet-4-6",
        "prompt_version": "v1",
        "level": "workflow",
        "prompt_template": "Was the refund resolved correctly? {trace}",
    }
    base.update(over)
    return ExecutableJudge(**base)  # type: ignore[arg-type]


def _certified_record(judge: ExecutableJudge) -> CalibrationRecord:
    return CalibrationRecord(
        provenance=TEST_STAMP,
        calib_id="calib-1",
        judge_id=judge.judge_id,
        model=judge.model,
        prompt_version=judge.prompt_version,
        judge_fingerprint=judge.fingerprint,
        hunted_set_id="hunted-1",
        n_pos=80,
        n_neg=80,
        tpr=Bound(point=0.95, lb95=0.90, n=80),
        tnr=Bound(point=0.95, lb95=0.90, n=80),
        youden_j=0.90,
        result="CERTIFIED",
    )


def test_executable_judge_satisfies_calibrate_judge_protocol() -> None:
    """ExecutableJudge exposes judge_id/model/prompt_version and certifies accepts it."""
    judge = _judge()
    assert judge.judge_id == "orderbot-refund-resolved"
    assert judge.model == "bedrock/global.anthropic.claude-sonnet-4-6"
    assert judge.prompt_version == "v1"
    # The structural Judge protocol pins to: certifies accepts it and reads the frozen pair.
    assert certifies(_certified_record(judge), judge) is True


def test_build_verdict_schema_orders_critique_before_verdict() -> None:
    """The emitted schema requires a non-empty critique and a binary verdict enum."""
    schema = build_verdict_schema(_judge())
    assert schema["type"] == "object"
    assert schema["required"] == ["critique", "verdict"]
    assert schema["properties"]["critique"]["minLength"] == 1
    assert schema["properties"]["verdict"]["enum"] == ["pass", "fail"]
    assert schema["additionalProperties"] is False


def test_schema_constraints_reach_inspect_json_schema() -> None:
    """The constraints (minLength, additionalProperties) survive the mapping into the
    Inspect JSONSchema sent to the model — not silently dropped."""
    from calibrated_agent_evaluation.execute.judge_model import _json_schema_from_dict

    js = _json_schema_from_dict(build_verdict_schema(_judge()))
    assert js.additionalProperties is False
    assert js.properties is not None
    assert js.properties["critique"].minLength == 1


def test_build_verdict_schema_rejects_verdict_before_critique() -> None:
    """A judge whose output_fields put verdict before critique is rejected."""
    bad = _judge(output_fields=("verdict", "critique"))
    with pytest.raises(JudgeRuntimeError, match="critique"):
        build_verdict_schema(bad)


def test_build_verdict_schema_rejects_missing_critique_field() -> None:
    """A judge whose output_fields omit critique entirely is rejected."""
    bad = _judge(output_fields=("verdict",))
    with pytest.raises(JudgeRuntimeError):
        build_verdict_schema(bad)
