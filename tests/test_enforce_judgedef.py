# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the JudgeDefinition contract.

Proves the binary default and the critique-before-verdict predicate (the invariant behind the
lint refusal).
"""

from __future__ import annotations

from calibrated_agent_evaluation.enforce import Axis, JudgeDefinition, critique_before_verdict


def _defn(**over: object) -> JudgeDefinition:
    base: dict[str, object] = {
        "judge_id": "orderbot-refund",
        "axes": (Axis(name="resolved"),),
        "output_fields": ("critique", "verdict"),
    }
    base.update(over)
    return JudgeDefinition(**base)  # type: ignore[arg-type]


# --- axis defaults to binary; critique must precede verdict ------------------


def test_axis_scale_defaults_to_binary() -> None:
    assert Axis(name="resolved").scale == "binary"


def test_critique_before_verdict_holds_for_correct_order() -> None:
    assert critique_before_verdict(_defn(output_fields=("critique", "verdict"))) is True


def test_critique_after_verdict_is_rejected() -> None:
    # Negative: verdict-first violates the critique-first invariant.
    assert critique_before_verdict(_defn(output_fields=("verdict", "critique"))) is False


def test_missing_critique_is_rejected() -> None:
    # Edge: a schema with no critique field cannot satisfy critique-first.
    assert critique_before_verdict(_defn(output_fields=("verdict",))) is False
