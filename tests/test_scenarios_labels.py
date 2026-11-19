# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each rater's newest label counts, and raters who record different expected outputs conflict."""

from __future__ import annotations

from typing import Any

from calibrated_agent_evaluation.scenarios.labels import newest_per_rater, resolve_expected
from calibrated_agent_evaluation.schema.types import Label, Outcome

DENIED = {"action": "REFUND_DENIED", "amount_usd": 0}
ISSUED = {"action": "REFUND_ISSUED", "amount_usd": 40}


def _label(
    outcome: Outcome, rater: str | None, expected: dict[str, Any] | None = None, **extra: Any
) -> Label:
    return Label(
        trace_id="ORD-1001",
        span_id=extra.pop("span_id", None),
        kind="human",
        outcome=outcome,
        critique="Out-of-window refund was issued.",
        failure_mode="trusted_broken_policy_tool" if outcome == "FAIL" else None,
        expected=expected,
        rater_id=rater,
        **extra,
    )


def test_a_raters_later_label_replaces_their_earlier_one() -> None:
    first, second = _label("PASS", "ana"), _label("FAIL", "ana", DENIED)

    newest = newest_per_rater([first, second, _label("FAIL", "bo", DENIED)])

    assert newest["ana"] is second
    assert set(newest) == {"ana", "bo"}


def test_raters_recording_the_same_expected_output_agree_on_it() -> None:
    resolved = resolve_expected([_label("FAIL", "ana", DENIED), _label("FAIL", "bo", DENIED)])

    assert resolved.value == DENIED
    assert not resolved.conflicting


def test_raters_recording_different_expected_outputs_conflict() -> None:
    disputed = resolve_expected([_label("FAIL", "ana", DENIED), _label("FAIL", "bo", ISSUED)])

    assert disputed.conflicting
    assert disputed.value is None


def test_a_label_without_an_expected_output_neither_conflicts_nor_supplies_one() -> None:
    partial = resolve_expected([_label("FAIL", "ana", DENIED), _label("FAIL", "bo")])

    assert partial == resolve_expected([_label("FAIL", "ana", DENIED)])
    assert resolve_expected([_label("FAIL", "ana")]).value is None


def test_key_order_does_not_make_equal_expected_outputs_conflict() -> None:
    reordered = {"amount_usd": 0, "action": "REFUND_DENIED"}

    resolved = resolve_expected([_label("FAIL", "ana", DENIED), _label("FAIL", "bo", reordered)])

    assert not resolved.conflicting
    assert resolved.value == DENIED
