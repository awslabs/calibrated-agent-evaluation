# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Critique-first label construction."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.review import PASS_DEFAULT_CRITIQUE, ReviewError, build_label


def test_submitted_label_is_human_with_nonempty_critique() -> None:
    label = build_label(trace_id="ORD-1001", outcome="FAIL", critique="over-refunded")
    assert label.kind == "human"
    assert label.outcome == "FAIL"
    assert label.critique == "over-refunded"


def test_kind_is_always_human_even_for_pass() -> None:
    # The review surface never emits a non-human label — kind is fixed by build_label.
    label = build_label(trace_id="ORD-1002", outcome="PASS", critique="looks correct")
    assert label.kind == "human"


def test_invalid_outcome_is_refused() -> None:
    with pytest.raises(ReviewError):
        build_label(trace_id="ORD-1001", outcome="MAYBE", critique="x")


def test_fail_with_empty_critique_is_refused() -> None:
    with pytest.raises(ReviewError):
        build_label(trace_id="ORD-1001", outcome="FAIL", critique="   ")


def test_fail_with_critique_is_accepted() -> None:
    label = build_label(trace_id="ORD-1001", outcome="FAIL", critique="the why")
    assert label.outcome == "FAIL"
    assert label.critique == "the why"


def test_pass_without_critique_gets_canonical_default() -> None:
    label = build_label(trace_id="ORD-1002", outcome="PASS", critique="")
    assert label.critique == PASS_DEFAULT_CRITIQUE


def test_pass_with_critique_keeps_the_labelers_text() -> None:
    label = build_label(trace_id="ORD-1002", outcome="PASS", critique="in-window, correct")
    assert label.critique == "in-window, correct"
