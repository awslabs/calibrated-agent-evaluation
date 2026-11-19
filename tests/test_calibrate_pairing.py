# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for observation pairing → confusion counts."""

from __future__ import annotations

import pytest

from _helpers import make_label
from calibrated_agent_evaluation.calibrate.errors import (
    UnpairedObservationError,
    UnratifiedGoldError,
)
from calibrated_agent_evaluation.calibrate.pairing import pair_observations
from calibrated_agent_evaluation.calibrate.record import LabelPair


def _pair(trace_id: str, gold_outcome: str, verdict_outcome: str, **over: object) -> LabelPair:
    """A (gold, verdict) pair for one item; gold is human by default (make_label default)."""
    span_id = over.pop("span_id", None)
    gold = make_label(trace_id=trace_id, span_id=span_id, kind="human", outcome=gold_outcome)
    verdict = make_label(
        trace_id=trace_id, span_id=span_id, kind="llm-judge", outcome=verdict_outcome, **over
    )
    return LabelPair(gold=gold, verdict=verdict)


# --- FAIL is the positive class -----------------------------------------


def test_pairing_counts_with_fail_as_the_positive_class() -> None:
    # 2 caught fails (tp), 1 missed fail (fn), 3 correct clears (tn), 1 false alarm (fp).
    pairs = [
        _pair("t1", "FAIL", "FAIL"),  # tp
        _pair("t2", "FAIL", "FAIL"),  # tp
        _pair("t3", "FAIL", "PASS"),  # fn (missed failure)
        _pair("t4", "PASS", "PASS"),  # tn
        _pair("t5", "PASS", "PASS"),  # tn
        _pair("t6", "PASS", "PASS"),  # tn
        _pair("t7", "PASS", "FAIL"),  # fp (false alarm)
    ]
    counts = pair_observations(pairs)
    assert (counts.tp, counts.fn, counts.tn, counts.fp) == (2, 1, 3, 1)
    # TPR = of the real failures, how many caught = 2/3; TNR = of the real passes, correct = 3/4.
    assert counts.n_pos == 3
    assert counts.n_neg == 4


def test_polarity_swap_is_detectably_different() -> None:
    # Guard against a polarity inversion: if PASS were (wrongly) treated as positive, the same data
    # would produce different counts. Assert the FAIL-positive counts, so a swap fails this test.
    pairs = [_pair("t1", "FAIL", "FAIL"), _pair("t2", "PASS", "PASS")]
    counts = pair_observations(pairs)
    # FAIL-positive: one tp (the FAIL/FAIL), one tn (the PASS/PASS). A swap would call these fp/fn.
    assert counts.tp == 1
    assert counts.tn == 1
    assert counts.fp == 0
    assert counts.fn == 0


# --- human-gold boundary ----------------------------------------


def test_refuses_non_human_gold() -> None:
    # Negative path: a gold set containing an LLM-produced label is refused — certifying a
    # judge against a judge turns every confidence bound into theater.
    gold = make_label(trace_id="t1", kind="llm-judge", outcome="FAIL")
    verdict = make_label(trace_id="t1", kind="llm-judge", outcome="FAIL")
    with pytest.raises(UnratifiedGoldError):
        pair_observations([LabelPair(gold=gold, verdict=verdict)])


def test_accepts_all_human_gold() -> None:
    # Positive companion: an all-human gold set pairs without complaint.
    counts = pair_observations([_pair("t1", "FAIL", "FAIL"), _pair("t2", "PASS", "FAIL")])
    assert counts.n_pos == 1
    assert counts.n_neg == 1


# --- exact pairing, never drop ------------------------------------------


def test_refuses_mismatched_trace_id() -> None:
    # Negative path: gold and verdict for different items can't be paired — typed error, not a
    # silent drop (which would bias TPR/TNR).
    gold = make_label(trace_id="t1", kind="human", outcome="FAIL")
    verdict = make_label(trace_id="t2", kind="llm-judge", outcome="FAIL")
    with pytest.raises(UnpairedObservationError):
        pair_observations([LabelPair(gold=gold, verdict=verdict)])


def test_refuses_mismatched_span_id() -> None:
    gold = make_label(trace_id="t1", span_id="t1/a", kind="human", outcome="FAIL")
    verdict = make_label(trace_id="t1", span_id="t1/b", kind="llm-judge", outcome="FAIL")
    with pytest.raises(UnpairedObservationError):
        pair_observations([LabelPair(gold=gold, verdict=verdict)])


def test_clean_one_to_one_pairs() -> None:
    # Positive: matched (trace_id, span_id) at the span level pairs correctly.
    counts = pair_observations([_pair("t1", "FAIL", "FAIL", span_id="t1/x")])
    assert counts.tp == 1
