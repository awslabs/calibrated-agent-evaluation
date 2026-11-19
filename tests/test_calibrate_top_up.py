# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Growing a locked test refused short of its plan: allowed for the same judge, and disclosed.

The single-use ledger refuses every reuse of a consumed set's gold, except a strict superset of a
set refused for having fewer labels than its plan, tested by the same judge. The record counts the
looks so the retest is visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from _helpers import make_label
from calibrated_agent_evaluation.calibrate import (
    CalibrationSet,
    InMemoryLockedTestLedger,
    LabelPair,
    LockedTestReuseError,
    open_for_top_up,
    plan_calibration,
)
from calibrated_agent_evaluation.calibrate.engine import GATE_THRESHOLD, certify

_PLAN = plan_calibration(0.90, GATE_THRESHOLD)


@dataclass(frozen=True)
class _Judge:
    judge_id: str = "j"
    model: str = "m"
    prompt_version: str = "v1"
    fingerprint: str | None = "fp-1"


def _set(set_id: str, fail: range, passing: range, *, wrong: int = 0) -> CalibrationSet:
    pairs = []
    for k, i in enumerate(fail):
        verdict = "PASS" if k < wrong else "FAIL"
        pairs.append(
            LabelPair(
                gold=make_label(trace_id=f"f{i}", kind="human", outcome="FAIL"),
                verdict=make_label(trace_id=f"f{i}", kind="llm-judge", outcome=verdict),
            )
        )
    for i in passing:
        pairs.append(
            LabelPair(
                gold=make_label(trace_id=f"p{i}", kind="human", outcome="PASS"),
                verdict=make_label(trace_id=f"p{i}", kind="llm-judge", outcome="PASS"),
            )
        )
    return CalibrationSet(pairs=tuple(pairs), split="locked_test", set_id=set_id)


def test_a_superset_of_a_set_refused_short_is_tested_and_its_looks_counted():
    ledger = InMemoryLockedTestLedger()
    first = certify(_Judge(), _set("a", range(5), range(5)), ledger=ledger, plan=_PLAN)
    assert first.result == "REFUSED"
    assert first.looks == 1
    grown = certify(_Judge(), _set("b", range(60), range(60)), ledger=ledger, plan=_PLAN)
    assert grown.result == "CERTIFIED"
    assert grown.looks == 2


def test_each_top_up_in_a_chain_adds_a_look():
    ledger = InMemoryLockedTestLedger()
    certify(_Judge(), _set("a", range(5), range(5)), ledger=ledger, plan=_PLAN)
    second = certify(_Judge(), _set("b", range(10), range(10)), ledger=ledger, plan=_PLAN)
    third = certify(_Judge(), _set("c", range(60), range(60)), ledger=ledger, plan=_PLAN)
    assert (second.looks, third.looks) == (2, 3)


def test_a_changed_judge_cannot_grow_the_set():
    ledger = InMemoryLockedTestLedger()
    certify(_Judge(), _set("a", range(5), range(5)), ledger=ledger, plan=_PLAN)
    with pytest.raises(LockedTestReuseError, match="already used"):
        certify(
            _Judge(fingerprint="fp-2"), _set("b", range(60), range(60)), ledger=ledger, plan=_PLAN
        )


def test_dropping_a_trace_of_the_refused_set_is_a_partial_overlap_and_refused():
    ledger = InMemoryLockedTestLedger()
    certify(_Judge(), _set("a", range(5), range(5)), ledger=ledger, plan=_PLAN)
    with pytest.raises(LockedTestReuseError, match="already used"):
        certify(_Judge(), _set("b", range(1, 60), range(60)), ledger=ledger, plan=_PLAN)


def test_a_certified_set_is_never_open_to_growth():
    ledger = InMemoryLockedTestLedger()
    done = certify(_Judge(), _set("a", range(60), range(60)), ledger=ledger, plan=_PLAN)
    assert done.result == "CERTIFIED"
    assert not open_for_top_up(done)
    with pytest.raises(LockedTestReuseError):
        certify(_Judge(), _set("b", range(70), range(70)), ledger=ledger, plan=_PLAN)


def test_a_set_refused_at_its_planned_size_is_never_open_to_growth():
    ledger = InMemoryLockedTestLedger()
    refused = certify(
        _Judge(), _set("a", range(55), range(55), wrong=15), ledger=ledger, plan=_PLAN
    )
    assert refused.result == "REFUSED"
    assert not open_for_top_up(refused)
    with pytest.raises(LockedTestReuseError):
        certify(_Judge(), _set("b", range(70), range(70)), ledger=ledger, plan=_PLAN)


def test_without_a_plan_a_refusal_is_not_open_to_growth():
    ledger = InMemoryLockedTestLedger()
    certify(_Judge(), _set("a", range(5), range(5)), ledger=ledger)
    with pytest.raises(LockedTestReuseError):
        certify(_Judge(), _set("b", range(60), range(60)), ledger=ledger, plan=_PLAN)


def test_a_judge_without_a_fingerprint_cannot_grow_a_set():
    ledger = InMemoryLockedTestLedger()
    certify(_Judge(fingerprint=None), _set("a", range(5), range(5)), ledger=ledger, plan=_PLAN)
    with pytest.raises(LockedTestReuseError):
        certify(
            _Judge(fingerprint=None), _set("b", range(60), range(60)), ledger=ledger, plan=_PLAN
        )


def test_a_ledger_seeded_from_records_keeps_the_rule_and_closes_a_grown_set():
    ledger = InMemoryLockedTestLedger()
    short = certify(_Judge(), _set("a", range(5), range(5)), ledger=ledger, plan=_PLAN)
    seeded = InMemoryLockedTestLedger.from_records([short])
    grown = certify(_Judge(), _set("b", range(10), range(10)), ledger=seeded, plan=_PLAN)
    assert grown.looks == 2

    reseeded = InMemoryLockedTestLedger.from_records([short, grown])
    # "a" was grown into "b"; a set holding "a" but not all of "b" overlaps "b" and is refused.
    with pytest.raises(LockedTestReuseError):
        certify(_Judge(), _set("c", range(5), range(5, 70)), ledger=reseeded, plan=_PLAN)
    third = certify(_Judge(), _set("d", range(60), range(60)), ledger=reseeded, plan=_PLAN)
    assert third.looks == 3


def test_allow_top_up_refuses_a_set_it_never_consumed():
    with pytest.raises(LockedTestReuseError, match="nothing to top up"):
        InMemoryLockedTestLedger().allow_top_up("nope", judge_fingerprint="fp-1")
