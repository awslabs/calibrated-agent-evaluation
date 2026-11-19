# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gold behind a certification is counted once, and a locked test is used once.

Two raters' labels on one target are one observation: counting both doubled the Wilson n and
certified a judge the single-label set refused (30 targets per class, 27 correct: REFUSED at one
label per target, CERTIFIED at two). And a locked test was single-use only by its name, so the same
pairs under a new ``set_id`` certified again.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from _helpers import make_label
from _judge_helpers import FakeJudgeModel, judge
from calibrated_agent_evaluation.calibrate import (
    DuplicateGoldError,
    InMemoryLockedTestLedger,
    LockedTestReuseError,
    UnpairedObservationError,
    pair_observations,
)
from calibrated_agent_evaluation.calibrate.engine import certify
from calibrated_agent_evaluation.calibrate.record import CalibrationSet, LabelPair
from calibrated_agent_evaluation.judges import JudgeTarget, locked_test_set


@dataclass(frozen=True)
class _Judge:
    judge_id: str = "j"
    model: str = "m"
    prompt_version: str = "v1"


def _pairs(
    n_pos: int, tp: int, n_neg: int, tn: int, *, raters: int = 1, prefix: str = ""
) -> tuple[LabelPair, ...]:
    out: list[LabelPair] = []
    for cls, n, right in (("FAIL", n_pos, tp), ("PASS", n_neg, tn)):
        other = "PASS" if cls == "FAIL" else "FAIL"
        for i in range(n):
            tid = f"{prefix}{cls[0].lower()}{i}"
            for r in range(raters):
                gold = make_label(trace_id=tid, kind="human", outcome=cls, rater_id=f"r{r}")
                verdict = make_label(
                    trace_id=tid, kind="llm-judge", outcome=cls if i < right else other
                )
                out.append(LabelPair(gold=gold, verdict=verdict))
    return tuple(out)


def test_double_coded_gold_is_refused_instead_of_doubling_the_sample() -> None:
    ledger = InMemoryLockedTestLedger()
    single = certify(
        _Judge(), CalibrationSet(_pairs(30, 27, 30, 27), "locked_test", "s1"), ledger=ledger
    )
    assert single.result == "REFUSED"

    doubled = CalibrationSet(_pairs(30, 27, 30, 27, raters=2, prefix="x"), "locked_test", "s2")
    with pytest.raises(DuplicateGoldError) as excinfo:
        certify(_Judge(), doubled, ledger=InMemoryLockedTestLedger())
    assert len(excinfo.value.duplicates) == 60
    assert "xf0 (x2)" in str(excinfo.value)
    assert "and 55 more" in str(excinfo.value)


def test_a_duplicate_target_is_an_unpaired_observation_for_existing_handlers() -> None:
    pairs = _pairs(3, 3, 3, 3)
    with pytest.raises(UnpairedObservationError, match="more than one gold label"):
        pair_observations((*pairs, pairs[0]))


def test_one_label_per_target_counts_every_target_once() -> None:
    counts = pair_observations(_pairs(30, 27, 30, 27))
    assert counts.n_pos + counts.n_neg == 60


def test_spans_of_one_trace_are_distinct_targets() -> None:
    pairs = tuple(
        LabelPair(
            gold=make_label(trace_id="t1", span_id=f"s{i}", kind="human", outcome="FAIL"),
            verdict=make_label(trace_id="t1", span_id=f"s{i}", kind="llm-judge", outcome="FAIL"),
        )
        for i in range(3)
    )
    assert pair_observations(pairs).n_pos == 3


def test_a_renamed_copy_of_a_consumed_locked_test_is_refused() -> None:
    ledger = InMemoryLockedTestLedger()
    certify(_Judge(), CalibrationSet(_pairs(30, 27, 30, 27), "locked_test", "s2"), ledger=ledger)
    renamed = CalibrationSet(tuple(reversed(_pairs(30, 27, 30, 27))), "locked_test", "s2-renamed")
    with pytest.raises(LockedTestReuseError, match="holds exactly the gold of 's2'"):
        certify(_Judge(prompt_version="v2"), renamed, ledger=ledger)


def test_a_set_sharing_a_trace_with_a_consumed_one_is_refused() -> None:
    ledger = InMemoryLockedTestLedger()
    certify(_Judge(), CalibrationSet(_pairs(30, 27, 30, 27), "locked_test", "a"), ledger=ledger)
    grown = _pairs(30, 27, 30, 27, prefix="new-") + _pairs(1, 1, 0, 0)
    with pytest.raises(LockedTestReuseError, match="1 of locked test 'b'"):
        certify(_Judge(), CalibrationSet(grown, "locked_test", "b"), ledger=ledger)


def test_a_disjoint_set_still_certifies_after_a_consumed_one() -> None:
    ledger = InMemoryLockedTestLedger()
    certify(_Judge(), CalibrationSet(_pairs(30, 27, 30, 27), "locked_test", "a"), ledger=ledger)
    fresh = certify(
        _Judge(),
        CalibrationSet(_pairs(80, 79, 80, 79, prefix="b-"), "locked_test", "b"),
        ledger=ledger,
    )
    assert fresh.result == "CERTIFIED"


def test_a_ledger_seeded_with_stored_traces_refuses_them() -> None:
    ledger = InMemoryLockedTestLedger(consumed_trace_ids=["f3"])
    with pytest.raises(LockedTestReuseError, match="f3"):
        certify(_Judge(), CalibrationSet(_pairs(5, 5, 5, 5), "locked_test", "c"), ledger=ledger)


def test_the_judge_runtime_refuses_a_target_graded_twice_before_any_model_call() -> None:
    model = FakeJudgeModel()
    target = JudgeTarget(trace_id="T1", span_id=None, content="render of T1")
    gold = [
        (target, make_label(trace_id="T1", outcome="FAIL", rater_id="r1")),
        (target, make_label(trace_id="T1", outcome="FAIL", rater_id="r2")),
    ]
    with pytest.raises(DuplicateGoldError, match="T1"):
        locked_test_set(judge(), gold, set_id="g", model_backend=model)
    assert model.call_count == 0
