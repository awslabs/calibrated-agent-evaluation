# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the paired McNemar + discordant adjudication.

Behavioural: paired_mcnemar auto-selects mid-p under 25 discordant and asymptotic above, carries the
effect as an Interval, and reports an honest undefined result at zero discordant; discordant_pairs +
discovery_enqueue surface the flipped pairs tagged frame="discovery"; require_adjudication refuses
judge-labeled discordant pairs until each carries a kind="human" adjudicating Label.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.improve import (
    AdjudicationRequiredError,
    ImproveError,
    PairedObs,
    discordant_pairs,
    discovery_enqueue,
    paired_mcnemar,
    require_adjudication,
)
from calibrated_agent_evaluation.schema import Label


def _pairs(before_fail: int, after_fail: int, n: int, *, judge: bool = False) -> list[PairedObs]:
    """n paired obs: the first `before_fail` fail before, the first `after_fail` fail after."""
    return [
        PairedObs(
            f"t{i}",
            before_fail=(i < before_fail),
            after_fail=(i < after_fail),
            labels_are_judge_produced=judge,
        )
        for i in range(n)
    ]


def test_small_discordant_uses_midp_and_carries_interval_effect() -> None:
    """< 25 discordant -> mid-p; the effect is an Interval (a fall in fail-rate < 0)."""
    # 8 fail before, 3 fail after over 20 -> discordant = 5 (b) + 0 (c) = 5 < 25.
    result = paired_mcnemar(_pairs(before_fail=8, after_fail=3, n=20))
    assert result.method == "midp"
    assert result.p_value is not None
    assert result.effect.point is not None
    assert result.effect.point < 0.0  # fewer failures after


def test_large_discordant_uses_asymptotic() -> None:
    """>= 25 discordant pairs uses the asymptotic McNemar."""
    # 30 fail before, 5 after over 60 -> b = 25 discordant.
    result = paired_mcnemar(_pairs(before_fail=30, after_fail=5, n=60))
    assert result.method == "asymptotic"
    assert result.n_discordant >= 25


def test_zero_discordant_is_undefined_not_fabricated() -> None:
    """Identical before/after (no flips) -> method='undefined', p_value=None, reason set."""
    result = paired_mcnemar(_pairs(before_fail=5, after_fail=5, n=20))
    assert result.method == "undefined"
    assert result.p_value is None
    assert result.undefined_reason is not None
    assert result.effect.point == pytest.approx(0.0)


def test_empty_paired_input_refused() -> None:
    """An empty confirmation slice cannot be tested -> typed refusal, not a vacuous pass."""
    with pytest.raises(ImproveError):
        paired_mcnemar([])


def test_discordant_pairs_and_discovery_enqueue() -> None:
    """discordant_pairs returns only flipped pairs; enqueue tags them frame='discovery'."""
    pairs = _pairs(before_fail=8, after_fail=3, n=20)  # t3..t7 are flipped (fail->pass)
    disc = discordant_pairs(pairs)
    assert {p.trace_id for p in disc} == {"t3", "t4", "t5", "t6", "t7"}
    enq = discovery_enqueue(disc)
    assert all(e.frame == "discovery" for e in enq)
    assert {e.trace_id for e in enq} == {p.trace_id for p in disc}


def test_concordant_pair_not_enqueued() -> None:
    """Edge: a pair with no flip is not discordant and is not enqueued."""
    pairs = [PairedObs("t0", before_fail=True, after_fail=True)]  # concordant
    assert discordant_pairs(pairs) == ()
    assert discovery_enqueue(discordant_pairs(pairs)) == ()


def test_human_gold_labels_need_no_adjudication() -> None:
    """With human-gold labels (not judge-produced), no adjudication is required."""
    disc = discordant_pairs(_pairs(before_fail=8, after_fail=3, n=20, judge=False))
    require_adjudication(disc, {}, labels_are_judge_produced=False, target_metric="m")  # no raise


def test_judge_labels_refuse_until_every_discordant_adjudicated() -> None:
    """Judge labels + any unadjudicated discordant pair -> AdjudicationRequiredError."""
    disc = discordant_pairs(_pairs(before_fail=8, after_fail=3, n=20, judge=True))
    # No adjudications at all -> refuse.
    with pytest.raises(AdjudicationRequiredError):
        require_adjudication(disc, {}, labels_are_judge_produced=True, target_metric="m")
    # A non-human (unratified LLM) adjudication for one pair -> still refuse.
    partial = {
        p.trace_id: Label(
            trace_id=p.trace_id,
            span_id=None,
            kind="human",
            outcome="FAIL",
            critique="human adjudicated",
            failure_mode="m",
        )
        for p in disc[:-1]
    }
    partial[disc[-1].trace_id] = Label(
        trace_id=disc[-1].trace_id,
        span_id=None,
        kind="llm-judge",
        outcome="FAIL",
        critique="judge said fail",
    )
    with pytest.raises(AdjudicationRequiredError):
        require_adjudication(disc, partial, labels_are_judge_produced=True, target_metric="m")


def test_judge_labels_pass_once_every_discordant_is_human_adjudicated() -> None:
    """Once every discordant pair has a kind='human' adjudication, the gate passes."""
    disc = discordant_pairs(_pairs(before_fail=8, after_fail=3, n=20, judge=True))
    adjudications = {
        p.trace_id: Label(
            trace_id=p.trace_id,
            span_id=None,
            kind="human",
            outcome="FAIL",
            critique="human adjudicated the flip",
            failure_mode="m",
        )
        for p in disc
    }
    require_adjudication(
        disc, adjudications, labels_are_judge_produced=True, target_metric="m"
    )  # no raise


def test_degenerate_discordant_cells_no_longer_give_a_zero_width_effect() -> None:
    """B=5, c=0 (and b=0, c=4) with a=d=0 used to read [-1, -1]."""
    improved = paired_mcnemar([PairedObs(f"t{i}", True, False) for i in range(5)])
    assert improved.effect.point == -1.0
    assert improved.effect.lo == -1.0
    assert improved.effect.hi == pytest.approx(-(1.0 - 2**0.5 * 0.434482), abs=1e-5)
    assert improved.effect.hi - improved.effect.lo > 0.6
    assert improved.effect.method == "newcombe_paired_hybrid_score"

    regressed = paired_mcnemar([PairedObs(f"t{i}", False, True) for i in range(4)])
    assert regressed.effect.point == 1.0
    assert regressed.effect.hi == 1.0
    assert regressed.effect.hi - regressed.effect.lo > 0.6


def test_a_trace_id_in_two_pairs_is_refused() -> None:
    """Listing the same 20 pairs twice used to shrink p from 0.016 to 0.0002."""
    once = [PairedObs(f"t{i}", before_fail=i < 6, after_fail=False) for i in range(20)]
    with pytest.raises(ImproveError, match="appear more than once"):
        paired_mcnemar(once + once)
