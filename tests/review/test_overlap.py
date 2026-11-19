# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The minimum overlap the agreement gate needs, and the gate's behaviour around it.

The size is deterministic (plug-in counts, exact bootstrap). The gate checks are simulations of
two raters through the real ``inter_rater_agreement``; each states its reps and seed, so every
assertion is exact for its seed.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from calibrated_agent_evaluation.review.agreement import inter_rater_agreement
from calibrated_agent_evaluation.review.errors import ReviewError
from calibrated_agent_evaluation.review.gate import ALPHA_LOWER_BOUND_GATE, MINIMUM_OVERLAP
from calibrated_agent_evaluation.review.overlap import (
    alpha_lower_bound_at,
    clears_planned,
    plan_overlap,
    plug_in_counts,
    share_below_gate,
)
from calibrated_agent_evaluation.schema import Label
from calibrated_agent_evaluation.stats.agreement import bootstrap_agreement, krippendorff_alpha

RATERS = ("owner", "colleague")


def _units(counts: tuple[int, int, int]) -> list[tuple[str, str]]:
    both_fail, both_pass, split = counts
    halves = [("FAIL", "PASS"), ("PASS", "FAIL")]
    return (
        [("FAIL", "FAIL")] * both_fail
        + [("PASS", "PASS")] * both_pass
        + [halves[i % 2] for i in range(split)]
    )


def _labels(units: list[tuple[str, str]]) -> list[Label]:
    return [
        Label(
            trace_id=f"T-{i}",
            span_id=None,
            kind="human",
            outcome=outcome,  # type: ignore[arg-type]
            critique="c",
            rater_id=rater,
            codebook_version="v1",
        )
        for i, pair in enumerate(units)
        for rater, outcome in zip(RATERS, pair, strict=True)
    ]


def _gate(units: list[tuple[str, str]], *, n_resamples: int = 500, seed: int = 0) -> str:
    report = inter_rater_agreement(
        _labels(units), raters=RATERS, codebook_version="v1", n_resamples=n_resamples, seed=seed
    )
    return report.gate


def _simulated(n: int, alpha: float, rng: random.Random) -> list[tuple[str, str]]:
    """Two raters with equal marginals at prevalence 0.5 and population alpha ``alpha``."""
    split = (1.0 - alpha) * 0.5
    units = []
    for _ in range(n):
        u = rng.random()
        if u < split:
            units.append(("FAIL", "PASS") if rng.random() < 0.5 else ("PASS", "FAIL"))
        elif u < split + (1.0 - split) / 2.0:
            units.append(("FAIL", "FAIL"))
        else:
            units.append(("PASS", "PASS"))
    return units


@pytest.mark.slow
def test_the_recorded_minimum_is_what_the_planner_computes() -> None:
    assert plan_overlap().n_units == MINIMUM_OVERLAP == 130


def test_the_minimum_holds_at_every_larger_size_and_not_just_the_first_that_clears() -> None:
    bound = {n: alpha_lower_bound_at(plug_in_counts(n)) for n in range(90, 260)}
    first_clear = min(n for n, lb in bound.items() if lb >= ALPHA_LOWER_BOUND_GATE)
    assert first_clear == 100  # the sawtooth: 100 clears, then 110 and 122 fall short again
    assert bound[110] < ALPHA_LOWER_BOUND_GATE
    assert not clears_planned(plug_in_counts(MINIMUM_OVERLAP - 1))
    assert all(clears_planned(plug_in_counts(n)) for n in range(MINIMUM_OVERLAP, 400))


def test_without_the_resampling_margin_the_minimum_would_be_123() -> None:
    lowest = [n for n in range(90, 400) if alpha_lower_bound_at(plug_in_counts(n)) < 0.80]
    assert lowest[-1] + 1 == 123


def test_the_closed_form_alpha_is_krippendorffs() -> None:
    for counts in [(58, 58, 7), (13, 34, 3), (5, 0, 2), (40, 41, 0)]:
        n = sum(counts)
        both_fail, both_pass, split = counts
        n_fail, n_pass = 2 * both_fail + split, 2 * both_pass + split
        closed = 1.0 - (2 * n - 1) * split / (n_fail * n_pass)
        assert closed == pytest.approx(krippendorff_alpha(_units(counts)), abs=1e-12)


@pytest.mark.slow
def test_the_exact_share_below_the_gate_is_what_resampling_estimates() -> None:
    # 20000 resamples, seed 3: the standard error of the sampled share is about 0.001.
    rng = random.Random(3)
    for n in (60, MINIMUM_OVERLAP, 200):
        counts = plug_in_counts(n)
        units = _units(counts)
        draws = []
        for _ in range(20000):
            resample = [units[rng.randrange(n)] for _ in range(n)]
            if len({v for pair in resample for v in pair}) > 1:
                draws.append(krippendorff_alpha(resample))
        sampled = sum(d < ALPHA_LOWER_BOUND_GATE - 1e-12 for d in draws) / len(draws)
        assert share_below_gate(counts) == pytest.approx(sampled, abs=0.004), n


def test_a_rarer_failure_needs_a_larger_overlap() -> None:
    assert plan_overlap(prevalence=0.3, scan_limit=400).n_units > MINIMUM_OVERLAP


def test_an_expected_alpha_at_the_gate_is_refused() -> None:
    with pytest.raises(ReviewError, match="no overlap size clears it"):
        plan_overlap(expected_alpha=ALPHA_LOWER_BOUND_GATE)


def test_impossible_counts_are_refused() -> None:
    with pytest.raises(ReviewError):
        plug_in_counts(1)
    with pytest.raises(ReviewError):
        plug_in_counts(50, expected_alpha=0.9, prevalence=1.0)
    with pytest.raises(ReviewError):
        alpha_lower_bound_at((1, 0, 0))


def test_perfect_agreement_on_a_small_overlap_is_not_usable() -> None:
    """The hole the minimum closes: no disagreement on 20 traces has a lower bound of 1.0."""
    perfect = [("FAIL", "FAIL")] * 10 + [("PASS", "PASS")] * 10
    report = inter_rater_agreement(
        _labels(perfect), raters=RATERS, codebook_version="v1", n_resamples=200
    )
    assert report.alpha.lower == 1.0
    assert report.gate == "top_up_overlap"


def test_simulated_good_raters_below_the_minimum_are_never_usable() -> None:
    # 100 sets of 20 shared traces from raters at alpha 0.95, 300 resamples each, seed 20261026.
    rng = random.Random(20261026)
    gates = []
    old_rule_usable = 0
    for _ in range(100):
        units = _simulated(20, 0.95, rng)
        try:
            lower = bootstrap_agreement(units, krippendorff_alpha, n_resamples=300).lower
        except Exception:
            continue
        old_rule_usable += lower >= ALPHA_LOWER_BOUND_GATE
        gates.append(_gate(units, n_resamples=300))
    assert old_rule_usable > 30  # without the minimum, a third or more of these sets passed
    assert set(gates) == {"top_up_overlap"}


@pytest.mark.slow
def test_simulated_good_raters_below_the_minimum_are_never_told_to_revise() -> None:
    # 100 sets each of 20, 60 and 100 shared traces from raters at alpha 0.90, 300 resamples
    # each, seed 20261027: an upper bound under 0.80 never happens to raters this good.
    for n in (20, 60, 100):
        rng = random.Random(20261027)
        gates = {_gate(_simulated(n, 0.90, rng), n_resamples=300) for _ in range(100)}
        assert gates == {"top_up_overlap"}, n


def test_simulated_poor_raters_below_the_minimum_are_told_so_without_topping_up() -> None:
    # 100 sets of 60 shared traces from raters at alpha 0.30, 300 resamples each, seed 20261027.
    # Every one's upper bound is under the gate, so none is sent to double-code more.
    rng = random.Random(20261027)
    gates = Counter(_gate(_simulated(60, 0.30, rng), n_resamples=300) for _ in range(100))
    assert gates == {"codebook_broken": 90, "revise_codebook": 10}


def test_raters_agreeing_as_planned_are_usable_from_the_minimum_up() -> None:
    # The plug-in sets through the real gate, 2000 resamples, seed 0, at sizes from the minimum.
    for n in (MINIMUM_OVERLAP, 150, 200, 300):
        assert _gate(_units(plug_in_counts(n)), n_resamples=2000) == "usable", n


@pytest.mark.slow
def test_simulated_raters_at_the_minimum_clear_the_gate_in_most_but_not_all_sets() -> None:
    """A plug-in size plans for raters who agree as expected, not for every unlucky draw.

    60 simulated sets of 130 shared traces from raters at alpha 0.90, 500 resamples each,
    seed 7: 41 come out usable; the rest are told to double-code more.
    """
    rng = random.Random(7)
    usable = sum(_gate(_simulated(MINIMUM_OVERLAP, 0.90, rng)) == "usable" for _ in range(60))
    assert usable == 41
