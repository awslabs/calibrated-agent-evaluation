# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The committed coverage map stays true of the shipped estimator.

``fixtures/expected/coverage/rg_interval_coverage.json`` is the measured coverage of
``rg_interval`` over the planner-reachable grid. The slow tests re-measure a few named cells at the
map's full replicate count and seeds; a change to ``rg_interval``, the bootstrap method, the planner
or the numeric dependencies that moves coverage beyond Monte Carlo noise fails here, so the map
cannot silently go stale. A deliberate estimator fix re-runs the study
and recommits the map.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from calibrated_agent_evaluation.calibrate.errors import StatisticsInputError
from calibrated_agent_evaluation.stats.studies.coverage import (
    ACCEPTANCE_ARM,
    EVAL_SIZES,
    P_HATS,
    PREVALENCES,
    Cell,
    acceptance_band,
    asymmetric_grid,
    grid,
    simulate_cell,
    summarize_arm,
)

_REPO = Path(__file__).resolve().parent.parent
MAP_PATH = _REPO / "fixtures" / "expected" / "coverage" / "rg_interval_coverage.json"
ASYMMETRIC_MAP_PATH = MAP_PATH.with_name("rg_interval_coverage_asymmetric.json")

# Named cells: cheap (eval size 100) and chosen to pin each behaviour the map reports. Keys are
# (p_hat, prevalence, n_eval); n_class comes from the planner.
NAMED_CELLS: tuple[tuple[float, float, int], ...] = (
    (0.95, 0.02, 100),  # the bootstrap covered 0.806 at 20/class
    (0.90, 0.06, 100),  # near the documented operating point (90/90, 35/class, low prevalence)
    (0.85, 0.06, 100),  # nominal: the large calibration set the planner returns at p_hat=0.85
    (0.98, 0.50, 100),  # p_hat=0.98, planned at 130/class by the coverage floor (was 11)
)


def _map() -> dict:
    return json.loads(MAP_PATH.read_text())


def _asymmetric_map() -> dict:
    return json.loads(ASYMMETRIC_MAP_PATH.read_text())


def _holm_rejections(doc: dict) -> tuple[int, int]:
    """Exact two-sided binomial test of every cell's covered count against 0.95, Holm across the
    map: the number of cells shown off nominal, and the family size."""
    from scipy.stats import binomtest

    from calibrated_agent_evaluation.stats.multiplicity import multiplicity

    reps = doc["replicates"]
    pvals = [
        binomtest(c["arms"][ACCEPTANCE_ARM]["covered"], reps, doc["nominal"]).pvalue
        for c in doc["cells"]
    ]
    reject, m = multiplicity(pvals, method="holm")
    return sum(reject), m


def _cell_doc(doc: dict, p_hat: float, prevalence: float, n_eval: int) -> dict:
    return next(
        c
        for c in doc["cells"]
        if c["p_hat"] == p_hat and c["prevalence"] == prevalence and c["n_eval"] == n_eval
    )


def test_the_map_covers_exactly_the_decided_grid() -> None:
    doc = _map()
    got = {(c["p_hat"], c["n_class"], c["prevalence"], c["n_eval"]) for c in doc["cells"]}
    want = {(c.p_hat, c.n_class, c.prevalence, c.n_eval) for c in grid()}
    assert got == want
    assert len(want) == len(P_HATS) * len(PREVALENCES) * len(EVAL_SIZES) == 168
    assert doc["replicates"] == 2000
    assert doc["n_resamples"] == 9999


def test_the_acceptance_band_is_derived_from_the_replicate_count() -> None:
    lo, hi = acceptance_band(2000)
    assert (lo, hi) == (0.9405, 0.9595)
    assert _map()["acceptance_band"] == [lo, hi]
    # Fewer replicates, wider band: the tolerance is a property of the study size.
    lo_small, hi_small = acceptance_band(300)
    assert lo_small < lo
    assert hi_small > hi


def test_every_verdict_matches_its_own_coverage_interval() -> None:
    for c in _map()["cells"]:
        arm = c["arms"][ACCEPTANCE_ARM]
        lo, hi = arm["coverage_ci"]
        expected = "over" if lo > 0.95 else "under" if hi < 0.95 else "nominal"
        assert arm["verdict"] == expected, c
        assert arm["covered"] + arm["miss_above"] + arm["miss_below"] == arm["answered"]


def test_a_cell_is_reproducible_and_its_prefix_is_stable() -> None:
    """Replicate r always sees the same data, so 5 replicates are the first 5 of 10."""
    cell = Cell(p_hat=0.95, n_class=20, prevalence=0.06, n_eval=100)
    a = simulate_cell(cell, replicates=10, n_resamples=99, arms=("rg_interval",))
    b = simulate_cell(cell, replicates=10, n_resamples=99, arms=("rg_interval",))
    assert a["arms"] == b["arms"]
    head = simulate_cell(cell, replicates=5, n_resamples=99, arms=("rg_interval",))
    assert head["arms"]["rg_interval"]["covered"] <= a["arms"]["rg_interval"]["covered"]


def test_the_planners_p_hat_095_pick_no_longer_sits_above_the_truth() -> None:
    """p_hat=0.95, prevalence 0.02, 100 eval traces.

    At the planner's old 20 per class the joint percentile bootstrap covered 0.806 here and sat
    entirely above the truth in 372 of 2000 replicates, because a calibration class with no errors
    resampled to exactly no errors. The planner now asks 80 per class and the interval is the
    Jeffreys one. It must not undercover, and its misses above the truth must be no more than
    one tail's 2.5% allows: an exact one-sided binomial test does not reject at 0.05.
    """
    from scipy.stats import binomtest

    doc = _map()
    cell = _cell_doc(doc, 0.95, 0.02, 100)
    assert cell["n_class"] == 80
    arm = cell["arms"][ACCEPTANCE_ARM]
    assert arm["verdict"] != "under"
    tail = binomtest(arm["miss_above"], doc["replicates"], 0.025, alternative="greater")
    assert tail.pvalue >= 0.05


def test_no_cell_is_off_nominal_after_holm_across_the_map() -> None:
    """The done-when, read with the family's size in view.

    Each cell's verdict uses its own 95% interval, so an exactly nominal estimator still fails
    about one cell in twenty of 168. Testing every cell's covered count against 0.95 with an exact
    binomial test and correcting with Holm across the map, no planner-reachable cell is off
    nominal. The shipped bootstrap failed this in 97 cells.
    """
    assert _holm_rejections(_map()) == (0, 168)


# --- asymmetric judges ----------------------------------------


def test_the_asymmetric_map_covers_every_ordered_pair_at_the_planners_size() -> None:
    doc = _asymmetric_map()
    got = {(c["p_hat"], c["tnr"], c["n_class"], c["prevalence"], c["n_eval"]) for c in doc["cells"]}
    want = {(c.p_hat, c.tnr, c.n_class, c.prevalence, c.n_eval) for c in asymmetric_grid()}
    assert got == want
    assert len(want) == len(P_HATS) * (len(P_HATS) - 1) * len(PREVALENCES) * len(EVAL_SIZES)
    assert all(c["p_hat"] != c["tnr"] for c in doc["cells"])
    assert doc["replicates"] == 2000
    assert doc["n_resamples"] == 9999
    assert doc["acceptance_band"] == list(acceptance_band(2000))
    for c in doc["cells"]:
        arm = c["arms"][ACCEPTANCE_ARM]
        lo, hi = arm["coverage_ci"]
        assert arm["verdict"] == ("over" if lo > 0.95 else "under" if hi < 0.95 else "nominal")


def test_no_asymmetric_cell_is_off_nominal_after_holm_across_the_map() -> None:
    """The planner sizes an asymmetric judge with the certification size at the lower rate and
    the coverage floor at the higher. At those sizes, over 30 ordered (TPR, TNR) pairs, 7
    prevalences and 4 eval sizes, no cell is shown off 0.95 after Holm, under or over."""
    assert _holm_rejections(_asymmetric_map()) == (0, 840)


def test_a_symmetric_cell_keeps_its_key_and_seed_when_tnr_is_omitted() -> None:
    sym = Cell(p_hat=0.90, n_class=35, prevalence=0.06, n_eval=100)
    assert sym.key == "p0.90-n35-pi0.06-e100"
    assert sym.seed_key() == (900, 35, 60, 100)
    assert "tnr" not in sym.as_dict()
    asym = Cell(p_hat=0.90, n_class=35, prevalence=0.06, n_eval=100, tnr=0.95)
    assert asym.key != sym.key
    assert asym.seed_key() != sym.seed_key()


def test_an_unknown_arm_is_refused() -> None:
    cell = Cell(p_hat=0.9, n_class=35, prevalence=0.06, n_eval=100)
    with pytest.raises(StatisticsInputError, match="unknown arms"):
        simulate_cell(cell, replicates=1, n_resamples=9, arms=("bogus",))


# Named asymmetric cells, cheap at eval size 100: (TPR, TNR, prevalence).
NAMED_ASYMMETRIC_CELLS: tuple[tuple[float, float, float], ...] = (
    # The floor at 0.98 sets 130/class. Flooring at 0.90 (35) covered 0.966 here, 33 of its 34
    # misses below the truth (planner_asymmetric_floor_at_min.json).
    (0.90, 0.98, 0.02),
)


@pytest.mark.slow
@pytest.mark.parametrize(("tpr", "tnr", "prevalence"), NAMED_ASYMMETRIC_CELLS)
def test_named_asymmetric_cell_still_measures_what_the_map_says(
    tpr: float, tnr: float, prevalence: float
) -> None:
    doc = _asymmetric_map()
    want = next(
        c
        for c in doc["cells"]
        if (c["p_hat"], c["tnr"], c["prevalence"], c["n_eval"]) == (tpr, tnr, prevalence, 100)
    )
    cell = Cell(p_hat=tpr, n_class=want["n_class"], prevalence=prevalence, n_eval=100, tnr=tnr)
    row = simulate_cell(
        cell,
        replicates=doc["replicates"],
        n_resamples=doc["n_resamples"],
        base_seed=doc["base_seed"],
        arms=(ACCEPTANCE_ARM,),
    )
    got = summarize_arm(row["arms"][ACCEPTANCE_ARM], doc["replicates"])
    pinned = want["arms"][ACCEPTANCE_ARM]
    tolerance = 1.96 * math.sqrt(pinned["coverage"] * (1 - pinned["coverage"]) / doc["replicates"])
    assert abs(got["coverage"] - pinned["coverage"]) <= tolerance
    assert got["verdict"] == pinned["verdict"]


@pytest.mark.slow
@pytest.mark.parametrize(("p_hat", "prevalence", "n_eval"), NAMED_CELLS)
def test_named_cell_still_measures_what_the_map_says(
    p_hat: float, prevalence: float, n_eval: int
) -> None:
    doc = _map()
    want = _cell_doc(doc, p_hat, prevalence, n_eval)
    cell = Cell(p_hat=p_hat, n_class=want["n_class"], prevalence=prevalence, n_eval=n_eval)
    row = simulate_cell(
        cell,
        replicates=doc["replicates"],
        n_resamples=doc["n_resamples"],
        base_seed=doc["base_seed"],
        arms=(ACCEPTANCE_ARM,),
    )
    got = summarize_arm(row["arms"][ACCEPTANCE_ARM], doc["replicates"])
    pinned = want["arms"][ACCEPTANCE_ARM]
    # Same seeds and versions reproduce the count exactly; allow Monte Carlo noise so a numpy or
    # scipy bump that reshuffles the stream does not fail, but a real shift in coverage does.
    tolerance = 1.96 * math.sqrt(pinned["coverage"] * (1 - pinned["coverage"]) / doc["replicates"])
    assert abs(got["coverage"] - pinned["coverage"]) <= tolerance
    assert got["verdict"] == pinned["verdict"]
