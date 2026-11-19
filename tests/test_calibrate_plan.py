# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the calibration sample-size planner.

The planned n is re-derivable from the gate's own Wilson lower bound: at p_hat=0.90, tau=0.80 it is
55 per class, the smallest size from which a judge right 90% of the time (rounded down) clears the
bound at every larger size too. It was 35 when the plan was the first size that cleared with the
count rounded to nearest; that plan let more labels un-certify a judge as good as planned. Since
the planner also applies a coverage floor so the corrected rate's
interval holds at the planned size; the floor is read from measured coverage, and
``wilson_calibration_size`` keeps the certification-only number.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from calibrated_agent_evaluation.calibrate.errors import InfeasiblePlanError
from calibrated_agent_evaluation.calibrate.plan import (
    COVERAGE_FLOOR,
    CalibrationPlan,
    coverage_floor,
    double_coded_per_class,
    plan_calibration,
    plan_calibration_size,
    rate_safe_at,
    rate_safe_calibration_size,
    wilson_calibration_size,
)
from calibrated_agent_evaluation.stats.proportions import wilson_lower

# --- the exact minimum, re-derivable ------------------------------------


def test_plan_returns_exact_minimum_at_the_gate_threshold() -> None:
    # 55 is the smallest n from which every size clears 0.80 at p_hat=0.90, counts rounded down:
    #   wilson_lower(49, 55) ~= 0.8026 (clears), and every larger size clears too
    #   wilson_lower(48, 54) ~= 0.7992 (falls short)
    assert plan_calibration_size(0.90, 0.80) == wilson_calibration_size(0.90, 0.80) == 55
    assert wilson_lower(49, 55) >= 0.80
    assert wilson_lower(48, 54) < 0.80


@pytest.mark.parametrize("p_hat", [0.85, 0.88, 0.90, 0.92, 0.95, 0.98, 0.99, 1.0])
def test_more_labels_than_planned_never_un_certify_a_judge_as_good_as_planned(
    p_hat: float,
) -> None:
    """The bound is a sawtooth in n: at 0.90 it first clears at 48, then 51 to 54 fall short.
    The plan holds for every larger size, and the size below it falls short."""
    import math

    planned = wilson_calibration_size(p_hat, 0.80)
    for n in range(planned, planned + 600):
        assert wilson_lower(math.floor(p_hat * n + 1e-9), n) >= 0.80, n
    below = planned - 1
    assert wilson_lower(math.floor(p_hat * below + 1e-9), below) < 0.80


def test_the_first_size_that_clears_is_not_the_plan() -> None:
    assert wilson_lower(43, 48) >= 0.80  # 48 at 90% clears
    assert wilson_lower(45, 51) < 0.80  # 51 at 90% does not


def test_certification_size_falls_with_expected_accuracy() -> None:
    # A more accurate judge needs fewer labels to certify: p_hat=0.95 holds from n=28 (< 55).
    assert wilson_calibration_size(0.95, 0.80) == 28
    assert wilson_calibration_size(0.95, 0.80) < wilson_calibration_size(0.90, 0.80)


# --- the coverage floor --------------------------------------

_FLOOR_FILE = (
    Path(__file__).resolve().parent.parent / "fixtures/expected/coverage/planner_floor.json"
)


def test_the_floor_table_is_the_one_derived_from_measured_coverage() -> None:
    """The floor is not a picked constant: the table, the derived file and a fresh derivation
    from the committed scan all agree."""
    from calibrated_agent_evaluation.stats.studies.coverage import REPLICATES, acceptance_band
    from calibrated_agent_evaluation.stats.studies.planner_floor import accuracy_key, derive

    shipped = {accuracy_key(p): n for p, n in COVERAGE_FLOOR}
    interior = json.loads(_FLOOR_FILE.read_text())["floor"]
    boundary_file = json.loads((_FLOOR_FILE.parent / "planner_floor_boundary.json").read_text())
    boundary = boundary_file["floor"]
    # Rows up to 0.99 are the 1000-replicate interior derivation; the rows above 0.99 are the
    # 3000-replicate boundary derivation, which supersedes the interior scan's 1.00 row.
    assert shipped == {k: n for k, n in interior.items() if float(k) <= 0.99} | boundary
    assert all(float(k) > 0.99 for k in boundary)

    scan = json.loads((_FLOOR_FILE.parent / "planner_floor_scan.json").read_text())["rows"]
    table, _ = derive(scan, acceptance_band(REPLICATES))
    assert {accuracy_key(p): n for p, n in table.items()} == interior

    boundary_scan_file = _FLOOR_FILE.parent / "planner_floor_boundary_scan.json"
    boundary_scan = json.loads(boundary_scan_file.read_text())
    assert boundary_scan["replicates"] == 3000
    assert boundary_scan["seeds"] == [20261014, 20261015, 20261016]
    assert boundary_file["one_sided_above"] == 0.99
    table, _ = derive(boundary_scan["rows"], acceptance_band(REPLICATES), one_sided_above=0.99)
    assert {accuracy_key(p): n for p, n in table.items()} == boundary


def test_the_planner_raises_high_accuracy_plans_to_the_coverage_floor() -> None:
    """20 per class certifies a 0.95 judge; the corrected rate's interval needs 80.

    At 20 per class the interval covered 0.958 to 0.982 (over in 26 of 28 grid cells); the
    measured floor at 0.95 is 80. The floor flattens, and above 0.92 reverses, the steep fall of
    the certification size with expected accuracy.
    """
    assert plan_calibration_size(0.95, 0.80) == 80
    assert plan_calibration_size(0.98, 0.80) == 130
    assert plan_calibration_size(0.99, 0.80) == 162
    assert plan_calibration_size(0.92, 0.80) == 80


def test_the_floor_does_not_bind_where_certification_already_needs_more() -> None:
    assert plan_calibration_size(0.85, 0.80) == wilson_calibration_size(0.85, 0.80) == 208
    assert plan_calibration_size(0.88, 0.80) == wilson_calibration_size(0.88, 0.80) == 86
    assert plan_calibration_size(0.90, 0.80) == wilson_calibration_size(0.90, 0.80) == 55


def test_between_measured_accuracies_the_floor_takes_the_larger_neighbour() -> None:
    # 0.91 sits between 0.90 (floor 35) and 0.92 (floor 80); 0.96 between 0.95 (80) and 0.98 (130).
    assert coverage_floor(0.91) == 80
    assert coverage_floor(0.96) == 130
    assert coverage_floor(0.90) == 35
    # Below the lowest measured accuracy the lowest measured floor applies.
    assert coverage_floor(0.70) == coverage_floor(0.85) == 28


def test_the_boundary_accuracies_have_a_measured_floor() -> None:
    """With the boundary rule the interval covers at least nominally from 11 per class at 0.995,
    0.999 and 1.0, so a plan there is the certification size."""
    assert coverage_floor(0.995) == coverage_floor(0.999) == coverage_floor(1.0) == 11
    assert plan_calibration_size(1.0, 0.80) == wilson_calibration_size(1.0, 0.80) == 11
    assert plan_calibration_size(0.995, 0.80) == max(11, wilson_calibration_size(0.995, 0.80))


def test_between_0_99_and_0_995_the_floor_is_the_larger_neighbour() -> None:
    assert coverage_floor(0.993) == 162
    assert coverage_floor(0.997) == 11


def test_the_planner_refuses_where_no_measured_size_covers_nominally(monkeypatch) -> None:
    """An accuracy whose table entry is None is refused, not guessed."""
    from calibrated_agent_evaluation.calibrate import plan as plan_module

    table = dict(COVERAGE_FLOOR) | {1.00: None}
    monkeypatch.setattr(plan_module, "COVERAGE_FLOOR", tuple(table.items()))
    with pytest.raises(InfeasiblePlanError, match="coverage"):
        plan_calibration_size(1.0, 0.80)
    with pytest.raises(InfeasiblePlanError, match="coverage"):
        plan_calibration_size(0.9995, 0.80)
    # The certification-only number is still available for a caller who wants just that.
    assert wilson_calibration_size(1.0, 0.80) == 11


# --- asymmetric judges (TPR != TNR) -------------------------


@pytest.mark.parametrize("p_hat", [0.85, 0.88, 0.90, 0.91, 0.92, 0.95, 0.96, 0.98, 0.99])
def test_a_symmetric_caller_gets_the_same_plan_with_or_without_tnr_hat(p_hat: float) -> None:
    assert plan_calibration_size(p_hat, 0.80, tnr_hat=p_hat) == plan_calibration_size(p_hat, 0.80)


# (TPR, TNR) -> planned per-class size. The certification size is taken at the lower rate and the
# coverage floor at the higher, so the plan is max(Wilson at min, floor at max). Each pair is a
# cell of the asymmetric coverage map (fixtures/expected/coverage/
# rg_interval_coverage_asymmetric.json), measured nominal at these sizes after Holm.
ASYMMETRIC_PLANS: tuple[tuple[float, float, int], ...] = (
    (0.90, 0.95, 80),  # floor at 0.95 (80) binds; certifying 0.90 needs only 55
    (0.95, 0.90, 80),  # the same plan with the classes swapped
    (0.90, 0.98, 130),  # floor at 0.98 binds; at the floor for 0.90 (35) the interval overcovers
    (0.88, 0.95, 86),  # certifying 0.88 (86) binds over the floor at 0.95 (80)
    (0.88, 0.90, 86),  # certifying 0.88 binds over the floor at 0.90 (35)
    (0.98, 0.85, 208),  # certifying 0.85 binds over every floor
)


@pytest.mark.parametrize(("tpr", "tnr", "planned"), ASYMMETRIC_PLANS)
def test_an_asymmetric_plan_certifies_the_weaker_rate_and_floors_at_the_stronger(
    tpr: float, tnr: float, planned: int
) -> None:
    got = plan_calibration_size(tpr, 0.80, tnr_hat=tnr)
    assert got == planned
    assert got == max(wilson_calibration_size(min(tpr, tnr), 0.80), coverage_floor(max(tpr, tnr)))


def _holm_off_nominal(path: Path) -> tuple[int, int, int]:
    """Cells off 0.95 after Holm across the file (exact two-sided binomial), split by direction."""
    from scipy.stats import binomtest

    from calibrated_agent_evaluation.stats.multiplicity import multiplicity

    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    covered = [
        r["replicates"] - r["arms"]["jeffreys"]["miss_above"] - r["arms"]["jeffreys"]["miss_below"]
        for r in rows
    ]
    pvals = [binomtest(k, r["replicates"], 0.95).pvalue for k, r in zip(covered, rows, strict=True)]
    reject, m = multiplicity(pvals, method="holm")
    hits = [k / r["replicates"] for k, r, x in zip(covered, rows, reject, strict=True) if x]
    return m, sum(c < 0.95 for c in hits), sum(c > 0.95 for c in hits)


def test_an_asymmetric_plan_floors_at_the_stronger_rate_not_the_weaker() -> None:
    """Flooring at the weaker rate would plan 55 per class for a 0.90/0.98 judge. Over the 30
    ordered pairs of the asymmetric grid (1000 replicates per cell), that rule leaves 5 of 840
    cells off nominal after Holm, 4 of them over-covering where the stronger class made almost no
    errors; the floor at the stronger rate, the shipped rule, leaves none."""
    assert plan_calibration_size(0.90, 0.80, tnr_hat=0.98) == 130
    assert coverage_floor(0.90) == 35
    shipped = _FLOOR_FILE.with_name("planner_asymmetric_floor_at_max.json")
    rejected = _FLOOR_FILE.with_name("planner_asymmetric_floor_at_min.json")
    assert _holm_off_nominal(shipped) == (840, 0, 0)
    assert _holm_off_nominal(rejected) == (840, 1, 4)
    planned = {
        (r["p_hat"], r["tnr"]): r["n_class"] for r in json.loads(shipped.read_text())["rows"]
    }
    assert all(plan_calibration_size(t, 0.80, tnr_hat=n) == s for (t, n), s in planned.items())


def test_an_asymmetric_plan_refuses_when_the_weaker_rate_cannot_certify() -> None:
    # The gate certifies both rates, so a TNR the threshold already meets is refused however
    # accurate the TPR, and the other way round.
    with pytest.raises(InfeasiblePlanError):
        plan_calibration_size(0.98, 0.80, tnr_hat=0.80)
    with pytest.raises(InfeasiblePlanError):
        plan_calibration_size(0.75, 0.80, tnr_hat=0.95)


def test_an_asymmetric_plan_reads_the_floor_at_the_stronger_rate() -> None:
    assert plan_calibration(0.90, 0.80, tnr_hat=0.995) == CalibrationPlan(55, 55, 11)


def test_an_asymmetric_plan_refuses_when_the_stronger_rate_has_no_floor(monkeypatch) -> None:
    from calibrated_agent_evaluation.calibrate import plan as plan_module

    table = dict(COVERAGE_FLOOR) | {0.995: None}
    monkeypatch.setattr(plan_module, "COVERAGE_FLOOR", tuple(table.items()))
    with pytest.raises(InfeasiblePlanError, match="coverage"):
        plan_calibration_size(0.90, 0.80, tnr_hat=0.995)


def test_an_asymmetric_plan_refuses_an_out_of_range_tnr() -> None:
    with pytest.raises(InfeasiblePlanError, match="tnr_hat"):
        plan_calibration_size(0.90, 0.80, tnr_hat=1.5)
    with pytest.raises(InfeasiblePlanError, match="tnr_hat"):
        plan_calibration_size(0.90, 0.80, tnr_hat=0.0)


# --- infeasible target refused ------------------------------------------


def test_plan_refuses_when_threshold_meets_or_exceeds_expected_accuracy() -> None:
    # Negative path: tau >= p_hat means the point estimate never clears the bar, so its lower bound
    # cannot either — refused rather than searching unboundedly.
    with pytest.raises(InfeasiblePlanError):
        plan_calibration_size(0.80, 0.80)  # tau == p_hat
    with pytest.raises(InfeasiblePlanError):
        plan_calibration_size(0.75, 0.80)  # tau > p_hat


def test_plan_rejects_out_of_range_inputs() -> None:
    with pytest.raises(InfeasiblePlanError):
        plan_calibration_size(1.5, 0.80)
    with pytest.raises(InfeasiblePlanError):
        plan_calibration_size(0.90, 1.0)


# --- planning for a corrected rate ---------------------------------------------


@pytest.mark.parametrize("p_hat", [0.88, 0.90, 0.92, 0.95, 0.98, 0.99])
def test_a_rate_safe_plan_is_162_per_class_at_every_measured_accuracy(p_hat: float) -> None:
    """Any certifying accuracy between 0.98 and 0.99 needs 162, and below 162 per class some count
    of errors always lands there. At 0.85 certifying alone needs more (208)."""
    assert rate_safe_calibration_size(p_hat, 0.80) == 162
    plan = plan_calibration(p_hat, 0.80, rate_safe=True)
    assert plan.per_class == 162
    # The parts are the certification-only plan's, unchanged.
    certification_only = plan_calibration(p_hat, 0.80)
    assert (plan.certification, plan.coverage_floor) == (
        certification_only.certification,
        certification_only.coverage_floor,
    )


def test_certification_only_plans_keep_their_numbers() -> None:
    assert plan_calibration(0.90, 0.80) == CalibrationPlan(55, 55, 35)
    assert plan_calibration(0.95, 0.80) == CalibrationPlan(80, 28, 80)
    assert plan_calibration(0.90, 0.80, rate_safe=True) == CalibrationPlan(162, 55, 35)
    assert rate_safe_calibration_size(0.85, 0.80) == 208
    assert [plan_calibration_size(p, 0.80) for p in (0.85, 0.88, 0.90, 0.92, 0.95, 0.98)] == [
        208,
        86,
        55,
        80,
        80,
        130,
    ]


def test_a_rate_safe_plan_never_falls_below_the_certification_size() -> None:
    # At 0.84 certifying alone needs 314 per class, above every measured floor.
    assert rate_safe_calibration_size(0.84, 0.80) == 314
    assert rate_safe_calibration_size(0.95, 0.80, tnr_hat=0.85) == 208
    # A stricter gate needs more to certify 0.90 (172), and still reaches the 0.98-0.99 band.
    assert rate_safe_calibration_size(0.90, 0.85) == wilson_calibration_size(0.90, 0.85) == 172
    assert rate_safe_calibration_size(0.95, 0.90) == 162


def test_rate_safety_at_a_size_names_the_accuracies_that_break_it() -> None:
    # 80 per class: 79 of 80 (0.9875) certifies and needs 162.
    assert not rate_safe_at(80, 0.80)
    # 161 per class: 159 of 161 (0.9876) needs 162.
    assert not rate_safe_at(161, 0.80)
    assert rate_safe_at(162, 0.80)
    assert rate_safe_at(200, 0.80)


def test_the_rate_safe_size_is_derived_from_the_floor_table(monkeypatch) -> None:
    from calibrated_agent_evaluation.calibrate import plan as plan_module

    table = dict(COVERAGE_FLOOR) | {0.99: 200}
    monkeypatch.setattr(plan_module, "COVERAGE_FLOOR", tuple(table.items()))
    assert rate_safe_calibration_size(0.90, 0.80) == 200


def test_a_rate_safe_plan_refuses_where_no_floor_was_measured(monkeypatch) -> None:
    from calibrated_agent_evaluation.calibrate import plan as plan_module

    table = dict(COVERAGE_FLOOR) | {0.995: None, 0.999: None, 1.00: None}
    monkeypatch.setattr(plan_module, "COVERAGE_FLOOR", tuple(table.items()))
    with pytest.raises(InfeasiblePlanError, match="coverage"):
        rate_safe_calibration_size(0.995, 0.80)
    with pytest.raises(InfeasiblePlanError, match="coverage"):
        plan_calibration(0.90, 0.80, tnr_hat=0.995, rate_safe=True)


def test_a_rate_safe_plan_refuses_an_unreachable_threshold() -> None:
    with pytest.raises(InfeasiblePlanError):
        rate_safe_calibration_size(0.80, 0.80)


def test_a_rate_safe_plan_at_the_boundary_is_the_largest_floor() -> None:
    """A judge planned at 0.995 can still show 0.98 to 0.99 at a small size, whose floor is 162."""
    assert rate_safe_calibration_size(0.995, 0.80) == 162
    assert plan_calibration(0.90, 0.80, tnr_hat=0.995, rate_safe=True).per_class == 162


# --- double-coded gold --------------------------------------------------


def test_double_coded_gold_is_planned_at_the_agreement_overlap() -> None:
    from calibrated_agent_evaluation.review.gate import MINIMUM_OVERLAP

    assert double_coded_per_class() == 65 == math.ceil(MINIMUM_OVERLAP / 2)
    assert plan_calibration(0.90, 0.80, raters=2) == CalibrationPlan(65, 55, 35, 65)
    assert plan_calibration(0.90, 0.80, raters=3) == CalibrationPlan(65, 55, 35, 65)
    assert plan_calibration(0.90, 0.80, raters=1) == CalibrationPlan(55, 55, 35, None)
    assert plan_calibration_size(0.90, 0.80, raters=2) == 65
    assert plan_calibration_size(0.90, 0.80) == 55


def test_double_coding_never_lowers_a_larger_plan() -> None:
    assert plan_calibration_size(0.95, 0.80, raters=2) == 80
    assert plan_calibration_size(0.85, 0.80, raters=2) == 208
    rate_safe = plan_calibration(0.90, 0.80, rate_safe=True, raters=2)
    assert rate_safe == CalibrationPlan(162, 55, 35, 65)


def test_double_coded_size_follows_the_overlap_it_is_given() -> None:
    assert double_coded_per_class(131) == 66
    assert double_coded_per_class(1) == 1
    with pytest.raises(InfeasiblePlanError, match="minimum_overlap"):
        double_coded_per_class(0)


@pytest.mark.parametrize("raters", [0, -1])
def test_fewer_than_one_rater_is_refused(raters: int) -> None:
    with pytest.raises(InfeasiblePlanError, match="raters must be at least 1"):
        plan_calibration(0.90, 0.80, raters=raters)


def test_the_calibrate_help_states_the_sizes_the_planner_computes(capsys) -> None:
    from calibrated_agent_evaluation.cli import main
    from calibrated_agent_evaluation.review.gate import MINIMUM_OVERLAP

    with pytest.raises(SystemExit):
        main(["calibrate", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert (
        f"at least {double_coded_per_class()} per class ({MINIMUM_OVERLAP} shared traces)" in text
    )
