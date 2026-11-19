# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Krippendorff's alpha, Fleiss' kappa, specific agreement and their unit bootstrap (stats/).

Reference cases are published worked examples with their published values:

* Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability", section A (binary, two
  observers, no missing data: alpha = 0.095) and section C (nominal, four observers, missing data:
  alpha = 0.743).
* Fleiss (1971), "Measuring nominal scale agreement among many raters", Table 1 (30 patients, six
  psychiatrists, five diagnoses: kappa = 0.430). The table is the transcription statsmodels ships
  in its own test suite; ``statsmodels.stats.inter_rater.fleiss_kappa`` is the dev-only oracle, the
  way sklearn is the oracle for ``cohen_kappa``.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from calibrated_agent_evaluation.calibrate.errors import StatisticsInputError
from calibrated_agent_evaluation.stats.agreement import (
    UndefinedAgreementError,
    bootstrap_agreement,
    fleiss_kappa,
    krippendorff_alpha,
    specific_agreement,
)

_ = None  # a value the rater did not code

# Krippendorff 2011, section A: observers A and B over ten units.
_K2011_BINARY = list(
    zip([0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [1, 1, 1, 0, 0, 1, 0, 0, 0, 0], strict=True)
)

# Krippendorff 2011, section C: observers A-D (columns) over twelve units (rows), with gaps.
_K2011_MISSING = [
    list(row)
    for row in zip(
        [1, 2, 3, 3, 2, 1, 4, 1, 2, _, _, _],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, _, 3],
        [_, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, _],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, _],
        strict=True,
    )
]

# Fleiss 1971, Table 1: the diagnosis each of six raters assigned to each of 30 patients
# (1 depression, 2 personality disorder, 3 schizophrenia, 4 neurosis, 5 other).
_FLEISS_1971 = [
    [4, 4, 4, 4, 4, 4],
    [2, 2, 2, 5, 5, 5],
    [2, 3, 3, 3, 3, 5],
    [5, 5, 5, 5, 5, 5],
    [2, 2, 2, 4, 4, 4],
    [1, 1, 3, 3, 3, 3],
    [3, 3, 3, 3, 5, 5],
    [1, 1, 3, 3, 3, 4],
    [1, 1, 4, 4, 4, 4],
    [5, 5, 5, 5, 5, 5],
    [1, 4, 4, 4, 4, 4],
    [1, 2, 4, 4, 4, 4],
    [2, 2, 2, 3, 3, 3],
    [1, 4, 4, 4, 4, 4],
    [2, 2, 4, 4, 4, 5],
    [3, 3, 3, 3, 3, 5],
    [1, 1, 1, 4, 5, 5],
    [1, 1, 1, 1, 1, 2],
    [2, 2, 4, 4, 4, 4],
    [1, 3, 3, 5, 5, 5],
    [5, 5, 5, 5, 5, 5],
    [2, 4, 4, 4, 4, 4],
    [2, 2, 4, 5, 5, 5],
    [1, 1, 4, 4, 4, 4],
    [1, 4, 4, 4, 4, 5],
    [2, 2, 2, 2, 2, 4],
    [1, 1, 1, 1, 5, 5],
    [2, 2, 4, 4, 4, 4],
    [1, 3, 3, 3, 3, 3],
    [5, 5, 5, 5, 5, 5],
]


# --- krippendorff_alpha -------------------------------------------------------------------


def test_alpha_matches_krippendorff_2011_binary_example() -> None:
    # Published: 0.095. By hand: n=20 pairable values, 14 zeros and 6 ones, 12 agreeing
    # coincidences, so alpha = 1 - 19 * 8 / (400 - 196 - 36) = 2/21.
    assert krippendorff_alpha(_K2011_BINARY) == pytest.approx(2 / 21)
    assert round(krippendorff_alpha(_K2011_BINARY), 3) == 0.095


def test_alpha_matches_krippendorff_2011_missing_data_example() -> None:
    # Published nominal alpha for the four-observer table with gaps: 0.743. Unit 12 has a single
    # value and is not pairable, so it must drop out rather than count as agreement.
    assert round(krippendorff_alpha(_K2011_MISSING), 3) == 0.743


def test_alpha_ignores_a_unit_only_one_rater_coded() -> None:
    # Adding unpairable units (one value each) must not move alpha at all.
    padded = [*_K2011_BINARY, (1, None), (None, 0), (None, None)]
    assert krippendorff_alpha(padded) == pytest.approx(krippendorff_alpha(_K2011_BINARY))


def test_alpha_is_one_under_perfect_agreement() -> None:
    assert krippendorff_alpha([("a", "a"), ("b", "b"), ("a", "a", "a"), ("c", None, "c")]) == 1.0


def test_alpha_is_negative_under_systematic_disagreement() -> None:
    # Two raters who always disagree on a balanced binary item do worse than chance.
    assert krippendorff_alpha([(0, 1), (1, 0), (0, 1), (1, 0)]) < 0.0


def test_alpha_refuses_a_single_observed_category() -> None:
    # Every pairable value is "PASS": expected disagreement is 0, alpha is 0/0. Reporting 1.0
    # would certify a codebook nobody has tested, so the statistic refuses.
    with pytest.raises(UndefinedAgreementError):
        krippendorff_alpha([("PASS", "PASS"), ("PASS", "PASS")])


def test_alpha_refuses_data_with_no_pairable_unit() -> None:
    # One rater, or two raters who never coded the same unit: nothing to agree on.
    with pytest.raises(UndefinedAgreementError):
        krippendorff_alpha([("a",), ("b",)])
    with pytest.raises(UndefinedAgreementError):
        krippendorff_alpha([("a", None), (None, "b")])


def test_alpha_refuses_empty_input() -> None:
    with pytest.raises(StatisticsInputError):
        krippendorff_alpha([])


def test_undefined_agreement_is_a_statistics_input_error() -> None:
    # Callers catching the existing stats error family also catch the undefined case.
    assert issubclass(UndefinedAgreementError, StatisticsInputError)


# --- fleiss_kappa -------------------------------------------------------------------------


def test_fleiss_kappa_matches_fleiss_1971_table_1() -> None:
    assert round(fleiss_kappa(_FLEISS_1971), 3) == 0.430


def test_fleiss_kappa_matches_statsmodels_oracle() -> None:
    inter_rater = pytest.importorskip("statsmodels.stats.inter_rater")
    np = pytest.importorskip("numpy")
    table, _cats = inter_rater.aggregate_raters(np.asarray(_FLEISS_1971))
    assert fleiss_kappa(_FLEISS_1971) == pytest.approx(float(inter_rater.fleiss_kappa(table)))


def test_fleiss_kappa_matches_statsmodels_on_string_labels() -> None:
    inter_rater = pytest.importorskip("statsmodels.stats.inter_rater")
    np = pytest.importorskip("numpy")
    units = [
        ["PASS", "PASS", "FAIL"],
        ["FAIL", "FAIL", "FAIL"],
        ["PASS", "PASS", "PASS"],
        ["PASS", "FAIL", "FAIL"],
        ["PASS", "PASS", "PASS"],
    ]
    table, _cats = inter_rater.aggregate_raters(np.asarray(units))
    assert fleiss_kappa(units) == pytest.approx(float(inter_rater.fleiss_kappa(table)))


def test_fleiss_kappa_is_one_under_perfect_agreement() -> None:
    assert fleiss_kappa([[1, 1, 1], [2, 2, 2], [3, 3, 3]]) == pytest.approx(1.0)


def test_fleiss_kappa_refuses_missing_ratings() -> None:
    # Fleiss assumes every unit has the same number of ratings; a gap is alpha's job, not a
    # silently shortened row.
    with pytest.raises(StatisticsInputError):
        fleiss_kappa([[1, 1, 2], [1, None, 2]])


def test_fleiss_kappa_refuses_unequal_rating_counts() -> None:
    with pytest.raises(StatisticsInputError):
        fleiss_kappa([[1, 1, 2], [1, 2]])


def test_fleiss_kappa_refuses_one_rater_per_unit() -> None:
    with pytest.raises(StatisticsInputError):
        fleiss_kappa([[1], [2]])


def test_fleiss_kappa_refuses_a_single_observed_category() -> None:
    with pytest.raises(UndefinedAgreementError):
        fleiss_kappa([["PASS", "PASS"], ["PASS", "PASS"]])


def test_fleiss_kappa_refuses_empty_input() -> None:
    with pytest.raises(StatisticsInputError):
        fleiss_kappa([])


_complete_units = st.integers(min_value=2, max_value=5).flatmap(
    lambda m: st.lists(
        st.lists(st.integers(min_value=0, max_value=2), min_size=m, max_size=m),
        min_size=2,
        max_size=25,
    )
)


@settings(max_examples=200)
@given(_complete_units)
def test_alpha_and_fleiss_kappa_satisfy_the_complete_data_identity(units: list[list[int]]) -> None:
    # On complete data with m ratings per unit over N units, alpha is Fleiss' kappa corrected for
    # sample size: alpha = 1 - (1 - kappa) * (N*m - 1) / (N*m). The two are computed by different
    # formulas here, so the identity is an independent check on both.
    if len({v for row in units for v in row}) < 2:
        with pytest.raises(UndefinedAgreementError):
            krippendorff_alpha(units)
        return
    total = len(units) * len(units[0])
    kappa = fleiss_kappa(units)
    assert krippendorff_alpha(units) == pytest.approx(1 - (1 - kappa) * (total - 1) / total)


# --- specific_agreement -------------------------------------------------------------------


def test_specific_agreement_is_positive_and_negative_agreement_for_two_raters() -> None:
    # Cicchetti and Feinstein (1990) p_pos = 2a / (2a + b + c), p_neg = 2d / (2d + b + c). The
    # binary example has a=1 (both 1), d=5 (both 0), b+c=4 discordant units.
    per_class = specific_agreement(_K2011_BINARY)
    assert per_class[1] == pytest.approx(2 * 1 / (2 * 1 + 4))
    assert per_class[0] == pytest.approx(2 * 5 / (2 * 5 + 4))


def test_specific_agreement_exposes_a_rare_class_the_headline_hides() -> None:
    # 18 agreed PASS and 2 units where only one rater saw FAIL: raw agreement is 90%, but the
    # raters never agreed on a single FAIL.
    units = [("PASS", "PASS")] * 18 + [("FAIL", "PASS"), ("PASS", "FAIL")]
    per_class = specific_agreement(units)
    assert per_class["FAIL"] == 0.0
    assert per_class["PASS"] == pytest.approx(36 / 38)


def test_specific_agreement_skips_unpairable_units() -> None:
    per_class = specific_agreement([("a", "a"), ("b", None), ("a", "b")])
    assert set(per_class) == {"a", "b"}
    assert per_class["a"] == pytest.approx(2 / 3)
    assert per_class["b"] == 0.0


def test_specific_agreement_refuses_data_with_no_pairable_unit() -> None:
    with pytest.raises(UndefinedAgreementError):
        specific_agreement([("a", None)])


# --- bootstrap_agreement ------------------------------------------------------------------


def test_bootstrap_brackets_the_point_estimate_and_reports_its_treatment() -> None:
    interval = bootstrap_agreement(
        _FLEISS_1971, krippendorff_alpha, n_resamples=400, confidence=0.95, seed=7
    )
    assert interval.point == pytest.approx(krippendorff_alpha(_FLEISS_1971))
    assert interval.lower < interval.point < interval.upper
    assert interval.confidence == 0.95
    assert interval.n_resamples == 400
    assert interval.n_units == 30
    assert interval.seed == 7


def test_bootstrap_is_reproducible_from_its_seed() -> None:
    first = bootstrap_agreement(_FLEISS_1971, fleiss_kappa, n_resamples=200, seed=3)
    again = bootstrap_agreement(_FLEISS_1971, fleiss_kappa, n_resamples=200, seed=3)
    other = bootstrap_agreement(_FLEISS_1971, fleiss_kappa, n_resamples=200, seed=4)
    assert (first.lower, first.upper) == (again.lower, again.upper)
    assert (first.lower, first.upper) != (other.lower, other.upper)


def test_bootstrap_interval_narrows_with_more_units() -> None:
    small = bootstrap_agreement(_K2011_BINARY * 2, krippendorff_alpha, n_resamples=400, seed=1)
    large = bootstrap_agreement(_K2011_BINARY * 20, krippendorff_alpha, n_resamples=400, seed=1)
    assert (large.upper - large.lower) < (small.upper - small.lower)


def test_bootstrap_counts_resamples_where_the_statistic_is_undefined() -> None:
    # A rare class: many resamples draw no FAIL at all, where alpha is 0/0. Those are counted,
    # not silently scored, so a reader can see how much of the distribution was undefined.
    units = [("PASS", "PASS")] * 12 + [("FAIL", "FAIL")]
    interval = bootstrap_agreement(units, krippendorff_alpha, n_resamples=300, seed=0)
    assert 0 < interval.n_undefined < 300
    assert interval.point == pytest.approx(1.0)


def test_bootstrap_refuses_when_the_point_estimate_is_undefined() -> None:
    with pytest.raises(UndefinedAgreementError):
        bootstrap_agreement([("PASS", "PASS")] * 5, krippendorff_alpha, n_resamples=50)


@pytest.mark.parametrize(
    ("n_resamples", "confidence"), [(0, 0.95), (-5, 0.95), (100, 0.0), (100, 1.0), (100, 1.5)]
)
def test_bootstrap_refuses_out_of_domain_settings(n_resamples: int, confidence: float) -> None:
    with pytest.raises(StatisticsInputError):
        bootstrap_agreement(
            _FLEISS_1971, krippendorff_alpha, n_resamples=n_resamples, confidence=confidence
        )
