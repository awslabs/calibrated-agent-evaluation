# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dual-coding inter-rater agreement over stored human labels.

The agreement set is the pre-adjudication labels of the named raters within one codebook version.
Alpha is the headline with a bootstrap interval, Cohen's kappa rides alongside for two raters,
per-class specific agreement is always reported, and the gold-set gate reads the interval's lower
bound.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from calibrated_agent_evaluation.review.agreement import (
    MINIMUM_OVERLAP,
    NO_FAILURE_MODE,
    alpha_gate,
    inter_rater_agreement,
)
from calibrated_agent_evaluation.review.errors import AgreementDataError
from calibrated_agent_evaluation.schema import Label
from calibrated_agent_evaluation.stats.agreement import (
    AgreementInterval,
    UndefinedAgreementError,
    cohen_kappa,
    krippendorff_alpha,
)

V1 = "codebook-v1"


def _label(trace_id: str, rater: str | None, outcome: str, **kw: Any) -> Label:
    return Label(
        trace_id=trace_id,
        span_id=kw.pop("span_id", None),
        kind=kw.pop("kind", "human"),
        outcome=outcome,  # type: ignore[arg-type]
        critique=kw.pop("critique", "why"),
        codebook_version=kw.pop("codebook_version", V1),
        rater_id=rater,
        **kw,
    )


def _dual_coded(a: list[str], b: list[str], **kw: Any) -> list[Label]:
    labels = [_label(f"T-{i}", "owner", o, **kw) for i, o in enumerate(a)]
    labels += [_label(f"T-{i}", "colleague", o, **kw) for i, o in enumerate(b)]
    return labels


_A = ["FAIL", "PASS", "FAIL", "FAIL", "FAIL", "FAIL", "FAIL", "FAIL", "PASS", "FAIL"]
_B = ["PASS", "PASS", "PASS", "FAIL", "FAIL", "PASS", "FAIL", "FAIL", "FAIL", "FAIL"]


def _interval(point: float, lower: float) -> AgreementInterval:
    return AgreementInterval(
        point=point,
        lower=lower,
        upper=1.0,
        confidence=0.95,
        n_units=100,
        n_resamples=2000,
        n_undefined=0,
        seed=0,
    )


# --- the report ---------------------------------------------------------------------------


def test_outcome_alpha_kappa_and_per_class_come_from_the_named_raters() -> None:
    report = inter_rater_agreement(
        _dual_coded(_A, _B), raters=("owner", "colleague"), codebook_version=V1, n_resamples=200
    )
    pairs = list(zip(_A, _B, strict=True))
    assert report.alpha.point == pytest.approx(krippendorff_alpha(pairs))
    assert report.alpha.lower <= report.alpha.point <= report.alpha.upper
    assert report.cohen_kappa is not None
    assert report.cohen_kappa.point == pytest.approx(cohen_kappa(_A, _B))
    # 5 units both called FAIL, 1 both PASS, 4 split: positive and negative specific agreement.
    assert report.per_class["FAIL"].point == pytest.approx(10 / 14)
    assert report.per_class["PASS"].point == pytest.approx(2 / 6)
    assert report.n_units == 10
    assert report.n_pairable_units == 10
    assert report.raters == ("owner", "colleague")
    assert report.codebook_version == V1
    assert report.axis == "outcome"


def test_adjudication_rows_and_other_raters_stay_out_of_the_number() -> None:
    # A third rater's adjudication of every split unit would push agreement up if pooled.
    labels = _dual_coded(_A, _B)
    labels += [_label(f"T-{i}", "adjudicator", _A[i]) for i in range(10) if _A[i] != _B[i]]
    labels += [_label("T-0", None, "FAIL")]  # unattributable: never part of an agreement set
    with_adjudication = inter_rater_agreement(
        labels, raters=("owner", "colleague"), codebook_version=V1, n_resamples=100
    )
    without = inter_rater_agreement(
        _dual_coded(_A, _B), raters=("owner", "colleague"), codebook_version=V1, n_resamples=100
    )
    assert with_adjudication.alpha == without.alpha


def test_labels_under_another_codebook_version_are_not_pooled() -> None:
    labels = _dual_coded(_A, _B) + _dual_coded(_A, _A, codebook_version="codebook-v2")
    v1 = inter_rater_agreement(
        labels, raters=("owner", "colleague"), codebook_version=V1, n_resamples=100
    )
    v2 = inter_rater_agreement(
        labels, raters=("owner", "colleague"), codebook_version="codebook-v2", n_resamples=100
    )
    assert v1.alpha.point == pytest.approx(krippendorff_alpha(list(zip(_A, _B, strict=True))))
    assert v2.alpha.point == pytest.approx(1.0)


def test_failure_mode_axis_treats_no_mode_as_its_own_category() -> None:
    # A PASS carries no failure mode; one rater saying PASS where the other names a mode is a
    # disagreement on the failure_mode axis, not a missing value.
    modes_a = ["refund", "refund", None, "tone", "tone", None]
    modes_b = ["refund", "tone", None, "tone", None, None]
    labels = [
        _label(f"T-{i}", "owner", "FAIL" if m else "PASS", failure_mode=m)
        for i, m in enumerate(modes_a)
    ]
    labels += [
        _label(f"T-{i}", "colleague", "FAIL" if m else "PASS", failure_mode=m)
        for i, m in enumerate(modes_b)
    ]
    report = inter_rater_agreement(
        labels,
        raters=("owner", "colleague"),
        codebook_version=V1,
        axis="failure_mode",
        n_resamples=100,
    )
    as_values = [
        (a or NO_FAILURE_MODE, b or NO_FAILURE_MODE) for a, b in zip(modes_a, modes_b, strict=True)
    ]
    assert report.alpha.point == pytest.approx(krippendorff_alpha(as_values))
    assert set(report.per_class) == {"refund", "tone", NO_FAILURE_MODE}


def test_incomplete_overlap_counts_only_pairable_units_and_three_raters_drop_cohen() -> None:
    labels = [
        _label("T-1", "r1", "FAIL"),
        _label("T-1", "r2", "FAIL"),
        _label("T-1", "r3", "PASS"),
        _label("T-2", "r1", "PASS"),
        _label("T-2", "r3", "PASS"),
        _label("T-3", "r2", "FAIL"),  # coded once: not pairable
        _label("T-4", "r1", "FAIL"),
        _label("T-4", "r2", "FAIL"),
    ]
    report = inter_rater_agreement(
        labels, raters=("r1", "r2", "r3"), codebook_version=V1, n_resamples=100
    )
    assert report.n_units == 4
    assert report.n_pairable_units == 3
    assert report.cohen_kappa is None  # Cohen's kappa is a two-rater statistic
    by_unit = [("FAIL", "FAIL", "PASS"), ("PASS", "PASS"), ("FAIL",), ("FAIL", "FAIL")]
    assert report.alpha.point == pytest.approx(krippendorff_alpha(by_unit))


def test_span_labels_are_separate_units_from_the_trace_label() -> None:
    labels = [
        _label("T-1", "owner", "FAIL"),
        _label("T-1", "colleague", "FAIL"),
        _label("T-1", "owner", "PASS", span_id="s1"),
        _label("T-1", "colleague", "FAIL", span_id="s1"),
        _label("T-2", "owner", "PASS"),
        _label("T-2", "colleague", "PASS"),
    ]
    report = inter_rater_agreement(
        labels, raters=("owner", "colleague"), codebook_version=V1, n_resamples=50
    )
    assert report.n_units == 3


# --- refusals -----------------------------------------------------------------------------


def test_refuses_fewer_than_two_raters() -> None:
    with pytest.raises(AgreementDataError):
        inter_rater_agreement(_dual_coded(_A, _B), raters=("owner",), codebook_version=V1)
    with pytest.raises(AgreementDataError):
        inter_rater_agreement(_dual_coded(_A, _B), raters=("owner", "owner"), codebook_version=V1)


def test_a_relabel_counts_once_with_the_raters_newest_label() -> None:
    base = _dual_coded(_A, _B)
    first = next(lb for lb in base if lb.trace_id == "T-0" and lb.rater_id == "owner")
    flipped = "PASS" if first.outcome == "FAIL" else "FAIL"
    relabeled = [*base, _label("T-0", "owner", flipped)]
    report = inter_rater_agreement(
        relabeled, raters=("owner", "colleague"), codebook_version=V1, n_resamples=100
    )
    assert report.n_superseded == 1
    assert report.n_pairable_units == len({lb.trace_id for lb in base})
    replaced = [lb for lb in base if lb is not first] + [_label("T-0", "owner", flipped)]
    expected = inter_rater_agreement(
        replaced, raters=("owner", "colleague"), codebook_version=V1, n_resamples=100
    )
    assert report.alpha == expected.alpha


def test_the_unversioned_bucket_is_its_own_version() -> None:
    unversioned = _dual_coded(_A, _B, codebook_version=None)
    report = inter_rater_agreement(
        [*unversioned, *_dual_coded(_A, _A)],
        raters=("owner", "colleague"),
        codebook_version=None,
        n_resamples=100,
    )
    assert report.codebook_version is None
    assert report.n_units == len({lb.trace_id for lb in unversioned})
    with pytest.raises(AgreementDataError, match="unversioned"):
        inter_rater_agreement(
            _dual_coded(_A, _B), raters=("owner", "colleague"), codebook_version=None
        )


def test_refuses_a_non_human_label_under_a_named_rater() -> None:
    labels = [*_dual_coded(_A, _B), _label("T-99", "colleague", "FAIL", kind="llm-judge")]
    with pytest.raises(AgreementDataError):
        inter_rater_agreement(labels, raters=("owner", "colleague"), codebook_version=V1)


def test_refuses_when_the_raters_share_no_unit() -> None:
    labels = [_label("T-1", "owner", "FAIL"), _label("T-2", "colleague", "PASS")]
    with pytest.raises(UndefinedAgreementError):
        inter_rater_agreement(labels, raters=("owner", "colleague"), codebook_version=V1)


def test_refuses_when_no_label_matches_the_codebook_version() -> None:
    with pytest.raises(AgreementDataError):
        inter_rater_agreement(
            _dual_coded(_A, _B), raters=("owner", "colleague"), codebook_version="codebook-v9"
        )


def test_refuses_a_single_observed_outcome() -> None:
    # Every unit both raters called FAIL: alpha is 0/0, not a certified 1.0.
    with pytest.raises(UndefinedAgreementError):
        inter_rater_agreement(
            _dual_coded(["FAIL"] * 5, ["FAIL"] * 5),
            raters=("owner", "colleague"),
            codebook_version=V1,
        )


# --- the gold-set gate ------------------------------------------------


@pytest.mark.parametrize(
    ("point", "lower", "verdict"),
    [
        (0.92, 0.80, "usable"),  # the lower bound, not the point, clears 0.80
        (0.95, 0.85, "usable"),
        (0.85, 0.79, "top_up_overlap"),  # point clears 0.80, bound does not
        (0.80, 0.70, "top_up_overlap"),
        (0.79, 0.60, "revise_codebook"),
        (0.667, 0.50, "revise_codebook"),
        (0.666, 0.50, "codebook_broken"),
        (0.10, -0.20, "codebook_broken"),
    ],
)
def test_gate_reads_the_lower_bound_and_the_revision_bands(
    point: float, lower: float, verdict: str
) -> None:
    assert alpha_gate(_interval(point, lower), n_pairable_units=MINIMUM_OVERLAP) == verdict


@pytest.mark.parametrize(
    ("point", "lower", "upper", "verdict"),
    [
        (1.0, 1.0, 1.0, "top_up_overlap"),  # perfect on a small overlap is not evidence
        (0.95, 0.85, 1.0, "top_up_overlap"),
        (0.60, 0.20, 0.85, "top_up_overlap"),  # a low point the interval cannot rule out yet
        (0.10, -0.20, 0.80, "top_up_overlap"),  # an upper bound at the gate is not under it
        (0.60, 0.30, 0.79, "revise_codebook"),
        (0.40, 0.10, 0.667, "revise_codebook"),  # at the floor, not under it
        (0.40, 0.10, 0.666, "codebook_broken"),
        (0.10, -0.20, 0.40, "codebook_broken"),
    ],
)
def test_below_the_minimum_overlap_only_the_upper_bound_can_rule_the_gold_out(
    point: float, lower: float, upper: float, verdict: str
) -> None:
    interval = dataclasses.replace(_interval(point, lower), upper=upper)
    gate = alpha_gate(interval, n_pairable_units=MINIMUM_OVERLAP - 1)
    assert gate == verdict


def test_report_carries_the_gate_on_its_alpha_and_overlap() -> None:
    perfect_but_small = inter_rater_agreement(
        _dual_coded(["FAIL", "PASS", "FAIL"], ["FAIL", "PASS", "FAIL"]),
        raters=("owner", "colleague"),
        codebook_version=V1,
        n_resamples=200,
    )
    assert perfect_but_small.alpha.lower == 1.0
    assert perfect_but_small.gate == "top_up_overlap"
    assert perfect_but_small.minimum_overlap == MINIMUM_OVERLAP
    noisy = inter_rater_agreement(
        _dual_coded(_A, _B),
        raters=("owner", "colleague"),
        codebook_version=V1,
        n_resamples=200,
        minimum_overlap=len(_A),
    )
    assert noisy.gate == "codebook_broken"
