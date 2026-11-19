# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trend report + production-frame guard.

Behavioural: the report emits one corrected-rate+CI TrendPoint per mode (no blended aggregate
scalar), keys crew modes per (agent, failure_type), keeps a refused-correction mode present with its
widening interval, and refuses a discovery-frame row from a headline rate.
"""

from __future__ import annotations

import pytest

from _helpers import TEST_STAMP, rate_ids_by_mode
from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord
from calibrated_agent_evaluation.run.report import ModeCounts, ModeKey, TrendPoint
from calibrated_agent_evaluation.run.trend import assert_production_rate_frame, trend_report
from calibrated_agent_evaluation.sampling import FrameContaminationError, FrameRow


def _record(judge_id: str, *, youden_j: float, tpr: float, tnr: float) -> CalibrationRecord:
    return CalibrationRecord(
        provenance=TEST_STAMP,
        calib_id=f"calib-{judge_id}",
        judge_id=judge_id,
        model="m",
        prompt_version="v",
        hunted_set_id=f"hunted-{judge_id}",
        n_pos=80,
        n_neg=80,
        tpr=Bound(point=tpr, lb95=tpr - 0.05, n=80),
        tnr=Bound(point=tnr, lb95=tnr - 0.05, n=80),
        youden_j=youden_j,
        result="CERTIFIED",
    )


def test_one_corrected_rate_point_per_mode_no_blended_scalar() -> None:
    """One corrected-rate+CI TrendPoint per mode, no aggregate scalar.

    The return is a list of TrendPoints -- there is no single blended accuracy number anywhere in
    the report (the "no aggregate accuracy" invariant is structural: the type cannot hold
    one).
    """
    a, b = ModeKey(failure_type="over_refund"), ModeKey(failure_type="dropped_fact")
    counts = {
        a: ModeCounts(mode=a, n_total=50, n_judged_fail=8),
        b: ModeCounts(mode=b, n_total=30, n_judged_fail=3),
    }
    calib = {
        a: _record("jA", youden_j=0.90, tpr=0.95, tnr=0.95),
        b: _record("jB", youden_j=0.90, tpr=0.95, tnr=0.95),
    }
    rate = {a: [1] * 32 + [0] * 168, b: [1] * 15 + [0] * 135}

    points = trend_report(counts, calib, rate, rate_trace_ids=rate_ids_by_mode(rate))

    assert len(points) == 2
    assert all(isinstance(p, TrendPoint) for p in points)
    for p in points:
        assert p.corrected_rate.point is not None  # strong judge -> a real corrected point
        assert 0.0 <= p.corrected_rate.lo <= p.corrected_rate.hi <= 1.0


def test_crew_modes_keyed_per_agent_each_own_judge() -> None:
    """Crew modes keyed per (agent, failure_type), each own-judge corrected."""
    intake = ModeKey(failure_type="dropped_fact", agent="intake")
    payout = ModeKey(failure_type="misread_amount", agent="payout")
    counts = {
        intake: ModeCounts(mode=intake, n_total=40, n_judged_fail=6),
        payout: ModeCounts(mode=payout, n_total=40, n_judged_fail=4),
    }
    calib = {
        intake: _record("intake-judge", youden_j=0.90, tpr=0.95, tnr=0.95),
        payout: _record("payout-judge", youden_j=0.90, tpr=0.95, tnr=0.95),
    }
    rate = {intake: [1] * 24 + [0] * 136, payout: [1] * 16 + [0] * 144}

    points = {
        p.mode: p for p in trend_report(counts, calib, rate, rate_trace_ids=rate_ids_by_mode(rate))
    }

    assert set(points) == {intake, payout}
    # Each mode is corrected by its OWN judge's calibration record.
    assert points[intake].calib_id == "calib-intake-judge"
    assert points[payout].calib_id == "calib-payout-judge"


def test_weak_judge_mode_kept_with_widening_interval_not_dropped() -> None:
    """A mode whose judge has J < 0.5 stays present with point=None + a refusal_reason."""
    mode = ModeKey(failure_type="dropped_fact")
    counts = {mode: ModeCounts(mode=mode, n_total=40, n_judged_fail=20)}
    calib = {mode: _record("weak", youden_j=0.40, tpr=0.70, tnr=0.70)}
    rate = {mode: [1] * 80 + [0] * 80}

    points = trend_report(counts, calib, rate, rate_trace_ids=rate_ids_by_mode(rate))

    assert len(points) == 1  # the mode is NOT dropped
    interval = points[0].corrected_rate
    assert interval.point is None  # NOT a fabricated point
    assert interval.refusal_reason is not None
    assert interval.lo <= interval.hi  # the widening interval is the honest signal


def test_mode_outside_identifiable_region_kept_with_reason() -> None:
    """A measured rate outside the RG identifiable region surfaces point=None + a reason.

    A strong judge (TPR=TNR=0.90 -> identifiable region [0.10, 0.90]) with a raw measured rate of
    0.95 (above TPR) is non-identifiable: the point is refused and the bound reported.
    """
    mode = ModeKey(failure_type="over_refund")
    counts = {mode: ModeCounts(mode=mode, n_total=40, n_judged_fail=38)}
    calib = {mode: _record("strong", youden_j=0.80, tpr=0.90, tnr=0.90)}
    rate = {mode: [1] * 152 + [0] * 8}  # raw 0.95, above TPR=0.90 -> non-identifiable

    points = trend_report(counts, calib, rate, rate_trace_ids=rate_ids_by_mode(rate))

    assert points[0].corrected_rate.point is None
    assert points[0].corrected_rate.refusal_reason == "non-identifiable"


def test_mode_without_calib_or_rate_sample_is_skipped() -> None:
    """A mode with no own-judge record or no rate sample is skipped."""
    a, b = ModeKey(failure_type="has_calib"), ModeKey(failure_type="no_calib")
    counts = {
        a: ModeCounts(mode=a, n_total=30, n_judged_fail=3),
        b: ModeCounts(mode=b, n_total=30, n_judged_fail=3),
    }
    calib = {a: _record("jA", youden_j=0.90, tpr=0.95, tnr=0.95)}  # b absent
    rate = {a: [1] * 15 + [0] * 135}  # b absent

    points = trend_report(counts, calib, rate, rate_trace_ids=rate_ids_by_mode(rate))

    assert {p.mode for p in points} == {a}


def test_discovery_frame_refused_from_production_rate() -> None:
    """A discovery-frame row in a headline rate raises contamination."""
    discovery = [FrameRow(cell={"feature": "refund"}, frame="discovery")]
    with pytest.raises(FrameContaminationError):
        assert_production_rate_frame(discovery)


def test_production_frame_passes() -> None:
    """A production-representative row passes the guard silently (no raise)."""
    prod = [FrameRow(cell={"feature": "refund"}, frame="production-representative")]
    assert_production_rate_frame(prod)  # returns None, raises nothing


def test_raw_rate_with_its_wilson_interval_sits_beside_the_corrected_rate() -> None:
    """The naive rate a reader would otherwise compute is reported, bounded."""
    from calibrated_agent_evaluation.stats.proportions import wilson_lower, wilson_upper

    a = ModeKey(failure_type="over_refund")
    counts = {a: ModeCounts(mode=a, n_total=200, n_judged_fail=32)}
    calib = {a: _record("jA", youden_j=0.90, tpr=0.95, tnr=0.95)}
    rate = {a: [1] * 32 + [0] * 168}

    (point,) = trend_report(
        counts, calib, rate, n_resamples=199, rate_trace_ids=rate_ids_by_mode(rate)
    )

    raw = point.raw_rate
    assert raw.point == pytest.approx(32 / 200)
    assert raw.lo == pytest.approx(wilson_lower(32, 200, alpha=0.05))
    assert raw.hi == pytest.approx(wilson_upper(32, 200, alpha=0.05))
    assert (raw.method, raw.level, raw.sided, raw.n) == ("wilson", 0.95, "two-sided", 200)
    assert raw.refusal_reason is None
    # The correction moves the number; that difference is what the pairing exists to show.
    assert point.corrected_rate.point is not None
    assert raw.point is not None
    assert point.corrected_rate.point < raw.point


def test_a_refused_correction_still_reports_the_raw_rate() -> None:
    """A weak judge refuses the corrected point; the raw rate is still shown, labelled as raw."""
    b = ModeKey(failure_type="dropped_fact")
    counts = {b: ModeCounts(mode=b, n_total=20, n_judged_fail=10)}
    calib = {b: _record("jB", youden_j=0.40, tpr=0.70, tnr=0.70)}
    rate = {b: [1] * 80 + [0] * 80}

    (point,) = trend_report(
        counts, calib, rate, n_resamples=199, rate_trace_ids=rate_ids_by_mode(rate)
    )

    assert point.corrected_rate.point is None
    assert point.raw_rate.point == pytest.approx(0.5)
    assert point.raw_rate.method == "wilson"
