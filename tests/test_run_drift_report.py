# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the input + judge drift reports.

Behavioural: input_drift reports PSI/JSD/Wasserstein with correct exceeded flags (zero on identity),
judge_drift reports ΔTPR/ΔTNR with finite CIs of the right sign, and empty inputs raise rather than
returning a spurious zero-drift.
"""

from __future__ import annotations

import pytest

from _helpers import TEST_STAMP, make_label
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.calibrate.errors import SampleMisuseError, StatisticsInputError
from calibrated_agent_evaluation.calibrate.record import (
    METHOD_NEWCOMBE_PAIRED,
    Bound,
    CalibrationRecord,
    LabelPair,
)
from calibrated_agent_evaluation.run.drift_report import (
    DEFAULT_WASSERSTEIN_THRESHOLD,
    DriftThresholds,
    input_drift,
    judge_drift,
)


def _record(judge_id: str, *, tpr: float, tnr: float) -> CalibrationRecord:
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
        youden_j=tpr + tnr - 1,
        result="CERTIFIED",
    )


def test_shifted_sample_exceeds_thresholds() -> None:
    """A shifted fresh sample is flagged on the gated metrics, each with its p-value."""
    baseline = [1.0] * 40 + [2.0] * 10
    fresh = [1.0] * 10 + [2.0] * 40
    report = input_drift(baseline, fresh, dimension="taxonomy_cell", n_permutations=199)

    names = {m.name for m in report.metrics}
    assert names == {"psi", "jsd", "wasserstein"}
    assert all(m.value >= 0.0 for m in report.metrics)
    # Every relabelling of this split is less extreme, so p is the floor 1 / (1 + 199).
    assert all(m.p_value == pytest.approx(1 / 200) for m in report.metrics)
    assert report.any_exceeded is True
    assert report.multiplicity is not None
    assert (report.multiplicity.family, report.multiplicity.method, report.multiplicity.m) == (
        "drift_flags",
        "fdr_by",
        2,
    )
    assert (report.n_permutations, report.seed, report.n_baseline, report.n_fresh) == (
        199,
        0,
        50,
        50,
    )


def test_identical_sample_yields_zero_no_flags() -> None:
    """An identical sample yields zero drift on all metrics and raises no flag."""
    sample = [float(x) for x in range(50)]
    report = input_drift(sample, sample, n_permutations=99)
    assert all(m.value == 0.0 for m in report.metrics)
    assert all(m.p_value == 1.0 for m in report.metrics)
    assert report.any_exceeded is False


def test_thresholds_are_configurable() -> None:
    """A high severity bar suppresses a significant flag; None leaves significance alone."""
    baseline = [1.0] * 35 + [2.0] * 15
    fresh = [1.0] * 15 + [2.0] * 35
    strict = input_drift(
        baseline,
        fresh,
        thresholds=DriftThresholds(psi=None, wasserstein=None),
        n_permutations=199,
    )
    lax = input_drift(
        baseline,
        fresh,
        thresholds=DriftThresholds(psi=99, wasserstein=99),
        n_permutations=199,
    )
    assert strict.any_exceeded is True
    assert lax.any_exceeded is False


def test_jsd_is_reported_but_never_flags() -> None:
    """JSD is reported with its p-value but sits outside the gated family and never flags."""
    baseline = [1.0] * 40 + [2.0] * 10
    fresh = [1.0] * 10 + [2.0] * 40
    report = input_drift(baseline, fresh, n_permutations=199)
    by_name = {m.name: m for m in report.metrics}
    jsd = by_name["jsd"]
    assert jsd.value > 0.0
    assert jsd.p_value == pytest.approx(1 / 200)
    assert (jsd.exceeded, jsd.threshold) == (False, None)
    assert by_name["psi"].exceeded is True
    assert report.multiplicity is not None
    assert report.multiplicity.m == 2
    assert len(report.multiplicity.reject) == 2


def test_wasserstein_flag_is_scale_invariant() -> None:
    """The same shift in different units flags the same way."""
    import numpy as np

    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 1.0, 60)
    fresh = rng.normal(0.1, 1.0, 60)
    unit = input_drift(base.tolist(), fresh.tolist(), n_permutations=199)
    scaled = input_drift((base * 1000).tolist(), (fresh * 1000).tolist(), n_permutations=199)
    u = {m.name: m for m in unit.metrics}["wasserstein"]
    k = {m.name: m for m in scaled.metrics}["wasserstein"]
    assert u.value == pytest.approx(k.value, rel=1e-9)
    assert u.p_value == pytest.approx(k.p_value)
    assert u.exceeded == k.exceeded
    # The scaled distance is unit-free, so one bar means the same thing in any unit.
    assert u.threshold == DEFAULT_WASSERSTEIN_THRESHOLD


def test_a_tiny_shift_on_a_huge_sample_is_significant_but_not_flagged() -> None:
    """A 0.03 SD shift at 20,000 traces per side is detectable and too small to act on."""
    import numpy as np

    rng = np.random.default_rng(1)
    base = rng.normal(size=20_000)
    fresh = rng.normal(loc=0.03, size=20_000)
    report = input_drift(base.tolist(), fresh.tolist(), seed=1, n_permutations=199)
    wass = {m.name: m for m in report.metrics}["wasserstein"]
    assert wass.p_value < 0.01
    assert DEFAULT_WASSERSTEIN_THRESHOLD is not None
    assert wass.value < DEFAULT_WASSERSTEIN_THRESHOLD
    assert wass.exceeded is False


def test_a_half_sd_shift_on_a_small_sample_is_flagged() -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    base = rng.normal(size=100)
    fresh = rng.normal(loc=0.5, size=100)
    report = input_drift(base.tolist(), fresh.tolist(), seed=1)
    wass = {m.name: m for m in report.metrics}["wasserstein"]
    assert DEFAULT_WASSERSTEIN_THRESHOLD is not None
    assert wass.value > DEFAULT_WASSERSTEIN_THRESHOLD
    assert wass.exceeded is True


def test_drift_below_the_bin_floor_is_refused() -> None:
    """Fewer than 5 * bins values per side is refused rather than read off empty bins."""
    with pytest.raises(StatisticsInputError, match="at least 50 values per side"):
        input_drift([1.0] * 20 + [2.0] * 5, [1.0] * 5 + [2.0] * 20)
    report = input_drift([1.0] * 20 + [2.0] * 5, [1.0] * 5 + [2.0] * 20, bins=5, n_permutations=99)
    assert len(report.metrics) == 3


def test_degraded_judge_delta_ci_sits_below_zero() -> None:
    """A judge whose TPR dropped yields a ΔTPR point < 0 with a finite CI (right sign)."""
    before = _record("j", tpr=0.95, tnr=0.95)
    # TPR degraded by 0.15, measured on a fresh hunted set (independent samples)
    after = replace(_record("j", tpr=0.80, tnr=0.95), hunted_set_id="hunted-j-2")
    report = judge_drift(before, after)

    assert report.delta_tpr.point is not None
    assert report.delta_tpr.point < 0  # TPR went down
    assert report.delta_tpr.lo <= report.delta_tpr.point <= report.delta_tpr.hi
    # TNR unchanged -> its delta straddles zero.
    assert report.delta_tnr.point == pytest.approx(0.0, abs=1e-9)


def test_unchanged_judge_deltas_are_zero() -> None:
    """Equal rates on independent samples yield ~zero deltas straddling 0."""
    before = _record("j", tpr=0.90, tnr=0.90)
    after = replace(before, calib_id="calib-j-2", hunted_set_id="hunted-j-2")
    report = judge_drift(before, after)
    assert report.delta_tpr.point == pytest.approx(0.0, abs=1e-9)
    assert report.delta_tnr.point == pytest.approx(0.0, abs=1e-9)
    assert report.delta_tpr.lo <= 0.0 <= report.delta_tpr.hi


def test_judge_drift_refuses_records_that_share_a_test_set() -> None:
    """Shared items make before/after paired, not independent.

    The independent two-sample interval is the wrong tool there, so the record-level drift refuses
    and names the paired path. Sharing is read off the hunted set id or the member trace ids.
    """
    rec = _record("j", tpr=0.90, tnr=0.90)
    with pytest.raises(SampleMisuseError, match="judge_drift_paired"):
        judge_drift(rec, replace(rec, calib_id="calib-j-2"))
    a = replace(rec, trace_ids=("t1", "t2", "t3"))
    b = replace(rec, calib_id="calib-j-2", hunted_set_id="hunted-j-2", trace_ids=("t3", "t4"))
    with pytest.raises(SampleMisuseError, match="1 shared trace"):
        judge_drift(a, b)


def _pairs(spec: list[tuple[str, str, str]]) -> list[LabelPair]:
    """(trace_id, gold, verdict) triples as locked-test pairs."""
    return [
        LabelPair(
            gold=make_label(trace_id=t, kind="human", outcome=g),
            verdict=make_label(trace_id=t, kind="llm-judge", outcome=v),
        )
        for t, g, v in spec
    ]


def _paired_runs() -> tuple[list[LabelPair], list[LabelPair]]:
    """80 true FAILs: 62 caught by both runs, 2 by after only, 10 by before only, 6 by neither.
    80 true PASSes, cleared by both runs throughout."""
    before, after = [], []
    for i in range(80):
        b_ok, a_ok = (i < 62 or 64 <= i < 74), (i < 64)
        before.append((f"p{i}", "FAIL", "FAIL" if b_ok else "PASS"))
        after.append((f"p{i}", "FAIL", "FAIL" if a_ok else "PASS"))
    for i in range(80):
        before.append((f"n{i}", "PASS", "PASS"))
        after.append((f"n{i}", "PASS", "PASS"))
    return _pairs(before), _pairs(after)


def test_judge_drift_paired_uses_the_newcombe_paired_interval() -> None:
    """The same 160 items judged before and after.

    TPR falls from 72/80 to 64/80. The paired interval is Newcombe's hybrid score on the table
    [[62, 2], [10, 6]]; it is narrower than the independent interval on the same margins because the
    two runs agree on most items.
    """
    from calibrated_agent_evaluation.run.drift_report import judge_drift_paired
    from calibrated_agent_evaluation.stats.drift import delta_proportion_ci
    from calibrated_agent_evaluation.stats.proportions import newcombe_paired_diff

    before, after = _paired_runs()
    report = judge_drift_paired(before, after, judge_id="j")
    _, lo, hi = newcombe_paired_diff(62, 2, 10, 6)
    assert report.delta_tpr.point == pytest.approx(-0.1, abs=1e-12)
    assert (report.delta_tpr.lo, report.delta_tpr.hi) == (lo, hi)
    assert report.delta_tpr.method == METHOD_NEWCOMBE_PAIRED
    assert report.delta_tpr.n == 80
    _, ind_lo, ind_hi = delta_proportion_ci(64, 80, 72, 80)
    assert hi - lo < ind_hi - ind_lo
    assert report.delta_tnr.point == pytest.approx(0.0, abs=1e-12)
    assert report.delta_tnr.lo <= 0.0 <= report.delta_tnr.hi


def test_judge_drift_paired_refuses_different_items() -> None:
    from calibrated_agent_evaluation.run.drift_report import judge_drift_paired

    before, after = _paired_runs()
    with pytest.raises(SampleMisuseError, match="same items"):
        judge_drift_paired(before, after[:-1], judge_id="j")
    flipped = _pairs([("p0", "PASS", "PASS")]) + after[1:]
    with pytest.raises(SampleMisuseError, match="gold"):
        judge_drift_paired(before, flipped, judge_id="j")


def test_empty_input_drift_raises() -> None:
    """input_drift over an empty baseline/sample raises, not a spurious zero."""
    with pytest.raises(StatisticsInputError):
        input_drift([], [1.0, 2.0])
    with pytest.raises(StatisticsInputError):
        input_drift([1.0, 2.0], [])
