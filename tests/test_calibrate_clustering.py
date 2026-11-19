# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clustered evidence is counted at its effective size, and a contradicted judge is refused.

Spans of one trace share what made the trace easy or hard to judge, so a span-level locked test
of 10 traces by 8 spans is not 80 independent observations. A clustered rate sample with no FAIL
used to get design effect 1, the most optimistic value, so 0 FAILs in 1500 spans of 150 traces
read as 1500 independent clean items. And a rate sample far below the judge's false-positive
floor clipped to a zero-width [0, 0] interval, a claim that the true rate is exactly 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from _helpers import make_label, rate_ids
from calibrated_agent_evaluation.calibrate import (
    ErrorRatesDoNotTransferError,
    InMemoryLockedTestLedger,
    SampleMisuseError,
)
from calibrated_agent_evaluation.calibrate.engine import (
    MIN_TRACES_FOR_DESIGN_EFFECT,
    certify,
    corrected_rate,
)
from calibrated_agent_evaluation.calibrate.record import (
    METHOD_WILSON_CLUSTERED,
    CalibrationSet,
    LabelPair,
)
from calibrated_agent_evaluation.stats.rogan_gladen import cluster_design_effect, rg_interval


@dataclass(frozen=True)
class _Judge:
    judge_id: str = "span-judge"
    model: str = "m"
    prompt_version: str = "v1"


def _span_pairs(
    traces: int, spans: int, *, wrong: frozenset[int] = frozenset(), missed: int | None = None
) -> tuple[LabelPair, ...]:
    """Both classes from ``traces`` traces of ``spans`` spans.

    The judge misses the first ``missed`` spans (all of them by default) of each trace in ``wrong``.
    """
    out: list[LabelPair] = []
    for cls, other in (("FAIL", "PASS"), ("PASS", "FAIL")):
        for t in range(traces):
            for s in range(spans):
                tid, sid = f"{cls}-t{t}", f"s{s}"
                verdict = other if t in wrong and s < (spans if missed is None else missed) else cls
                out.append(
                    LabelPair(
                        gold=make_label(trace_id=tid, span_id=sid, kind="human", outcome=cls),
                        verdict=make_label(
                            trace_id=tid, span_id=sid, kind="llm-judge", outcome=verdict
                        ),
                    )
                )
    return tuple(out)


def _flat_pairs(n: int) -> tuple[LabelPair, ...]:
    out: list[LabelPair] = []
    for cls in ("FAIL", "PASS"):
        for i in range(n):
            tid = f"{cls}-{i}"
            out.append(
                LabelPair(
                    gold=make_label(trace_id=tid, kind="human", outcome=cls),
                    verdict=make_label(trace_id=tid, kind="llm-judge", outcome=cls),
                )
            )
    return tuple(out)


def test_eighty_spans_of_ten_traces_do_not_certify_like_eighty_traces() -> None:
    flat = certify(
        _Judge(),
        CalibrationSet(_flat_pairs(80), "locked_test", "flat"),
        ledger=InMemoryLockedTestLedger(),
    )
    assert flat.result == "CERTIFIED"
    assert flat.tpr_design_effect == 1.0

    spans = certify(
        _Judge(),
        CalibrationSet(_span_pairs(10, 8), "locked_test", "spans"),
        ledger=InMemoryLockedTestLedger(),
    )
    assert spans.result == "REFUSED"
    assert spans.n_pos == 80
    assert spans.tpr_design_effect == 8.0  # fewer traces than the minimum: each trace one draw
    assert spans.tpr.method == METHOD_WILSON_CLUSTERED
    assert spans.tpr.lb95 < 0.80 < flat.tpr.lb95


def test_enough_traces_estimate_the_design_effect_from_the_spans() -> None:
    traces = MIN_TRACES_FOR_DESIGN_EFFECT + 20
    record = certify(
        _Judge(),
        CalibrationSet(
            _span_pairs(traces, 3, wrong=frozenset(range(6)), missed=1), "locked_test", "est"
        ),
        ledger=InMemoryLockedTestLedger(),
    )
    # Six traces each missed on one of three spans: errors spread over traces, so the estimated
    # design effect stays well below the worst case of one draw per trace.
    assert 1.0 < record.tpr_design_effect < 2.0
    assert record.effective_n_pos == pytest.approx(record.n_pos / record.tpr_design_effect)


def test_clustered_gold_is_compared_to_the_coverage_floor_at_its_effective_size() -> None:
    record = certify(
        _Judge(),
        CalibrationSet(_span_pairs(60, 2, wrong=frozenset(range(3))), "locked_test", "floor"),
        ledger=InMemoryLockedTestLedger(),
    )
    assert record.result == "CERTIFIED"
    assert record.coverage_floor is not None
    assert record.n_pos >= record.coverage_floor > record.effective_n_pos
    assert record.below_coverage_floor


def test_a_clustered_sample_with_no_fail_counts_each_trace_once() -> None:
    ids = [f"t{i // 10}" for i in range(1500)]
    assert cluster_design_effect([0] * 1500, ids) == pytest.approx(10.0)
    assert cluster_design_effect([1] * 1500, ids) == pytest.approx(10.0)

    cal_pos, cal_neg = [1] * 52 + [0] * 3, [0] * 55
    spans = rg_interval(cal_pos, cal_neg, [0] * 1500, eval_cluster_ids=ids, seed=4)
    traces = rg_interval(cal_pos, cal_neg, [0] * 150, seed=4)
    assert spans.eval_design_effect == pytest.approx(10.0)
    assert spans.hi == pytest.approx(traces.hi)
    assert spans.hi > 0.015  # was 0.0018 with design effect 1, ten times too tight


def test_a_sample_below_the_false_positive_floor_is_refused_not_reported_as_zero() -> None:
    cal_pos, cal_neg = [1] * 52 + [0] * 3, [1] * 2 + [0] * 53
    with pytest.raises(ErrorRatesDoNotTransferError, match="false-positive floor"):
        rg_interval(cal_pos, cal_neg, [0] * 150, seed=0)


def test_the_corrected_rate_refuses_the_interval_when_error_rates_do_not_transfer() -> None:
    pairs = _flat_pairs(80)
    # 4 false positives in 80 true passes: the judge's false-positive floor is about 5%.
    noisy = tuple(
        LabelPair(
            gold=p.gold,
            verdict=make_label(trace_id=p.gold.trace_id, kind="llm-judge", outcome="FAIL"),
        )
        if p.gold.outcome == "PASS" and p.gold.trace_id in {f"PASS-{i}" for i in range(4)}
        else p
        for p in pairs
    )
    record = certify(
        _Judge(), CalibrationSet(noisy, "locked_test", "noisy"), ledger=InMemoryLockedTestLedger()
    )
    assert record.result == "CERTIFIED"
    scored = corrected_rate(
        record, [0] * 400, rate_set_id="prod", n_resamples=999, rate_trace_ids=rate_ids([0] * 400)
    )
    interval = scored.corrected_rate
    assert interval is not None
    assert (interval.lo, interval.hi) == (0.0, 1.0)
    assert interval.refusal_reason is not None
    assert "judge error rates don't transfer" in interval.refusal_reason
    with pytest.raises(ErrorRatesDoNotTransferError):
        interval.require_point()


def test_a_rate_sample_reusing_calibration_traces_is_refused() -> None:
    record = certify(
        _Judge(),
        CalibrationSet(_flat_pairs(80), "locked_test", "cal"),
        ledger=InMemoryLockedTestLedger(),
    )
    rate_traces = [f"prod-{i}" for i in range(148)] + ["FAIL-0", "FAIL-1"]
    verdicts = [0] * 140 + [1] * 10
    with pytest.raises(SampleMisuseError, match="2 trace\\(s\\) of the rate sample") as excinfo:
        corrected_rate(record, verdicts, rate_set_id="prod", rate_trace_ids=rate_traces)
    assert "FAIL-0" in str(excinfo.value)
    with pytest.raises(SampleMisuseError, match="FAIL-1"):
        corrected_rate(
            record,
            verdicts,
            rate_set_id="prod",
            rate_cluster_ids=rate_traces,
            rate_trace_ids=rate_ids(verdicts),
        )


def test_the_rate_sample_trace_ids_are_required() -> None:
    record = certify(
        _Judge(),
        CalibrationSet(_flat_pairs(80), "locked_test", "cal"),
        ledger=InMemoryLockedTestLedger(),
    )
    verdicts = [0] * 140 + [1] * 10
    with pytest.raises(TypeError, match="rate_trace_ids"):
        corrected_rate(record, verdicts, rate_set_id="prod")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "ids",
    [
        [f"prod-{i}" for i in range(149)],
        [f"prod-{i}" for i in range(151)],
        [*(f"prod-{i}" for i in range(149)), ""],
        [],
    ],
)
def test_rate_trace_ids_that_do_not_match_the_verdicts_are_refused(ids: list[str]) -> None:
    record = certify(
        _Judge(),
        CalibrationSet(_flat_pairs(80), "locked_test", "cal"),
        ledger=InMemoryLockedTestLedger(),
    )
    with pytest.raises(SampleMisuseError, match="rate_trace_ids must name the trace of each"):
        corrected_rate(record, [0] * 140 + [1] * 10, rate_set_id="prod", rate_trace_ids=ids)


def test_a_trend_mode_without_rate_trace_ids_is_refused() -> None:
    from calibrated_agent_evaluation.run import ModeCounts, ModeKey, trend_report

    record = certify(
        _Judge(),
        CalibrationSet(_flat_pairs(80), "locked_test", "cal"),
        ledger=InMemoryLockedTestLedger(),
    )
    mode = ModeKey(failure_type="refund")
    counts = {mode: ModeCounts(mode=mode, n_total=150, n_judged_fail=10)}
    with pytest.raises(SampleMisuseError, match="rate_trace_ids"):
        trend_report(counts, {mode: record}, {mode: [0] * 140 + [1] * 10}, rate_trace_ids={})


def test_a_disjoint_rate_sample_is_scored() -> None:
    record = certify(
        _Judge(),
        CalibrationSet(_flat_pairs(80), "locked_test", "cal"),
        ledger=InMemoryLockedTestLedger(),
    )
    rate_traces = [f"prod-{i}" for i in range(150)]
    scored = corrected_rate(
        record,
        [0] * 140 + [1] * 10,
        rate_set_id="prod",
        rate_trace_ids=rate_traces,
        n_resamples=999,
    )
    assert scored.corrected_rate is not None
    assert scored.corrected_rate.point is not None


def test_the_corrected_rate_widens_with_the_calibration_design_effect() -> None:
    traces = 300
    clustered = certify(
        _Judge(),
        CalibrationSet(_span_pairs(traces, 2, wrong=frozenset(range(10))), "locked_test", "c"),
        ledger=InMemoryLockedTestLedger(),
    )
    assert clustered.tpr_design_effect > 1.0
    flat = clustered.model_copy(update={"tpr_design_effect": 1.0, "tnr_design_effect": 1.0})
    verdicts = [1] * 30 + [0] * 170
    wide = corrected_rate(
        clustered, verdicts, rate_set_id="prod", n_resamples=1999, rate_trace_ids=rate_ids(verdicts)
    ).corrected_rate
    narrow = corrected_rate(
        flat, verdicts, rate_set_id="prod", n_resamples=1999, rate_trace_ids=rate_ids(verdicts)
    ).corrected_rate
    assert wide is not None
    assert narrow is not None
    assert wide.hi - wide.lo > narrow.hi - narrow.lo
