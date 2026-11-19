# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The results table: one flat row per estimate, refusals included.

Also covers the interval kind every Bound/Interval now carries, so a two-sided 95% interval and a
one-sided 95% bound are distinguishable in the type, and the RunReport -> RunRecord builder.
"""

from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pytest

from _helpers import TEST_INTERVAL_KIND, TEST_STAMP, make_label, rate_ids
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.bias import BiasBatteryRecord, ProbeResult
from calibrated_agent_evaluation.calibrate import (
    InMemoryLockedTestLedger,
    dump_records,
    load_records,
)
from calibrated_agent_evaluation.calibrate.engine import certify, corrected_rate
from calibrated_agent_evaluation.calibrate.errors import CalibrationError
from calibrated_agent_evaluation.calibrate.record import (
    METHOD_RG_JEFFREYS,
    METHOD_WILSON,
    Bound,
    CalibrationSet,
    Interval,
    LabelPair,
)
from calibrated_agent_evaluation.dumpload import (
    RESULTS_COLUMNS,
    Estimate,
    ResultsTableError,
    results_table,
    write_results_csv,
)
from calibrated_agent_evaluation.run import (
    ModeKey,
    RunGateDecision,
    RunReport,
    TrendPoint,
    run_record_of,
)


@dataclass(frozen=True)
class _Judge:
    judge_id: str = "j"
    model: str = "bedrock/claude"
    prompt_version: str = "v1"


def _certified(tp: int = 72, tn: int = 72):
    pairs = []
    for i in range(80):
        pairs.append(
            LabelPair(
                gold=make_label(trace_id=f"p{i}", kind="human", outcome="FAIL"),
                verdict=make_label(
                    trace_id=f"p{i}", kind="llm-judge", outcome="FAIL" if i < tp else "PASS"
                ),
            )
        )
        pairs.append(
            LabelPair(
                gold=make_label(trace_id=f"n{i}", kind="human", outcome="PASS"),
                verdict=make_label(
                    trace_id=f"n{i}", kind="llm-judge", outcome="PASS" if i < tn else "FAIL"
                ),
            )
        )
    locked = CalibrationSet(pairs=tuple(pairs), split="locked_test", set_id="h")
    return certify(_Judge(), locked, ledger=InMemoryLockedTestLedger())


EVAL = [1] * 148 + [0] * 852


def test_a_bound_is_one_sided_and_the_corrected_rate_is_two_sided() -> None:
    """The 95-one-sided vs 95-two-sided ambiguity is resolved by the types themselves."""
    rec = corrected_rate(
        _certified(), EVAL, rate_set_id="prod", n_resamples=199, rate_trace_ids=rate_ids(EVAL)
    )
    assert (rec.tpr.sided, rec.tpr.level, rec.tpr.method, rec.tpr.n) == (
        "lower",
        0.95,
        METHOD_WILSON,
        80,
    )
    rate = rec.corrected_rate
    assert rate is not None
    assert (rate.sided, rate.level, rate.method, rate.n) == (
        "two-sided",
        0.95,
        METHOD_RG_JEFFREYS,
        len(EVAL),
    )


def test_an_interval_cannot_be_built_without_saying_what_it_is() -> None:
    with pytest.raises(CalibrationError, match="method"):
        Interval(point=0.1, lo=0.0, hi=0.2)  # type: ignore[call-arg]
    with pytest.raises(CalibrationError, match="n"):
        Bound(point=0.9, lb95=0.8)  # type: ignore[call-arg]


def test_the_interval_kind_survives_the_calibration_store(tmp_path: Path) -> None:
    rec = corrected_rate(
        _certified(), EVAL, rate_set_id="prod", n_resamples=199, rate_trace_ids=rate_ids(EVAL)
    )
    dump_records([rec], tmp_path, fmt="csv")
    (loaded,) = load_records(tmp_path, fmt="csv")
    assert loaded.corrected_rate == rec.corrected_rate
    assert loaded.tpr == rec.tpr


def test_a_calibration_record_flattens_to_bounds_and_the_corrected_rate() -> None:
    rec = corrected_rate(
        _certified(),
        EVAL,
        rate_set_id="prod",
        seed=3,
        n_resamples=199,
        rate_trace_ids=rate_ids(EVAL),
    )
    rows = results_table([rec])
    by = {r.estimand: r for r in rows}
    assert set(by) == {"tpr", "tnr", "corrected_rate"}
    assert by["tpr"].sided == "lower"
    assert by["tpr"].hi == 1.0
    assert by["tpr"].seed is None
    assert by["corrected_rate"].seed == 3
    assert by["corrected_rate"].n_resamples == 199
    assert by["corrected_rate"].n == len(EVAL)
    assert all(r.provenance == rec.provenance for r in rows)


def test_a_refused_correction_keeps_its_row_with_an_empty_point(tmp_path: Path) -> None:
    """A weak judge's refused point must still appear: empty point, populated reason."""
    # A CERTIFIED record whose point TPR/TNR give J=0.4 (the engine's own J<0.5 refusal path).
    weak_judge = replace(
        _certified(), tpr=Bound(point=0.70, lb95=0.65, n=80), tnr=Bound(point=0.70, lb95=0.65, n=80)
    )
    # A measured rate inside the weak judge's [1 - TNR, TPR] range, so only J refuses the point.
    weak = corrected_rate(
        weak_judge,
        [1] * 45 + [0] * 105,
        rate_set_id="prod",
        n_resamples=199,
        rate_trace_ids=rate_ids([1] * 45 + [0] * 105),
    )
    rows = [r for r in results_table([weak]) if r.estimand == "corrected_rate"]
    assert len(rows) == 1
    assert rows[0].point is None
    assert rows[0].refusal_reason

    out = tmp_path / "results.csv"
    write_results_csv([weak], out)
    with out.open(newline="") as fh:
        table = list(csv.DictReader(fh))
    refused = next(r for r in table if r["estimand"] == "corrected_rate")
    assert refused["point"] == ""
    assert refused["refusal_reason"].startswith("youden_j=")


def test_the_csv_has_one_flat_column_per_field(tmp_path: Path) -> None:
    rec = corrected_rate(
        _certified(),
        EVAL,
        rate_set_id="prod",
        seed=1,
        n_resamples=199,
        rate_trace_ids=rate_ids(EVAL),
    )
    out = tmp_path / "t.csv"
    assert write_results_csv(results_table([rec]), out) == 3
    with out.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames or ()) == RESULTS_COLUMNS
        first = next(reader)
    assert (
        first["calibrated_agent_evaluation_version"]
        == rec.provenance.calibrated_agent_evaluation_version
    )
    assert float(first["lo"]) == pytest.approx(rec.tpr.lb95)


def test_a_bias_battery_row_is_an_upper_bound_and_not_applicable_is_a_refusal() -> None:
    battery = BiasBatteryRecord(
        battery_id="b1",
        judge_id="j",
        model="m",
        prompt_version="v1",
        is_pairwise=False,
        results=(
            ProbeResult(
                bias="length",
                status="run",
                n=20,
                biased_count=1,
                magnitude=0.05,
                magnitude_ub95=0.236,
                threshold=0.3,
                passed=True,
                is_must_pass=True,
            ),
            ProbeResult(
                bias="positional",
                status="not_applicable",
                n=0,
                biased_count=0,
                magnitude=0.0,
                magnitude_ub95=0.0,
                threshold=None,
                passed=None,
                is_must_pass=False,
            ),
        ),
        verdict="CLEAR",
        provenance=TEST_STAMP,
    )
    length, positional = results_table([battery])
    assert (length.sided, length.lo, length.hi, length.n) == ("upper", 0.0, 0.236, 20)
    assert positional.point is None
    assert positional.refusal_reason == "not_applicable"


def test_run_record_of_keeps_the_gate_threshold_and_trend() -> None:
    mode = ModeKey(failure_type="refund")
    trend = (
        TrendPoint(
            mode=mode,
            n_total=50,
            corrected_rate=Interval(point=0.06, lo=0.02, hi=0.11, **TEST_INTERVAL_KIND),
            calib_id="c1",
            raw_rate=Interval(point=0.16, lo=0.08, hi=0.29, **TEST_INTERVAL_KIND),
        ),
    )
    report = RunReport(
        run_id="r1",
        mode="ci",
        trend=trend,
        gate=RunGateDecision(passed=False, failing_modes=(mode,), threshold=0.85, n_modes=1),
    )
    record = run_record_of(report, inputs={"labels": "9"}, judge_ids=("j",), seed=2, n_resamples=9)
    assert record.gate_passed is False
    assert record.gate_n_modes == 1
    assert record.provenance.thresholds == {"gate": 0.85}
    assert (record.provenance.seed, record.provenance.n_resamples) == (2, 9)
    assert record.calib_ids == ("c1",)
    assert record.trend == trend
    assert record.multiplicity == ()  # the CI gate records no multiplicity family
    corrected, raw = results_table([record])
    assert (corrected.estimand, corrected.slice, corrected.seed) == ("corrected_rate", "refund", 2)
    # The raw rate is a closed-form Wilson interval: no seed, and it sits on the next row.
    assert (raw.estimand, raw.slice, raw.seed) == ("raw_rate", "refund", None)


def test_run_record_of_refuses_empty_inputs() -> None:
    from calibrated_agent_evaluation.run import RunError

    report = RunReport(run_id="r", mode="cadence", trend=())
    with pytest.raises(RunError, match="inputs"):
        run_record_of(report, inputs={})


def test_an_unknown_record_type_is_refused_not_skipped() -> None:
    with pytest.raises(ResultsTableError, match="str"):
        results_table(["not a record"])  # type: ignore[list-item]


def test_mixing_rows_and_records_is_refused(tmp_path: Path) -> None:
    rec = corrected_rate(
        _certified(), EVAL, rate_set_id="prod", n_resamples=99, rate_trace_ids=rate_ids(EVAL)
    )
    row = results_table([rec])[0]
    with pytest.raises(ResultsTableError, match="mix"):
        write_results_csv([row, rec], tmp_path / "x.csv")  # type: ignore[list-item]


def test_a_write_failure_is_typed(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(ResultsTableError, match="failed to write"):
        write_results_csv([], blocker / "results.csv")


def test_estimate_is_frozen() -> None:
    rec = corrected_rate(
        _certified(), EVAL, rate_set_id="prod", n_resamples=99, rate_trace_ids=rate_ids(EVAL)
    )
    row: Estimate = results_table([rec])[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.point = 0.5  # type: ignore[misc]


def test_agentcore_claims_never_become_estimate_rows() -> None:
    import json

    from calibrated_agent_evaluation._provenance import capture_provenance
    from calibrated_agent_evaluation.agentcore.audit_types import AbTestClaimRecord, ArmAssignment
    from calibrated_agent_evaluation.dumpload.results import ResultsTableError, results_table

    golden = Path(__file__).resolve().parents[1] / "fixtures" / "expected" / "audit"
    doc = json.loads((golden / "abtest_verdict.json").read_text(encoding="utf-8"))
    claim = AbTestClaimRecord.model_validate(
        {
            **doc,
            "record_id": "abtest-1",
            "read_at": "2026-11-02T10:00:00+00:00",
            "provenance": capture_provenance().model_dump(mode="json"),
        }
    )
    arn = claim.ab_test_arn
    assignment = ArmAssignment(
        trace_id="t",
        otlp_trace_ids=("t",),
        read_id="arms-1",
        runtime_session_id=None,
        span_session_ids=(),
        ab_test_arn=arn,
        runtime="r",
        window_start="2026-10-28T19:00:00+00:00",
        window_end="2026-10-28T20:00:00+00:00",
        omni=False,
        rekey_sessions=False,
        status="assigned",
        experiment_arn=arn,
        variant="C",
        spans_total=1,
        spans_stamped=1,
    )

    assert results_table([claim, assignment]) == []
    from test_agentcore_audit_types import audit_record, recommendation_claim_record, run_provenance
    from test_improve_store import _record as improve_record

    improve = improve_record()
    audit_kinds = [
        claim,
        assignment,
        recommendation_claim_record(),
        audit_record(),
        run_provenance(),
    ]
    rows = results_table([*audit_kinds, improve])
    assert rows == results_table([improve])
    assert rows
    agentcore_figures = {0.031, 0.48, 0.71, 0.78, 0.64, 0.66, 0.006, 0.134}
    assert not any(r.point in agentcore_figures for r in rows)
    with pytest.raises(ResultsTableError):
        results_table([object()])  # type: ignore[list-item]
