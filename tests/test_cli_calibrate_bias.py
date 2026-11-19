# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation calibrate` and `calibrated-agent-evaluation bias`, the gates
`calibrated-agent-evaluation judge` admits a judge on.

On the fixture corpus the OrderBot judge has one human FAIL and one PASS to be graded against, so
calibration is honestly REFUSED and the chain stops there. The success path runs the fixture clean
judge (the synthetic controlled judge) over a generated store with enough gold, its verdicts
replayed from a cassette built here, then its bias battery from the committed CLI cassette, then
`calibrated-agent-evaluation judge`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.judgefile import load_judge_file
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.execute.cassette import Cassette, Mode, fingerprint
from calibrated_agent_evaluation.judges import build_verdict_schema, render_trace
from calibrated_agent_evaluation.schema.types import Label, Trace
from cli_store_helpers import (
    certified_calibration,
    clear_battery,
    store_battery,
    store_calibration,
    trace,
)

_REPO = Path(__file__).resolve().parent.parent
_FILES = _REPO / "fixtures" / "judge-files"
ORDERBOT = _FILES / "orderbot-refund-resolved.json"
ORDERBOT_CASSETTE = (
    _REPO / "fixtures" / "raw" / "cassettes" / "judge_orderbot_resolved.cassette.json"
)
CLEAN = _FILES / "fixture-clean-judge.json"
LENGTH_BIASED = _FILES / "fixture-length-biased-judge.json"
BIAS_CASSETTE = _REPO / "fixtures" / "raw" / "cassettes" / "bias_fixture_judges_cli.cassette.json"
PROBES = _REPO / "fixtures" / "bias" / "probes"
THRESHOLDS = _REPO / "fixtures" / "bias" / "thresholds.json"
MODE = "wrong_refund_decision"


def _calibrated_agent_evaluation(root: Path, *argv: str) -> int:
    return main(["--root", str(root), *argv])


def _calibrate_orderbot(root: Path, *extra: str) -> int:
    return _calibrated_agent_evaluation(
        root,
        "calibrate",
        "--judge",
        str(ORDERBOT),
        "--cassette",
        str(ORDERBOT_CASSETTE),
        *extra,
    )


def _judge_labels(root: Path) -> list[Label]:
    with LocalBackend(root=root) as backend:
        return [
            lb
            for t in backend.query()
            for lb in backend.get_scores(t.trace_id)
            if lb.kind == "llm-judge"
        ]


# --- on the fixture corpus: the honest refusal --------------------------------------------------


def test_the_corpus_gold_is_too_small_to_certify_and_the_record_says_so(seeded_local_root, capsys):
    assert _calibrate_orderbot(seeded_local_root, "--cell", "feature=refund_handling") == (
        EXIT_DOMAIN_ERROR
    )
    err = capsys.readouterr().err
    assert "REFUSED" in err
    assert "on 2 human gold label(s) from 2 stored trace(s) in cell feature=refund_handling" in err
    assert "TPR 1.000 (95% lower bound 0.270, n=1 FAIL)" in err
    assert "plan at expected accuracy 0.90: 55 gold label(s) per class" in err
    assert "short by 54 FAIL and 54 PASS" in err
    (record,) = load_records_of(seeded_local_root, "calibration")
    assert record.result == "REFUSED"
    assert (record.n_pos, record.n_neg) == (1, 1)
    assert sorted(record.trace_ids) == ["ORD-1001", "ORD-1002"]
    # The calibration verdicts certify the judge; they are not scores.
    assert _judge_labels(seeded_local_root) == []


def test_the_same_gold_cannot_be_used_twice(seeded_local_root, capsys):
    _calibrate_orderbot(seeded_local_root, "--cell", "feature=refund_handling")
    capsys.readouterr()
    assert _calibrate_orderbot(seeded_local_root, "--cell", "feature=refund_handling") == (
        EXIT_DOMAIN_ERROR
    )
    assert "single-use" in capsys.readouterr().err
    assert len(load_records_of(seeded_local_root, "calibration")) == 1


def test_a_missing_verdict_writes_no_record(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _gold_store(root, 3, 3)
    partial = _verdict_cassette(tmp_path / "c.json", CLEAN, [(t, "fail") for t in traces[:5]])
    assert _calibrate(root, CLEAN, partial) == EXIT_DOMAIN_ERROR
    assert "on trace 'G005': no recorded verdict" in capsys.readouterr().err
    assert load_records_of(root, "calibration") == []


def test_a_refused_calibration_stops_bias_and_judge(seeded_local_root, capsys):
    _calibrate_orderbot(seeded_local_root, "--cell", "feature=refund_handling")
    capsys.readouterr()
    bias = ["bias", "--judge", str(ORDERBOT), "--probes", str(PROBES), "--thresholds"]
    assert (
        _calibrated_agent_evaluation(seeded_local_root, *bias, str(THRESHOLDS)) == EXIT_DOMAIN_ERROR
    )
    err = capsys.readouterr().err
    assert "isn't certified yet: its newest calibration record" in err
    # The hint repeats the store the command read, so pasting it calibrates the same store.
    assert f"calibrated-agent-evaluation --root {seeded_local_root} calibrate --judge" in err
    assert load_records_of(seeded_local_root, "bias") == []

    judge = ["judge", "--judge", str(ORDERBOT), "--cassette", str(ORDERBOT_CASSETTE)]
    assert (
        _calibrated_agent_evaluation(seeded_local_root, *judge, "--trace", "ORD-1001")
        == EXIT_DOMAIN_ERROR
    )
    assert "no bias battery" in capsys.readouterr().err
    assert _judge_labels(seeded_local_root) == []


# --- the success path: certify, CLEAR, score -----------------------------------------------------


def _gold_store(root: Path, n_fail: int, n_pass: int, *, extra: list[Label] | None = None) -> list:
    """Traces with one human gold label each (FAIL for the mode, or PASS); returns the traces."""
    traces = []
    with LocalBackend(root=root) as backend:
        for i in range(n_fail + n_pass):
            t = trace(f"G{i:03d}")
            backend.put_trace(t)
            outcome = "FAIL" if i < n_fail else "PASS"
            mode = MODE if outcome == "FAIL" else None
            backend.put_score(
                Label(
                    trace_id=t.trace_id,
                    span_id=None,
                    kind="human",
                    outcome=outcome,
                    critique="gold",
                    failure_mode=mode,
                )
            )
            traces.append(t)
        for label in extra or []:
            backend.put_score(label)
    return traces


def _verdict_cassette(path: Path, judge_file: Path, verdicts: list[tuple[Trace, str]]) -> Path:
    """A cassette of the judge's replies on each trace, keyed exactly as the judge requests them."""
    judge = load_judge_file(judge_file).judge
    schema = build_verdict_schema(judge)
    cassette = Cassette(path, mode=Mode.AUTO)
    for t, verdict in verdicts:
        request = {
            "system": judge.prompt_template,
            "user": render_trace(t),
            "schema": schema,
            "via": "response_schema",
        }
        cassette.record(
            fingerprint(request),
            request,
            {
                "fields": {"critique": f"graded {t.trace_id}", "verdict": verdict},
                "via": "response_schema",
            },
        )
    return path


def _calibrate(root: Path, judge_file: Path, cassette: Path, *extra: str) -> int:
    return _calibrated_agent_evaluation(
        root, "calibrate", "--judge", str(judge_file), "--cassette", str(cassette), *extra
    )


def test_calibrate_then_bias_then_judge(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _gold_store(root, 55, 55)
    # The judge agrees with the gold on every trace but one per class.
    replies = [(t, "fail" if i < 55 else "pass") for i, t in enumerate(traces)]
    replies[0] = (traces[0], "pass")
    replies[109] = (traces[109], "fail")
    cassette = _verdict_cassette(tmp_path / "judge.cassette.json", CLEAN, replies)

    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    out = capsys.readouterr().out
    assert "on 110 human gold label(s) from 110 stored trace(s): CERTIFIED" in out
    assert "TPR 0.982 (95% lower bound 0.923, n=55 FAIL)" in out
    assert "this set has 55 FAIL and 55 PASS, enough" in out
    (record,) = load_records_of(root, "calibration")
    assert record.result == "CERTIFIED"
    assert record.locked_test_consumed
    assert _judge_labels(root) == []

    bias = ["bias", "--judge", str(CLEAN), "--cassette", str(BIAS_CASSETTE)]
    assert (
        _calibrated_agent_evaluation(
            root, *bias, "--probes", str(PROBES), "--thresholds", str(THRESHOLDS)
        )
        == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "bias battery for judge 'fixture-clean-judge'" in out
    assert ": CLEAR" in out
    assert (
        "length (must-pass): 0 of 6 trial(s) disagreed, 0 of 6 placebo(s) did; excess 0.000 "
        "(95% upper bound "
    ) in out
    assert "positional (advisory): not_applicable" in out
    (battery,) = load_records_of(root, "bias")
    assert battery.verdict == "CLEAR"

    judge = ["judge", "--judge", str(CLEAN), "--cassette", str(cassette)]
    assert _calibrated_agent_evaluation(root, *judge, "--trace", "G001") == EXIT_OK
    assert "admitted" in capsys.readouterr().out
    (verdict,) = _judge_labels(root)
    assert (verdict.trace_id, verdict.outcome, verdict.failure_mode) == ("G001", "FAIL", MODE)


def test_a_biased_judge_is_recorded_and_refused(tmp_path, capsys):
    root = tmp_path / "store"
    judge = load_judge_file(LENGTH_BIASED).judge
    store_calibration(
        root,
        certified_calibration(
            judge.judge_id,
            model=judge.model,
            prompt_version=judge.prompt_version,
            fingerprint=judge.fingerprint,
        ),
    )
    bias = ["bias", "--judge", str(LENGTH_BIASED), "--cassette", str(BIAS_CASSETTE)]
    argv = [*bias, "--probes", str(PROBES), "--thresholds", str(THRESHOLDS)]
    assert _calibrated_agent_evaluation(root, *argv) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "BIASED, must-pass bias exceeded: length" in err
    assert "length (must-pass): 6 of 6 trial(s) disagreed, 0 of 6 placebo(s) did" in err
    assert "OVER the 0.35 threshold" in err
    (battery,) = load_records_of(root, "bias")
    assert battery.verdict == "BIASED"


def test_gold_that_is_not_gold_for_the_judge_is_counted(tmp_path, capsys):
    root = tmp_path / "store"
    other = Label(
        trace_id="G000",
        span_id=None,
        kind="human",
        outcome="FAIL",
        critique="other",
        failure_mode="misread_amount",
    )
    disagree = Label(
        trace_id="G001",
        span_id=None,
        kind="human",
        outcome="PASS",
        critique="a second rater disagrees",
    )
    span = Label(
        trace_id="G002",
        span_id="G002/agent",
        kind="human",
        outcome="FAIL",
        critique="span gold",
        failure_mode=MODE,
    )
    traces = _gold_store(root, 3, 2, extra=[other, disagree, span])
    replies = [(t, "fail") for t in traces]
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, replies)
    assert _calibrate(root, CLEAN, cassette) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "on 4 human gold label(s) from 5 stored trace(s): REFUSED" in err
    assert "left out: 1 FAIL label(s) for another failure mode or agent" in err
    assert "left out: 1 target(s) whose human labels disagree for this mode" in err
    assert "left out: 1 span-level label(s), which do not grade a workflow judge" in err


def test_gold_without_both_classes_runs_no_judge(tmp_path, capsys):
    root = tmp_path / "store"
    _gold_store(root, 3, 0)
    # No cassette entry exists for these traces: a model call would be a miss, not this refusal.
    empty = _verdict_cassette(tmp_path / "c.json", CLEAN, [])
    assert _calibrate(root, CLEAN, empty) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "3 FAIL for wrong_refund_decision and 0 PASS" in err
    assert "needs both classes" in err
    assert load_records_of(root, "calibration") == []


def test_an_expected_accuracy_the_gate_cannot_certify_is_refused_before_the_judge_runs(
    tmp_path, capsys
):
    root = tmp_path / "store"
    _gold_store(root, 3, 3)
    empty = _verdict_cassette(tmp_path / "c.json", CLEAN, [])
    assert _calibrate(root, CLEAN, empty, "--expected-accuracy", "0.80") == EXIT_DOMAIN_ERROR
    assert "InfeasiblePlanError" in capsys.readouterr().err
    assert load_records_of(root, "calibration") == []


def test_an_asymmetric_judge_is_planned_on_both_rates(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _gold_store(root, 3, 3)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, [(t, "fail") for t in traces])
    argv = ["--expected-accuracy", "0.95", "--expected-tnr", "0.85"]
    assert _calibrate(root, CLEAN, cassette, *argv) == EXIT_DOMAIN_ERROR
    # The certification size at the lower rate (0.85) is 208, above the 0.95 coverage floor of 80.
    assert "plan at expected TPR 0.95 and TNR 0.85: 208 gold label(s) per class" in (
        capsys.readouterr().err
    )


def test_bias_without_a_calibration_record_is_refused(tmp_path, capsys):
    argv = ["bias", "--judge", str(CLEAN), "--probes", str(PROBES), "--thresholds", str(THRESHOLDS)]
    assert _calibrated_agent_evaluation(tmp_path / "store", *argv) == EXIT_DOMAIN_ERROR
    assert "has no calibration record" in capsys.readouterr().err


def test_bias_needs_its_probe_set(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _calibrated_agent_evaluation(tmp_path, "bias", "--judge", str(CLEAN))
    assert exc.value.code == EXIT_USAGE


def test_of_two_records_stamped_in_the_same_second_the_later_one_counts():
    """Provenance stamps have one-second resolution, so a tie is broken by write order."""
    from calibrated_agent_evaluation.cli.records import latest_for_judge

    judge = load_judge_file(CLEAN).judge
    pinned = {
        "model": judge.model,
        "prompt_version": judge.prompt_version,
        "fingerprint": judge.fingerprint,
    }
    refused = certified_calibration(judge.judge_id, calib_id="first", result="REFUSED", **pinned)
    certified = certified_calibration(judge.judge_id, calib_id="second", **pinned)
    tied = [refused, replace(certified, provenance=refused.provenance)]
    assert latest_for_judge(tied, judge).calib_id == "second"
    assert latest_for_judge(list(reversed(tied)), judge).calib_id == "first"


# --- certified on gold below the coverage floor ------------------------


def _replies(traces: list, n_fail: int, wrong: int) -> list[tuple[Trace, str]]:
    """The judge's replies on a gold store: right on all but the first ``wrong`` of each class."""
    replies = []
    for i, t in enumerate(traces):
        is_fail = i < n_fail
        flipped = (i if is_fail else i - n_fail) < wrong
        replies.append((t, "fail" if is_fail != flipped else "pass"))
    return replies


def _admit_and_run(root: Path, cassette: Path, capsys) -> str:
    """Admit the clean judge, score the store with it, and return
    `calibrated-agent-evaluation run`'s output.

    The run still gates and records; only the corrected rate is withheld, with its reason.
    """
    judge = load_judge_file(CLEAN).judge
    pinned = {
        "model": judge.model,
        "prompt_version": judge.prompt_version,
        "fingerprint": judge.fingerprint,
    }
    store_battery(root, clear_battery(judge.judge_id, **pinned))
    assert (
        _calibrated_agent_evaluation(
            root, "judge", "--judge", str(CLEAN), "--cassette", str(cassette)
        )
        == EXIT_OK
    )
    assert "admitted" in capsys.readouterr().out
    run = ["run", "--judge", str(CLEAN), "--n-resamples", "199"]
    assert _calibrated_agent_evaluation(root, *run) == EXIT_OK
    (record,) = load_records_of(root, "run")
    (point,) = record.trend
    assert point.corrected_rate.point is None
    assert point.corrected_rate.refusal_reason.startswith("gold below the coverage floor:")
    captured = capsys.readouterr()
    return captured.out + captured.err


def test_a_certification_below_the_coverage_floor_is_rate_unsafe(tmp_path, capsys):
    """Certified and admitted by calibrated-agent-evaluation judge; calibrated-agent-evaluation run
    gates it but withholds its rate."""
    root = tmp_path / "store"
    traces = _gold_store(root, 20, 20)
    cassette = _verdict_cassette(tmp_path / "judge.cassette.json", CLEAN, _replies(traces, 20, 1))

    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    out = capsys.readouterr().out
    assert "on 40 human gold label(s) from 40 stored trace(s): CERTIFIED" in out
    assert "this set has 20 FAIL and 20 PASS, short by 35 FAIL and 35 PASS" in out
    assert (
        "  rate-unsafe: 20 FAIL and 20 PASS is below the coverage floor of 80 per class (35 at "
        "the planned accuracy, 80 at the observed TPR 0.950 and TNR 0.950); the judge is "
        "certified, but `calibrated-agent-evaluation run` refuses a corrected rate from this record"
    ) in out
    (record,) = load_records_of(root, "calibration")
    assert record.result == "CERTIFIED"
    assert (record.planned_per_class, record.planned_coverage_floor) == (55, 35)
    assert record.below_coverage_floor
    assert len(_judge_labels(root)) == 0

    out = _admit_and_run(root, cassette, capsys)
    assert "REFUSED (gold below the coverage floor:" in out
    assert "20 FAIL and 20 PASS" in out
    assert "coverage floor of 80 per class" in out


def test_a_judge_more_accurate_than_planned_is_rate_unsafe_at_the_planned_floor(tmp_path, capsys):
    """55 per class meets the plan, but 52 of 55 right (0.945) needs 80."""
    root = tmp_path / "store"
    traces = _gold_store(root, 55, 55)
    cassette = _verdict_cassette(tmp_path / "judge.cassette.json", CLEAN, _replies(traces, 55, 3))

    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    out = capsys.readouterr().out
    assert ": CERTIFIED" in out
    assert "this set has 55 FAIL and 55 PASS, enough" in out
    assert (
        "  rate-unsafe: 55 FAIL and 55 PASS is below the coverage floor of 80 per class (35 at "
        "the planned accuracy, 80 at the observed TPR 0.945 and TNR 0.945)"
    ) in out
    (record,) = load_records_of(root, "calibration")
    assert (record.planned_coverage_floor, record.observed_coverage_floor) == (35, 80)

    out = _admit_and_run(root, cassette, capsys)
    assert "REFUSED (gold below the coverage floor:" in out
    assert "coverage floor of 80 per class (the larger of 35 at the planned accuracy and 80 at" in (
        out
    )


def test_a_perfect_judge_reads_the_boundary_floor(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _gold_store(root, 35, 35)
    cassette = _verdict_cassette(tmp_path / "judge.cassette.json", CLEAN, _replies(traces, 35, 0))
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    assert "rate-unsafe" not in capsys.readouterr().out
    (record,) = load_records_of(root, "calibration")
    assert (record.observed_coverage_floor, record.observed_floor_unmeasured) == (11, False)


def test_an_accuracy_without_a_measured_floor_is_reported(tmp_path, capsys, monkeypatch):
    from calibrated_agent_evaluation.calibrate import plan as plan_module

    table = dict(plan_module.COVERAGE_FLOOR) | {1.00: None}
    monkeypatch.setattr(plan_module, "COVERAGE_FLOOR", tuple(table.items()))
    root = tmp_path / "store"
    traces = _gold_store(root, 35, 35)
    cassette = _verdict_cassette(tmp_path / "judge.cassette.json", CLEAN, _replies(traces, 35, 0))
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    assert (
        "  rate-unsafe: at the observed TPR 1.000 and TNR 1.000 the corrected rate's interval was "
        "measured and did not cover nominally at any scanned gold size, so 35 FAIL and 35 PASS is "
        "not known to be enough"
    ) in capsys.readouterr().out
    (record,) = load_records_of(root, "calibration")
    assert record.observed_floor_unmeasured


def test_a_rate_safe_plan_asks_for_162_per_class(tmp_path, capsys):
    """--rate-safe plans the size at which any certifying accuracy is
    rate-safe; 160 of 162 right per class (observed 0.988, floor 162) is not marked."""
    root = tmp_path / "store"
    traces = _gold_store(root, 162, 162)
    cassette = _verdict_cassette(tmp_path / "judge.cassette.json", CLEAN, _replies(traces, 162, 2))
    assert _calibrate(root, CLEAN, cassette, "--rate-safe") == EXIT_OK
    out = capsys.readouterr().out
    assert (
        "  rate-safe plan at expected accuracy 0.90: 162 gold label(s) per class (55 certify the "
        "judge; 162 keep the corrected rate's interval covering at every certifying accuracy); "
        "this set has 162 FAIL and 162 PASS, enough"
    ) in out
    assert "rate-unsafe" not in out
    (record,) = load_records_of(root, "calibration")
    assert (record.planned_per_class, record.planned_coverage_floor) == (162, 35)
    assert record.observed_coverage_floor == 162
    assert not record.below_coverage_floor


def test_a_rate_safe_plan_without_a_measured_floor_is_refused_before_the_judge_runs(
    tmp_path, capsys, monkeypatch
):
    from calibrated_agent_evaluation.calibrate import plan as plan_module

    table = dict(plan_module.COVERAGE_FLOOR) | {0.995: None, 0.999: None, 1.00: None}
    monkeypatch.setattr(plan_module, "COVERAGE_FLOOR", tuple(table.items()))
    root = tmp_path / "store"
    _gold_store(root, 3, 3)
    empty = tmp_path / "none.json"
    argv = ("--rate-safe", "--expected-accuracy", "0.995")
    assert _calibrate(root, CLEAN, empty, *argv) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "InfeasiblePlanError" in err
    assert "coverage_floor" in err
    assert load_records_of(root, "calibration") == []


def test_gold_at_both_floors_is_not_marked(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _gold_store(root, 80, 80)
    cassette = _verdict_cassette(tmp_path / "judge.cassette.json", CLEAN, _replies(traces, 80, 4))
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    assert "rate-unsafe" not in capsys.readouterr().out
    (record,) = load_records_of(root, "calibration")
    assert (record.planned_coverage_floor, record.observed_coverage_floor) == (35, 80)
    assert not record.below_coverage_floor


def test_bias_without_probes_is_a_usage_error_with_no_default_set(tmp_path, capsys):
    """No probe set is assumed: a battery built for another judge's mode would not test this one."""
    argv = ["bias", "--judge", str(ORDERBOT), "--thresholds", str(THRESHOLDS)]
    with pytest.raises(SystemExit) as exit_info:
        main(["--root", str(tmp_path / "store"), *argv])
    assert exit_info.value.code == EXIT_USAGE
    assert "--probes" in capsys.readouterr().err
    assert not (tmp_path / "store" / "records" / "bias").exists()
