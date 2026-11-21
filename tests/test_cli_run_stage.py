# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation run` gates at a chosen threshold, corrects each judged mode's
rate with its calibration record, scopes to `--trace`, and persists a RunRecord.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.records import load_records_of
from cli_store_helpers import (
    admit_default_judge,
    certified_calibration,
    clear_battery,
    default_judge_rater_id,
    seed_verdicts,
    store_battery,
    store_calibration,
    write_judge_file,
)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    return tmp_path / "store"


def _run(root, *argv: str) -> int:
    return main(["--root", str(root), "run", "--n-resamples", "199", *argv])


def test_threshold_changes_the_gate_and_is_printed_both_ways(root, capsys):
    # 48 of 50 pass: the Wilson 95% lower bound is about 0.865.
    seed_verdicts(root, 48, 2)
    admit_default_judge(root)
    assert _run(root, "--ci") == EXIT_OK
    out = capsys.readouterr().out
    assert "threshold 0.80" in out
    assert "PASSED" in out

    assert _run(root, "--ci", "--threshold", "0.90") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "threshold 0.90" in err
    assert "FAILED" in err
    assert "over_refund" in err


def test_without_ci_a_failing_gate_is_reported_with_exit_zero(root, capsys):
    seed_verdicts(root, 48, 2)
    admit_default_judge(root)
    assert _run(root, "--threshold", "0.90") == EXIT_OK
    assert "FAILED" in capsys.readouterr().out


def test_a_threshold_outside_zero_one_is_refused(root, capsys):
    seed_verdicts(root, 48, 2)
    assert _run(root, "--threshold", "1.5") == EXIT_USAGE
    assert "--threshold" in capsys.readouterr().err


def test_an_empty_store_refuses_a_vacuous_pass(root, capsys):
    assert _run(root, "--ci") == EXIT_DOMAIN_ERROR
    assert "no judge verdicts with a failure mode on 0 stored trace(s)" in (capsys.readouterr().err)
    assert load_records_of(root, "run") == []


def test_every_run_persists_a_run_record_with_its_seed_resamples_and_threshold(root, capsys):
    seed_verdicts(root, 48, 2)
    admit_default_judge(root)
    assert _run(root, "--seed", "7", "--threshold", "0.85") == EXIT_OK
    assert _run(root) == EXIT_OK
    records = load_records_of(root, "run")
    assert len(records) == 2
    first = records[0]
    assert first.provenance.seed == 7
    assert first.provenance.n_resamples == 199
    assert first.provenance.thresholds == {"gate": 0.85}
    assert first.gate_passed is True
    assert first.mode == "cadence"
    assert first.inputs["verdict_source"] == "judge"
    assert f"recorded run {first.run_id}" in capsys.readouterr().out


def test_ci_runs_are_recorded_as_ci(root, capsys):
    seed_verdicts(root, 48, 2)
    admit_default_judge(root)
    assert _run(root, "--ci") == EXIT_OK
    assert load_records_of(root, "run")[0].mode == "ci"


def test_the_judge_verdicts_are_the_default_source(root, capsys):
    seed_verdicts(root, 48, 2)
    admit_default_judge(root)
    seed_verdicts(root, 0, 5, kind="human", prefix="H")
    assert _run(root, "--ci") == EXIT_OK
    out = capsys.readouterr().out
    assert "50 judge verdicts" in out


def test_verdicts_human_gates_only_the_human_labels_and_records_it(root, capsys):
    seed_verdicts(root, 48, 2)
    seed_verdicts(root, 0, 5, kind="human", prefix="H")
    assert _run(root, "--ci", "--verdicts", "human") == EXIT_DOMAIN_ERROR
    assert "CI gate over 5 human verdicts on 55 stored trace(s)" in capsys.readouterr().err
    (record,) = load_records_of(root, "run")
    assert record.inputs["verdict_source"] == "human"
    assert record.inputs["n_gated"] == "5"


def test_a_store_of_only_human_labels_is_not_gated_as_judge_verdicts(root, capsys):
    """The gate never falls back to the other source when the chosen one is empty."""
    seed_verdicts(root, 0, 5, kind="human")
    assert _run(root) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "no judge verdicts with a failure mode on 5 stored trace(s)" in err
    assert "--verdicts human" in err
    assert load_records_of(root, "run") == []

    assert _run(root, "--verdicts", "human") == EXIT_OK
    assert "5 human verdicts" in capsys.readouterr().out


def test_a_store_of_only_judge_verdicts_has_no_human_labels_to_gate(root, capsys):
    seed_verdicts(root, 48, 2)
    assert _run(root, "--verdicts", "human") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "no human verdicts with a failure mode on 50 stored trace(s)" in err
    assert "--verdicts judge" in err


def test_verdicts_judge_gates_only_the_llm_judge_verdicts(root, capsys):
    """Code-scorer verdicts are their own source, not part of `judge`."""
    seed_verdicts(root, 48, 2)
    admit_default_judge(root)
    seed_verdicts(root, 0, 7, kind="code", prefix="S")
    assert _run(root, "--ci") == EXIT_OK
    assert "CI gate over 50 judge verdicts on 57 stored trace(s)" in capsys.readouterr().out
    (record,) = load_records_of(root, "run")
    assert record.inputs["verdict_source"] == "judge"
    assert record.inputs["n_gated"] == "50"


def test_verdicts_scorer_gates_only_the_code_scorer_verdicts_and_records_it(root, capsys):
    seed_verdicts(root, 48, 2)
    seed_verdicts(root, 0, 7, kind="code", prefix="S")
    assert _run(root, "--ci", "--verdicts", "scorer") == EXIT_DOMAIN_ERROR
    assert "CI gate over 7 code-scorer verdicts on 57 stored trace(s)" in capsys.readouterr().err
    (record,) = load_records_of(root, "run")
    assert record.inputs["verdict_source"] == "scorer"
    assert record.inputs["n_gated"] == "7"


def test_a_store_of_only_scorer_verdicts_is_not_gated_as_judge_verdicts(root, capsys):
    seed_verdicts(root, 20, 0, kind="code")
    assert _run(root) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert (
        "no judge verdicts with a failure mode on 20 stored trace(s) to gate (--verdicts judge)"
        in (err)
    )
    assert "--verdicts scorer or --verdicts human" in err
    assert load_records_of(root, "run") == []
    assert _run(root, "--verdicts", "scorer") == EXIT_OK
    assert "20 code-scorer verdicts" in capsys.readouterr().out


def test_a_store_without_scorer_verdicts_refuses_the_scorer_source(root, capsys):
    seed_verdicts(root, 48, 2)
    seed_verdicts(root, 0, 5, kind="human", prefix="H")
    assert _run(root, "--verdicts", "scorer") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "no code-scorer verdicts with a failure mode on 55 stored trace(s)" in err
    assert "--verdicts judge or --verdicts human" in err
    assert load_records_of(root, "run") == []


def test_an_unknown_verdict_source_is_a_usage_error(root):
    with pytest.raises(SystemExit) as exc:
        _run(root, "--verdicts", "both")
    assert exc.value.code == 2


def test_no_judge_file_means_no_trend_and_the_output_says_why(root, capsys):
    seed_verdicts(root, 48, 2)
    admit_default_judge(root)
    assert _run(root) == EXIT_OK
    out = capsys.readouterr().out
    assert "trend: none" in out
    assert "--judge" in out
    assert load_records_of(root, "run")[0].trend == ()


def test_a_judge_without_a_certifying_calibration_gets_no_trend(root, tmp_path, capsys):
    seed_verdicts(root, 160, 40)
    seed_verdicts(root, 0, 5, kind="human", prefix="H")
    judge = write_judge_file(tmp_path / "judge.json")
    store_calibration(root, certified_calibration(result="REFUSED"))
    store_battery(root, clear_battery())
    # Its verdicts can't gate, so the judge source is refused; the human gate still reports why
    # the judge got no corrected rate.
    assert _run(root, "--judge", str(judge)) == EXIT_DOMAIN_ERROR
    assert "calibration gate refused" in capsys.readouterr().err
    assert _run(root, "--verdicts", "human", "--judge", str(judge)) == EXIT_OK
    out = capsys.readouterr().out
    assert "failure rate: not estimated (this judge isn't certified" in out


def test_a_calibrated_judge_gets_a_corrected_rate_with_its_interval(root, tmp_path, capsys):
    seed_verdicts(root, 160, 40)
    judge = write_judge_file(tmp_path / "judge.json")
    store_calibration(root, certified_calibration())
    store_battery(root, clear_battery())
    assert _run(root, "--judge", str(judge), "--seed", "3", "--explain") == EXIT_OK
    out = capsys.readouterr().out
    assert "trend (1 mode(s)" in out
    assert "corrected 0." in out
    assert "95% CI [" in out
    assert "calibration calib-refund" in out
    assert "raw 0.200" in out

    (record,) = load_records_of(root, "run")
    (point,) = record.trend
    assert point.calib_id == "calib-refund"
    assert point.corrected_rate.n == 200
    assert record.calib_ids == ("calib-refund",)
    assert record.judge_ids == ("refund-judge",)
    assert record.provenance.seed == 3


def test_the_trend_is_reproducible_from_its_seed(root, tmp_path, capsys):
    seed_verdicts(root, 160, 40)
    judge = write_judge_file(tmp_path / "judge.json")
    store_calibration(root, certified_calibration())
    store_battery(root, clear_battery())
    for _ in range(2):
        assert _run(root, "--judge", str(judge), "--seed", "11") == EXIT_OK
    a, b = load_records_of(root, "run")
    assert a.trend == b.trend


def test_discovery_frame_verdicts_stay_out_of_the_rate(root, tmp_path, capsys):
    seed_verdicts(root, 160, 40)
    seed_verdicts(root, 0, 20, frame="discovery", prefix="D")
    judge = write_judge_file(tmp_path / "judge.json")
    store_calibration(root, certified_calibration())
    store_battery(root, clear_battery())
    assert _run(root, "--judge", str(judge)) == EXIT_OK
    out = capsys.readouterr().out
    assert "20 verdict(s) on hand-picked traces were left out of the rate" in out
    (record,) = load_records_of(root, "run")
    assert record.trend[0].corrected_rate.n == 200


def test_trace_scopes_the_run_to_one_trace(root, capsys):
    ids = seed_verdicts(root, 48, 2)
    admit_default_judge(root)
    assert _run(root, "--trace", ids[0]) == EXIT_OK
    out = capsys.readouterr().out
    assert f"CI gate over 1 judge verdicts on trace {ids[0]!r}" in out


def test_an_unknown_trace_id_is_refused_not_ignored(root, capsys):
    seed_verdicts(root, 48, 2)
    assert _run(root, "--trace", "NOPE") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "NOPE" in err
    assert load_records_of(root, "run") == []


def test_a_malformed_judge_file_is_a_typed_refusal(root, tmp_path, capsys):
    seed_verdicts(root, 48, 2)
    bad = tmp_path / "bad.json"
    bad.write_text('{"judge_id": "x"}')
    assert _run(root, "--judge", str(bad)) == EXIT_DOMAIN_ERROR
    assert "missing required field" in capsys.readouterr().err


def test_a_judge_is_never_corrected_with_another_judge_versions_verdicts(root, tmp_path, capsys):
    """Verdicts written by another version of the judge are left out of the gate and the rate."""
    seed_verdicts(root, 160, 40, rater_id=default_judge_rater_id(prompt_template="Other prompt."))
    seed_verdicts(root, 150, 50, prefix="M")
    judge = write_judge_file(tmp_path / "judge.json")
    store_calibration(root, certified_calibration())
    store_battery(root, clear_battery())
    assert _run(root, "--judge", str(judge)) == EXIT_OK
    out = capsys.readouterr().out
    assert "200 verdict(s) made by another judge or judge version were left out" in out
    (record,) = load_records_of(root, "run")
    assert record.inputs["n_gated"] == "200"
    (point,) = record.trend
    assert point.raw_rate.n == 200
    assert point.raw_rate.point == pytest.approx(0.25)


def test_run_refuses_to_gate_a_judge_whose_verdicts_are_all_another_versions(
    root, tmp_path, capsys
):
    seed_verdicts(root, 160, 40, rater_id=default_judge_rater_id(prompt_template="Other prompt."))
    judge = write_judge_file(tmp_path / "judge.json")
    store_calibration(root, certified_calibration())
    store_battery(root, clear_battery())
    assert _run(root, "--judge", str(judge), "--ci") == EXIT_DOMAIN_ERROR
    assert "none of the judge verdicts" in capsys.readouterr().err
    assert load_records_of(root, "run") == []


def test_a_judge_that_failed_its_bias_check_gets_no_corrected_rate(root, tmp_path, capsys):
    seed_verdicts(root, 160, 40)
    seed_verdicts(root, 0, 5, kind="human", prefix="H")
    judge = write_judge_file(tmp_path / "judge.json")
    store_calibration(root, certified_calibration())
    store_battery(root, clear_battery(verdict="BIASED"))
    assert _run(root, "--judge", str(judge)) == EXIT_DOMAIN_ERROR
    assert "bias gate refused" in capsys.readouterr().err
    assert _run(root, "--verdicts", "human", "--judge", str(judge)) == EXIT_OK
    out = capsys.readouterr().out
    assert "failure rate: not estimated (this judge didn't pass its bias check" in out
    assert load_records_of(root, "run")[0].trend == ()


def test_a_judge_never_checked_for_bias_gets_no_corrected_rate(root, tmp_path, capsys):
    seed_verdicts(root, 160, 40)
    seed_verdicts(root, 0, 5, kind="human", prefix="H")
    judge = write_judge_file(tmp_path / "judge.json")
    store_calibration(root, certified_calibration())
    assert _run(root, "--judge", str(judge)) == EXIT_DOMAIN_ERROR
    assert "but no bias battery" in capsys.readouterr().err
    assert _run(root, "--verdicts", "human", "--judge", str(judge)) == EXIT_OK
    out = capsys.readouterr().out
    assert "this judge hasn't been checked for bias" in out
    assert load_records_of(root, "run")[0].trend == ()


def test_the_default_output_says_when_no_rate_was_estimated(root, capsys):
    seed_verdicts(root, 48, 2)
    admit_default_judge(root)
    assert _run(root) == EXIT_OK
    out = capsys.readouterr().out.split("Run with --explain")[0]
    assert "No failure rate estimated: pass --judge FILE" in out


def _gold_on(*trace_ids: str):
    from calibrated_agent_evaluation._model import replace

    return replace(certified_calibration(), trace_ids=trace_ids)


def test_production_verdicts_on_the_calibration_gold_stay_out_of_the_rate(
    root, tmp_path, capsys, monkeypatch
):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_EXPLAIN")
    seed_verdicts(root, 160, 40)
    gold = seed_verdicts(root, 5, 5, prefix="G")
    judge = write_judge_file(tmp_path / "judge.json")
    store_calibration(root, _gold_on(*gold, "elsewhere-1"))
    store_battery(root, clear_battery())
    assert _run(root, "--judge", str(judge)) == EXIT_OK
    out = capsys.readouterr().out
    assert "10 verdict(s) on 10 trace(s) this judge was calibrated on were left out" in out
    assert "G000" not in out
    (record,) = load_records_of(root, "run")
    assert record.trend[0].corrected_rate.n == 200


def test_explain_names_the_gold_traces_left_out_of_the_rate(root, tmp_path, capsys):
    seed_verdicts(root, 160, 40)
    gold = seed_verdicts(root, 1, 1, prefix="G")
    judge = write_judge_file(tmp_path / "judge.json")
    store_calibration(root, _gold_on(*gold))
    store_battery(root, clear_battery())
    assert _run(root, "--judge", str(judge), "--explain") == EXIT_OK
    out = capsys.readouterr().out
    assert "2 verdict(s) on 2 production trace(s) in the calibration gold of 'calib-refund'" in out
    assert "G000, G001" in out


def test_a_rate_sample_made_only_of_gold_is_not_estimated_and_the_gate_still_runs(
    root, tmp_path, capsys
):
    ids = seed_verdicts(root, 48, 2)
    judge = write_judge_file(tmp_path / "judge.json")
    store_calibration(root, _gold_on(*ids))
    store_battery(root, clear_battery())
    assert _run(root, "--judge", str(judge)) == EXIT_OK
    out = capsys.readouterr().out
    assert "every random-sample trace this judge scored is in its calibration gold" in out
    assert "PASSED" in out
    (record,) = load_records_of(root, "run")
    assert record.trend == ()


def test_no_scorer_verdicts_points_at_calibrated_agent_evaluation_score(root, capsys):
    seed_verdicts(root, 48, 2)
    assert _run(root, "--verdicts", "scorer") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "no code-scorer verdicts" in err
    assert "`calibrated-agent-evaluation score`" in err
