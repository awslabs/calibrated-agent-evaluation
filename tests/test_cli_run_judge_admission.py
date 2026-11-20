# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation run` gates LLM-judge verdicts only from an admitted judge.

`judge` refuses to score with a judge that is not certified and bias-checked, but verdicts can
also arrive with `load` or sit in an older store. `run` reads the judge version each verdict
carries (`<judge_id>@<fingerprint>`) and refuses to gate unless a CERTIFIED calibration and a CLEAR
bias battery for that version are on record. Human labels and code-scorer verdicts are unaffected.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
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
    return main(["--root", str(root), "--explain", "run", "--n-resamples", "199", *argv])


def _refused(capsys) -> str:
    err = capsys.readouterr().err
    assert "refusing to gate verdicts from a judge that is not admitted" in err
    return err


def test_verdicts_from_a_judge_with_no_calibration_record_are_refused(root, capsys):
    seed_verdicts(root, 48, 2)
    assert _run(root, "--ci") == EXIT_DOMAIN_ERROR
    err = _refused(capsys)
    assert "'refund-judge' (no calibration record for this version of the judge)" in err
    assert load_records_of(root, "run") == []


def test_a_refused_calibration_does_not_admit_the_judge(root, capsys):
    seed_verdicts(root, 48, 2)
    store_calibration(root, certified_calibration(result="REFUSED"))
    store_battery(root, clear_battery())
    assert _run(root, "--ci") == EXIT_DOMAIN_ERROR
    assert "its newest calibration 'calib-refund' is REFUSED" in _refused(capsys)


def test_a_newer_refused_calibration_outranks_an_older_certified_one(root, capsys):
    seed_verdicts(root, 48, 2)
    store_calibration(root, certified_calibration(calib_id="calib-old"))
    newer = certified_calibration(calib_id="calib-new", result="REFUSED")
    store_calibration(root, newer.model_copy(update={"hunted_set_id": "hunted-refund-2"}))
    store_battery(root, clear_battery())
    assert _run(root, "--ci") == EXIT_DOMAIN_ERROR
    assert "its newest calibration 'calib-new' is REFUSED" in _refused(capsys)


def test_a_certified_judge_with_no_bias_battery_is_refused(root, capsys):
    seed_verdicts(root, 48, 2)
    store_calibration(root, certified_calibration())
    assert _run(root, "--ci") == EXIT_DOMAIN_ERROR
    assert "calibration 'calib-refund', but no bias battery" in _refused(capsys)


def test_a_biased_battery_does_not_admit_the_judge(root, capsys):
    seed_verdicts(root, 48, 2)
    store_calibration(root, certified_calibration())
    store_battery(root, clear_battery(verdict="BIASED"))
    assert _run(root, "--ci") == EXIT_DOMAIN_ERROR
    assert "is BIASED" in _refused(capsys)


def test_records_for_another_version_of_the_judge_do_not_admit_its_verdicts(root, capsys):
    other = default_judge_rater_id(prompt_template="Another prompt.")
    seed_verdicts(root, 48, 2, rater_id=other)
    admit_default_judge(root)
    assert _run(root, "--ci") == EXIT_DOMAIN_ERROR
    assert "no calibration record for this version of the judge" in _refused(capsys)


def test_a_verdict_with_no_judge_fingerprint_is_refused(root, capsys):
    seed_verdicts(root, 48, 2, rater_id="hand-written")
    assert _run(root, "--ci") == EXIT_DOMAIN_ERROR
    assert "'hand-written' (the verdicts carry no judge fingerprint)" in _refused(capsys)


def test_a_verdict_with_no_rater_id_is_refused(root, capsys):
    seed_verdicts(root, 48, 2, rater_id=None)
    assert _run(root, "--ci") == EXIT_DOMAIN_ERROR
    assert "a judge with no rater id (the verdicts carry no judge fingerprint)" in _refused(capsys)


def test_an_admitted_judge_gates(root, capsys):
    seed_verdicts(root, 48, 2)
    admit_default_judge(root)
    assert _run(root, "--ci") == EXIT_OK
    assert "CI gate passed" in capsys.readouterr().out


def test_one_unadmitted_judge_refuses_the_whole_gate(root, capsys):
    seed_verdicts(root, 48, 2)
    seed_verdicts(root, 48, 2, prefix="O", rater_id="other-judge@0123456789abcdef")
    admit_default_judge(root)
    assert _run(root, "--ci") == EXIT_DOMAIN_ERROR
    err = _refused(capsys)
    assert "'other-judge'" in err
    assert "'refund-judge'" not in err


def test_with_a_judge_file_the_full_admission_chain_runs(root, tmp_path, capsys):
    seed_verdicts(root, 48, 2)
    judge = write_judge_file(tmp_path / "judge.json")
    store_calibration(root, certified_calibration())
    store_battery(root, clear_battery(verdict="BIASED"))
    assert _run(root, "--ci", "--judge", str(judge)) == EXIT_DOMAIN_ERROR
    assert "bias gate refused" in _refused(capsys)


def test_with_an_admitted_judge_file_the_gate_runs(root, tmp_path):
    seed_verdicts(root, 48, 2)
    judge = write_judge_file(tmp_path / "judge.json")
    admit_default_judge(root)
    assert _run(root, "--ci", "--judge", str(judge)) == EXIT_OK


def test_human_labels_gate_without_any_judge_records(root):
    seed_verdicts(root, 48, 2)
    seed_verdicts(root, 48, 2, kind="human", prefix="H")
    assert _run(root, "--ci", "--verdicts", "human") == EXIT_OK


def test_code_scorer_verdicts_gate_without_any_judge_records(root):
    seed_verdicts(root, 48, 2)
    seed_verdicts(root, 48, 2, kind="code", prefix="S", rater_id="scorer:exact@{}")
    assert _run(root, "--ci", "--verdicts", "scorer") == EXIT_OK


def test_judge_verdicts_on_discovery_traces_alone_do_not_trigger_the_check(root, capsys):
    seed_verdicts(root, 48, 2)
    admit_default_judge(root)
    seed_verdicts(root, 0, 5, frame="discovery", prefix="D", rater_id="unknown@0123456789abcdef")
    assert _run(root, "--ci") == EXIT_OK
    assert "left out of the gate" in capsys.readouterr().out


def test_the_plain_output_names_the_judge_and_the_next_step(root, capsys):
    seed_verdicts(root, 48, 2)
    assert main(["--root", str(root), "run", "--ci"]) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "hasn't passed its check against your labels" in err
    assert "refund-judge" in err
    assert "--verdicts human" in err
