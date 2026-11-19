# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation calibrate --check` re-measures a certified judge on fresh gold and
records the drift.

It reads the certifying record, takes only gold no calibration of the mode used, refuses short gold
before any model call with the count to label (sized from the record), and writes a judge-check
record. The calibration records are never written or changed.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.run.judge_check import check_size
from test_cli_calibrate_agreement import _replies, _store
from test_cli_calibrate_bias import CLEAN, _calibrate, _verdict_cassette


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)


@pytest.fixture
def certified(tmp_path, capsys):
    """A store whose clean judge is certified on 55 + 55 gold it judged with one miss per class."""
    root = tmp_path / "store"
    traces = _store(root, 55, 55)
    replies = _replies(traces, 55, wrong=1)
    replies[109] = (traces[109], "fail")
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, replies)
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    capsys.readouterr()
    return root, cassette


def test_a_judge_that_held_up_gets_a_judge_check_record(certified, capsys):
    root, cassette = certified
    (calibration,) = load_records_of(root, "calibration")
    need_fail, need_pass = check_size(calibration)
    fresh = _store(root, need_fail, need_pass, start=300)
    _verdict_cassette(cassette, CLEAN, _replies(fresh, need_fail))
    assert _calibrate(root, CLEAN, cassette, "--check") == EXIT_OK
    out = capsys.readouterr().out
    assert "shows no drop since it was certified" in out
    assert "drift flagged: no" in out
    assert "no calibration record was written or changed" in out
    (check,) = load_records_of(root, "judge-check")
    assert check.against_calib_id == calibration.calib_id
    assert (check.needed_fail, check.needed_pass) == (need_fail, need_pass)
    assert set(check.trace_ids) == {t.trace_id for t in fresh}
    assert not check.drifted
    assert load_records_of(root, "calibration") == [calibration]


def test_a_judge_that_got_worse_is_flagged_and_told_to_recertify(certified, capsys):
    root, cassette = certified
    (calibration,) = load_records_of(root, "calibration")
    fresh = _store(root, 40, 40, start=300)
    _verdict_cassette(cassette, CLEAN, _replies(fresh, 40, wrong=20))
    assert _calibrate(root, CLEAN, cassette, "--check") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "has got worse since it was certified" in err
    assert "certify it again on new gold" in err
    (check,) = load_records_of(root, "judge-check")
    assert check.drifted
    assert check.drift.delta_tpr.hi < 0
    assert load_records_of(root, "calibration") == [calibration]


def test_short_fresh_gold_is_refused_before_any_model_call(certified, capsys):
    root, _ = certified
    (calibration,) = load_records_of(root, "calibration")
    need_fail, need_pass = check_size(calibration)
    _store(root, 2, 2, start=300)
    empty = _verdict_cassette(root.parent / "empty.json", CLEAN, [])
    assert _calibrate(root, CLEAN, empty, "--check") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert f"detected with {need_fail} FAIL and {need_pass} PASS" in err
    assert f"Label {need_fail - 2} more FAIL and {need_pass - 2} more PASS" in err
    assert "110 gold label(s) on traces a calibration or an earlier check" in err
    assert load_records_of(root, "judge-check") == []


def test_gold_used_by_an_earlier_check_is_not_fresh(certified, capsys):
    root, cassette = certified
    (calibration,) = load_records_of(root, "calibration")
    need_fail, need_pass = check_size(calibration)
    fresh = _store(root, need_fail, need_pass, start=300)
    _verdict_cassette(cassette, CLEAN, _replies(fresh, need_fail))
    assert _calibrate(root, CLEAN, cassette, "--check") == EXIT_OK
    capsys.readouterr()
    assert _calibrate(root, CLEAN, cassette, "--check") == EXIT_DOMAIN_ERROR
    assert "not enough fresh gold" in capsys.readouterr().err
    assert len(load_records_of(root, "judge-check")) == 1


def test_an_uncertified_judge_cannot_be_rechecked(tmp_path, capsys):
    root = tmp_path / "store"
    _store(root, 5, 5)
    empty = _verdict_cassette(tmp_path / "empty.json", CLEAN, [])
    assert _calibrate(root, CLEAN, empty, "--check") == EXIT_DOMAIN_ERROR
    assert "it has no calibration record" in capsys.readouterr().err
    assert load_records_of(root, "judge-check") == []


def test_a_plan_flag_is_refused_with_check(certified, capsys):
    root, cassette = certified
    assert _calibrate(root, CLEAN, cassette, "--check", "--rate-safe") == EXIT_USAGE
    assert "--rate-safe plans a certification" in capsys.readouterr().err
