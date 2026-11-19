# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation calibrate` checks the gold's raters first, discloses top-ups, and
explains the judge.

Gold from two raters who overlapped is measured with Krippendorff's alpha before any model call:
usable agreement certifies and is stored on the record, anything else is refused with nothing
recorded. One rater's gold certifies with a note. Growing a set refused short of its plan records
the looks. Under --explain the judge's agreement with the gold and its drift since the previous
record are shown.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.schema.types import Label, Trace
from cli_store_helpers import trace
from test_cli_calibrate_bias import CLEAN, MODE, _calibrate, _verdict_cassette


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)


def _label(t: Trace, fail: bool, rater: str | None, version: str | None = None) -> Label:
    return Label(
        trace_id=t.trace_id,
        span_id=None,
        kind="human",
        outcome="FAIL" if fail else "PASS",
        critique="gold",
        failure_mode=MODE if fail else None,
        rater_id=rater,
        codebook_version=version,
    )


def _store(
    root: Path,
    n_fail: int,
    n_pass: int,
    *,
    start: int = 0,
    second: str | None = None,
    overlap: int = 0,
    flip: int = 0,
    version: str | None = None,
    second_version: str | None = None,
    first: str | None = "ana",
) -> list[Trace]:
    """Gold by ``first`` on every trace; ``second`` double-codes the first ``overlap`` FAILs and
    PASSes, disagreeing on ``flip`` of each."""
    traces = []
    with LocalBackend(root=root) as backend:
        for i in range(n_fail + n_pass):
            t = trace(f"G{start + i:03d}")
            backend.put_trace(t)
            fail = i < n_fail
            backend.put_score(_label(t, fail, first, version))
            k = i if fail else i - n_fail
            if second is not None and k < overlap:
                agree = k >= flip
                backend.put_score(
                    _label(t, fail if agree else not fail, second, second_version or version)
                )
            traces.append(t)
    return traces


def _replies(traces: list[Trace], n_fail: int, wrong: int = 0) -> list[tuple[Trace, str]]:
    out = []
    for i, t in enumerate(traces):
        verdict = "fail" if i < n_fail else "pass"
        if i < wrong:
            verdict = "pass"
        out.append((t, verdict))
    return out


def test_gold_from_two_raters_who_agree_certifies_and_stores_alpha(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _store(root, 65, 65, second="ben", overlap=65)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _replies(traces, 65))
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    out = capsys.readouterr().out
    assert "Krippendorff's alpha 1.000" in out
    assert "130 of 130 unit(s) coded by two or more" in out
    assert "not measured" not in out
    (record,) = load_records_of(root, "calibration")
    assert record.agreement_alpha is not None
    assert record.agreement_alpha.point == 1.0
    assert record.agreement_alpha.n == 130


def test_gold_double_coded_on_too_few_traces_is_refused_even_when_the_raters_agree(
    tmp_path, capsys
):
    root = tmp_path / "store"
    _store(root, 55, 55, second="ben", overlap=30)
    empty = _verdict_cassette(tmp_path / "empty.json", CLEAN, [])
    assert _calibrate(root, CLEAN, empty) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "verdict top_up_overlap" in err
    assert "until at least 130 of its traces are coded by both (70 more)" in err
    assert load_records_of(root, "calibration") == []


def test_gold_whose_raters_disagree_is_refused_before_any_model_call(tmp_path, capsys):
    root = tmp_path / "store"
    _store(root, 65, 65, second="ben", overlap=65, flip=10)
    empty = _verdict_cassette(tmp_path / "empty.json", CLEAN, [])
    assert _calibrate(root, CLEAN, empty) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "do not agree well enough to certify against it" in err
    assert "No judge was run and nothing was recorded" in err
    assert "cassette" not in err.lower()
    assert load_records_of(root, "calibration") == []


def test_gold_from_one_rater_certifies_with_a_note(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _store(root, 55, 55)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _replies(traces, 55))
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    out = capsys.readouterr().out
    assert "Rater agreement on the gold was not measured (the gold has one rater)" in out
    (record,) = load_records_of(root, "calibration")
    assert record.agreement_alpha is None


def test_gold_with_no_rater_recorded_certifies_with_a_note(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _store(root, 55, 55, first=None)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _replies(traces, 55))
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    assert "record no rater" in capsys.readouterr().out


def test_two_raters_who_never_overlapped_are_not_measured(tmp_path, capsys):
    root = tmp_path / "store"
    first = _store(root, 30, 30, first="ana")
    more = _store(root, 25, 25, start=60, first="ben")
    replies = [*_replies(first, 30), *_replies(more, 25)]
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, replies)
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    assert "was not measured" in capsys.readouterr().out


def test_gold_across_codebook_versions_is_refused(tmp_path, capsys):
    root = tmp_path / "store"
    _store(root, 55, 55, second="ben", overlap=10, version="v1", second_version="v2")
    empty = _verdict_cassette(tmp_path / "empty.json", CLEAN, [])
    assert _calibrate(root, CLEAN, empty) == EXIT_DOMAIN_ERROR
    assert "codebook versions v1, v2" in capsys.readouterr().err


def test_a_raters_newest_label_on_a_trace_counts(tmp_path, capsys):
    """A rater who relabeled a trace to agree is read at the newest label, not refused."""
    root = tmp_path / "store"
    traces = _store(root, 65, 65, second="ben", overlap=65, flip=10)
    with LocalBackend(root=root) as backend:
        for i in [*range(10), *range(65, 75)]:
            backend.put_score(_label(traces[i], i < 65, "ben"))
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _replies(traces, 65))
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    assert "Krippendorff's alpha 1.000" in capsys.readouterr().out


def test_explain_shows_the_judge_against_the_gold_and_the_looks(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _store(root, 55, 55)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _replies(traces, 55, wrong=1))
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    out = capsys.readouterr().out
    assert "judge vs gold: raw agreement 0.991" in out
    assert "looks: 1 (the first test" in out
    assert "judge drift: no earlier record for this judge id" in out


def test_explain_shows_kappa_with_its_cells_on_imbalanced_gold(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _store(root, 55, 600)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _replies(traces, 55, wrong=1))
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    out = capsys.readouterr().out
    assert "judge vs gold: Cohen's kappa" in out
    assert "cells TP 54, FN 1, FP 0, TN 600" in out


def test_a_grown_set_records_two_looks_and_the_drift_is_not_compared(tmp_path, capsys):
    root = tmp_path / "store"
    first = _store(root, 3, 3)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _replies(first, 3))
    assert _calibrate(root, CLEAN, cassette) == EXIT_DOMAIN_ERROR
    capsys.readouterr()
    more = _store(root, 52, 52, start=6)
    _verdict_cassette(cassette, CLEAN, [*_replies(first, 3), *_replies(more, 52)])
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    out = capsys.readouterr().out
    assert "tested 2 times on this gold as it grew" in out
    assert "looks: 2" in out
    assert "judge drift: not compared with" in out
    _, grown = load_records_of(root, "calibration")
    assert grown.looks == 2


def test_the_previous_record_on_independent_gold_gives_the_drift(tmp_path, capsys):
    """Two cells' gold, calibrated one cell at a time, are independent samples."""
    root = tmp_path / "store"
    first = _store(root, 55, 55)
    second = _store(root, 55, 55, start=200)
    with LocalBackend(root=root) as backend:
        for t in second:
            backend.put_trace(t.model_copy(update={"taxonomy_cell": {"feature": "exchange"}}))
    replies = [*_replies(first, 55), *_replies(second, 55, wrong=5)]
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, replies)
    assert _calibrate(root, CLEAN, cassette, "--cell", "feature=refund") == EXIT_OK
    capsys.readouterr()
    _calibrate(root, CLEAN, cassette, "--cell", "feature=exchange")
    out = capsys.readouterr()
    assert "judge drift since 'calib-fixture-clean-judge-gold-" in out.out + out.err
    assert "change in TPR -0.091" in out.out + out.err


def test_gold_two_raters_double_coded_is_planned_at_the_agreement_overlap(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _store(root, 65, 65, second="ben", overlap=65)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _replies(traces, 65))
    assert _calibrate(root, CLEAN, cassette, "--explain") == EXIT_OK
    out = capsys.readouterr().out
    assert (
        "The plan is 65 labels per class, not 55: a second rater double-codes this gold, and "
        "their agreement needs 130 shared traces."
    ) in out
    assert "plan at expected accuracy 0.90: 65 gold label(s) per class" in out
    assert "raters: two or more (in the gold: ana, ben)" in out
    assert "which raises the plan from 55 to 65" in out
    (record,) = load_records_of(root, "calibration")
    assert record.planned_per_class == 65


def test_raters_flag_plans_for_a_second_rater_who_has_not_labeled_yet(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _store(root, 55, 55)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _replies(traces, 55))
    assert _calibrate(root, CLEAN, cassette, "--raters", "2", "--explain") == EXIT_OK
    out = capsys.readouterr().out
    assert "The plan is 65 labels per class, not 55" in out
    assert "raters: two or more (--raters 2; in the gold: ana)" in out
    assert "this set has 55 FAIL and 55 PASS, short by 10 FAIL and 10 PASS" in out
    assert "at least 130 of its traces" in out
    (record,) = load_records_of(root, "calibration")
    assert record.planned_per_class == 65


def test_one_rater_keeps_the_single_rater_plan(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _store(root, 55, 55)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _replies(traces, 55))
    assert _calibrate(root, CLEAN, cassette, "--explain") == EXIT_OK
    out = capsys.readouterr().out
    assert "The plan is" not in out
    assert "raters: one (in the gold: ana), so the gold needs no agreement overlap" in out
    (record,) = load_records_of(root, "calibration")
    assert record.planned_per_class == 55


def test_raters_flag_below_the_gold_raters_is_overruled_by_the_gold(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _store(root, 65, 65, second="ben", overlap=65)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _replies(traces, 65))
    assert _calibrate(root, CLEAN, cassette, "--raters", "1", "--explain") == EXIT_OK
    out = capsys.readouterr().out
    assert "--raters 1 is fewer than the raters already in the gold, so the plan is for 2" in out
    (record,) = load_records_of(root, "calibration")
    assert record.planned_per_class == 65


def test_rate_safe_plan_already_covers_the_overlap(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _store(root, 65, 65, second="ben", overlap=65)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _replies(traces, 65))
    _calibrate(root, CLEAN, cassette, "--rate-safe", "--explain")
    out = capsys.readouterr().out
    assert "which the plan of 162 already covers" in out
    assert "The plan is" not in out


@pytest.mark.parametrize("value", ["0", "-2"])
def test_raters_below_one_is_refused_before_reading_the_store(tmp_path, capsys, value):
    root = tmp_path / "store"
    empty = _verdict_cassette(tmp_path / "empty.json", CLEAN, [])
    assert _calibrate(root, CLEAN, empty, "--raters", value) != EXIT_OK
    assert f"--raters must be at least 1, got {value}" in capsys.readouterr().err
    assert load_records_of(root, "calibration") == []


def test_poor_agreement_on_a_small_overlap_is_refused_as_a_codebook_problem(tmp_path, capsys):
    """Below 130 shared traces an upper bound under the bar is conclusive: no top-up advice."""
    root = tmp_path / "store"
    _store(root, 55, 55, second="ben", overlap=20, flip=10)
    empty = _verdict_cassette(tmp_path / "empty.json", CLEAN, [])
    assert _calibrate(root, CLEAN, empty) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "verdict codebook_broken (alpha could be at most" in err
    assert "so double-coding more of the gold would not change this" in err
    assert "rewrite the failure-mode definitions" in err
    assert "until at least 130" not in err
    assert load_records_of(root, "calibration") == []
