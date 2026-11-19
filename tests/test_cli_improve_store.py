# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation improve --evaluate` pairs the before and after verdicts from two
stores.

The run before the fix and the run after it are two local stores holding the same trace ids. The
target's pairs come from each trace's verdict for the pre-registered metric on both sides; every
other judged failure mode is a guard slice tested with McNemar for a regression and Holm corrected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.pairing import guard_slice, pair_stores
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.improve import LedgerIntegrityError
from calibrated_agent_evaluation.schema.types import Label
from cli_store_helpers import trace

TARGET = "over_refund"
GUARD = "misread_amount"
N = 40


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    return tmp_path / "records-root"


def _store(path: Path, fails: dict[str, set[int]], *, kind: str = "llm-judge", n: int = N) -> Path:
    """``n`` traces, each with one verdict per mode in ``fails`` (FAIL for the listed indices)."""
    with LocalBackend(root=path) as backend:
        for i in range(n):
            tid = f"T{i:03d}"
            backend.put_trace(trace(tid))
            for mode, failing in fails.items():
                outcome = "FAIL" if i in failing else "PASS"
                backend.put_score(
                    Label(
                        trace_id=tid,
                        span_id=None,
                        kind=kind,  # type: ignore[arg-type]
                        outcome=outcome,
                        critique="v",
                        failure_mode=mode,
                    )
                )
    return path


def _adjudicate(path: Path, ids: range) -> None:
    with LocalBackend(root=path) as backend:
        for i in ids:
            backend.put_score(
                Label(
                    trace_id=f"T{i:03d}",
                    span_id=None,
                    kind="human",
                    outcome="PASS",
                    critique="fixed, checked",
                )
            )


def _development(tmp_path: Path, *ids: str) -> Path:
    path = tmp_path / "dev.txt"
    path.write_text("\n".join(ids or ("D-1", "D-2")) + "\n")
    return path


def _register(root: Path, reduction: str = "0.05") -> None:
    assert (
        main(["--root", str(root), "improve", "--register", TARGET, "--min-reduction", reduction])
        == EXIT_OK
    )


def _evaluate(root: Path, before: Path, after: Path, dev: Path, *extra: str) -> int:
    return main(
        [
            "--root",
            str(root),
            "improve",
            "--evaluate",
            TARGET,
            "--before-root",
            str(before),
            "--after-root",
            str(after),
            "--development",
            str(dev),
            *extra,
        ]
    )


def _fixed(tmp_path: Path, *, guard_after: set[int] | None = None) -> tuple[Path, Path]:
    """12 of 40 failed the target before the fix and 2 after; the guard mode fails twice each."""
    before = _store(tmp_path / "before", {TARGET: set(range(12)), GUARD: {30, 31}})
    after = _store(
        tmp_path / "after",
        {TARGET: {0, 1}, GUARD: {30, 31} if guard_after is None else guard_after},
    )
    _adjudicate(after, range(2, 12))
    return before, after


def test_a_fix_is_accepted_from_the_paired_stores(root, tmp_path, capsys):
    _register(root)
    before, after = _fixed(tmp_path)
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_OK
    out = capsys.readouterr().out
    assert f"improve {TARGET!r}: ACCEPTED" in out
    assert "paired fail-rate change (after - before) -0.250" in out
    assert "10 discordant pair(s)" in out
    assert f"paired 40 trace(s) by id between {before} (before) and {after} (after)" in out
    assert "guard panel: 1 slice(s), Holm; regressed: none" in out
    assert f"{GUARD}: fail rate 0.050 [" in out
    assert "one-sided McNemar regression p=1" in out
    (record,) = load_records_of(root, "improve")
    assert record.verdict == "accepted"
    assert record.inputs["verdict_source"] == "judge"
    assert record.inputs["n_pairs"] == "40"
    assert record.inputs["n_adjudicated"] == "10"
    assert record.inputs["before_root"] == str(before)


def test_a_guard_mode_that_regresses_rejects_the_fix(root, tmp_path, capsys):
    _register(root)
    before, after = _fixed(tmp_path, guard_after=set(range(20, 32)))
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "REJECTED" in err
    assert f"regressed: {GUARD}" in err


def test_a_trace_on_one_side_only_is_refused(root, tmp_path, capsys):
    _register(root)
    before = _store(tmp_path / "before", {TARGET: set(range(12)), GUARD: set()})
    after = _store(tmp_path / "after", {TARGET: {0}, GUARD: set()}, n=N - 1)
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "'T039' only before, none only after" in err
    assert load_records_of(root, "improve") == []


def test_a_verdict_on_one_side_only_is_refused(root, tmp_path, capsys):
    _register(root)
    before = _store(tmp_path / "before", {TARGET: set(range(12)), GUARD: set()})
    after = _store(tmp_path / "after", {TARGET: {0}, GUARD: set()})
    with LocalBackend(root=tmp_path / "after") as backend:
        stale = Label(
            trace_id="T005",
            span_id=None,
            kind="llm-judge",
            outcome="PASS",
            critique="v",
            failure_mode=TARGET,
        )
        backend.retract_score(stale)
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert f"the judge verdicts for the target {TARGET!r} are one-sided: 1 without one after" in err
    assert "'T005'" in err


def test_a_guard_mode_judged_on_one_side_only_is_refused(root, tmp_path, capsys):
    _register(root)
    before = _store(tmp_path / "before", {TARGET: set(range(12)), GUARD: set()})
    after = _store(tmp_path / "after", {TARGET: {0}})
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_DOMAIN_ERROR
    assert f"the guard mode {GUARD!r} are one-sided: 40 without one after" in (
        capsys.readouterr().err
    )


def test_no_other_judged_mode_means_no_guard_and_a_refusal(root, tmp_path, capsys):
    _register(root)
    before = _store(tmp_path / "before", {TARGET: set(range(12))})
    after = _store(tmp_path / "after", {TARGET: {0}})
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_DOMAIN_ERROR
    assert "no guard slice" in capsys.readouterr().err


def test_a_judged_flip_needs_a_human_adjudication(root, tmp_path, capsys):
    _register(root)
    before = _store(tmp_path / "before", {TARGET: set(range(12)), GUARD: set()})
    after = _store(tmp_path / "after", {TARGET: set(), GUARD: set()})
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_DOMAIN_ERROR
    assert "AdjudicationRequiredError" in capsys.readouterr().err


def _human_store(path: Path, fails: dict[int, str]) -> Path:
    """One human label per trace, as a labeler writes them: FAIL with its mode, else PASS."""
    with LocalBackend(root=path) as backend:
        for i in range(N):
            tid = f"T{i:03d}"
            backend.put_trace(trace(tid))
            mode = fails.get(i)
            backend.put_score(
                Label(
                    trace_id=tid,
                    span_id=None,
                    kind="human",
                    outcome="FAIL" if mode else "PASS",
                    critique="h",
                    failure_mode=mode,
                )
            )
    return path


def test_human_verdicts_pair_without_adjudication(root, tmp_path, capsys):
    _register(root)
    before_fails = dict.fromkeys(range(12), TARGET) | {30: GUARD}
    after_fails = {0: TARGET, 1: TARGET, 30: GUARD}
    before = _human_store(tmp_path / "before", before_fails)
    after = _human_store(tmp_path / "after", after_fails)
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path), "--verdicts", "human") == EXIT_OK
    out = capsys.readouterr().out
    assert "ACCEPTED" in out
    assert "human verdicts" in out
    (record,) = load_records_of(root, "improve")
    assert record.inputs["verdict_source"] == "human"
    assert record.inputs["n_adjudicated"] == "0"


def test_a_development_trace_among_the_pairs_is_refused(root, tmp_path, capsys):
    _register(root)
    before, after = _fixed(tmp_path)
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path, "T003")) == EXIT_DOMAIN_ERROR
    assert "ConfirmationContaminationError" in capsys.readouterr().err


def test_evaluate_needs_its_development_set_and_two_stores(root, tmp_path, capsys):
    _register(root)
    before, after = _fixed(tmp_path)
    capsys.readouterr()
    base = ["--root", str(root), "improve", "--evaluate", TARGET]
    assert main([*base, "--before-root", str(before), "--after-root", str(after)]) == (EXIT_USAGE)
    assert "--development FILE" in capsys.readouterr().err
    assert _evaluate(root, before, before, _development(tmp_path)) == EXIT_USAGE
    assert "the same store" in capsys.readouterr().err
    assert _evaluate(root, before, tmp_path / "nope", _development(tmp_path)) == EXIT_DOMAIN_ERROR
    assert "--after-root" in capsys.readouterr().err


def test_evaluate_reads_local_stores_only(root, tmp_path, capsys):
    before, after = _fixed(tmp_path)
    dev = _development(tmp_path)
    assert _evaluate(root, before, after, dev, "--backend", "langfuse") == EXIT_DOMAIN_ERROR
    assert "local stores" in capsys.readouterr().err


def test_exactly_one_improve_action(root, tmp_path, capsys):
    argv = ["--root", str(root), "improve", "--evaluate", TARGET, "--request", "r.json"]
    assert main(argv) == EXIT_USAGE
    assert "exactly one of --register" in capsys.readouterr().err


# --- the per-slice test ------------------------------------------------------------------------


def _pairs(improved: int, regressed: int, same: int = 20) -> dict[str, tuple[bool, bool]]:
    out = {f"i{i}": (True, False) for i in range(improved)}
    out |= {f"r{i}": (False, True) for i in range(regressed)}
    out |= {f"s{i}": (False, False) for i in range(same)}
    return out


def test_the_regression_p_value_is_one_sided():
    # 0 improved, 8 regressed: the two-sided mid-p is 2 * 0.5**8 - 0.5**8, and the one-sided
    # regression tail P(X > 8) + P(X = 8) / 2 is half of it.
    assert guard_slice("g", _pairs(0, 8)).regression_p == pytest.approx(0.5**9)
    # The mirror image is strong evidence of improvement, so no evidence of a regression.
    assert guard_slice("g", _pairs(8, 0)).regression_p == pytest.approx(1 - 0.5**9)
    assert guard_slice("g", _pairs(0, 0)).regression_p == 1.0


def test_a_slice_carries_its_rates_with_intervals():
    sl = guard_slice("g", _pairs(0, 8, same=12))
    assert (sl.rate_before.point, sl.rate_after.point) == (0.0, 0.4)
    assert sl.rate_after.method == "wilson"
    assert sl.rate_after.sided == "two-sided"
    assert sl.rate_after.lo < 0.4 < sl.rate_after.hi
    assert sl.rate_after.n == 20


def test_disagreeing_verdicts_on_one_trace_are_refused():
    from calibrated_agent_evaluation.cli.errors import CliError

    labels = {
        "T1": [
            Label(
                trace_id="T1",
                span_id=None,
                kind="llm-judge",
                outcome="FAIL",
                critique="a",
                failure_mode=TARGET,
            ),
            Label(
                trace_id="T1",
                span_id=None,
                kind="llm-judge",
                outcome="PASS",
                critique="b",
                failure_mode=TARGET,
            ),
        ]
    }
    with pytest.raises(CliError, match="disagree"):
        pair_stores(labels, labels, TARGET, "judge")


def test_a_human_pass_for_another_mode_says_nothing_about_the_target():
    from calibrated_agent_evaluation.cli.pairing import verdicts_for

    other_pass = [
        Label(
            trace_id="T1",
            span_id=None,
            kind="human",
            outcome="PASS",
            critique="h",
            failure_mode=GUARD,
        )
    ]
    assert verdicts_for(other_pass, TARGET, "human") == (None, False)
    other_fail = [
        Label(
            trace_id="T1",
            span_id=None,
            kind="human",
            outcome="FAIL",
            critique="h",
            failure_mode=GUARD,
        )
    ]
    assert verdicts_for(other_fail, TARGET, "human") == (False, False)
    clash = [
        Label(
            trace_id="T1",
            span_id=None,
            kind="human",
            outcome="FAIL",
            critique="h",
            failure_mode=TARGET,
        ),
        Label(trace_id="T1", span_id=None, kind="human", outcome="PASS", critique="a second rater"),
    ]
    assert verdicts_for(clash, TARGET, "human") == (True, True)


# --- a FAIL for another mode is not evidence about the target --------------------


def test_a_human_fail_for_another_mode_is_excluded_from_the_targets_pairs(root, tmp_path, capsys):
    """T030 fails the guard mode on both sides; T031 fails the target before and the guard after.

    Neither says whether the trace failed the target on the side labelled with the guard mode, so
    both are left out of the target's pairs, counted and named, instead of read as a target PASS.
    """
    _register(root)
    before_fails = dict.fromkeys(range(12), TARGET) | {30: GUARD, 31: TARGET}
    after_fails = {0: TARGET, 1: TARGET, 30: GUARD, 31: GUARD}
    before = _human_store(tmp_path / "before", before_fails)
    after = _human_store(tmp_path / "after", after_fails)
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path), "--verdicts", "human") == EXIT_OK
    out = capsys.readouterr().out
    assert "paired fail-rate change (after - before) -0.263" in out
    assert f"paired 38 trace(s) by id between {before} (before) and {after} (after)" in out
    assert (
        f"  excluded 2 trace(s) from the pairs for {TARGET!r}: a human FAIL for another failure "
        "mode is not evidence about it ('T030', 'T031')"
    ) in out
    (record,) = load_records_of(root, "improve")
    assert record.inputs["n_pairs"] == "38"
    assert record.inputs["n_excluded"] == "2"
    assert record.inputs["excluded_ids"] == "T030,T031"


def test_exclusion_that_leaves_no_target_pairs_is_refused(root, tmp_path, capsys):
    _register(root)
    before = _human_store(tmp_path / "before", dict.fromkeys(range(N), GUARD))
    after = _human_store(tmp_path / "after", {})
    capsys.readouterr()
    argv = ("--verdicts", "human")
    assert _evaluate(root, before, after, _development(tmp_path), *argv) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert f"no trace has a human verdict for the target {TARGET!r} on both sides" in err
    assert "40 trace(s) were excluded" in err
    assert "'T000', 'T001', 'T002', 'T003', 'T004' and 35 more" in err
    assert load_records_of(root, "improve") == []


def _human(trace_id: str, outcome: str, mode: str | None) -> list[Label]:
    return [
        Label(
            trace_id=trace_id,
            span_id=None,
            kind="human",
            outcome=outcome,  # type: ignore[arg-type]
            critique="h",
            failure_mode=mode,
        )
    ]  # type: ignore[arg-type]


def test_pair_stores_names_the_excluded_traces():
    before = {"T1": _human("T1", "FAIL", GUARD), "T2": _human("T2", "FAIL", TARGET)}
    after = {"T1": _human("T1", "PASS", None), "T2": _human("T2", "PASS", None)}
    paired = pair_stores(before, after, TARGET, "human")
    assert paired.target == {"T2": (True, False)}
    assert paired.excluded == ("T1",)
    # A judge verdict is stamped with the mode it judged, so the judge source excludes nothing.
    judged = {
        "T1": [
            Label(
                trace_id="T1",
                span_id=None,
                kind="llm-judge",
                outcome="FAIL",
                critique="j",
                failure_mode=TARGET,
            )
        ]
    }
    assert pair_stores(judged, judged, TARGET, "judge").excluded == ()


# --- guard slices exclude another mode's FAIL too; scorers are their own source ---


def test_pair_stores_excludes_another_modes_fail_from_a_guard_slice():
    """T1 fails the target before the fix, which says nothing about whether it failed the guard."""
    before = {"T1": _human("T1", "FAIL", TARGET), "T2": _human("T2", "FAIL", GUARD)}
    after = {"T1": _human("T1", "PASS", None), "T2": _human("T2", "PASS", None)}
    paired = pair_stores(before, after, TARGET, "human")
    assert paired.guards == {GUARD: {"T2": (True, False)}}
    assert paired.guard_excluded == {GUARD: ("T1",)}
    # The judge source stamps each verdict with its mode, so it excludes nothing from a guard.
    judged = {
        t: [
            Label(
                trace_id=t,
                span_id=None,
                kind="llm-judge",
                outcome="FAIL",
                critique="j",
                failure_mode=TARGET,
            ),
            Label(
                trace_id=t,
                span_id=None,
                kind="llm-judge",
                outcome="PASS",
                critique="j",
                failure_mode=GUARD,
            ),
        ]
        for t in ("T1", "T2")
    }
    assert pair_stores(judged, judged, TARGET, "judge").guard_excluded == {}


def test_a_guard_slice_leaves_out_traces_whose_label_is_another_modes_fail(root, tmp_path, capsys):
    """T000-T011 fail the target before the fix and so carry no evidence about the guard mode."""
    _register(root)
    before_fails = dict.fromkeys(range(12), TARGET) | {30: GUARD}
    after_fails = {0: TARGET, 1: TARGET, 30: GUARD}
    before = _human_store(tmp_path / "before", before_fails)
    after = _human_store(tmp_path / "after", after_fails)
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path), "--verdicts", "human") == EXIT_OK
    out = capsys.readouterr().out
    assert (
        f"  excluded 12 trace(s) from the guard slice {GUARD!r}: a human FAIL for another "
        "failure mode is not evidence about it ('T000', 'T001', 'T002', 'T003', 'T004' and 7 "
        "more)"
    ) in out
    assert f"    {GUARD}: fail rate 0.036 [" in out
    assert "(n=28) after" in out
    (record,) = load_records_of(root, "improve")
    assert record.inputs["n_guard_excluded"] == "12"
    excluded = ",".join(f"T{i:03d}" for i in range(12))
    assert record.inputs["guard_excluded_ids"] == f"{GUARD}={excluded}"
    # T030 fails the guard mode on both sides, so it is left out of the target's pairs instead.
    assert record.inputs["excluded_ids"] == "T030"


def test_a_guard_slice_emptied_by_exclusion_is_named_in_the_refusal(root, tmp_path, capsys):
    """Every trace fails the target on one side, so none says whether it failed the guard."""
    _register(root)
    before_fails = dict.fromkeys(range(N), TARGET) | {10: GUARD}
    after_fails = {10: TARGET}
    before = _human_store(tmp_path / "before", before_fails)
    after = _human_store(tmp_path / "after", after_fails)
    capsys.readouterr()
    argv = ("--verdicts", "human")
    assert _evaluate(root, before, after, _development(tmp_path), *argv) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert f"no guard slice for {TARGET!r}" in err
    assert (
        f"the guard mode {GUARD!r} has no pair left: all 40 of its traces were excluded because "
        "a side's only human label is a FAIL for another failure mode"
    ) in err
    assert load_records_of(root, "improve") == []


def _score(path: Path, mode: str, outcome: str) -> None:
    """A code-scorer verdict for ``mode`` on every trace in the store at ``path``."""
    with LocalBackend(root=path) as backend:
        for i in range(N):
            label = Label(
                trace_id=f"T{i:03d}",
                span_id=None,
                kind="code",
                outcome=outcome,  # type: ignore[arg-type]
                critique="s",
                failure_mode=mode,
            )
            backend.put_score(label)


def test_judge_verdicts_are_paired_without_the_scorer_verdicts(root, tmp_path, capsys):
    """A scorer that disagrees with the judge on T000-T011 before the fix no longer conflicts."""
    _register(root)
    before, after = _fixed(tmp_path)
    _score(before, TARGET, "PASS")
    _score(after, TARGET, "PASS")
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_OK
    out = capsys.readouterr().out
    assert "paired fail-rate change (after - before) -0.250" in out
    assert f"{after} (after), judge verdicts" in out


def test_scorer_verdicts_are_their_own_source(root, tmp_path, capsys):
    _register(root)
    before = _store(tmp_path / "before", {TARGET: set(range(12)), GUARD: {30, 31}}, kind="code")
    after = _store(tmp_path / "after", {TARGET: {0, 1}, GUARD: {30, 31}}, kind="code")
    _adjudicate(after, range(2, 12))
    capsys.readouterr()
    dev = _development(tmp_path)
    assert _evaluate(root, before, after, dev, "--verdicts", "scorer") == EXIT_OK
    out = capsys.readouterr().out
    assert "paired fail-rate change (after - before) -0.250" in out
    assert f"{after} (after), code-scorer verdicts" in out
    (record,) = load_records_of(root, "improve")
    assert record.inputs["verdict_source"] == "scorer"
    assert record.inputs["n_adjudicated"] == "10"


def test_a_source_with_no_verdicts_for_the_target_is_refused_not_swapped(root, tmp_path, capsys):
    _register(root)
    before = _store(tmp_path / "before", {TARGET: set(range(12)), GUARD: {30, 31}}, kind="code")
    after = _store(tmp_path / "after", {TARGET: {0, 1}, GUARD: {30, 31}}, kind="code")
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert (
        f"no judge verdicts for the target {TARGET!r} in either store (--verdicts judge); score "
        "both stores with `calibrated-agent-evaluation judge` first, or pair another source with "
        "--verdicts scorer or "
        "--verdicts human"
    ) in err
    assert load_records_of(root, "improve") == []


def _label_after(path: Path, ids: range, outcome: str, mode: str | None, rater: str) -> None:
    with LocalBackend(root=path) as backend:
        for i in ids:
            backend.put_score(
                Label(
                    trace_id=f"T{i:03d}",
                    span_id=None,
                    kind="human",
                    outcome=outcome,  # type: ignore[arg-type]
                    critique="reviewed the after run",
                    failure_mode=mode,
                    rater_id=rater,
                )
            )


def _judged_fix(tmp_path: Path) -> tuple[Path, Path]:
    """12 of 40 failed the target before; the judge says only 2 fail after, nobody adjudicated."""
    before = _store(tmp_path / "before", {TARGET: set(range(12)), GUARD: {30, 31}})
    after = _store(tmp_path / "after", {TARGET: {0, 1}, GUARD: {30, 31}})
    return before, after


def test_humans_who_say_the_judged_fixes_still_fail_reject_the_fix(root, tmp_path, capsys):
    _register(root)
    before, after = _judged_fix(tmp_path)
    _label_after(after, range(2, 12), "FAIL", TARGET, "alice")
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_DOMAIN_ERROR
    out = capsys.readouterr().err
    assert f"improve {TARGET!r}: REJECTED" in out
    assert "0 discordant pair(s)" in out
    assert "10 judge-scored flip(s) adjudicated by a human" in out
    assert "10 overturned" in out
    (record,) = load_records_of(root, "improve")
    assert record.verdict == "rejected"
    assert record.inputs["n_adjudication_overturned"] == "10"


def test_a_human_fail_for_another_mode_does_not_adjudicate_a_flip(root, tmp_path, capsys):
    _register(root)
    before, after = _judged_fix(tmp_path)
    _label_after(after, range(2, 12), "FAIL", "other_mode", "alice")
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "AdjudicationRequiredError" in err
    assert "about another failure mode ('other_mode')" in err
    assert load_records_of(root, "improve") == []


def test_a_human_label_in_the_before_store_does_not_adjudicate_the_fix(root, tmp_path, capsys):
    _register(root)
    before, after = _judged_fix(tmp_path)
    _label_after(before, range(2, 12), "PASS", None, "alice")
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_DOMAIN_ERROR
    assert "no human label for 10 trace(s)" in capsys.readouterr().err


def test_raters_who_disagree_on_a_flip_are_refused(root, tmp_path, capsys):
    _register(root)
    before, after = _judged_fix(tmp_path)
    _label_after(after, range(2, 12), "PASS", None, "alice")
    _label_after(after, range(2, 3), "FAIL", TARGET, "bob")
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert f"the human raters disagree about {TARGET!r} on 1 judge-scored flip(s)" in err
    assert "'T002' (alice PASS, bob FAIL)" in err


def test_a_raters_newer_label_replaces_their_older_one(root, tmp_path, capsys):
    _register(root)
    before, after = _judged_fix(tmp_path)
    _label_after(after, range(2, 12), "FAIL", TARGET, "alice")
    _label_after(after, range(2, 12), "PASS", None, "alice")
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_OK


def _stamped_store(path: Path, captured_at: str) -> Path:
    from calibrated_agent_evaluation._model import replace

    with LocalBackend(root=path) as backend:
        for t in backend.query():
            backend.put_trace(replace(t, captured_at=captured_at))
    return path


def test_an_after_run_captured_before_the_target_was_frozen_is_refused(root, tmp_path, capsys):
    before, after = _fixed(tmp_path)
    _stamped_store(after, "2026-01-01T00:00:00+00:00")
    _register(root)
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "LedgerIntegrityError" in err
    assert "after the after run began (2026-01-01T00:00:00+00:00)" in err
    assert load_records_of(root, "improve") == []


def test_the_freeze_and_after_run_times_are_recorded(root, tmp_path, capsys):
    _register(root)
    before, after = _fixed(tmp_path)
    _stamped_store(after, "2999-01-01T00:00:00+00:00")
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_OK
    assert "before the after run began at 2999-01-01 00:00:00+00:00" in capsys.readouterr().out
    (record,) = load_records_of(root, "improve")
    assert record.inputs["after_run_started_at"] == "2999-01-01T00:00:00+00:00"
    assert record.inputs["prereg_frozen_at"] == record.prereg.frozen_at
    assert record.inputs["ledger_head"] == record.prereg.content_hash


def test_an_after_run_without_times_is_evaluated_and_says_the_order_was_not_checked(
    root, tmp_path, capsys
):
    _register(root)
    before, after = _fixed(tmp_path)
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_OK
    assert "not checked against the after run: no trace in" in capsys.readouterr().out
    (record,) = load_records_of(root, "improve")
    assert record.inputs["after_run_started_at"] == "not recorded"


def test_the_unchecked_order_is_in_the_summary_without_explain(root, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_EXPLAIN", raising=False)
    _register(root)
    before, after = _fixed(tmp_path)
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_OK
    out = capsys.readouterr().out
    assert "Not checked: whether the target was registered before the after run began" in out
    assert "--explain" in out


_SWEBENCH_ROW = Path(__file__).resolve().parent.parent / "fixtures/raw/hf/swebench-session.jsonl"


def _ingest_convention_trace(store: Path) -> Any:
    from calibrated_agent_evaluation.ingest.core import run_ingest

    with LocalBackend(root=store) as backend:
        (result,) = run_ingest([_SWEBENCH_ROW], backend=backend).results
        assert result.trace_id is not None
        for mode in (TARGET, GUARD):
            backend.put_score(
                Label(
                    trace_id=result.trace_id,
                    span_id=None,
                    kind="llm-judge",
                    outcome="PASS",
                    critique="v",
                    failure_mode=mode,
                )
            )
        return backend.get_trace(result.trace_id)


def test_a_convention_ingested_after_run_is_checked_against_the_freeze(root, tmp_path, capsys):
    """The swebench row was captured in April 2026, before a target registered now."""
    before, after = _fixed(tmp_path)
    _ingest_convention_trace(before)
    ingested = _ingest_convention_trace(after)
    _register(root)
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "LedgerIntegrityError" in err
    assert f"after the after run began ({ingested.spans[0].started_at})" in err
    assert load_records_of(root, "improve") == []


def test_a_convention_ingested_start_passes_a_freeze_that_precedes_it(tmp_path):
    from datetime import datetime, timedelta

    from calibrated_agent_evaluation.improve import PreRegEntry, freeze, require_frozen_before

    started = datetime.fromisoformat(
        _ingest_convention_trace(tmp_path / "after").spans[0].started_at
    )
    entry = PreRegEntry(prereg_id="p", target_metric=TARGET, success_threshold=0.1)
    require_frozen_before(freeze(entry, now=started - timedelta(days=1)), started)
    with pytest.raises(LedgerIntegrityError, match="after the after run began"):
        require_frozen_before(freeze(entry, now=started), started)


def test_a_judge_rerun_on_the_after_side_counts_its_newest_verdict_once(root, tmp_path, capsys):
    _register(root)
    before = _store(tmp_path / "before", {TARGET: set(range(12)), GUARD: {30, 31}})
    after = tmp_path / "after"
    with LocalBackend(root=after) as backend:
        for i in range(N):
            tid = f"T{i:03d}"
            backend.put_trace(trace(tid))
            for run, failing in (("first", set(range(12))), ("second", {0, 1})):
                for mode, fails in ((TARGET, failing), (GUARD, {30, 31})):
                    backend.put_score(
                        Label(
                            trace_id=tid,
                            span_id=None,
                            kind="llm-judge",
                            outcome="FAIL" if i in fails else "PASS",
                            critique=f"{run} run",
                            failure_mode=mode,
                            rater_id="judge@v1",
                        )
                    )
    _adjudicate(after, range(2, 12))
    capsys.readouterr()
    assert _evaluate(root, before, after, _development(tmp_path)) == EXIT_OK
    out = capsys.readouterr().out
    assert f"improve {TARGET!r}: ACCEPTED" in out
    assert "10 discordant pair(s)" in out
