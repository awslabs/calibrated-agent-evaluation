# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CLI's default output is plain language; --explain adds the statistics behind it.

Covers the presenter on its own (no CLI), the plain summary builders, and the real commands on the
bundled corpus with and without --explain.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from calibrated_agent_evaluation.calibrate.errors import RateSampleTooSmallError
from calibrated_agent_evaluation.cli import plain
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, StageOutcome, main
from calibrated_agent_evaluation.cli.present import EXPLAIN_HINT, Presenter


@pytest.fixture(autouse=True)
def _plain_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_EXPLAIN", raising=False)


def _presenter(*, explain: bool) -> tuple[Presenter, io.StringIO, io.StringIO]:
    out, err = io.StringIO(), io.StringIO()
    return (
        Presenter(out=out, err=err, explain=explain, explainers=plain.default_error_explainers()),
        out,
        err,
    )


def test_the_summary_prints_and_the_detail_waits_for_explain() -> None:
    outcome = StageOutcome(stage="run", exit_code=0, summary="Gate passed.", detail="lb 0.91")
    presenter, out, _ = _presenter(explain=False)
    presenter.outcome("run", outcome, ok=True)
    assert out.getvalue() == f"Gate passed.\n{EXPLAIN_HINT}\n"

    presenter, out, _ = _presenter(explain=True)
    presenter.outcome("run", outcome, ok=True)
    assert out.getvalue() == "Gate passed.\n\nlb 0.91\n"


def test_an_outcome_without_detail_prints_no_hint() -> None:
    presenter, out, _ = _presenter(explain=False)
    presenter.outcome("ingest", StageOutcome(stage="ingest", exit_code=0, summary="done"), ok=True)
    assert out.getvalue() == "done\n"


def test_a_failed_outcome_is_prefixed_with_the_command_once() -> None:
    presenter, _, err = _presenter(explain=False)
    presenter.outcome(
        "run", StageOutcome(stage="run", exit_code=1, summary="Gate failed."), ok=False
    )
    assert err.getvalue() == "calibrated-agent-evaluation run: Gate failed.\n"

    presenter, _, err = _presenter(explain=False)
    already = StageOutcome(
        stage="ingest",
        exit_code=1,
        summary="calibrated-agent-evaluation ingest: 0 ingested, 1 refused",
    )
    presenter.outcome("ingest", already, ok=False)
    assert err.getvalue() == "calibrated-agent-evaluation ingest: 0 ingested, 1 refused\n"


def test_a_known_error_is_plain_with_a_next_step_and_its_type_only_under_explain() -> None:
    exc = RateSampleTooSmallError("rate sample of 40 is below 150", n=40, minimum=150)
    presenter, _, err = _presenter(explain=False)
    presenter.error("run", exc)
    text = err.getvalue()
    assert text.startswith(
        "calibrated-agent-evaluation run: Too few scored traces to estimate a failure rate: 40"
    )
    assert "Next: score 110 more production trace(s)" in text
    assert "(Run with --explain for the details.)" in text
    assert "RateSampleTooSmallError" not in text
    assert "below 150" not in text

    presenter, _, err = _presenter(explain=True)
    presenter.error("run", exc)
    assert "RateSampleTooSmallError: rate sample of 40 is below 150" in err.getvalue()


def test_an_unexplained_error_shows_its_message_and_its_type_only_under_explain() -> None:
    """The class name is for a bug report, not for the reader of a refusal."""
    presenter, _, err = _presenter(explain=False)
    presenter.error("ingest", ValueError("bad file"))
    assert err.getvalue() == "calibrated-agent-evaluation ingest: bad file\n"
    presenter, _, err = _presenter(explain=True)
    presenter.error("ingest", ValueError("bad file"))
    assert err.getvalue() == "calibrated-agent-evaluation ingest: ValueError: bad file\n"


def _record(result: str, *, tpr: float, tnr: float, n: int, below_floor: bool = False) -> object:
    return SimpleNamespace(
        result=result,
        tpr=SimpleNamespace(point=tpr),
        tnr=SimpleNamespace(point=tnr),
        n_pos=n,
        n_neg=n,
        below_coverage_floor=below_floor,
        coverage_floor=162,
    )


def test_a_refused_calibration_says_how_many_labels_are_missing() -> None:
    lines = plain.calibration(
        _record("REFUSED", tpr=1.0, tnr=1.0, n=1),
        plan_per_class=55,
        judge_path="judge.json",
        gold="2 gold label(s)",
    )
    assert lines[0] == "Not enough labels yet to trust this judge (checked on 2 gold label(s))."
    assert "label 54 more FAIL and 54 more PASS traces" in lines[1]


def test_a_certified_but_rate_unsafe_judge_says_what_a_rate_needs() -> None:
    lines = plain.calibration(
        _record("CERTIFIED", tpr=0.95, tnr=0.97, n=60, below_floor=True),
        plan_per_class=55,
        judge_path="judge.json",
        gold="120 gold label(s)",
    )
    assert lines[0].startswith("Judge certified on 120 gold label(s)")
    assert "caught 95% of the failures and cleared 97% of the passes" in lines[0]
    assert "label 102 more FAIL and 102 more PASS traces" in lines[1]
    assert "calibrated-agent-evaluation bias --judge judge.json" in lines[2]


def test_synthetic_gold_is_named_as_synthetic() -> None:
    assert plain.gold_note(["synthetic-rule-oracle"] * 3) == (
        "3 gold label(s) (synthetic, from the sample policy rule)"
    )
    assert plain.gold_note(["alice", "bob"]) == "2 gold label(s)"
    assert plain.gold_note(["alice", "synthetic-x"]) == (
        "2 gold label(s) (1 of them synthetic, rater 'synthetic-x')"
    )


def test_a_biased_judge_is_described_by_what_it_does() -> None:
    record = SimpleNamespace(verdict="BIASED", failed_biases=("length", "sycophancy"))
    lines = plain.bias(record, judge_path="judge.json")
    assert lines[0] == (
        "Bias check failed: the judge changes its verdict when answers get longer and goes "
        "along with claims asserted in the conversation."
    )


def test_a_rate_reads_as_a_percentage_with_its_range() -> None:
    assert plain.rate_range(0.112, 0.062, 0.148) == "11% (likely between 6% and 15%)"
    assert plain.rate_range(None, 0.0, 0.2) == "somewhere between 0% and 20%"


def test_run_on_the_corpus_is_plain_by_default_and_detailed_with_explain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = str(tmp_path / "store")
    assert main(["--root", root, "quickstart"]) == EXIT_OK
    capsys.readouterr()

    assert main(["--root", root, "run", "--verdicts", "human", "--ci"]) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "CI gate failed: 4 of 4 failure mode(s) don't clear the bar yet" in err
    assert "at least 80% of each mode's traces must pass" in err
    assert EXPLAIN_HINT in err
    assert "lower bound" not in err

    assert main(["--root", root, "run", "--verdicts", "human", "--ci", "--explain"]) == 1
    err = capsys.readouterr().err
    assert "pass-rate 95% lower bound: FAILED" in err


def test_explain_is_accepted_before_the_subcommand_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = str(tmp_path / "store")
    assert main(["--root", root, "quickstart"]) == EXIT_OK
    capsys.readouterr()
    assert main(["--explain", "--root", root, "cluster"]) == EXIT_OK
    assert "Good-Turing coverage" in capsys.readouterr().out


def test_calibrated_agent_evaluation_explain_env_turns_explain_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = str(tmp_path / "store")
    assert main(["--root", root, "quickstart"]) == EXIT_OK
    capsys.readouterr()
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_EXPLAIN", "1")
    assert main(["--root", root, "cluster"]) == EXIT_OK
    assert "Good-Turing coverage" in capsys.readouterr().out
