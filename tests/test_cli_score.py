# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation score` writes code-scorer verdicts the run gate reads with
`--verdicts scorer`.

Runs the real autoevals scorers over a local store: expected values come from the human FAIL
labels for the mode or from a JSONL file, a trace without one is counted and named, and the
verdicts carry the mode and a rater id naming the scorer and its parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.schema.types import Label
from cli_store_helpers import trace

MODE = "wrong_reply"


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)


def _store(root: Path, n: int, *, expected: dict | None = None, labeled: int | None = None) -> None:
    """``n`` traces replying ``done``; the first ``labeled`` carry a human FAIL and ``expected``."""
    with LocalBackend(root=root) as backend:
        for i in range(n):
            t = trace(f"S{i:03d}")
            backend.put_trace(t)
            if labeled is None or i < labeled:
                backend.put_score(
                    Label(
                        trace_id=t.trace_id,
                        span_id=None,
                        kind="human",
                        outcome="FAIL",
                        critique="wrong reply",
                        failure_mode=MODE,
                        expected=expected if i % 2 == 0 else {"text": "done"},
                        rater_id="ana",
                    )
                )


def _code(root: Path) -> list[Label]:
    with LocalBackend(root=root) as backend:
        return [
            lb
            for t in backend.query()
            for lb in backend.get_scores(t.trace_id)
            if lb.kind == "code"
        ]


def _score(root: Path, *argv: str) -> int:
    return main(["--root", str(root), "score", *argv])


def test_exact_scores_against_the_human_expected_and_run_gates_them(tmp_path, capsys):
    root = tmp_path / "store"
    _store(root, 40, expected={"text": "refunded"})
    assert _score(root, "--scorer", "exact", "--mode", MODE) == EXIT_OK
    out = capsys.readouterr().out
    assert f"Scored 40 trace(s) for {MODE} with the exact scorer: 20 FAIL, 20 PASS." in out
    assert "calibrated-agent-evaluation run --verdicts scorer" in out
    verdicts = _code(root)
    assert len(verdicts) == 40
    assert {lb.failure_mode for lb in verdicts} == {MODE}
    (rater,) = {lb.rater_id for lb in verdicts}
    assert rater is not None
    assert rater.startswith("scorer:exact@")

    assert main(["--root", str(root), "run", "--verdicts", "scorer"]) == EXIT_OK
    assert "code-scorer verdicts" in capsys.readouterr().out


def test_other_parameters_are_another_rater(tmp_path, capsys):
    root = tmp_path / "store"
    _store(root, 4, expected={"text": "dune"})
    assert _score(root, "--scorer", "levenshtein", "--mode", MODE) == EXIT_OK
    assert _score(root, "--scorer", "levenshtein", "--mode", MODE, "--threshold", "0.5") == EXIT_OK
    raters = {lb.rater_id for lb in _code(root)}
    assert len(raters) == 2


def test_traces_without_an_expected_value_are_counted_and_named(tmp_path, capsys):
    root = tmp_path / "store"
    _store(root, 10, expected={"text": "refunded"}, labeled=4)
    assert _score(root, "--scorer", "exact", "--mode", MODE) == EXIT_OK
    out = capsys.readouterr().out
    assert "6 trace(s) in scope had no expected value" in out
    assert "not scored: 6 trace(s) with no expected value for this mode: 'S004'" in out
    assert len(_code(root)) == 4


def test_raters_who_recorded_different_expected_values_are_not_scored(tmp_path, capsys):
    root = tmp_path / "store"
    _store(root, 2, expected={"text": "refunded"})
    with LocalBackend(root=root) as backend:
        backend.put_score(
            Label(
                trace_id="S000",
                span_id=None,
                kind="human",
                outcome="FAIL",
                critique="wrong",
                failure_mode=MODE,
                expected={"text": "other"},
                rater_id="ben",
            )
        )
    assert _score(root, "--scorer", "exact", "--mode", MODE) == EXIT_OK
    assert "raters recorded different expected values" in capsys.readouterr().out
    assert [lb.trace_id for lb in _code(root)] == ["S001"]


def test_no_expected_value_in_scope_is_refused_and_writes_nothing(tmp_path, capsys):
    root = tmp_path / "store"
    _store(root, 3, labeled=0)
    assert _score(root, "--scorer", "exact", "--mode", MODE) == EXIT_DOMAIN_ERROR
    assert "nothing to compare its output with" in capsys.readouterr().err
    assert _code(root) == []


def test_expected_values_from_a_file(tmp_path, capsys):
    root = tmp_path / "store"
    _store(root, 3, labeled=0)
    path = tmp_path / "expected.jsonl"
    rows = [
        {"trace_id": "S000", "expected": {"text": "done"}},
        {"trace_id": "S001", "expected": {"text": "nope"}},
        {"trace_id": "ELSEWHERE", "expected": {"text": "done"}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert _score(root, "--scorer", "exact", "--mode", MODE, "--expected", str(path)) == EXIT_OK
    out = capsys.readouterr().out
    assert "1 FAIL, 1 PASS" in out
    assert "1 trace id(s) in" in out
    assert "1 trace(s) in scope had no expected value" in out


@pytest.mark.parametrize(
    "line",
    ["not json", '{"trace_id": "S000"}', '{"trace_id": 1, "expected": {}}', "[1]"],
)
def test_a_malformed_expected_file_is_refused(tmp_path, capsys, line):
    root = tmp_path / "store"
    _store(root, 1, labeled=0)
    path = tmp_path / "expected.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    assert _score(root, "--scorer", "exact", "--mode", MODE, "--expected", str(path)) == (
        EXIT_DOMAIN_ERROR
    )
    assert "--expected" in capsys.readouterr().err
    assert _code(root) == []


def test_valid_json_needs_no_expected_value(tmp_path, capsys):
    root = tmp_path / "store"
    _store(root, 2, labeled=0)
    assert _score(root, "--scorer", "valid-json", "--mode", MODE, "--key", "text") == EXIT_OK
    assert "2 FAIL, 0 PASS" in capsys.readouterr().out


def test_a_threshold_on_a_pass_fail_scorer_is_refused(tmp_path, capsys):
    root = tmp_path / "store"
    _store(root, 1)
    assert _score(root, "--scorer", "exact", "--mode", MODE, "--threshold", "0.5") == (EXIT_USAGE)
    assert "--threshold applies to" in capsys.readouterr().err


def test_scoring_before_any_human_label_warns_that_error_analysis_comes_first(tmp_path, capsys):
    root = tmp_path / "store"
    _store(root, 2, labeled=0)
    path = tmp_path / "expected.jsonl"
    path.write_text(json.dumps({"trace_id": "S000", "expected": {"text": "done"}}) + "\n")
    assert _score(root, "--scorer", "exact", "--mode", MODE, "--expected", str(path)) == EXIT_OK
    assert "error analysis" in capsys.readouterr().out


def test_a_missing_key_in_the_output_is_a_typed_refusal(tmp_path, capsys):
    root = tmp_path / "store"
    _store(root, 2, expected={"answer": "x"})
    rc = _score(root, "--scorer", "exact", "--mode", MODE, "--key", "answer")
    assert rc == EXIT_DOMAIN_ERROR
    assert "not present in the output" in capsys.readouterr().err
    assert _code(root) == []


def test_one_trace_and_a_frame_narrow_the_scope(tmp_path, capsys):
    root = tmp_path / "store"
    _store(root, 4, expected={"text": "refunded"})
    assert _score(root, "--scorer", "exact", "--mode", MODE, "--trace", "S002") == EXIT_OK
    assert [lb.trace_id for lb in _code(root)] == ["S002"]
    assert _score(root, "--scorer", "exact", "--mode", MODE, "--frame", "discovery") == (
        EXIT_DOMAIN_ERROR
    )
    assert "no stored traces to score" in capsys.readouterr().err
