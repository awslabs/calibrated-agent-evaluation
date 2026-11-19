# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation prefer`: import pairwise preferences into the store, then rank one
subjective dimension.

The store is the real local store seeded by `calibrated-agent-evaluation quickstart` (the OrderBot
and ClaimsCrew corpus traces). Ratings are Bradley-Terry with Wald intervals, tracked and never
gated.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.subjective import PreferencePair, bradley_terry

ATOMS = {
    "criterion": "claim letter",
    "objective": [{"name": "cites_exclusion"}],
    "subjective": [{"name": "tone"}, {"name": "clarity"}],
}
# A connected, two-way win graph: 4471 wins most, 4477 least, every trace both won and lost.
WINS = [
    ("CLM-4471", "CLM-4473"),
    ("CLM-4471", "CLM-4475"),
    ("CLM-4471", "CLM-4477"),
    ("CLM-4473", "CLM-4475"),
    ("CLM-4473", "CLM-4477"),
    ("CLM-4475", "CLM-4477"),
    ("CLM-4477", "CLM-4471"),
    ("CLM-4475", "CLM-4473"),
    ("CLM-4471", "CLM-4473"),
]


@pytest.fixture
def root(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    store = tmp_path / "store"
    assert main(["--root", str(store), "quickstart"]) == EXIT_OK
    capsys.readouterr()
    return store


@pytest.fixture
def atoms(tmp_path) -> Path:
    path = tmp_path / "atoms.json"
    path.write_text(json.dumps(ATOMS))
    return path


def _line(a: str, b: str, winner: str = "a", dimension: str = "tone", kind: str = "human") -> str:
    return json.dumps(
        {
            "trace_id_a": a,
            "trace_id_b": b,
            "dimension": dimension,
            "winner": winner,
            "judge_kind": kind,
        }
    )


def _pairs(tmp_path: Path, lines: list[str], name: str = "pairs.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


def _wins(kind: str = "human", dimension: str = "tone") -> list[str]:
    # Alternate sides so winner "a" and "b" both appear.
    return [
        _line(w, lose, "a", dimension, kind) if i % 2 else _line(lose, w, "b", dimension, kind)
        for i, (w, lose) in enumerate(WINS)
    ]


def _prefer(root: Path, *argv: str) -> int:
    return main(["--root", str(root), "prefer", *argv])


def test_imported_preferences_are_stored_and_a_second_import_adds_nothing(
    root, atoms, tmp_path, capsys
):
    pairs = _pairs(tmp_path, _wins())
    assert _prefer(root, "--atoms", str(atoms), "--pairs", str(pairs)) == EXIT_OK
    out = capsys.readouterr().out
    assert "Imported 9 preference(s) on 'tone'" in out
    assert "calibrated-agent-evaluation prefer --rank tone" in out
    stored = load_records_of(root, "preference")
    assert len(stored) == 9
    assert {r.source for r in stored} == {"file"}
    assert stored[0].source_ref.endswith("pairs.jsonl:1")

    assert _prefer(root, "--atoms", str(atoms), "--pairs", str(pairs)) == EXIT_OK
    assert "9 were already imported" in capsys.readouterr().out
    assert len(load_records_of(root, "preference")) == 9


@pytest.mark.parametrize(
    ("line", "match"),
    [
        (_line("CLM-4471", "CLM-4473", dimension="cites_exclusion"), "is objective"),
        (_line("CLM-4471", "CLM-4473", dimension="warmth"), "not found"),
        (_line("CLM-4471", "CLM-4471"), "itself"),
        (_line("CLM-4471", "CLM-4473", winner="tie"), "winner"),
    ],
)
def test_a_bad_preference_refuses_the_file_and_stores_nothing(
    root, atoms, tmp_path, capsys, line, match
):
    pairs = _pairs(tmp_path, [_line("CLM-4471", "CLM-4473"), line])
    assert _prefer(root, "--atoms", str(atoms), "--pairs", str(pairs)) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "line 2" in err
    assert match in err
    assert load_records_of(root, "preference") == []


def test_a_trace_not_in_the_store_refuses_the_import(root, atoms, tmp_path, capsys):
    pairs = _pairs(tmp_path, [_line("CLM-4471", "CLM-9999")])
    assert _prefer(root, "--atoms", str(atoms), "--pairs", str(pairs)) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "'CLM-9999'" in err
    assert "not in the store" in err
    assert load_records_of(root, "preference") == []


def test_import_needs_an_atom_set_and_exactly_one_mode(root, atoms, tmp_path, capsys):
    pairs = _pairs(tmp_path, _wins())
    assert _prefer(root, "--pairs", str(pairs)) == EXIT_USAGE
    assert "--pairs needs --atoms" in capsys.readouterr().err
    assert _prefer(root) == EXIT_USAGE
    assert "exactly one of" in capsys.readouterr().err
    assert _prefer(root, "--pairs", str(pairs), "--rank", "tone") == EXIT_USAGE
    assert "exactly one of" in capsys.readouterr().err


def test_rank_prints_the_bradley_terry_fit_with_intervals_tracked_never_gated(
    root, atoms, tmp_path, capsys, monkeypatch
):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_EXPLAIN")
    _prefer(root, "--atoms", str(atoms), "--pairs", str(_pairs(tmp_path, _wins())))
    capsys.readouterr()
    assert _prefer(root, "--rank", "tone") == EXIT_OK
    out = capsys.readouterr().out
    assert "tracked, never gated" in out
    assert "95% interval" in out
    assert "Elo" not in out  # sequential Elo is display-only, shown under --explain

    oracle = bradley_terry("tone", [r.pair() for r in load_records_of(root, "preference")])
    ranked = [
        line.split()[1] for line in out.splitlines() if line.strip()[:2] in {"1.", "2.", "3.", "4."}
    ]
    assert ranked == [r.trace_id for r in oracle.ratings]
    top = oracle.ratings[0].rating
    assert f"{top.point:.0f} (95% interval {top.lo:.0f} to {top.hi:.0f})" in out


def test_explain_adds_the_method_and_the_order_dependent_elo(root, atoms, tmp_path, capsys):
    _prefer(root, "--atoms", str(atoms), "--pairs", str(_pairs(tmp_path, _wins())))
    capsys.readouterr()
    assert main(["--root", str(root), "--explain", "prefer", "--rank", "tone"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "bradley_terry_mle_wald" in out
    assert "sequential Elo, display only" in out
    assert "no interval" in out


def test_human_and_llm_judge_preferences_are_ranked_separately(root, atoms, tmp_path, capsys):
    lines = _wins() + _wins(kind="llm-judge")
    _prefer(root, "--atoms", str(atoms), "--pairs", str(_pairs(tmp_path, lines)))
    capsys.readouterr()
    assert _prefer(root, "--rank", "tone") == EXIT_OK
    out = capsys.readouterr().out
    assert "human preferences (9 comparison(s))" in out
    assert "llm-judge preferences (9 comparison(s))" in out
    assert "never pooled" in out


def test_an_unbeaten_trace_is_not_ranked_and_exits_1(root, atoms, tmp_path, capsys):
    lines = [_line("CLM-4471", "CLM-4473"), _line("CLM-4471", "CLM-4475")]
    _prefer(root, "--atoms", str(atoms), "--pairs", str(_pairs(tmp_path, lines)))
    capsys.readouterr()
    assert _prefer(root, "--rank", "tone") == EXIT_DOMAIN_ERROR
    assert "not strongly connected" in capsys.readouterr().err


def test_rank_refuses_an_objective_or_empty_dimension(root, atoms, tmp_path, capsys):
    assert _prefer(root, "--rank", "tone") == EXIT_DOMAIN_ERROR
    assert "no stored preference on 'tone'" in capsys.readouterr().err
    assert _prefer(root, "--rank", "cites_exclusion", "--atoms", str(atoms)) == EXIT_DOMAIN_ERROR
    assert "objective atom" in capsys.readouterr().err
    assert _prefer(root, "--rank", "warmth", "--atoms", str(atoms)) == EXIT_DOMAIN_ERROR
    assert "not declared" in capsys.readouterr().err


def test_the_wald_intervals_rank_reports_cover_the_true_rating_about_95_percent() -> None:
    """Coverage of the 95% Wald interval the rank command prints: 400 reps, seed 20261026.

    Four traces with known log-strengths, every pair compared 25 times per rep, each fit checked
    against the true rating on the same sum-to-zero Elo scale. Observed: 1539 of 1600 (0.962).
    """
    rng = random.Random(20261026)
    theta = [0.6, 0.2, -0.1, -0.7]
    mean = sum(theta) / len(theta)
    truth = {f"t{i}": 1500.0 + 400.0 / math.log(10.0) * (t - mean) for i, t in enumerate(theta)}
    covered = total = 0
    for _ in range(400):
        pairs = []
        for i in range(4):
            for j in range(i + 1, 4):
                p = 1.0 / (1.0 + math.exp(theta[j] - theta[i]))
                for _ in range(25):
                    winner = "a" if rng.random() < p else "b"
                    pairs.append(PreferencePair(f"t{i}", f"t{j}", "tone", winner, "human"))
        fit = bradley_terry("tone", pairs)
        for rating in fit.ratings:
            total += 1
            covered += rating.rating.lo <= truth[rating.trace_id] <= rating.rating.hi
    assert 0.93 <= covered / total <= 0.97
