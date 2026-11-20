# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What `calibrated-agent-evaluation run` refuses to gate: a generic metric's verdicts in CI, and a
subjective dimension.

The scorer behind each code verdict is its `rater_id` lane (`scorer:<name>@<params>`); the run
records the lanes it gated and, under `--ci`, refuses a lane whose scorer is a generic
generation-quality metric. `--atoms FILE` declares which dimensions are objective (gated) and which
are subjective (tracked); a gated mode must be an objective atom.
"""

from __future__ import annotations

import json

import pytest

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
from calibrated_agent_evaluation.cli.records import NO_LEDGER_HEAD, load_records_of
from cli_store_helpers import admit_default_judge, seed_verdicts


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    return tmp_path / "store"


def _run(root, *argv: str) -> int:
    return main(["--root", str(root), "run", "--n-resamples", "199", *argv])


def _scored(root, lane: str | None) -> None:
    seed_verdicts(root, 48, 2, kind="code", rater_id=lane)


def test_a_generic_metric_lane_is_refused_in_ci_and_nothing_is_recorded(root, capsys):
    _scored(root, "scorer:rouge@1a2b3c")
    assert _run(root, "--ci", "--verdicts", "scorer") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "'rouge', a generic generation-quality metric" in err
    assert "cannot gate CI" in err
    assert "ENF" not in err
    assert load_records_of(root, "run") == []


def test_outside_ci_a_generic_metric_is_gated_but_labelled_exploration_only(root, capsys):
    _scored(root, "scorer:bertscore@9f")
    assert _run(root, "--verdicts", "scorer") == EXIT_OK
    assert "exploration-only, it cannot gate CI" in capsys.readouterr().out
    (record,) = load_records_of(root, "run")
    assert record.scorer_lanes == ("scorer:bertscore@9f",)
    (decision,) = record.decisions
    assert decision.tier == "display_default"
    assert decision.inputs["scorer"] == "bertscore"


def test_a_structured_scorer_gates_ci_and_its_lane_is_recorded(root, capsys):
    _scored(root, "scorer:exact@0d0d")
    assert _run(root, "--ci", "--verdicts", "scorer", "--explain") == EXIT_OK
    assert "scorer lanes gated: scorer:exact@0d0d" in capsys.readouterr().out
    (record,) = load_records_of(root, "run")
    assert record.scorer_lanes == ("scorer:exact@0d0d",)
    assert record.decisions == ()


def test_code_verdicts_with_no_recorded_scorer_gate_with_a_note(root, capsys):
    _scored(root, None)
    assert _run(root, "--ci", "--verdicts", "scorer", "--explain") == EXIT_OK
    assert "record no scorer" in capsys.readouterr().out
    (record,) = load_records_of(root, "run")
    assert record.scorer_lanes == ("unrecorded",)


def test_judge_verdicts_record_no_scorer_lanes(root, capsys):
    seed_verdicts(root, 48, 2)
    admit_default_judge(root)
    assert _run(root, "--ci") == EXIT_OK
    (record,) = load_records_of(root, "run")
    assert record.scorer_lanes == ()


def _atoms(tmp_path, *, objective=(), subjective=()) -> str:
    path = tmp_path / "atoms.json"
    doc = {
        "criterion": "refund handling",
        "objective_atoms": [{"name": n} for n in objective],
        "subjective_dimensions": [{"name": n, "description": "how it reads"} for n in subjective],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def test_an_objective_atom_is_gated(root, tmp_path, capsys):
    seed_verdicts(root, 48, 2)
    admit_default_judge(root)
    atoms = _atoms(tmp_path, objective=("over_refund",), subjective=("tone",))
    assert _run(root, "--ci", "--atoms", atoms) == EXIT_OK
    (record,) = load_records_of(root, "run")
    assert record.inputs["atoms_file"] == atoms


def test_a_subjective_dimension_cannot_be_gated(root, tmp_path, capsys):
    seed_verdicts(root, 48, 2)
    admit_default_judge(root)
    atoms = _atoms(tmp_path, objective=("tone",), subjective=("over_refund",))
    assert _run(root, "--ci", "--atoms", atoms) == EXIT_DOMAIN_ERROR
    assert "'over_refund' is subjective" in capsys.readouterr().err
    assert load_records_of(root, "run") == []


def test_a_mode_the_atoms_file_does_not_declare_is_refused(root, tmp_path, capsys):
    seed_verdicts(root, 48, 2)
    admit_default_judge(root)
    atoms = _atoms(tmp_path, objective=("wrong_item",))
    assert _run(root, "--atoms", atoms) == EXIT_DOMAIN_ERROR
    assert "'over_refund' not found in AtomSet" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "is not valid JSON"),
        (json.dumps({"objective_atoms": []}), "is not an AtomSet (criterion: Field required)"),
        (
            json.dumps(
                {
                    "criterion": "c",
                    "objective_atoms": [{"name": "x"}],
                    "subjective_dimensions": [{"name": "x"}],
                }
            ),
            "appears in both objective and subjective lanes",
        ),
    ],
)
def test_a_bad_atoms_file_is_refused(root, tmp_path, content, message, capsys):
    seed_verdicts(root, 48, 2)
    path = tmp_path / "atoms.json"
    path.write_text(content, encoding="utf-8")
    assert _run(root, "--atoms", str(path)) == EXIT_DOMAIN_ERROR
    assert message in capsys.readouterr().err


def test_a_missing_atoms_file_is_refused(root, tmp_path, capsys):
    seed_verdicts(root, 48, 2)
    assert _run(root, "--atoms", str(tmp_path / "nope.json")) == EXIT_DOMAIN_ERROR
    assert "cannot read the atoms file" in capsys.readouterr().err


def test_every_run_record_carries_the_ledger_head(root, capsys):
    seed_verdicts(root, 48, 2)
    admit_default_judge(root)
    assert _run(root) == EXIT_OK
    assert main(["--root", str(root), "improve", "--register", "over_refund",
                 "--min-reduction", "0.05"]) == EXIT_OK  # fmt: skip
    assert _run(root) == EXIT_OK
    from calibrated_agent_evaluation.cli.records import load_ledger

    before, after = load_records_of(root, "run")
    assert (before.inputs["ledger_head"], before.inputs["ledger_entries"]) == (NO_LEDGER_HEAD, "0")
    head = load_ledger(root).entries[-1].content_hash
    assert (after.inputs["ledger_head"], after.inputs["ledger_entries"]) == (head, "1")
