# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation label --agreement`: the gold-set agreement gate, run from the CLI on
a real local store.

Each test seeds a LocalBackend with two raters' labels on the same traces, runs the real CLI, and
reads the stored `agreement` record back through the CLI's own record codec.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.schema import Label
from cli_store_helpers import trace


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_EXPLAIN", raising=False)
    return tmp_path / "store"


def _label(trace_id: str, rater: str, outcome: str, version: str | None = "1", **kw) -> Label:
    return Label(
        trace_id=trace_id,
        span_id=None,
        kind="human",
        outcome=outcome,  # type: ignore[arg-type]
        critique=f"{rater} on {trace_id}",
        rater_id=rater,
        codebook_version=version,
        **kw,
    )


def seed(
    root: Path,
    n: int,
    n_fail: int,
    n_disagree: int,
    *,
    version: str | None = "1",
    cell: str = "refund",
) -> list[str]:
    """``n`` traces both raters coded: ``n_fail`` both FAIL, ``n_disagree`` split, rest PASS."""
    ids = [f"{cell}-{i:03d}" for i in range(n)]
    with LocalBackend(root=root) as backend:
        for i, tid in enumerate(ids):
            backend.put_trace(trace(tid).model_copy(update={"taxonomy_cell": {"feature": cell}}))
            a = "FAIL" if i < n_fail + n_disagree else "PASS"
            b = "FAIL" if i < n_fail else "PASS"
            backend.put_score(_label(tid, "alice", a, version))
            backend.put_score(_label(tid, "bob", b, version))
    return ids


def run(root: Path, *argv: str) -> int:
    return main(["--root", str(root), "label", "--agreement", *argv])


def test_raters_who_agree_on_enough_traces_are_usable_and_recorded(root, capsys):
    ids = seed(root, n=140, n_fail=60, n_disagree=0)
    assert run(root, "--raters", "alice,bob", "--codebook-version", "1") == EXIT_OK
    out = capsys.readouterr().out
    assert "the gold set is usable for calibration" in out
    [record] = load_records_of(root, "agreement")
    assert record.gate == "usable"
    assert record.raters == ("alice", "bob")
    assert record.codebook_version == "1"
    assert record.axis == "outcome"
    assert record.trace_ids == tuple(sorted(ids))
    assert record.n_pairable_units == 140
    assert record.minimum_overlap == 130
    assert record.cohen_kappa is not None
    assert set(record.per_class) == {"FAIL", "PASS"}
    assert record.provenance.seed == 0
    assert record.provenance.n_resamples == 2000


@pytest.mark.parametrize(
    ("n", "n_fail", "n_disagree", "gate", "next_step"),
    [
        (20, 6, 1, "top_up_overlap", "label 110 more traces of the calibration gold"),
        (140, 60, 10, "top_up_overlap", "the interval is still too wide"),
        (140, 60, 18, "revise_codebook", "tighten the codebook"),
        (140, 60, 40, "codebook_broken", "rewrite the codebook"),
        (40, 10, 20, "codebook_broken", "rewrite the codebook"),
        (60, 20, 12, "revise_codebook", "tighten the codebook"),
    ],
)
def test_every_other_verdict_exits_1_with_a_next_step_and_is_still_recorded(
    root, capsys, n, n_fail, n_disagree, gate, next_step
):
    seed(root, n=n, n_fail=n_fail, n_disagree=n_disagree)
    assert run(root, "--raters", "alice,bob", "--codebook-version", "1") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert gate.replace("_", " ") in err
    assert next_step in err
    [record] = load_records_of(root, "agreement")
    assert record.gate == gate


def test_perfect_agreement_on_a_small_overlap_is_not_usable(root, capsys):
    seed(root, n=20, n_fail=10, n_disagree=0)
    assert run(root, "--raters", "alice,bob", "--codebook-version", "1") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "alpha 1.00 (could be as low as 1.00) over 20 shared unit(s)" in err
    assert "The gate needs at least 130 traces labeled by two raters" in err
    [record] = load_records_of(root, "agreement")
    assert record.gate == "top_up_overlap"


def test_poor_agreement_on_a_small_overlap_is_conclusive_without_a_top_up(root, capsys):
    seed(root, n=40, n_fail=10, n_disagree=20)
    assert run(root, "--raters", "alice,bob", "--codebook-version", "1") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "not usable yet (codebook broken)" in err
    assert "This holds even though fewer than 130 traces are shared: alpha could be at most" in err
    assert "more traces of the calibration gold" not in err
    [record] = load_records_of(root, "agreement")
    assert record.gate == "codebook_broken"
    assert record.alpha.upper < 0.667


def test_explain_shows_the_intervals_and_the_minimum_overlap(root, capsys):
    from calibrated_agent_evaluation.review import MINIMUM_OVERLAP

    seed(root, n=20, n_fail=6, n_disagree=1)
    run(root, "--raters", "alice,bob", "--codebook-version", "1", "--explain")
    err = capsys.readouterr().err
    assert "Krippendorff's alpha" in err
    assert "95% bootstrap interval" in err
    assert "Cohen's kappa" in err
    assert "specific agreement on 'FAIL'" in err
    assert f"minimum overlap: {MINIMUM_OVERLAP} units" in err
    assert "double-code the calibration gold itself, no extra traces" in err
    assert "seed" in err


def test_without_a_codebook_version_only_unversioned_labels_count(root, capsys):
    seed(root, n=40, n_fail=12, n_disagree=0, version="1")
    assert run(root, "--raters", "alice,bob") == EXIT_DOMAIN_ERROR
    assert "unversioned" in capsys.readouterr().err
    assert load_records_of(root, "agreement") == []


def test_a_relabel_counts_once_with_the_newest_label(root, capsys):
    ids = seed(root, n=140, n_fail=60, n_disagree=1)
    with LocalBackend(root=root) as backend:
        # alice changes her mind on the one split trace: now both say PASS there.
        backend.put_score(_label(ids[60], "alice", "PASS"))
    assert run(root, "--raters", "alice,bob", "--codebook-version", "1") == EXIT_OK
    out = capsys.readouterr().out
    assert "1 older label(s) were replaced by a rater's relabel" in out
    [record] = load_records_of(root, "agreement")
    assert record.n_superseded == 1
    assert record.alpha.point == 1.0


def test_the_failure_mode_axis_compares_modes(root, capsys):
    with LocalBackend(root=root) as backend:
        for i in range(150):
            tid = f"t-{i:03d}"
            backend.put_trace(trace(tid))
            mode = ("wrong_tool", "late_refund", None)[i % 3]
            outcome = "PASS" if mode is None else "FAIL"
            for rater in ("alice", "bob"):
                backend.put_score(_label(tid, rater, outcome, failure_mode=mode))
    argv = ["--raters", "alice,bob", "--codebook-version", "1", "--axis", "failure_mode"]
    assert run(root, *argv) == EXIT_OK
    [record] = load_records_of(root, "agreement")
    assert record.axis == "failure_mode"
    assert set(record.per_class) == {"wrong_tool", "late_refund", "(none)"}


def test_cell_and_trace_narrow_the_scope(root, capsys):
    seed(root, n=140, n_fail=60, n_disagree=0, cell="refund")
    seed(root, n=60, n_fail=10, n_disagree=40, cell="claims")
    assert run(root, "--raters", "alice,bob", "--codebook-version", "1") == EXIT_DOMAIN_ERROR
    argv = ["--raters", "alice,bob", "--codebook-version", "1", "--cell", "feature=refund"]
    assert run(root, *argv) == EXIT_OK
    usable = [r for r in load_records_of(root, "agreement") if r.gate == "usable"]
    assert usable[0].cell == {"feature": "refund"}
    assert all(t.startswith("refund-") for t in usable[0].trace_ids)
    capsys.readouterr()
    one = ["--raters", "alice,bob", "--codebook-version", "1", "--trace", "refund-000"]
    assert run(root, *one) == EXIT_DOMAIN_ERROR  # one unit, one category: alpha is undefined
    assert "undefined" in capsys.readouterr().err


@pytest.mark.parametrize("raters", [None, "alice", "alice,alice", " , "])
def test_fewer_than_two_raters_is_refused(root, capsys, raters):
    seed(root, n=10, n_fail=3, n_disagree=0)
    argv = [] if raters is None else ["--raters", raters]
    assert run(root, *argv) == EXIT_USAGE
    assert "at least two rater ids" in capsys.readouterr().err


def test_a_rater_with_no_labels_is_refused(root, capsys):
    seed(root, n=10, n_fail=3, n_disagree=0)
    assert run(root, "--raters", "carol,dave", "--codebook-version", "1") == EXIT_DOMAIN_ERROR
    assert "no human label" in capsys.readouterr().err


def test_agreement_and_serve_together_is_refused(root, capsys):
    seed(root, n=10, n_fail=3, n_disagree=0)
    assert run(root, "--serve", "--raters", "alice,bob") == EXIT_USAGE
    assert "pick one" in capsys.readouterr().err


def test_an_empty_scope_is_refused(root, capsys):
    seed(root, n=10, n_fail=3, n_disagree=0)
    argv = ["--raters", "alice,bob", "--cell", "feature=nothing"]
    assert run(root, *argv) == EXIT_DOMAIN_ERROR
    assert "no stored trace matches" in capsys.readouterr().err


def test_agreement_records_survive_dump_and_load(root, tmp_path, capsys):
    seed(root, n=40, n_fail=12, n_disagree=0)
    run(root, "--raters", "alice,bob", "--codebook-version", "1")
    out = tmp_path / "dump"
    assert main(["--root", str(root), "dump", str(out)]) == EXIT_OK
    fresh = tmp_path / "fresh"
    assert main(["--root", str(fresh), "load", str(out)]) == EXIT_OK
    assert load_records_of(fresh, "agreement") == load_records_of(root, "agreement")
