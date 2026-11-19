# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation improve` pre-registers a target, then gates a fix on it and records
the verdict. The threshold comes from the ledger, never from the request.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.records import load_ledger, load_records_of, records_dir
from calibrated_agent_evaluation.schema.types import Label

METRIC = "trusted_broken_policy_tool"


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    return tmp_path / "store"


def _improve(root: Path, *argv: str) -> int:
    return main(["--root", str(root), "improve", *argv])


def _request(tmp_path: Path, n_fixed: int, n: int = 40, *, judged: bool = False, **extra) -> Path:
    """n pairs: the first n_fixed failed before and pass after, the rest pass both times."""
    pairs = [
        {
            "trace_id": f"c-{i}",
            "before_fail": i < n_fixed,
            "after_fail": False,
            "labels_are_judge_produced": judged,
        }
        for i in range(n)
    ]
    doc = {
        "target_metric": METRIC,
        "pairs": pairs,
        "development_ids": ["d-1", "d-2"],
        "slices": [_slice(0.7)],
        **extra,
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(doc))
    return path


def _slice(regression_p: float) -> dict:
    rate = {
        "point": 0.1,
        "lo": 0.05,
        "hi": 0.2,
        "method": "wilson",
        "level": 0.95,
        "sided": "two-sided",
        "n": 100,
    }
    return {
        "name": "misread_amount",
        "kind": "other",
        "regression_p": regression_p,
        "rate_before": rate,
        "rate_after": rate,
    }


def test_register_freezes_the_target_into_the_ledger(root, capsys):
    assert _improve(root, "--register", METRIC, "--min-reduction", "0.10") == EXIT_OK
    out = capsys.readouterr().out
    assert "cut its fail rate by at least 0.10" in out
    (entry,) = load_ledger(root).entries
    assert entry.target_metric == METRIC
    assert entry.success_threshold == 0.10
    assert entry.content_hash == entry.compute_hash()


def test_a_metric_registers_once(root, capsys):
    assert _improve(root, "--register", METRIC, "--min-reduction", "0.10") == EXIT_OK
    assert _improve(root, "--register", METRIC, "--min-reduction", "0.01") == EXIT_DOMAIN_ERROR
    assert "already pre-registered" in capsys.readouterr().err
    assert len(load_ledger(root).entries) == 1


def test_register_needs_a_minimum_reduction_in_range(root, capsys):
    assert _improve(root, "--register", METRIC) == EXIT_USAGE
    assert "--min-reduction" in capsys.readouterr().err
    assert _improve(root, "--register", METRIC, "--min-reduction", "1.5") == EXIT_DOMAIN_ERROR
    assert "not in (0, 1)" in capsys.readouterr().err


def test_a_fix_that_clears_its_registered_reduction_is_accepted_and_recorded(
    root, tmp_path, capsys
):
    _improve(root, "--register", METRIC, "--min-reduction", "0.10")
    capsys.readouterr()
    assert _improve(root, "--request", str(_request(tmp_path, 20)), "--improve-id", "i1") == EXIT_OK
    out = capsys.readouterr().out
    assert "ACCEPTED" in out
    assert "95% CI [" in out
    assert "McNemar" in out
    (record,) = load_records_of(root, "improve")
    assert record.improve_id == "i1"
    assert record.verdict == "accepted"
    assert record.provenance.thresholds["success_threshold"] == 0.10
    assert record.inputs["n_pairs"] == "40"


def test_a_fix_short_of_its_registered_reduction_is_rejected_with_exit_1(root, tmp_path, capsys):
    """Three fixes in 40 is a real but small drop; it does not show the 0.30 that was promised."""
    _improve(root, "--register", METRIC, "--min-reduction", "0.30")
    capsys.readouterr()
    assert _improve(root, "--request", str(_request(tmp_path, 3))) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "REJECTED" in err
    assert "reason:" in err
    assert load_records_of(root, "improve")[0].verdict == "rejected"


def test_an_unregistered_metric_is_refused_before_measuring(root, tmp_path, capsys):
    assert _improve(root, "--request", str(_request(tmp_path, 20))) == EXIT_DOMAIN_ERROR
    assert "ImproveError" in capsys.readouterr().err
    assert load_records_of(root, "improve") == []


def test_an_edited_ledger_is_refused(root, tmp_path, capsys):
    _improve(root, "--register", METRIC, "--min-reduction", "0.30")
    ledger_file = records_dir(root, "improve") / "prereg_ledger.json"
    doc = json.loads(ledger_file.read_text())
    doc[0]["success_threshold"] = 0.01  # lowered after seeing the result
    ledger_file.write_text(json.dumps(doc))
    capsys.readouterr()
    assert _improve(root, "--request", str(_request(tmp_path, 3))) == EXIT_DOMAIN_ERROR
    assert "mutated after it was frozen" in capsys.readouterr().err


def test_confirmation_traces_overlapping_development_are_refused(root, tmp_path, capsys):
    _improve(root, "--register", METRIC, "--min-reduction", "0.10")
    request = _request(tmp_path, 20)
    doc = json.loads(request.read_text())
    doc["development_ids"] = ["c-0"]
    request.write_text(json.dumps(doc))
    capsys.readouterr()
    assert _improve(root, "--request", str(request)) == EXIT_DOMAIN_ERROR
    assert "ConfirmationContaminationError" in capsys.readouterr().err


def test_judge_scored_discordant_pairs_need_a_human_adjudication(root, tmp_path, capsys):
    _improve(root, "--register", METRIC, "--min-reduction", "0.10")
    capsys.readouterr()
    request = _request(tmp_path, 20, judged=True)
    assert _improve(root, "--request", str(request)) == EXIT_DOMAIN_ERROR
    assert "AdjudicationRequiredError" in capsys.readouterr().err

    # A human label on each discordant trace in the store is the adjudication.
    from cli_store_helpers import trace

    with LocalBackend(root=root) as backend:
        for i in range(20):
            backend.put_trace(trace(f"c-{i}"))
            backend.put_score(
                Label(
                    trace_id=f"c-{i}",
                    span_id=None,
                    kind="human",
                    outcome="PASS",
                    critique="fixed",
                    rater_id="alice",
                )
            )
    assert _improve(root, "--request", str(request)) == EXIT_OK
    assert load_records_of(root, "improve")[0].inputs["n_adjudicated"] == "20"


def test_a_regressed_guard_slice_rejects_the_fix(root, tmp_path, capsys):
    _improve(root, "--register", METRIC, "--min-reduction", "0.10")
    capsys.readouterr()
    request = _request(tmp_path, 20, slices=[_slice(0.001)])
    assert _improve(root, "--request", str(request)) == EXIT_DOMAIN_ERROR
    assert "regressed: misread_amount" in capsys.readouterr().err


def test_an_empty_guard_panel_is_refused(root, tmp_path, capsys):
    _improve(root, "--register", METRIC, "--min-reduction", "0.10")
    capsys.readouterr()
    assert _improve(root, "--request", str(_request(tmp_path, 20, slices=[]))) == EXIT_DOMAIN_ERROR
    assert "panel is empty" in capsys.readouterr().err


def test_register_and_request_together_or_neither_are_refused(root, tmp_path, capsys):
    assert _improve(root) == EXIT_USAGE
    assert "exactly one of" in capsys.readouterr().err
    req = str(_request(tmp_path, 20))
    assert _improve(root, "--register", METRIC, "--request", req) == EXIT_USAGE


def test_a_crew_request_is_refused_with_the_library_path(root, tmp_path, capsys):
    request = _request(tmp_path, 20, touched_agent="intake")
    assert _improve(root, "--request", str(request)) == EXIT_DOMAIN_ERROR
    assert "evaluate_improvement" in capsys.readouterr().err


def test_a_request_pairing_a_trace_twice_is_refused(root, tmp_path, capsys):
    _improve(root, "--register", METRIC, "--min-reduction", "0.05")
    request = _request(tmp_path, 6, n=20)
    doc = json.loads(request.read_text())
    doc["pairs"] = doc["pairs"] * 2
    request.write_text(json.dumps(doc))
    capsys.readouterr()
    assert _improve(root, "--request", str(request)) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "pairs 20 trace id(s) more than once ('c-0', 'c-1'" in err
    assert load_records_of(root, "improve") == []


def test_humans_who_say_the_requested_fixes_still_fail_reject_it(root, tmp_path, capsys):
    from cli_store_helpers import trace

    _improve(root, "--register", METRIC, "--min-reduction", "0.10")
    with LocalBackend(root=root) as backend:
        for i in range(20):
            backend.put_trace(trace(f"c-{i}"))
            backend.put_score(
                Label(
                    trace_id=f"c-{i}",
                    span_id=None,
                    kind="human",
                    outcome="FAIL",
                    critique="still over-refunds",
                    failure_mode=METRIC,
                    rater_id="alice",
                )
            )
    capsys.readouterr()
    request = _request(tmp_path, 20, judged=True)
    assert _improve(root, "--request", str(request)) == EXIT_DOMAIN_ERROR
    assert "REJECTED" in capsys.readouterr().err
    (record,) = load_records_of(root, "improve")
    assert record.inputs["n_adjudication_overturned"] == "20"


def test_a_refrozen_last_entry_cannot_be_saved_over_the_ledger(root, capsys):
    from calibrated_agent_evaluation.improve import (
        LedgerIntegrityError,
        PreRegEntry,
        PreRegLedger,
        freeze,
    )
    from calibrated_agent_evaluation.improve.store import dump_prereg_ledger

    _improve(root, "--register", METRIC, "--min-reduction", "0.30")
    ledger_dir = records_dir(root, "improve")
    before = (ledger_dir / "prereg_ledger.json").read_text()
    tail = load_ledger(root).entries[-1]
    lowered = freeze(
        PreRegEntry(prereg_id=tail.prereg_id, target_metric=METRIC, success_threshold=0.01),
        prev_hash=tail.prev_hash,
    )
    with pytest.raises(LedgerIntegrityError, match="append-only"):
        dump_prereg_ledger(PreRegLedger(entries=(lowered,)), ledger_dir)
    assert (ledger_dir / "prereg_ledger.json").read_text() == before


def _evaluated(root: Path, tmp_path: Path, capsys) -> None:
    """Register at 0.30 and record one evaluation, which keeps a copy of the entry."""
    _improve(root, "--register", METRIC, "--min-reduction", "0.30")
    _improve(root, "--request", str(_request(tmp_path, 3)))
    assert len(load_records_of(root, "improve")) == 1
    capsys.readouterr()


def test_a_deleted_ledger_cannot_be_started_again_at_a_lower_bar(root, tmp_path, capsys):
    _evaluated(root, tmp_path, capsys)
    (records_dir(root, "improve") / "prereg_ledger.json").unlink()
    assert _improve(root, "--register", METRIC, "--min-reduction", "0.01") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "LedgerIntegrityError" in err
    assert "is missing from the ledger" in err
    assert load_ledger(root).entries == ()


def test_a_hand_rehashed_last_entry_is_refused_once_an_evaluation_recorded_it(
    root, tmp_path, capsys
):
    from calibrated_agent_evaluation._model import replace

    _evaluated(root, tmp_path, capsys)
    ledger_file = records_dir(root, "improve") / "prereg_ledger.json"
    (entry,) = load_ledger(root).entries
    lowered = replace(entry, success_threshold=0.01)
    lowered = replace(lowered, content_hash=lowered.compute_hash())
    doc = json.loads(ledger_file.read_text())
    doc[0].update(success_threshold=0.01, content_hash=lowered.content_hash)
    ledger_file.write_text(json.dumps(doc))
    assert _improve(root, "--request", str(_request(tmp_path, 3))) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "now reads threshold 0.01" in err
    assert len(load_records_of(root, "improve")) == 1
