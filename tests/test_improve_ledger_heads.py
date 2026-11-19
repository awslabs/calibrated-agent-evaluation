# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ledger head a run or judge record is stamped with, and the refusal when it is gone.

A record made against a ledger names the ledger's head. If the ledger is later edited, replaced or
deleted, the head is no longer in it, and registering or evaluating against the new ledger is
refused: the bar the record was made under is not the bar on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _helpers import TEST_STAMP
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
from calibrated_agent_evaluation.cli.records import load_ledger, records_dir
from calibrated_agent_evaluation.improve import (
    EMPTY_LEDGER_HEAD,
    LEDGER_HEAD_FIELD,
    LedgerIntegrityError,
    PreRegEntry,
    PreRegLedger,
    ledger_head,
    register_prereg,
    verify_heads,
)
from calibrated_agent_evaluation.run.report import RunRecord


def _ledger(*metrics: str) -> PreRegLedger:
    ledger = PreRegLedger()
    for i, metric in enumerate(metrics):
        ledger = register_prereg(
            ledger, PreRegEntry(prereg_id=f"p{i}", target_metric=metric, success_threshold=0.1)
        )
    return ledger


def test_the_head_of_an_empty_ledger_is_named_empty() -> None:
    assert ledger_head(PreRegLedger()) == EMPTY_LEDGER_HEAD


def test_the_head_is_the_last_entrys_hash_and_moves_as_entries_are_added() -> None:
    one = _ledger("mode_a")
    two = _ledger("mode_a", "mode_b")
    assert ledger_head(one) == one.entries[-1].content_hash
    assert ledger_head(two) == two.entries[-1].content_hash
    assert ledger_head(one) != ledger_head(two)


def test_a_head_still_in_the_ledger_passes_even_after_later_entries() -> None:
    one = _ledger("mode_a")
    two = register_prereg(
        one, PreRegEntry(prereg_id="p9", target_metric="mode_b", success_threshold=0.2)
    )
    verify_heads(two, [("run record 'run-1'", ledger_head(one))])
    verify_heads(two, [("run record 'run-0'", EMPTY_LEDGER_HEAD)])


def test_a_head_the_ledger_no_longer_holds_is_refused_naming_the_record() -> None:
    original = _ledger("mode_a")
    replaced = _ledger("mode_b")
    with pytest.raises(LedgerIntegrityError, match="run record 'run-7'") as info:
        verify_heads(replaced, [("run record 'run-7'", ledger_head(original))])
    assert ledger_head(original)[:12] in str(info.value)
    assert "Restore the ledger" in str(info.value)


def test_a_deleted_ledger_refuses_a_head_recorded_before_it_was_deleted() -> None:
    with pytest.raises(LedgerIntegrityError):
        verify_heads(PreRegLedger(), [("judge verdict", ledger_head(_ledger("mode_a")))])


def test_no_stamped_records_means_nothing_to_refuse() -> None:
    verify_heads(_ledger("mode_a"), [])


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    return tmp_path / "store"


def _register(root: Path, metric: str) -> int:
    return main(["--root", str(root), "improve", "--register", metric, "--min-reduction", "0.1"])


def _replace_ledger(root: Path) -> None:
    (records_dir(root, "improve") / "prereg_ledger.json").unlink()
    (records_dir(root, "improve") / "prereg_ledger.csv").unlink(missing_ok=True)


def test_register_refuses_when_a_stored_run_names_a_replaced_ledger(root, capsys):
    from calibrated_agent_evaluation.cli.records import append_records, load_records_of

    assert _register(root, "mode_a") == EXIT_OK
    head = ledger_head(load_ledger(root))
    record = RunRecord.model_validate(
        {
            "provenance": TEST_STAMP.model_dump(mode="json"),
            "run_id": "run-7",
            "mode": "ci",
            "gate_passed": True,
            "judge_ids": (),
            "calib_ids": (),
            "scorer_lanes": (),
            "trend": (),
            "input_drift": None,
            "judge_drift": (),
            "multiplicity": (),
            "decisions": (),
            "inputs": {"root": str(root), LEDGER_HEAD_FIELD: head},
        }
    )
    append_records(root, "run", [record])
    assert load_records_of(root, "run")
    _replace_ledger(root)
    capsys.readouterr()
    assert _register(root, "mode_b") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "LedgerIntegrityError" in err
    assert "run record 'run-7'" in err
    assert head[:12] in err
    assert load_ledger(root).entries == ()


def test_register_refuses_when_a_stored_judge_run_names_a_replaced_ledger(root, capsys):
    from calibrated_agent_evaluation.cli.records import append_records
    from calibrated_agent_evaluation.judges.run_record import JudgeRunRecord

    assert _register(root, "mode_a") == EXIT_OK
    head = ledger_head(load_ledger(root))
    judge_run = JudgeRunRecord(
        run_id="judge-run-3",
        judge_id="refund-judge",
        model="bedrock/model",
        prompt_version="v1",
        calib_id="calib-1",
        level="workflow",
        trace_ids=("t-1",),
        span_ids=(None,),
        n_cached=0,
        n_computed=1,
        ledger_head=head,
        ledger_entries=1,
    )
    append_records(root, "judge", [judge_run])
    _replace_ledger(root)
    capsys.readouterr()
    assert _register(root, "mode_b") == EXIT_DOMAIN_ERROR
    assert "judge record 'judge-run-3'" in capsys.readouterr().err


def test_register_with_no_stamped_records_still_works(root, capsys):
    assert _register(root, "mode_a") == EXIT_OK
    assert _register(root, "mode_b") == EXIT_OK
    assert len(load_ledger(root).entries) == 2


def test_the_cli_stamp_for_an_empty_ledger_is_the_head_improve_accepts():
    from calibrated_agent_evaluation.cli.records import NO_LEDGER_HEAD

    assert NO_LEDGER_HEAD == EMPTY_LEDGER_HEAD
