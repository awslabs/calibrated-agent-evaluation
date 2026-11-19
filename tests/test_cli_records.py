# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""One local root for every stage, and the stage records round-trip with it.

``--root`` (or ``CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT``) selects the local store whatever the
working directory, and every record kind a stage writes under ``<root>/records/<kind>/`` travels
through ``calibrated-agent-evaluation dump`` / ``calibrated-agent-evaluation load`` with the traces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
from calibrated_agent_evaluation.cli.records import (
    RECORD_KINDS,
    append_records,
    load_ledger,
    load_records_of,
    save_ledger,
)
from calibrated_agent_evaluation.improve import PreRegEntry, PreRegLedger, register_prereg
from test_agentcore_audit_types import arm_assignment as arm_assignment_record
from test_agentcore_audit_types import audit_record, recommendation_claim_record, run_provenance
from test_agentcore_audit_types import claim_record as abtest_claim_record
from test_agentcore_online_record import online_record, read_record, session_log
from test_bias_store import _record as bias_record
from test_calibrate_store import _record as calibration_record
from test_cli_probe_battery import _record as probe_record
from test_flagship_store import _report as flagship_record
from test_improve_store import _record as improve_record
from test_ingest_sample import sample_record
from test_judge_check import _record as judge_check_record
from test_judges_run_record import _record as judge_run_record
from test_review_store import _codebook as codebook_record
from test_review_store import _proposal as proposal_record
from test_review_store import _record as agreement_record
from test_run_store import _record as run_record
from test_subjective_store import _record as preference_record

_SAMPLES = {
    "calibration": calibration_record,
    "run": run_record,
    "improve": improve_record,
    "bias": bias_record,
    "flagship": flagship_record,
    "preference": preference_record,
    "judge": judge_run_record,
    "agreement": agreement_record,
    "codebook": codebook_record,
    "judge-check": judge_check_record,
    "sample": sample_record,
    "probe": probe_record,
    "proposed-mode": proposal_record,
    "agentcore-abtest": abtest_claim_record,
    "agentcore-arms": arm_assignment_record,
    "agentcore-recommendation": recommendation_claim_record,
    "agentcore-audit": audit_record,
    "agentcore-run-provenance": run_provenance,
    "agentcore-online": online_record,
    "agentcore-session": session_log,
    "agentcore-read": read_record,
}


@pytest.fixture(autouse=True)
def _no_ambient_root(monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)


def _trace_ids(dump_dir: Path) -> set[str]:
    return {row["trace_id"] for row in json.loads((dump_dir / "trace.json").read_text())}


def test_root_selects_the_same_store_from_any_working_directory(tmp_path, monkeypatch, capsys):
    store = tmp_path / "store"
    assert main(["--root", str(store), "quickstart"]) == EXIT_OK

    seen = []
    for cwd in (tmp_path / "a", tmp_path / "a" / "deeper"):
        cwd.mkdir(parents=True)
        monkeypatch.chdir(cwd)
        out = tmp_path / f"dump-{cwd.name}"
        assert main(["--root", str(store), "dump", str(out)]) == EXIT_OK
        seen.append(_trace_ids(out))
    assert seen[0] == seen[1]
    assert len(seen[0]) == 6
    # Nothing was written to a CWD-relative store on the way.
    assert not (tmp_path / "a" / ".calibrated-agent-evaluation").exists()


def test_the_env_var_selects_the_root_and_the_flag_overrides_it(tmp_path, monkeypatch, capsys):
    seeded, empty = tmp_path / "seeded", tmp_path / "empty"
    assert main(["--root", str(seeded), "quickstart"]) == EXIT_OK
    monkeypatch.chdir(tmp_path)

    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", str(seeded))
    assert main(["dump", str(tmp_path / "via-env")]) == EXIT_OK
    assert len(_trace_ids(tmp_path / "via-env")) == 6

    assert main(["--root", str(empty), "dump", str(tmp_path / "via-flag")]) == EXIT_OK
    assert _trace_ids(tmp_path / "via-flag") == set()


def test_root_is_accepted_after_the_subcommand_too(tmp_path, capsys):
    store = tmp_path / "store"
    assert main(["quickstart", "--root", str(store)]) == EXIT_OK
    assert main(["dump", str(tmp_path / "out"), "--root", str(store)]) == EXIT_OK
    assert len(_trace_ids(tmp_path / "out")) == 6


def test_every_record_kind_round_trips_through_dump_and_load(tmp_path, capsys):
    source, target, dump_dir = tmp_path / "source", tmp_path / "target", tmp_path / "dump"
    assert main(["--root", str(source), "quickstart"]) == EXIT_OK
    for kind, make in _SAMPLES.items():
        assert append_records(source, kind, [make()]) == 1
    ledger = register_prereg(
        PreRegLedger(),
        PreRegEntry(prereg_id="p1", target_metric="over_refund", success_threshold=0.05),
    )
    save_ledger(source, ledger)

    assert main(["--root", str(source), "dump", str(dump_dir)]) == EXIT_OK
    assert "1 calibration" in capsys.readouterr().out
    for kind in RECORD_KINDS:
        assert (dump_dir / "records" / kind).is_dir()

    assert main(["--root", str(target), "load", str(dump_dir)]) == EXIT_OK
    for kind, make in _SAMPLES.items():
        assert load_records_of(target, kind) == [make()]
    assert load_ledger(target) == ledger


def test_appending_the_same_record_twice_keeps_one(tmp_path):
    record = calibration_record()
    assert append_records(tmp_path, "calibration", [record]) == 1
    assert append_records(tmp_path, "calibration", [record]) == 0
    assert load_records_of(tmp_path, "calibration") == [record]


def test_a_clashing_record_id_is_refused_and_the_load_writes_nothing(tmp_path, capsys):
    from calibrated_agent_evaluation._model import replace

    source, target, dump_dir = tmp_path / "source", tmp_path / "target", tmp_path / "dump"
    append_records(source, "calibration", [calibration_record()])
    append_records(source, "run", [run_record()])
    assert main(["--root", str(source), "dump", str(dump_dir)]) == EXIT_OK

    clash = replace(calibration_record(), youden_j=0.1)
    append_records(target, "calibration", [clash])
    capsys.readouterr()

    assert main(["--root", str(target), "load", str(dump_dir)]) == EXIT_DOMAIN_ERROR
    assert "already exists" in capsys.readouterr().err
    assert load_records_of(target, "calibration") == [clash]
    assert load_records_of(target, "run") == []


def test_two_different_ledgers_are_refused(tmp_path, capsys):
    source, target, dump_dir = tmp_path / "source", tmp_path / "target", tmp_path / "dump"
    save_ledger(
        source,
        register_prereg(
            PreRegLedger(),
            PreRegEntry(prereg_id="p1", target_metric="over_refund", success_threshold=0.05),
        ),
    )
    save_ledger(
        target,
        register_prereg(
            PreRegLedger(),
            PreRegEntry(prereg_id="p2", target_metric="dropped_fact", success_threshold=0.05),
        ),
    )
    assert main(["--root", str(source), "dump", str(dump_dir)]) == EXIT_OK

    assert main(["--root", str(target), "load", str(dump_dir)]) == EXIT_DOMAIN_ERROR
    assert "ledger" in capsys.readouterr().err
