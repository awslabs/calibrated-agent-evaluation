# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation dump` / `calibrated-agent-evaluation load` round-trip tests.

Reuses the dump/load. `dump` then `load` round-trips the seeded artifacts; a corrupt dump
directory raises the typed ``CorruptDumpError`` (surfaced by the CLI boundary as exit 1), never
a bare parse traceback.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK


def test_dump_then_load_round_trips(seeded_local_root, tmp_path, capsys):
    """Dump the seeded backend to a dir, then load it back — both exit 0."""
    out_dir = tmp_path / "dump"
    assert main(["dump", str(out_dir)]) == EXIT_OK
    assert (out_dir).exists()

    # A CSV+JSON set was written; loading it back succeeds.
    assert main(["load", str(out_dir)]) == EXIT_OK
    combined = capsys.readouterr().out
    assert "dumped" in combined
    assert "loaded" in combined


def test_corrupt_dump_dir_raises_typed_error_exit_1(seeded_local_root, tmp_path, capsys):
    """Negative: loading a corrupted dump → typed error → exit 1, no bare traceback."""
    out_dir = tmp_path / "dump"
    assert main(["dump", str(out_dir)]) == EXIT_OK
    capsys.readouterr()  # drain the dump summary

    # Corrupt the CSV mirror so its row->doc parse fails, and read it back explicitly.
    trace_csv = out_dir / "trace.csv"
    assert trace_csv.exists()
    trace_csv.write_text("trace_id,doc\nT1,{not valid json]\n", encoding="utf-8")

    rc = main(["load", "--format", "csv", str(out_dir)])

    assert rc == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "Traceback (most recent call last)" not in err
    assert err.strip()


def test_load_refuses_a_missing_directory(seeded_local_root, tmp_path, capsys):
    missing = tmp_path / "definitely-not-here"
    assert main(["load", str(missing)]) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "CorruptDumpError" in err
    assert str(missing) in err


def test_load_refuses_a_file(seeded_local_root, tmp_path, capsys):
    a_file = tmp_path / "hosts"
    a_file.write_text("127.0.0.1 localhost\n")
    assert main(["load", str(a_file)]) == EXIT_DOMAIN_ERROR
    assert "not a directory" in capsys.readouterr().err


def test_load_refuses_a_directory_with_no_dump_in_it(seeded_local_root, tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["load", str(empty)]) == EXIT_DOMAIN_ERROR
    assert "no dump file" in capsys.readouterr().err


def test_load_refuses_a_damaged_file_whichever_format_it_reads(seeded_local_root, tmp_path, capsys):
    """The manifest seals both files of the pair, so a damaged CSV mirror refuses a JSON load."""
    out_dir = tmp_path / "dump"
    assert main(["dump", str(out_dir)]) == EXIT_OK
    assert main(["--root", str(tmp_path / "ok"), "load", str(out_dir)]) == EXIT_OK
    (out_dir / "trace.csv").write_text("trace_id,doc\nT1,{broken\n", encoding="utf-8")
    capsys.readouterr()
    for fmt in ("json", "csv"):
        fresh = tmp_path / f"fresh-{fmt}"
        assert main(["--root", str(fresh), "load", "--format", fmt, str(out_dir)]) == (
            EXIT_DOMAIN_ERROR
        )
        assert "DumpMismatchError: trace.csv" in capsys.readouterr().err


def test_load_refuses_a_changed_record_file_before_writing_anything(
    seeded_local_root, tmp_path, capsys
):
    from calibrated_agent_evaluation.cli.records import append_records, load_records_of
    from test_agentcore_audit_types import audit_record

    append_records(seeded_local_root, "agentcore-audit", [audit_record()])
    out_dir = tmp_path / "dump"
    assert main(["dump", str(out_dir)]) == EXIT_OK
    (record_file,) = (out_dir / "records" / "agentcore-audit").glob("*.json")
    record_file.write_text(record_file.read_text(encoding="utf-8") + " ", encoding="utf-8")
    fresh = tmp_path / "fresh"
    capsys.readouterr()

    assert main(["--root", str(fresh), "load", str(out_dir)]) == EXIT_DOMAIN_ERROR
    assert "DumpMismatchError: records/agentcore-audit/" in capsys.readouterr().err
    assert load_records_of(fresh, "agentcore-audit") == []


def test_a_dump_interrupted_before_its_manifest_is_refused(
    seeded_local_root, tmp_path, capsys, monkeypatch
):
    import calibrated_agent_evaluation.dumpload.io as dumpload_io

    def interrupted(out, manifest):
        raise KeyboardInterrupt

    out_dir = tmp_path / "dump"
    with monkeypatch.context() as patched:
        patched.setattr(dumpload_io, "_write_manifest", interrupted)
        with pytest.raises(KeyboardInterrupt):
            main(["dump", str(out_dir)])
    assert (out_dir / "trace.json").is_file()
    capsys.readouterr()

    fresh = tmp_path / "fresh"
    assert main(["--root", str(fresh), "load", str(out_dir)]) == EXIT_DOMAIN_ERROR
    assert "IncompleteDumpError" in capsys.readouterr().err


def test_audit_records_survive_dump_and_load(seeded_local_root, tmp_path):
    import json
    from pathlib import Path

    from _arms_fixture import AB_TEST_ARN, MAIN, VERSIONS, Double, Window, deliver
    from calibrated_agent_evaluation._provenance import capture_provenance
    from calibrated_agent_evaluation.agentcore.arms import read_arms
    from calibrated_agent_evaluation.agentcore.audit_types import AbTestClaimRecord
    from calibrated_agent_evaluation.cli.records import append_records, load_records_of

    golden = Path(__file__).resolve().parents[1] / "fixtures" / "expected" / "audit"
    doc = json.loads((golden / "abtest_verdict.json").read_text(encoding="utf-8"))
    claim = AbTestClaimRecord.model_validate(
        {
            **doc,
            "record_id": "abtest-1",
            "read_at": "2026-11-02T10:00:00+00:00",
            "provenance": capture_provenance().model_dump(mode="json"),
        }
    )
    arms = read_arms(
        Double(deliver(MAIN)),
        ab_test_arn=AB_TEST_ARN,
        runtime="fixture-runtime",
        window=Window(),
        bundle_versions=VERSIONS,
    ).assignments
    root = Path(seeded_local_root)
    append_records(root, "agentcore-abtest", [claim])
    append_records(root, "agentcore-arms", list(arms))
    from test_agentcore_audit_types import audit_record, recommendation_claim_record, run_provenance

    extra = {
        "agentcore-recommendation": recommendation_claim_record(),
        "agentcore-audit": audit_record(),
        "agentcore-run-provenance": run_provenance(),
    }
    for kind, record in extra.items():
        append_records(root, kind, [record])
    out_dir = tmp_path / "dump"
    assert main(["dump", str(out_dir)]) == EXIT_OK

    fresh = tmp_path / "fresh"
    assert main(["--root", str(fresh), "load", str(out_dir)]) == EXIT_OK

    assert load_records_of(fresh, "agentcore-abtest") == [claim]
    assert load_records_of(fresh, "agentcore-arms") == list(arms)
    for kind, record in extra.items():
        assert load_records_of(fresh, kind) == [record]

    # A different record under a stored id is refused, as for every other kind.
    clash = tmp_path / "clash"
    append_records(
        clash,
        "agentcore-audit",
        [extra["agentcore-audit"].model_copy(update={"improve_id": "other"})],
    )
    assert main(["--root", str(clash), "load", str(out_dir)]) != EXIT_OK


def test_labels_whose_trace_the_store_does_not_hold_round_trip(seeded_local_root, tmp_path):
    """The seeded store holds gold labels for three ClaimsCrew traces it never stored."""
    from calibrated_agent_evaluation.dumpload import load

    with LocalBackend(root=seeded_local_root) as source:
        traces, labels = source.all_traces(), source.all_scores()
    held = {t.trace_id for t in traces}
    assert {"CLM-4473", "CLM-4475", "CLM-4477"} <= {lb.trace_id for lb in labels} - held
    out_dir = tmp_path / "dump"
    assert main(["dump", str(out_dir)]) == EXIT_OK
    assert main(["load", str(out_dir)]) == EXIT_OK

    def key(lb):
        return (lb.trace_id, lb.span_id or "", lb.kind, lb.outcome, lb.critique)

    with LocalBackend(root=tmp_path / "fresh") as restored:
        load(restored, out_dir, "json")
        assert restored.all_traces() == traces
        assert sorted(restored.all_scores(), key=key) == sorted(labels, key=key)
