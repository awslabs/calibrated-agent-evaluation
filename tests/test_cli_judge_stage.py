# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation judge` scores through the three-gate trust chain.

Drives `calibrated-agent-evaluation` against a seeded local store with the committed OrderBot judge
file and the cassette its verdicts were recorded in, so the real gate, runtime and write-back run;
only the model call is a replay.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.judgefile import load_judge_file
from cli_store_helpers import (
    certified_calibration,
    clear_battery,
    store_battery,
    store_calibration,
    write_judge_file,
)

_REPO = Path(__file__).resolve().parent.parent
JUDGE_FILE = _REPO / "fixtures" / "judge-files" / "orderbot-refund-resolved.json"
CASSETTE = _REPO / "fixtures" / "raw" / "cassettes" / "judge_orderbot_resolved.cassette.json"
_JUDGE = load_judge_file(JUDGE_FILE).judge
_PINNED = {
    "model": _JUDGE.model,
    "prompt_version": _JUDGE.prompt_version,
    "fingerprint": _JUDGE.fingerprint,
}


@pytest.fixture
def trusted(seeded_local_root):
    """The seeded store plus a certifying calibration and a CLEAR battery for the judge."""
    store_calibration(
        seeded_local_root, certified_calibration(_JUDGE.judge_id, calib_id="calib-ob", **_PINNED)
    )
    store_battery(seeded_local_root, clear_battery(_JUDGE.judge_id, **_PINNED))
    return seeded_local_root


def _judge(*argv: str) -> int:
    return main(["judge", "--judge", str(JUDGE_FILE), "--cassette", str(CASSETTE), *argv])


def _judge_labels(root: Path, trace_id: str):
    with LocalBackend(root=root) as backend:
        return [lb for lb in backend.get_scores(trace_id) if lb.kind == "llm-judge"]


def test_an_admitted_judge_writes_its_verdict_stamped_with_its_mode(trusted, capsys):
    assert _judge("--trace", "ORD-1001") == EXIT_OK
    out = capsys.readouterr().out
    assert "admitted" in out
    assert "calib-ob" in out
    assert "wrote 1 verdict(s) for trusted_broken_policy_tool on trace 'ORD-1001'" in out
    assert "not a rate" in out

    (verdict,) = _judge_labels(trusted, "ORD-1001")
    assert verdict.failure_mode == "trusted_broken_policy_tool"
    assert verdict.critique.strip()
    assert verdict.outcome in ("PASS", "FAIL")
    # Only the scoped trace was judged.
    assert _judge_labels(trusted, "ORD-1002") == []


def test_a_replay_miss_writes_nothing(trusted, capsys):
    """CLM-4471 was never recorded in the OrderBot cassette, so the whole-store run misses."""
    assert _judge() == EXIT_DOMAIN_ERROR
    assert "no recorded verdict" in capsys.readouterr().err
    for trace_id in ("ORD-1001", "ORD-1002", "CLM-4471"):
        assert _judge_labels(trusted, trace_id) == []


def test_no_calibration_record_means_no_score(seeded_local_root, capsys):
    store_battery(seeded_local_root, clear_battery(_JUDGE.judge_id, **_PINNED))
    assert _judge("--trace", "ORD-1001") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "no calibration record" in err
    assert _judge_labels(seeded_local_root, "ORD-1001") == []


def test_a_refused_calibration_is_refused_by_the_gate(seeded_local_root, capsys):
    store_calibration(
        seeded_local_root,
        certified_calibration(_JUDGE.judge_id, result="REFUSED", **_PINNED),
    )
    store_battery(seeded_local_root, clear_battery(_JUDGE.judge_id, **_PINNED))
    assert _judge("--trace", "ORD-1001") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "refused" in err
    assert "calibration gate" in err
    assert _judge_labels(seeded_local_root, "ORD-1001") == []


def test_a_newer_refused_calibration_wins_over_an_older_certified_one(seeded_local_root, capsys):
    from calibrated_agent_evaluation._model import replace

    old = certified_calibration(_JUDGE.judge_id, calib_id="calib-old", **_PINNED)
    newer = replace(
        certified_calibration(_JUDGE.judge_id, calib_id="calib-new", result="REFUSED", **_PINNED),
        provenance=replace(old.provenance, run_at="2999-01-01T00:00:00+00:00"),
        # A newer calibration is scored on fresh gold: the store refuses a second record on one
        # gold set.
        hunted_set_id="hunted-refund-2",
    )
    store_calibration(seeded_local_root, old)
    store_calibration(seeded_local_root, newer)
    store_battery(seeded_local_root, clear_battery(_JUDGE.judge_id, **_PINNED))
    assert _judge("--trace", "ORD-1001") == EXIT_DOMAIN_ERROR
    assert "calib-new" in capsys.readouterr().err


def test_a_biased_judge_is_refused(seeded_local_root, capsys):
    store_calibration(seeded_local_root, certified_calibration(_JUDGE.judge_id, **_PINNED))
    store_battery(seeded_local_root, clear_battery(_JUDGE.judge_id, verdict="BIASED", **_PINNED))
    assert _judge("--trace", "ORD-1001") == EXIT_DOMAIN_ERROR
    assert "bias gate refused" in capsys.readouterr().err


def test_no_bias_battery_is_refused(seeded_local_root, capsys):
    store_calibration(seeded_local_root, certified_calibration(_JUDGE.judge_id, **_PINNED))
    assert _judge("--trace", "ORD-1001") == EXIT_DOMAIN_ERROR
    assert "no bias battery" in capsys.readouterr().err


def test_a_judge_whose_prompt_changed_since_calibration_is_refused(trusted, tmp_path, capsys):
    edited = write_judge_file(
        tmp_path / "edited.json",
        **{**load_judge_file_doc(), "prompt_version": "demo-v2"},
    )
    rc = main(["judge", "--judge", str(edited), "--cassette", str(CASSETTE), "--trace", "ORD-1001"])
    assert rc == EXIT_DOMAIN_ERROR
    assert "does not certify" in capsys.readouterr().err


def load_judge_file_doc() -> dict:
    import json

    return json.loads(JUDGE_FILE.read_text())


def test_judge_without_a_judge_file_is_refused(seeded_local_root, capsys):
    assert main(["judge"]) == EXIT_USAGE
    assert "--judge FILE" in capsys.readouterr().err


def test_an_unknown_trace_is_refused(trusted, capsys):
    assert _judge("--trace", "NOPE") == EXIT_DOMAIN_ERROR
    assert "NOPE" in capsys.readouterr().err


def test_judged_verdicts_flow_into_calibrated_agent_evaluation_run(trusted, capsys):
    """The two stages compose: `calibrated-agent-evaluation run --judge` gates the verdicts
    `calibrated-agent-evaluation judge` wrote."""
    for trace_id in ("ORD-1001", "ORD-1002"):
        assert _judge("--trace", trace_id) == EXIT_OK
    capsys.readouterr()
    assert main(["run", "--judge", str(JUDGE_FILE), "--n-resamples", "199"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "CI gate over 2 judge verdicts" in out
    # The two OrderBot traces are production-representative, so they are the mode's rate sample.
    assert "trend (1 mode(s)" in out
    assert "calibration calib-ob" in out


@pytest.fixture
def no_store_no_model(monkeypatch):
    """Make opening a store or a judge model fail, so a check that touches either one fails."""
    import calibrated_agent_evaluation.cli.backends
    import calibrated_agent_evaluation.cli.judging

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "calibrated-agent-evaluation judge --check must read no store and call no model"
        )

    monkeypatch.setattr(calibrated_agent_evaluation.cli.backends, "open_backend", refuse)
    monkeypatch.setattr(calibrated_agent_evaluation.cli.judging, "model_backend", refuse)


def test_check_passes_a_judge_file_calibrated_agent_evaluation_would_load(
    tmp_path, no_store_no_model, capsys
):
    """`calibrated-agent-evaluation judge --check FILE` runs the load and schema checks the scoring
    stages run, and nothing else: no store, no model, no file written."""
    root = tmp_path / "root"
    assert main(["--root", str(root), "judge", "--check", str(JUDGE_FILE)]) == EXIT_OK
    assert capsys.readouterr().out.strip() == "judge file ok"
    assert not root.exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"output_fields": ["verdict", "critique"]}, "verdict before the critique"),
        ({"scale": "1-5"}, "Likert"),
        ({"rubric": "x"}, "unknown field(s): rubric"),
        ({"level": "span", "agent": None}, "a span-level judge needs an agent"),
        ({"level": "trace"}, "level must be 'workflow' or 'span'"),
        ({"prompt_version": ""}, "missing required field(s): prompt_version"),
    ],
)
def test_check_refuses_a_judge_file_with_calibrated_agent_evaluation_own_error(
    tmp_path, no_store_no_model, capsys, change, message
):
    import json

    bad = tmp_path / "judge.json"
    bad.write_text(json.dumps({**load_judge_file_doc(), **change}), encoding="utf-8")
    rc = main(["--root", str(tmp_path / "root"), "judge", "--check", str(bad)])
    assert rc == EXIT_DOMAIN_ERROR
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("calibrated-agent-evaluation judge: ")
    assert message in captured.err
    assert set(tmp_path.iterdir()) == {bad}


def test_check_refuses_a_missing_file(tmp_path, no_store_no_model, capsys):
    assert main(["judge", "--check", str(tmp_path / "nope.json")]) == EXIT_DOMAIN_ERROR
    assert "cannot read judge file" in capsys.readouterr().err


def test_check_with_judge_is_refused_rather_than_checking_one_of_them(no_store_no_model, capsys):
    rc = main(["judge", "--check", str(JUDGE_FILE), "--judge", str(JUDGE_FILE)])
    assert rc == EXIT_USAGE
    assert "--check" in capsys.readouterr().err


def test_a_judge_file_edited_in_place_is_refused_until_recertified(trusted, tmp_path, capsys):
    """Editing the prompt without bumping prompt_version no longer keeps the old certification."""
    import json

    doc = json.loads(JUDGE_FILE.read_text(encoding="utf-8"))
    doc["prompt_template"] += " Be lenient with refunds."
    edited = tmp_path / "edited-judge.json"
    edited.write_text(json.dumps(doc), encoding="utf-8")
    rc = main(["judge", "--judge", str(edited), "--cassette", str(CASSETTE), "--trace", "ORD-1001"])
    assert rc == EXIT_DOMAIN_ERROR
    captured = capsys.readouterr()
    assert "prompt changed since it was certified" in captured.out + captured.err
    assert _judge_labels(trusted, "ORD-1001") == []


def test_each_judge_run_is_recorded_with_the_ledger_head_it_ran_against(trusted, capsys):
    from calibrated_agent_evaluation.cli.records import NO_LEDGER_HEAD, load_ledger, load_records_of

    assert _judge("--trace", "ORD-1001") == EXIT_OK
    assert main(["improve", "--register", "over_refund", "--min-reduction", "0.05"]) == EXIT_OK
    assert _judge("--trace", "ORD-1001") == EXIT_OK

    first, second = load_records_of(trusted, "judge")
    assert first.run_id != second.run_id
    assert (first.ledger_head, first.ledger_entries) == (NO_LEDGER_HEAD, 0)
    head = load_ledger(trusted).entries[-1].content_hash
    assert (second.ledger_head, second.ledger_entries) == (head, 1)
    assert second.judge_id == _JUDGE.judge_id
    assert second.trace_ids == ("ORD-1001",)


def test_a_refused_judge_records_no_judge_run(seeded_local_root, capsys):
    from calibrated_agent_evaluation.cli.records import load_records_of

    assert _judge("--trace", "ORD-1001") == EXIT_DOMAIN_ERROR
    assert load_records_of(seeded_local_root, "judge") == []


AGENTCORE_JUDGE_FILE = _REPO / "fixtures" / "judge-files" / "agentcore-correctness.json"


def test_check_passes_an_agentcore_judge_file_with_no_aws_call(tmp_path, no_store_no_model, capsys):
    root = tmp_path / "root"
    assert main(["--root", str(root), "judge", "--check", str(AGENTCORE_JUDGE_FILE)]) == EXIT_OK
    assert "judge file ok" in capsys.readouterr().out
    assert not root.exists()


@pytest.mark.parametrize(
    ("change", "named"),
    [
        ({"level": "span"}, "level"),
        ({"rubric": "x"}, "rubric"),
        ({"evaluator_id": None}, "evaluator_id"),
    ],
    ids=["span-level", "unknown-field", "missing-evaluator"],
)
def test_check_refuses_a_malformed_agentcore_judge_file_with_a_cli_error(
    tmp_path, no_store_no_model, capsys, change, named
):
    import json

    bad = tmp_path / "judge.json"
    document = {**json.loads(AGENTCORE_JUDGE_FILE.read_text(encoding="utf-8")), **change}
    bad.write_text(
        json.dumps({k: v for k, v in document.items() if v is not None}), encoding="utf-8"
    )
    assert main(["judge", "--check", str(bad)]) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert err.startswith("calibrated-agent-evaluation judge: ")
    assert named in err


def test_a_judges_only_dump_whose_judge_file_was_swapped_for_an_agentcore_one_is_refused_on_load(
    trusted, tmp_path, capsys
):
    import shutil

    kit = tmp_path / "prompt-kit"
    assert main(["dump", "--judges-only", "--judge", str(JUDGE_FILE), str(kit)]) == EXIT_OK
    (dumped,) = (kit / "judges").iterdir()
    shutil.copyfile(AGENTCORE_JUDGE_FILE, dumped)
    capsys.readouterr()
    assert main(["--root", str(tmp_path / "fresh"), "load", str(kit)]) == EXIT_DOMAIN_ERROR
    assert "edited after the dump" in capsys.readouterr().err
