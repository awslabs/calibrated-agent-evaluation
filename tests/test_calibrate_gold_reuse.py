# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation calibrate` uses up a locked test by content, whatever judge file
evaluates it.

A locked test is single-use so a judge cannot be tuned on the test it is scored on. The set is
recognized by its gold, for the failure mode it tests, so the same gold handed to a copy of the
judge file under a new id, or a set that shares traces with a used one, is refused. Growing a set
that was refused for having fewer labels than its plan, with the judge unchanged, is allowed:
that is what the refusal tells the user to do.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.schema.types import Label
from test_cli_calibrate_bias import CLEAN, MODE, _calibrate, _gold_store, _verdict_cassette


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)


def _judge_copy(tmp_path: Path, name: str, **changes: str) -> Path:
    doc = {**json.loads(CLEAN.read_text(encoding="utf-8")), **changes}
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _agreeing(traces: list, n_fail: int) -> list:
    return [(t, "fail" if i < n_fail else "pass") for i, t in enumerate(traces)]


def _add_gold(root: Path, start: int, n_fail: int, n_pass: int) -> list:
    """More traces with human gold, numbered from ``start``; returns them."""
    from cli_store_helpers import trace

    added = []
    with LocalBackend(root=root) as backend:
        for i in range(n_fail + n_pass):
            t = trace(f"G{start + i:03d}")
            backend.put_trace(t)
            fail = i < n_fail
            backend.put_score(
                Label(
                    trace_id=t.trace_id,
                    span_id=None,
                    kind="human",
                    outcome="FAIL" if fail else "PASS",
                    critique="gold",
                    failure_mode=MODE if fail else None,
                )
            )
            added.append(t)
    return added


def test_a_copy_of_the_judge_file_under_a_new_id_cannot_reuse_the_gold(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _gold_store(root, 55, 55)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _agreeing(traces, 55))
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    capsys.readouterr()

    copy = _judge_copy(tmp_path, "copy", judge_id="renamed-clean-judge")
    assert _calibrate(root, copy, cassette) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "this gold set was already used to test judge 'fixture-clean-judge'" in err
    assert len(load_records_of(root, "calibration")) == 1


def test_adding_one_trace_to_a_used_set_does_not_make_it_fresh(tmp_path, capsys):
    root = tmp_path / "store"
    traces = _gold_store(root, 55, 55)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _agreeing(traces, 55))
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    capsys.readouterr()

    (extra,) = _add_gold(root, 110, 1, 0)
    edited = _judge_copy(tmp_path, "v2", prompt_version="v2")
    _verdict_cassette(cassette, edited, [*_agreeing(traces, 55), (extra, "fail")])
    assert _calibrate(root, edited, cassette) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "110 of this gold set's 111 trace(s) were already used" in err
    assert len(load_records_of(root, "calibration")) == 1


def test_a_set_short_of_its_plan_can_be_grown_with_the_judge_unchanged(tmp_path, capsys):
    root = tmp_path / "store"
    first = _gold_store(root, 3, 3)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _agreeing(first, 3))
    assert _calibrate(root, CLEAN, cassette) == EXIT_DOMAIN_ERROR
    capsys.readouterr()

    more = _add_gold(root, 6, 52, 52)
    replies = [*_agreeing(first, 3), *_agreeing(more, 52)]
    _verdict_cassette(cassette, CLEAN, replies)
    assert _calibrate(root, CLEAN, cassette) == EXIT_OK
    first_record, second = load_records_of(root, "calibration")
    assert first_record.result == "REFUSED"
    assert second.result == "CERTIFIED"


def test_a_second_run_waiting_on_the_lock_is_refused_before_any_model_call(tmp_path, capsys):
    """A run that starts while another holds the records sees its record and spends nothing.

    The waiting run's cassette is empty, so a model call would fail with a cassette miss; the
    refusal it gets instead shows it read the stored records only after the lock was released.
    """
    import threading
    import time

    from calibrated_agent_evaluation.cli.records import append_records, records_lock

    first, second = tmp_path / "first", tmp_path / "second"
    traces = _gold_store(first, 55, 55)
    _gold_store(second, 55, 55)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _agreeing(traces, 55))
    assert _calibrate(first, CLEAN, cassette) == EXIT_OK
    (made,) = load_records_of(first, "calibration")
    empty = _verdict_cassette(tmp_path / "empty.json", CLEAN, [])
    capsys.readouterr()

    codes: list[int] = []
    with records_lock(second, "calibration"):
        waiting = threading.Thread(target=lambda: codes.append(_calibrate(second, CLEAN, empty)))
        waiting.start()
        time.sleep(0.5)
        append_records(second, "calibration", [made], locked=True)
    waiting.join(timeout=60)

    assert codes == [EXIT_DOMAIN_ERROR]
    err = capsys.readouterr().err
    assert "already used to test a judge" in err
    assert "no recorded verdict" not in err
    assert load_records_of(second, "calibration") == [made]


def test_a_changed_judge_cannot_reuse_a_short_set(tmp_path, capsys):
    """The top-up exception is for the same judge only: a revised prompt needs fresh traces."""
    root = tmp_path / "store"
    first = _gold_store(root, 3, 3)
    cassette = _verdict_cassette(tmp_path / "c.json", CLEAN, _agreeing(first, 3))
    assert _calibrate(root, CLEAN, cassette) == EXIT_DOMAIN_ERROR
    capsys.readouterr()

    more = _add_gold(root, 6, 52, 52)
    edited = _judge_copy(tmp_path, "edited", prompt_template="A different prompt.")
    _verdict_cassette(cassette, edited, [*_agreeing(first, 3), *_agreeing(more, 52)])
    assert _calibrate(root, edited, cassette) == EXIT_DOMAIN_ERROR
    assert "6 of this gold set's 110 trace(s) were already used" in capsys.readouterr().err
