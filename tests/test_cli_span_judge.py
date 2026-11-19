# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation judge` runs span-level judges, keyed on (trace_id, span_id).

A span judge for one crew agent reads each of that agent's spans with its ancestor path, and its
verdicts go to that agent's mode in `calibrated-agent-evaluation run`, the culprit side of the
walk-back. `calibrated-agent-evaluation calibrate` grades it against per-span human gold only. The
judge model is replayed from a cassette built from the shipped renderer; the stored ClaimsCrew trace
CLM-4471 is the corpus capture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _helpers import load_corpus_trace
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
from calibrated_agent_evaluation.cli.judgefile import load_judge_file
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.execute.cassette import Cassette, Mode, fingerprint
from calibrated_agent_evaluation.judges import build_verdict_schema, render_span
from calibrated_agent_evaluation.schema.types import Label
from cli_store_helpers import (
    certified_calibration,
    clear_battery,
    store_battery,
    store_calibration,
    write_judge_file,
)

MODE = "dropped_material_fact"
SPAN_JUDGE = {"level": "span", "agent": "intake", "failure_mode": MODE, "judge_id": "intake-judge"}


@pytest.fixture
def span_judge(tmp_path) -> Path:
    return write_judge_file(tmp_path / "span-judge.json", **SPAN_JUDGE)


def _cassette(path: Path, judge_file: Path, replies: list[tuple[str, str]]) -> Path:
    """Replies for the judge on each named trace's intake span, keyed as the judge requests them."""
    judge = load_judge_file(judge_file).judge
    cassette = Cassette(path, mode=Mode.AUTO)
    for trace_id, verdict in replies:
        trace = load_corpus_trace(trace_id)
        span = next(s for s in trace.spans if s.agent == "intake")
        request = {
            "system": judge.prompt_template,
            "user": render_span(trace, span),
            "schema": build_verdict_schema(judge),
            "via": "response_schema",
        }
        cassette.record(
            fingerprint(request),
            request,
            {
                "fields": {"critique": f"intake on {trace_id}", "verdict": verdict},
                "via": "response_schema",
            },
        )
    return path


def _trust(root: Path, judge_file: Path) -> None:
    judge = load_judge_file(judge_file).judge
    pinned = {
        "model": judge.model,
        "prompt_version": judge.prompt_version,
        "fingerprint": judge.fingerprint,
    }
    store_calibration(root, certified_calibration(judge.judge_id, calib_id="calib-span", **pinned))
    store_battery(root, clear_battery(judge.judge_id, **pinned))


def _judge_labels(root: Path) -> list[Label]:
    with LocalBackend(root=root) as backend:
        return [
            lb
            for t in backend.query()
            for lb in backend.get_scores(t.trace_id)
            if lb.kind == "llm-judge"
        ]


def test_a_span_judge_scores_each_span_of_its_agent(
    seeded_local_root, span_judge, tmp_path, capsys
):
    _trust(seeded_local_root, span_judge)
    cassette = _cassette(tmp_path / "c.json", span_judge, [("CLM-4471", "fail")])
    assert main(["judge", "--judge", str(span_judge), "--cassette", str(cassette)]) == EXIT_OK
    out = capsys.readouterr().out
    # The store holds ORD-1001, ORD-1002 and CLM-4471; only CLM-4471 has an intake span.
    assert (
        "wrote 1 verdict(s) for intake/dropped_material_fact on 1 span(s) in 1 of 3 stored" in out
    )
    (verdict,) = _judge_labels(seeded_local_root)
    assert (verdict.trace_id, verdict.span_id) == ("CLM-4471", "CLM-4471/intake")
    assert (verdict.outcome, verdict.failure_mode) == ("FAIL", MODE)


def test_span_verdicts_count_toward_the_judged_agents_mode(
    seeded_local_root, span_judge, tmp_path, capsys
):
    """The verdict on intake is intake's, though CLM-4471's harm manifested at payout."""
    _trust(seeded_local_root, span_judge)
    cassette = _cassette(tmp_path / "c.json", span_judge, [("CLM-4471", "fail")])
    main(["judge", "--judge", str(span_judge), "--cassette", str(cassette)])
    capsys.readouterr()
    assert main(["run", "--judge", str(span_judge), "--n-resamples", "199"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "CI gate over 1 judge verdicts" in out
    assert "intake/dropped_material_fact" in out
    (record,) = load_records_of(seeded_local_root, "run")
    (point,) = record.trend
    assert (point.mode.failure_type, point.mode.agent) == (MODE, "intake")


def test_the_mode_is_read_off_the_trace_not_the_span_id(seeded_local_root):
    """A span id need not spell its agent; the trace says whose span it is."""
    from calibrated_agent_evaluation.attribution import attributed_agent, span_agents

    trace = load_corpus_trace("CLM-4471")
    verdict = Label(
        trace_id="CLM-4471",
        span_id="CLM-4471/intake",
        kind="llm-judge",
        outcome="FAIL",
        critique="c",
        failure_mode=MODE,
    )
    renamed = {("CLM-4471", "CLM-4471/intake"): "claims-intake"}
    assert attributed_agent(verdict, renamed) == "claims-intake"
    assert attributed_agent(verdict, span_agents([trace])) == "intake"


def test_a_span_judge_must_name_its_agent(seeded_local_root, tmp_path, capsys):
    path = write_judge_file(tmp_path / "j.json", level="span")
    assert main(["judge", "--judge", str(path)]) == EXIT_DOMAIN_ERROR
    assert "a span-level judge needs an agent" in capsys.readouterr().err


def test_no_span_of_the_agent_in_scope_is_refused(seeded_local_root, span_judge, tmp_path, capsys):
    _trust(seeded_local_root, span_judge)
    # Any readable cassette: the refusal comes before a model call.
    empty = (
        Path(__file__).parent.parent
        / "fixtures/raw/cassettes/judge_orderbot_resolved.cassette.json"
    )
    argv = ["judge", "--judge", str(span_judge), "--trace", "ORD-1001", "--cassette", str(empty)]
    assert main(argv) == EXIT_DOMAIN_ERROR
    assert "no span of agent 'intake' on trace 'ORD-1001'" in capsys.readouterr().err
    assert _judge_labels(seeded_local_root) == []


def test_a_span_judge_is_calibrated_on_span_gold_only(
    seeded_local_root, span_judge, tmp_path, capsys
):
    """CLM-4471's trace-level FAIL never grades the span judge."""
    empty = _cassette(tmp_path / "c.json", span_judge, [])
    argv = ["calibrate", "--judge", str(span_judge), "--cassette", str(empty)]
    assert main(argv) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "1 FAIL for intake/dropped_material_fact and 0 PASS" in err
    # ORD-1001, ORD-1002 and CLM-4471 each carry a trace-level human label.
    assert "left out: 3 trace-level label(s), which do not grade a span judge" in err


def test_span_gold_from_two_traces_makes_a_locked_test(
    seeded_local_root, span_judge, tmp_path, capsys
):
    with LocalBackend(root=seeded_local_root) as backend:
        backend.put_trace(load_corpus_trace("CLM-4475"))
        backend.put_score(
            Label(
                trace_id="CLM-4475",
                span_id="CLM-4475/intake",
                kind="human",
                outcome="PASS",
                critique="intake fine",
            )
        )
    replies = [("CLM-4471", "fail"), ("CLM-4475", "pass")]
    cassette = _cassette(tmp_path / "c.json", span_judge, replies)
    argv = ["calibrate", "--judge", str(span_judge), "--cassette", str(cassette)]
    assert main(argv) == EXIT_DOMAIN_ERROR  # one gold label per class cannot certify
    err = capsys.readouterr().err
    assert "on 2 human gold label(s)" in err
    # CLM-4475's intake span also carries a FAIL for misread_amount, another mode: left out.
    assert "left out: 1 FAIL label(s) for another failure mode or agent" in err
    (record,) = load_records_of(seeded_local_root, "calibration")
    assert sorted(record.trace_ids) == ["CLM-4471", "CLM-4475"]
    assert (record.n_pos, record.n_neg) == (1, 1)
