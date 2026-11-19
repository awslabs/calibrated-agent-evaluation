# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What `calibrated-agent-evaluation run` counts: one verdict per trace, rater and mode, on the
random sample only.

Each case seeds a real LocalBackend and runs the real CLI. The scenarios are the gate failures a
review reproduced: two judges' modes merged into one, a judge re-run doubling n, a human gate that
could never pass, span defects gated as verdicts, and hunted discovery traces pooled into the bound.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _helpers import load_corpus_labels, load_corpus_trace
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.run import ModeKey
from calibrated_agent_evaluation.schema.types import Label, Span, Trace
from cli_store_helpers import seed_verdicts, trace


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    return tmp_path / "store"


def _run(root: Path, *argv: str) -> int:
    return main(["--root", str(root), "--explain", "run", "--n-resamples", "199", *argv])


def _verdict(tid: str, outcome: str, mode: str, rater: str, critique: str = "c") -> Label:
    return Label(
        trace_id=tid,
        span_id=None,
        kind="llm-judge",
        outcome=outcome,  # type: ignore[arg-type]
        critique=critique,
        failure_mode=mode,
        rater_id=rater,
    )


def _human(tid: str, outcome: str, mode: str | None = None, **over: object) -> Label:
    fields: dict[str, object] = {
        "trace_id": tid,
        "span_id": None,
        "kind": "human",
        "outcome": outcome,
        "critique": "looked at it",
        "failure_mode": mode,
        "rater_id": "r1",
        **over,
    }
    return Label.model_validate(fields)


def _store(
    root: Path, ids: list[str], labels: list[Label], frame: str = "production-representative"
):
    with LocalBackend(root=root) as backend:
        for tid in ids:
            backend.put_trace(trace(tid, frame))
        for label in labels:
            backend.put_score(label)


def _gate_record(root: Path):
    return load_records_of(root, "run")[-1]


def test_two_judges_on_one_trace_are_gated_under_their_own_modes(root, capsys):
    ids = [f"T{i:03d}" for i in range(200)]
    # judge A fails 40 of 200 (Wilson lower bound about 0.75), judge B passes every trace
    labels = [_verdict(t, "FAIL" if i < 40 else "PASS", "m_a", "A@1") for i, t in enumerate(ids)]
    labels += [_verdict(t, "PASS", "m_b", "B@1") for t in ids]
    _store(root, ids, labels)

    assert _run(root, "--ci") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "1 of 2 mode(s) did not: m_a" in err
    assert "m_a: 40 FAIL of 200" in err
    assert "m_b: 0 FAIL of 200" in err
    record = _gate_record(root)
    assert record.gate_failing_modes == (ModeKey(failure_type="m_a"),)
    assert {(g.mode.failure_type, g.n_total, g.n_fail) for g in record.gate_modes} == {
        ("m_a", 200, 40),
        ("m_b", 200, 0),
    }


def test_a_judge_re_run_with_new_critiques_does_not_double_the_sample(root, capsys):
    ids = [f"T{i:03d}" for i in range(40)]
    first = [
        _verdict(t, "FAIL" if i < 4 else "PASS", "m", "j@f", "run 1") for i, t in enumerate(ids)
    ]
    _store(root, ids, first)
    assert _run(root, "--ci") == EXIT_DOMAIN_ERROR  # 36/40: lower bound 0.795 < 0.80
    capsys.readouterr()

    rerun = [
        _verdict(t, "FAIL" if i < 4 else "PASS", "m", "j@f", "run 2") for i, t in enumerate(ids)
    ]
    _store(root, [], rerun)
    assert _run(root, "--ci") == EXIT_DOMAIN_ERROR
    assert "CI gate over 40 judge verdicts on 40 stored trace(s)" in capsys.readouterr().err
    assert _gate_record(root).gate_modes[0].n_total == 40


def test_the_newest_verdict_of_a_re_run_is_the_one_counted(root, capsys):
    ids = [f"T{i:03d}" for i in range(50)]
    _store(root, ids, [_verdict(t, "FAIL", "m", "j@f", "old") for t in ids])
    _store(root, [], [_verdict(t, "PASS", "m", "j@f", "new") for t in ids])
    assert _run(root, "--ci") == EXIT_OK
    assert _gate_record(root).gate_modes[0].n_fail == 0


def test_a_human_gate_counts_mode_free_passes_and_can_pass(root, capsys):
    ids = [f"H{i:03d}" for i in range(100)]
    labels = [_human(ids[0], "FAIL", "m")] + [_human(t, "PASS") for t in ids[1:]]
    _store(root, ids, labels)

    assert _run(root, "--ci", "--verdicts", "human") == EXIT_OK
    out = capsys.readouterr().out
    assert "CI gate over 100 human verdicts on 100 stored trace(s)" in out
    assert "PASSED" in out
    (mode,) = _gate_record(root).gate_modes
    assert (mode.n_total, mode.n_fail) == (100, 1)


def test_a_human_fail_for_another_mode_is_no_evidence_for_this_one(root, capsys):
    ids = ["H0", "H1", "H2"]
    labels = [_human("H0", "FAIL", "m_a"), _human("H1", "FAIL", "m_b"), _human("H2", "PASS")]
    _store(root, ids, labels)
    _run(root, "--verdicts", "human")
    counts = {g.mode.failure_type: (g.n_total, g.n_fail) for g in _gate_record(root).gate_modes}
    assert counts == {"m_a": (2, 1), "m_b": (2, 1)}


def test_a_mode_free_pass_counts_for_a_crew_mode_only_where_that_agent_ran(root, capsys):
    def crew_trace(tid: str, agents: list[str]) -> Trace:
        spans = [
            Span(
                span_id=f"{tid}/{a}",
                trace_id=tid,
                agent=a,
                role="specialist",
                parent=None,
                seq=i,
                input={},
                output={},
                status="completed",
            )
            for i, a in enumerate(agents)
        ]
        return Trace(
            trace_id=tid,
            frame="production-representative",
            ownership="owned",
            input={"m": tid},
            output={"a": "done"},
            taxonomy_cell={"feature": "x"},
            spans=spans,
        )

    with LocalBackend(root=root) as backend:
        backend.put_trace(crew_trace("C1", ["intake", "payout"]))
        backend.put_trace(crew_trace("C2", ["intake", "payout"]))
        backend.put_trace(crew_trace("O1", ["orderbot"]))
        backend.put_score(_human("C1", "FAIL", "misread", culprit_span="C1/intake"))
        backend.put_score(_human("C2", "PASS"))
        backend.put_score(_human("O1", "PASS"))
    _run(root, "--verdicts", "human")
    (gate,) = _gate_record(root).gate_modes
    assert gate.mode == ModeKey(failure_type="misread", agent="intake")
    assert (gate.n_total, gate.n_fail) == (2, 1)


def test_one_rater_failing_and_passing_a_mode_on_one_trace_is_refused(root, capsys):
    _store(root, ["H0"], [_human("H0", "FAIL", "m"), _human("H0", "PASS")])
    assert _run(root, "--verdicts", "human") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "both failed and passed" in err
    assert "'H0'" in err
    assert load_records_of(root, "run") == []


def test_two_raters_on_one_trace_are_two_observations(root, capsys):
    labels = [_human("H0", "FAIL", "m"), _human("H0", "PASS", rater_id="r2")]
    _store(root, ["H0"], labels)
    _run(root, "--verdicts", "human")
    (gate,) = _gate_record(root).gate_modes
    assert (gate.n_total, gate.n_fail) == (2, 1)


def test_span_defects_are_not_gated(root, capsys):
    labels = [
        _verdict("T0", "PASS", "m", "j@1"),
        Label(
            trace_id="T0",
            span_id="T0/orderbot",
            kind="llm-judge",
            outcome="FAIL",
            critique="x",
            failure_mode="m",
            span_defect=True,
            rater_id="j@1",
        ),
    ]
    _store(root, ["T0"], labels)
    _run(root)
    assert "CI gate over 1 judge verdicts" in capsys.readouterr().out
    (gate,) = _gate_record(root).gate_modes
    assert (gate.mode, gate.n_total, gate.n_fail) == (ModeKey(failure_type="m"), 1, 0)


def test_discovery_traces_are_left_out_of_the_gate_and_the_output_says_so(root, capsys):
    seed_verdicts(root, 60, 0, mode="m")
    seed_verdicts(root, 0, 15, mode="m", frame="discovery", prefix="D")
    assert _run(root, "--ci") == EXIT_OK
    out = capsys.readouterr().out
    assert "CI gate over 60 judge verdicts on 75 stored trace(s)" in out
    assert "15 judge verdicts on discovery-frame traces left out of the gate" in out
    assert _gate_record(root).gate_modes[0].n_total == 60


def test_a_store_of_only_discovery_verdicts_is_refused_and_says_why(root, capsys):
    seed_verdicts(root, 0, 5, frame="discovery")
    assert _run(root, "--ci") == EXIT_DOMAIN_ERROR
    assert "5 are on discovery-frame traces, which the gate does not read" in (
        capsys.readouterr().err
    )


def test_the_fixture_corpus_gates_one_verdict_per_trace_and_mode(root, capsys):
    with LocalBackend(root=root) as backend:
        for tid in ("ORD-1001", "ORD-1002", "CLM-4471", "CLM-4473", "CLM-4475", "CLM-4477"):
            backend.put_trace(load_corpus_trace(tid))
        for label in load_corpus_labels():
            backend.put_score(label)
    _run(root, "--verdicts", "human")
    counts = {
        (g.mode.agent, g.mode.failure_type): (g.n_total, g.n_fail)
        for g in _gate_record(root).gate_modes
    }
    # A span defect (CLM-4475 intake/misread_amount) is not a gated mode, and each FAIL trace
    # counts once even when a span defect repeats its mode (CLM-4471).
    assert counts == {
        ("intake", "dropped_material_fact"): (2, 1),
        ("orderbot", "trusted_broken_policy_tool"): (2, 1),
        ("policy-checker", "trusted_broken_policy_tool"): (2, 1),
        (None, "vacancy_threshold_margin_mishandled"): (3, 1),
    }
