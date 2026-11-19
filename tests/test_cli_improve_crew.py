# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation improve --evaluate --crew-file` evaluates a crew fix from two
ClaimsCrew stores.

The before and after stores hold 40 copies of the four ClaimsCrew corpus traces (each copy a new
trace id with its spans renamed to match), run through the real `LocalBackend`. The workflow
outcome is one trace-level failure mode, `wrong_claim_decision`; each FAIL names its culprit span
and, where the walk-back found one, the handoff, the way the corpus gold labels do.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from _helpers import load_corpus_trace
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.schema.types import Label, Trace

MODE = "wrong_claim_decision"
N = 40
BASES = ("CLM-4471", "CLM-4473", "CLM-4475", "CLM-4477")
INTAKE_SEAM = "intake->policy-checker"
CREW = {
    "agents": ["supervisor", "intake", "policy-checker", "fraud-scorer", "payout", "escalation"],
    "handoffs": [
        "intake->policy-checker",
        "policy-checker->fraud-scorer",
        "fraud-scorer->payout",
        "fraud-scorer->escalation",
    ],
    "scenarios": {
        "S2_vacancy_exclusion": {
            "expected_agents": ["supervisor", "intake", "policy-checker", "fraud-scorer", "payout"],
            "terminal_decision": "deny",
            "payout_allowed": False,
            "escalation_required": False,
        }
    },
    "rules": [
        {
            "rule_id": "R2",
            "antecedent": "fraud flag raised",
            "consequent_agent": "escalation",
            "deterministic": False,
        }
    ],
}


def _tid(i: int) -> str:
    return f"{BASES[i % len(BASES)]}-r{i:02d}"


def _copy(i: int) -> Trace:
    """Corpus trace ``BASES[i % 4]`` under a new trace id, its span ids renamed to match."""
    base = BASES[i % len(BASES)]
    new = _tid(i)
    doc = load_corpus_trace(base).model_dump(mode="json")
    doc["trace_id"] = new
    for span in doc["spans"]:
        span["trace_id"] = new
        span["span_id"] = span["span_id"].replace(base, new, 1)
        if span["parent"]:
            span["parent"] = span["parent"].replace(base, new, 1)
    return Trace.model_validate(doc)


def _human(i: int, culprit: str | None, handoff: str | None = None) -> Label:
    """A trace-level human verdict on copy ``i``: PASS when ``culprit`` is None, else a FAIL."""
    tid = _tid(i)
    if culprit is None:
        return Label(trace_id=tid, span_id=None, kind="human", outcome="PASS", critique="correct")
    return Label(
        trace_id=tid,
        span_id=None,
        kind="human",
        outcome="FAIL",
        critique="wrong decision on the claim",
        failure_mode=MODE,
        culprit_span=culprit if culprit == "disputed" else f"{tid}/{culprit}",
        manifestation_span=f"{tid}/payout",
        handoff_implicated=handoff,
        rater_id="rater-1",
    )


def _store(path: Path, culprits: dict[int, str], handoffs: dict[int, str] | None = None) -> Path:
    """40 corpus copies, a human FAIL blaming ``culprits[i]`` on copy ``i`` and a PASS elsewhere."""
    handoffs = handoffs or {}
    with LocalBackend(root=path) as backend:
        for i in range(N):
            backend.put_trace(_copy(i))
            backend.put_score(_human(i, culprits.get(i), handoffs.get(i)))
    return path


def _crew_file(tmp_path: Path, doc: object = None, name: str = "crew.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(CREW if doc is None else doc), encoding="utf-8")
    return path


def _dev(tmp_path: Path) -> Path:
    path = tmp_path / "dev.txt"
    path.write_text("CLM-4471\nCLM-4473\n", encoding="utf-8")
    return path


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    return tmp_path / "records-root"


def _register(root: Path, *, crew: bool = True) -> None:
    argv = ["--root", str(root), "improve", "--register", MODE, "--min-reduction", "0.05"]
    assert main([*argv, *(["--crew"] if crew else [])]) == EXIT_OK


def _evaluate(root: Path, before: Path, after: Path, dev: Path, *extra: str) -> int:
    return main(
        [
            "--root",
            str(root),
            "improve",
            "--evaluate",
            MODE,
            "--before-root",
            str(before),
            "--after-root",
            str(after),
            "--development",
            str(dev),
            *extra,
        ]
    )


def _crew_args(crew_file: Path, touched: str = "intake", source: str = "human") -> list[str]:
    return ["--crew-file", str(crew_file), "--touched-agent", touched, "--verdicts", source]


def _fixed(tmp_path: Path) -> tuple[Path, Path]:
    """Intake caused 14 of 40 wrong decisions before the fix and 2 after."""
    before = _store(
        tmp_path / "before",
        dict.fromkeys(range(14), "intake"),
        dict.fromkeys(range(14), INTAKE_SEAM),
    )
    after = _store(
        tmp_path / "after", dict.fromkeys(range(2), "intake"), dict.fromkeys(range(2), INTAKE_SEAM)
    )
    return before, after


def test_a_crew_fix_that_cuts_the_workflow_failures_is_accepted(root, tmp_path, capsys):
    _register(root)
    before, after = _fixed(tmp_path)
    crew_file = _crew_file(tmp_path)
    capsys.readouterr()
    assert _evaluate(root, before, after, _dev(tmp_path), *_crew_args(crew_file)) == EXIT_OK
    out = capsys.readouterr().out
    assert f"improve {MODE!r}: ACCEPTED" in out
    assert "workflow failures 14 of 40 before, 2 of 40 after" in out
    assert "relocation check: improved" in out
    assert "agent intake: 14 failed trace(s) before, 2 after (touched)" in out
    assert "agent policy-checker: 0 failed trace(s) before, 0 after" in out
    assert f"handoff {INTAKE_SEAM}: 14 failed trace(s) before, 2 after" in out
    # Escalation never runs in ClaimsCrew, so the human labels cannot speak for it or its seam.
    assert "Not measured by the human verdicts, so not guarded: escalation" in out
    assert "escalation (agent): not measured by the human verdicts; held at p=1" in out
    assert "guard panel: 9 slice(s), Holm; regressed: none" in out

    (record,) = load_records_of(root, "improve")
    assert record.verdict == "accepted"
    assert record.panel.relocation is not None
    assert record.panel.relocation.verdict == "improved"
    assert record.panel.m == 9
    assert {s.kind for s in record.panel.slices} == {"agent", "handoff"}
    assert "intake" not in {s.name for s in record.panel.slices}
    inputs = record.inputs
    assert inputs["crew_file"] == str(crew_file)
    assert inputs["crew_file_sha256"] == hashlib.sha256(crew_file.read_bytes()).hexdigest()
    assert inputs["touched_agent"] == "intake"
    assert inputs["agent_failures_before"] == (
        "supervisor=0,intake=14,policy-checker=0,fraud-scorer=0,payout=0,escalation=0"
    )
    assert inputs["agent_failures_after"].startswith("supervisor=0,intake=2,")
    assert inputs["handoff_failures_before"].startswith(f"{INTAKE_SEAM}=14,")
    assert inputs["not_measured"] == "escalation,fraud-scorer->escalation"
    assert inputs["disputed_before"] == inputs["disputed_after"] == "none"


def test_failures_that_move_to_another_agent_are_a_relocation(root, tmp_path, capsys):
    _register(root)
    before = _store(tmp_path / "before", dict.fromkeys(range(14), "intake"))
    after = _store(
        tmp_path / "after",
        {**dict.fromkeys(range(2), "intake"), **dict.fromkeys(range(2, 14), "policy-checker")},
    )
    capsys.readouterr()
    code = _evaluate(root, before, after, _dev(tmp_path), *_crew_args(_crew_file(tmp_path)))
    assert code == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "the failures moved to another step instead of going away" in err
    assert "relocation check: relocation" in err
    assert "agent policy-checker: 0 failed trace(s) before, 12 after" in err
    assert "regressed: policy-checker" in err
    (record,) = load_records_of(root, "improve")
    assert record.verdict == "rejected"
    assert record.panel.relocation is not None
    assert record.panel.relocation.relocated_from == "intake"
    assert any("failure relocation" in r for r in record.reasons)
    assert "guard slices regressed: policy-checker" in record.reasons
    assert not any("[" in r or "relocated_from=" in r for r in record.reasons)


def test_a_handoff_that_regresses_rejects_the_fix(root, tmp_path, capsys):
    _register(root)
    seam = "policy-checker->fraud-scorer"
    before = _store(tmp_path / "before", dict.fromkeys(range(14), "intake"))
    after = _store(
        tmp_path / "after",
        {**dict.fromkeys(range(2), "intake"), **dict.fromkeys(range(20, 32), "disputed")},
        dict.fromkeys(range(20, 32), seam),
    )
    capsys.readouterr()
    code = _evaluate(root, before, after, _dev(tmp_path), *_crew_args(_crew_file(tmp_path)))
    assert code == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert f"handoff {seam}: 0 failed trace(s) before, 12 after" in err
    assert f"regressed: {seam}" in err
    (record,) = load_records_of(root, "improve")
    assert record.verdict == "rejected"
    assert record.panel.regressed == (seam,)


def test_a_disputed_culprit_counts_for_no_agent_but_for_the_workflow(root, tmp_path, capsys):
    _register(root)
    # Copy 3 is CLM-4477, whose corpus gold could not choose between intake and policy-checker.
    culprits = {**dict.fromkeys(range(14), "intake"), 3: "disputed"}
    before = _store(tmp_path / "before", culprits, {3: INTAKE_SEAM})
    after = _store(tmp_path / "after", dict.fromkeys(range(2), "intake"))
    capsys.readouterr()
    assert (
        _evaluate(root, before, after, _dev(tmp_path), *_crew_args(_crew_file(tmp_path))) == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "1 trace(s) have a failure whose culprit is disputed" in out
    assert "'CLM-4477-r03'" in out
    assert "workflow failures 14 of 40 before" in out
    assert "agent intake: 13 failed trace(s) before, 2 after (touched)" in out
    assert f"handoff {INTAKE_SEAM}: 1 failed trace(s) before, 0 after" in out
    (record,) = load_records_of(root, "improve")
    assert record.inputs["disputed_before"] == "CLM-4477-r03"
    assert record.inputs["disputed_after"] == "none"
    assert "intake=13" in record.inputs["agent_failures_before"]
    assert record.panel.relocation is not None
    assert record.panel.relocation.workflow_outcome_rate_before == pytest.approx(14 / 40)


def _judged_store(path: Path, workflow_fails: set[int], intake_fails: set[int]) -> Path:
    """Judge verdicts only: a workflow judge on each trace and an intake span judge on each."""
    with LocalBackend(root=path) as backend:
        for i in range(N):
            tid = _tid(i)
            backend.put_trace(_copy(i))
            backend.put_score(
                Label(
                    trace_id=tid,
                    span_id=None,
                    kind="llm-judge",
                    outcome="FAIL" if i in workflow_fails else "PASS",
                    critique="workflow verdict",
                    failure_mode=MODE,
                )
            )
            backend.put_score(
                Label(
                    trace_id=tid,
                    span_id=f"{tid}/intake",
                    kind="llm-judge",
                    outcome="FAIL" if i in intake_fails else "PASS",
                    critique="intake verdict",
                    failure_mode="dropped_material_fact",
                )
            )
    return path


def _adjudicate(path: Path, ids: range) -> None:
    with LocalBackend(root=path) as backend:
        for i in ids:
            backend.put_score(_human(i, None))


def test_slices_the_judges_cannot_measure_stay_at_p_one_and_are_named(root, tmp_path, capsys):
    _register(root)
    before = _judged_store(tmp_path / "before", set(range(14)), set(range(14)))
    after = _judged_store(tmp_path / "after", {0, 1}, {0, 1})
    _adjudicate(after, range(2, 14))
    capsys.readouterr()
    args = _crew_args(_crew_file(tmp_path), source="judge")
    assert _evaluate(root, before, after, _dev(tmp_path), *args) == EXIT_OK
    out = capsys.readouterr().out
    # The judge runtime writes no handoff, and only intake has a span judge.
    assert (
        "Not measured by the judge verdicts, so not guarded: supervisor, policy-checker, "
        "fraud-scorer, payout, escalation, intake->policy-checker, policy-checker->fraud-scorer, "
        "fraud-scorer->payout, fraud-scorer->escalation."
    ) in out
    assert "agent intake: 14 failed trace(s) before, 2 after (touched)" in out
    assert "12 judge-scored flip(s) adjudicated by a human" in out
    (record,) = load_records_of(root, "improve")
    assert record.panel.m == 9
    assert all(s.regression_p == 1.0 for s in record.panel.slices)
    assert all(s.rate_before.point is None for s in record.panel.slices)
    assert record.inputs["not_measured"].split(",")[0] == "supervisor"


def test_a_slice_measured_on_one_side_only_is_refused(root, tmp_path, capsys):
    _register(root)
    before = _judged_store(tmp_path / "before", set(range(14)), set(range(14)))
    after = _judged_store(tmp_path / "after", {0, 1}, {0, 1})
    with LocalBackend(root=before) as backend:
        backend.put_score(
            Label(
                trace_id=_tid(0),
                span_id=f"{_tid(0)}/payout",
                kind="llm-judge",
                outcome="PASS",
                critique="payout verdict",
                failure_mode="paid_excluded_claim",
            )
        )
    capsys.readouterr()
    args = _crew_args(_crew_file(tmp_path), source="judge")
    assert _evaluate(root, before, after, _dev(tmp_path), *args) == EXIT_DOMAIN_ERROR
    assert "measure agent 'payout' (before only) on one side only" in capsys.readouterr().err
    assert load_records_of(root, "improve") == []


def test_a_culprit_the_crew_file_does_not_declare_is_refused(root, tmp_path, capsys):
    _register(root)
    before = _store(tmp_path / "before", dict.fromkeys(range(14), "intake"))
    after = _store(tmp_path / "after", {0: "policy-checker"})
    narrow = {"agents": ["supervisor", "intake", "fraud-scorer", "payout"], "handoffs": []}
    capsys.readouterr()
    args = _crew_args(_crew_file(tmp_path, narrow))
    assert _evaluate(root, before, after, _dev(tmp_path), *args) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "the after store blames agent 'policy-checker' on 'CLM-4471-r00'" in err
    assert "the crew file and the store describe different crews" in err


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "is not valid JSON"),
        ("[1, 2]", "must hold one JSON object"),
        (json.dumps({**CREW, "agnets": []}), "agnets: Extra inputs are not permitted"),
        (json.dumps({"agents": ["intake", "intake"]}), "lists agent(s) more than once"),
        (
            json.dumps({"agents": ["intake"], "handoffs": ["intake->payout"]}),
            "handoff 'intake->payout' names agent(s) ['payout'] not in the declared crew",
        ),
        (json.dumps({"agents": []}), "must declare at least one agent"),
    ],
)
def test_a_malformed_crew_file_is_refused_before_either_store_is_read(
    root, tmp_path, capsys, content, message
):
    _register(root)
    # Empty directories: a refusal that read them would complain about the stores instead.
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    crew_file = tmp_path / "crew.json"
    crew_file.write_text(content, encoding="utf-8")
    capsys.readouterr()
    assert _evaluate(root, before, after, _dev(tmp_path), *_crew_args(crew_file)) == (
        EXIT_DOMAIN_ERROR
    )
    err = capsys.readouterr().err
    assert message in err
    assert str(crew_file) in err
    assert not (before / "store.sqlite3").exists()


def test_an_unknown_touched_agent_is_refused(root, tmp_path, capsys):
    _register(root)
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    capsys.readouterr()
    args = _crew_args(_crew_file(tmp_path), touched="adjuster")
    assert _evaluate(root, before, after, _dev(tmp_path), *args) == EXIT_DOMAIN_ERROR
    assert "--touched-agent 'adjuster' is not an agent in" in capsys.readouterr().err


def test_a_missing_crew_file_is_refused(root, tmp_path, capsys):
    _register(root)
    before, after = _fixed(tmp_path)
    capsys.readouterr()
    args = _crew_args(tmp_path / "no-such-crew.json")
    assert _evaluate(root, before, after, _dev(tmp_path), *args) == EXIT_DOMAIN_ERROR
    assert "cannot read --crew-file" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--touched-agent", "intake"], "--touched-agent needs --crew-file"),
        (["--crew-file", "crew.json"], "needs --touched-agent NAME"),
    ],
)
def test_half_a_crew_is_refused(root, tmp_path, capsys, extra, message):
    _register(root)
    before, after = _fixed(tmp_path)
    capsys.readouterr()
    assert _evaluate(root, before, after, _dev(tmp_path), *extra) == EXIT_USAGE
    assert message in capsys.readouterr().err


def test_a_crew_file_for_a_single_agent_metric_is_refused(root, tmp_path, capsys):
    _register(root, crew=False)
    before, after = _fixed(tmp_path)
    capsys.readouterr()
    args = _crew_args(_crew_file(tmp_path))
    assert _evaluate(root, before, after, _dev(tmp_path), *args) == EXIT_DOMAIN_ERROR
    assert f"{MODE!r} was not registered as a crew metric" in capsys.readouterr().err
    assert load_records_of(root, "improve") == []


def test_a_crew_metric_without_a_crew_file_is_refused(root, tmp_path, capsys):
    _register(root)
    before, after = _fixed(tmp_path)
    capsys.readouterr()
    assert _evaluate(root, before, after, _dev(tmp_path), "--verdicts", "human") == (
        EXIT_DOMAIN_ERROR
    )
    assert f"{MODE!r} was registered as a crew metric (--crew)" in capsys.readouterr().err


def test_a_crew_metric_from_a_request_file_is_refused(root, tmp_path, capsys):
    _register(root)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "target_metric": MODE,
                "pairs": [{"trace_id": "c-1", "before_fail": True, "after_fail": False}],
                "development_ids": ["d-1"],
            }
        ),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert main(["--root", str(root), "improve", "--request", str(request)]) == EXIT_DOMAIN_ERROR
    assert "not --request" in capsys.readouterr().err


def test_crew_flags_without_evaluate_are_refused(root, tmp_path, capsys):
    argv = ["--root", str(root), "improve", "--register", MODE, "--min-reduction", "0.05"]
    capsys.readouterr()
    assert main([*argv, "--touched-agent", "intake"]) == EXIT_USAGE
    assert "go with --evaluate METRIC" in capsys.readouterr().err
