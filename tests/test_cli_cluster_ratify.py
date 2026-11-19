# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation cluster --ratify MAP` stamps ratified failure modes onto stored FAIL
labels.

Runs the real CLI over a real LocalBackend; the map is a real file and the codebook record is read
back through the CLI's record codec.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.schema import Label
from cli_store_helpers import trace


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    root = tmp_path / "store"
    with LocalBackend(root=root) as backend:
        for tid in ("tr1", "tr2", "tr3"):
            backend.put_trace(trace(tid))
        backend.put_score(_fail("tr1", "alice", "refunded 41 days after delivery"))
        backend.put_score(_fail("tr2", "alice", "called the wrong tool"))
        backend.put_score(_fail("tr2", "bob", "used lookup instead of refund"))
        backend.put_score(
            Label(
                trace_id="tr3",
                span_id=None,
                kind="human",
                outcome="PASS",
                critique="fine",
                rater_id="alice",
            )
        )
    return root


def _fail(trace_id: str, rater: str, critique: str, **kw: Any) -> Label:
    return Label(
        trace_id=trace_id,
        span_id=None,
        kind="human",
        outcome="FAIL",
        critique=critique,
        rater_id=rater,
        **kw,
    )


def _map(tmp_path: Path, assign: list[dict[str, Any]], **over: Any) -> Path:
    doc = {"codebook_version": "1", "ratified_by": "carol", "assign": assign, **over}
    path = tmp_path / "ratify.json"
    path.write_text(json.dumps(doc))
    return path


def ratify(root: Path, path: Path) -> int:
    return main(["--root", str(root), "cluster", "--ratify", str(path)])


def _labels(root: Path, trace_id: str) -> list[Label]:
    with LocalBackend(root=root) as backend:
        return backend.get_scores(trace_id)


ASSIGN = [
    {"trace_id": "tr1", "mode": "refund_outside_window"},
    {"trace_id": "tr2", "rater_id": "alice", "mode": "wrong_tool"},
    {"trace_id": "tr2", "rater_id": "bob", "mode": "wrong_tool"},
]


def test_ratify_tags_each_label_and_stores_the_codebook(root, tmp_path, capsys):
    modes = [
        {"name": "refund_outside_window", "description": "past the window"},
        {"name": "wrong_tool"},
    ]
    assert ratify(root, _map(tmp_path, ASSIGN, modes=modes)) == EXIT_OK
    out = capsys.readouterr().out
    assert "ratified 2 failure mode(s) under codebook version '1'" in out
    assert "tagged 3 FAIL label(s)" in out
    [t1] = _labels(root, "tr1")
    assert (t1.failure_mode, t1.codebook_version, t1.critique) == (
        "refund_outside_window",
        "1",
        "refunded 41 days after delivery",
    )
    assert {(lb.rater_id, lb.failure_mode) for lb in _labels(root, "tr2")} == {
        ("alice", "wrong_tool"),
        ("bob", "wrong_tool"),
    }
    [record] = load_records_of(root, "codebook")
    assert record.version == "1"
    assert record.ratified_by == "carol"
    assert [m.name for m in record.modes] == ["refund_outside_window", "wrong_tool"]
    assert record.modes[0].description == "past the window"
    assert record.n_tagged == 3


def test_ratifying_the_same_map_again_changes_no_label(root, tmp_path, capsys):
    path = _map(tmp_path, ASSIGN)
    ratify(root, path)
    before = [_labels(root, t) for t in ("tr1", "tr2")]
    assert ratify(root, path) == EXIT_OK
    assert "tagged 0 FAIL label(s); 3 already carried their mode" in capsys.readouterr().out
    assert [_labels(root, t) for t in ("tr1", "tr2")] == before


def test_a_unit_two_raters_failed_needs_a_rater_id(root, tmp_path, capsys):
    path = _map(tmp_path, [{"trace_id": "tr2", "mode": "wrong_tool"}])
    assert ratify(root, path) == EXIT_DOMAIN_ERROR
    assert "name the rater_id" in capsys.readouterr().err
    assert all(lb.failure_mode is None for lb in _labels(root, "tr2"))


def test_a_pass_or_a_missing_trace_is_refused_and_nothing_is_written(root, tmp_path, capsys):
    for target in ("tr3", "tr404"):
        path = _map(tmp_path, [ASSIGN[0], {"trace_id": target, "mode": "wrong_tool"}])
        assert ratify(root, path) == EXIT_DOMAIN_ERROR
        assert "no human FAIL label" in capsys.readouterr().err
    assert _labels(root, "tr1")[0].failure_mode is None
    assert load_records_of(root, "codebook") == []


def test_retagging_a_label_under_the_same_version_is_refused(root, tmp_path, capsys):
    ratify(root, _map(tmp_path, ASSIGN))
    changed = _map(tmp_path, [{"trace_id": "tr1", "mode": "wrong_tool"}])
    assert ratify(root, changed) == EXIT_DOMAIN_ERROR
    assert "new codebook_version" in capsys.readouterr().err
    assert _labels(root, "tr1")[0].failure_mode == "refund_outside_window"


def test_a_new_version_retags_and_records_its_parent(root, tmp_path, capsys):
    ratify(root, _map(tmp_path, ASSIGN))
    v2 = _map(tmp_path, [{"trace_id": "tr1", "mode": "late_refund"}], codebook_version="2")
    assert ratify(root, v2) == EXIT_OK
    [t1] = _labels(root, "tr1")
    assert (t1.failure_mode, t1.codebook_version) == ("late_refund", "2")
    newest = max(load_records_of(root, "codebook"), key=lambda r: r.version)
    assert newest.parent_version == "1"


def test_a_version_that_does_not_advance_is_refused(root, tmp_path, capsys):
    ratify(root, _map(tmp_path, ASSIGN, codebook_version="5"))
    old = _map(tmp_path, [{"trace_id": "tr1", "mode": "x"}], codebook_version="3")
    assert ratify(root, old) == EXIT_DOMAIN_ERROR
    assert "must sort strictly after" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        ("not json", "not valid JSON"),
        ("[]", "must be a JSON object"),
        ('{"codebook_version": "1", "ratified_by": "carol", "assign": []}', "assign"),
        (
            '{"codebook_version": "1", "ratified_by": "carol", "modes": [{"name": "a"}], '
            '"assign": [{"trace_id": "tr1", "mode": "b"}]}',
            "not listed in modes",
        ),
        (
            '{"ratified_by": "carol", "assign": [{"trace_id": "tr1", "mode": "b"}]}',
            "codebook_version",
        ),
    ],
)
def test_a_malformed_map_is_a_typed_refusal(root, tmp_path, capsys, doc, message):
    path = tmp_path / "bad.json"
    path.write_text(doc)
    assert ratify(root, path) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert message in err
    assert "Traceback" not in err


def test_two_modes_for_one_label_are_refused(root, tmp_path, capsys):
    path = _map(
        tmp_path,
        [{"trace_id": "tr1", "mode": "a"}, {"trace_id": "tr1", "mode": "b"}],
    )
    assert ratify(root, path) == EXIT_DOMAIN_ERROR
    assert "give it one mode" in capsys.readouterr().err


def test_ratified_modes_feed_the_saturation_count(root, tmp_path, capsys):
    ratify(root, _map(tmp_path, ASSIGN))
    capsys.readouterr()
    assert main(["--root", str(root), "cluster"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "no failure mode yet" not in out


# --- settling the modes labelers proposed from the review form --------------------------------


def _propose(root: Path, name: str, note: str, at: str, trace_id: str = "tr1") -> None:
    from _helpers import TEST_STAMP
    from calibrated_agent_evaluation.cli.records import append_records
    from calibrated_agent_evaluation.review.record import ProposedModeRecord, with_content_id

    record = with_content_id(
        ProposedModeRecord(
            proposal_id="",
            codebook_version="1",
            name=name,
            note=note,
            proposed_by="alice",
            trace_id=trace_id,
            proposed_at=at,
            provenance=TEST_STAMP,
        ),
        "proposal_id",
        "proposal",
    )
    append_records(root, "proposed-mode", [record])


@pytest.fixture
def proposed_root(root: Path) -> Path:
    """tr1 and tr2 (alice) FAIL with proposed modes under version 1, as the review form saves."""
    with LocalBackend(root=root) as backend:
        for tid in ("tr1", "tr2"):
            for old in backend.get_scores(tid):
                if old.rater_id == "alice":
                    backend.retract_score(old)
        backend.put_score(
            _fail(
                "tr1", "alice", "refund late", failure_mode="refund too late", codebook_version="1"
            )
        )
        backend.put_score(
            _fail(
                "tr2",
                "alice",
                "promised a call",
                failure_mode="Promised callback",
                codebook_version="1",
            )
        )
    _propose(root, "refund too late", "refunded after the window", "2026-10-26T10:00:00+00:00")
    _propose(
        root, "Promised callback", "callback promised, none set", "2026-10-26T10:05:00+00:00", "tr2"
    )
    return root


def _settle(tmp_path: Path, **over: Any) -> Path:
    doc = {"codebook_version": "1", "ratified_by": "carol", **over}
    path = tmp_path / "settle.json"
    path.write_text(json.dumps(doc))
    return path


def _alices(root: Path, trace_id: str) -> Label:
    [label] = [lb for lb in _labels(root, trace_id) if lb.rater_id == "alice"]
    return label


def test_ratify_adopts_one_proposal_and_merges_another(proposed_root, tmp_path, capsys):
    path = _settle(
        tmp_path,
        modes=[{"name": "refund_outside_window", "description": "past the window"}],
        adopt=["promised callback"],
        merge={"Refund Too Late": "refund_outside_window"},
    )
    assert ratify(proposed_root, path) == EXIT_OK
    out = capsys.readouterr().out
    assert (
        "Proposed modes settled: adopted Promised callback; merged refund too late into "
        "refund_outside_window." in out
    )
    assert _alices(proposed_root, "tr1").failure_mode == "refund_outside_window"
    assert _alices(proposed_root, "tr2").failure_mode == "Promised callback"
    [record] = load_records_of(proposed_root, "codebook")
    assert {m.name: m.description for m in record.modes} == {
        "refund_outside_window": "past the window",
        "Promised callback": "callback promised, none set",
    }
    assert record.merged == {"refund too late": "refund_outside_window"}
    from calibrated_agent_evaluation.cli.proposals import RecordsProposalStore

    assert RecordsProposalStore(proposed_root).pending("1") == ()


def test_an_adopted_proposal_is_a_ratified_mode_in_the_next_session(proposed_root, tmp_path):
    from argparse import Namespace

    from calibrated_agent_evaluation.cli.stages.codebook import resolve_session_codebook

    assert (
        ratify(proposed_root, _settle(tmp_path, adopt=["Promised callback", "refund too late"]))
        == EXIT_OK
    )
    book = resolve_session_codebook(Namespace(codebook_version="1"), proposed_root)
    assert [m.name for m in book.modes] == ["Promised callback", "refund too late"]


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"adopt": ["never proposed"]}, "is not a pending proposed mode"),
        ({"merge": {"refund too late": "no_such_mode"}}, "merge target 'no_such_mode'"),
        (
            {"adopt": ["refund too late"], "merge": {"Refund too late": "x"}},
            "both adopted and merged",
        ),
    ],
)
def test_settling_a_proposal_that_does_not_fit_is_refused_and_nothing_changes(
    proposed_root, tmp_path, capsys, over, message
):
    assert ratify(proposed_root, _settle(tmp_path, **over)) == EXIT_DOMAIN_ERROR
    assert message in capsys.readouterr().err
    assert load_records_of(proposed_root, "codebook") == []
    assert _alices(proposed_root, "tr1").failure_mode == "refund too late"
