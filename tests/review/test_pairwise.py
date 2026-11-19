# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The A/B page: one rater picks the better of two traces per subjective dimension.

Driven over the real local store seeded by `calibrated-agent-evaluation quickstart`, with choices
written through the same record store `calibrated-agent-evaluation prefer --rank` reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.records import append_records, load_records_of
from calibrated_agent_evaluation.subjective import SubjectiveStoreError, atom_set_from_dict

pytest.importorskip("fasthtml")

PAGE = {"HX-Request": "true"}
RATER = "rater-1"
ATOMS = {
    "criterion": "claim letter",
    "objective": [{"name": "cites_exclusion"}],
    "subjective": [{"name": "tone", "description": "courteous and plain"}],
}
CLAIMS = ("CLM-4471", "CLM-4473", "CLM-4475")


@pytest.fixture
def root(tmp_path, monkeypatch, capsys) -> Path:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    store = tmp_path / "store"
    assert main(["--root", str(store), "quickstart"]) == EXIT_OK
    capsys.readouterr()
    return store


def _client(root: Path, *, store=None, trace_ids=CLAIMS, rater=RATER):
    from starlette.testclient import TestClient

    from calibrated_agent_evaluation.review.pairwise import build_pairwise_app

    backend = LocalBackend(root=root)
    traces = [backend.get_trace(t) for t in trace_ids]
    app = build_pairwise_app(
        backend,
        atom_set=atom_set_from_dict(ATOMS),
        rater_id=rater,
        traces=traces,
        store=store or (lambda rs: append_records(root, "preference", rs)),
        done=load_records_of(root, "preference"),
    )
    return app, TestClient(app, base_url="http://127.0.0.1:5001")


def _choose(client, app, winner: str, headers=PAGE):
    item = app.state.pairwise.pending[0]
    response = client.post(
        "/choose", data={"item_id": item.item_id, "winner": winner}, headers=headers
    )
    return item, response


def test_the_page_queues_every_pair_once_per_subjective_dimension(root) -> None:
    app, client = _client(root)
    queue = app.state.pairwise.pending
    assert len(queue) == 3  # three traces, one dimension: C(3, 2)
    assert {frozenset((c.trace_id_a, c.trace_id_b)) for c in queue} == {
        frozenset(p) for p in (CLAIMS[:2], CLAIMS[1:], (CLAIMS[0], CLAIMS[2]))
    }
    page = client.get("/").text
    assert "Which is better on &#x27;tone&#x27;?" in page or "Which is better on 'tone'?" in page
    assert "courteous and plain" in page
    assert 'data-choice="a"' in page
    assert 'data-choice="b"' in page


def test_a_choice_is_stored_as_a_human_preference_and_the_page_advances(root) -> None:
    app, client = _client(root)
    item, response = _choose(client, app, "b")
    assert response.status_code == 200
    (record,) = load_records_of(root, "preference")
    assert (record.trace_id_a, record.trace_id_b, record.winner) == (
        item.trace_id_a,
        item.trace_id_b,
        "b",
    )
    assert record.judge_kind == "human"
    assert record.rater_id == RATER
    assert record.source == "review"
    assert len(app.state.pairwise.pending) == 2
    assert "2 of 3" in response.text


def test_skip_records_nothing_and_a_finished_queue_says_how_to_rank(root) -> None:
    app, client = _client(root)
    _choose(client, app, "skip")
    _choose(client, app, "a")
    _, last = _choose(client, app, "a")
    assert len(load_records_of(root, "preference")) == 2
    assert "All comparisons done" in last.text
    assert "Recorded 2, skipped 1" in last.text
    assert "calibrated-agent-evaluation prefer --rank" in last.text


def test_a_restarted_session_leaves_out_the_comparisons_already_recorded(root) -> None:
    app, client = _client(root)
    done, _ = _choose(client, app, "a")
    again, _ = _client(root)
    remaining = {frozenset((c.trace_id_a, c.trace_id_b)) for c in again.state.pairwise.pending}
    assert len(remaining) == 2
    assert frozenset((done.trace_id_a, done.trace_id_b)) not in remaining
    other, _ = _client(root, rater="second-rater")
    assert len(other.state.pairwise.pending) == 3


def test_a_stale_or_bad_choice_records_nothing_and_says_why(root) -> None:
    app, client = _client(root)
    stale = client.post("/choose", data={"item_id": "tone:x:y", "winner": "a"}, headers=PAGE)
    assert "no longer the current one" in stale.text
    _, bad = _choose(client, app, "tie")
    assert "Pick a or b" in bad.text
    assert load_records_of(root, "preference") == []
    assert len(app.state.pairwise.pending) == 3


def test_a_failed_write_keeps_the_comparison_current(root) -> None:
    def refuse(_records):
        raise SubjectiveStoreError("disk full")

    app, client = _client(root, store=refuse)
    item, response = _choose(client, app, "a")
    assert "Not recorded: disk full" in response.text
    assert app.state.pairwise.pending[0] == item


@pytest.mark.parametrize(
    ("headers", "reason"),
    [
        ({}, "HX-Request"),
        ({**PAGE, "Origin": "http://evil.example"}, "origin"),
        ({**PAGE, "Referer": "http://evil.example/page"}, "another site"),
    ],
)
def test_a_write_from_outside_the_page_is_refused(root, headers, reason) -> None:
    app, client = _client(root)
    _, response = _choose(client, app, "a", headers=headers)
    assert response.status_code == 403
    assert reason in response.text
    assert load_records_of(root, "preference") == []


def test_a_rebinding_host_is_refused(root) -> None:
    _, client = _client(root)
    assert client.get("/", headers={"Host": "attacker.example:5001"}).status_code == 400


def test_fewer_than_two_traces_or_no_rater_is_refused(root) -> None:
    from calibrated_agent_evaluation.review.errors import ReviewError

    with pytest.raises(ReviewError, match="two traces"):
        _client(root, trace_ids=("CLM-4471",))
    with pytest.raises(ReviewError, match="rater_id"):
        _client(root, rater=" ")


def test_an_atom_set_without_subjective_dimensions_is_refused() -> None:
    from calibrated_agent_evaluation.review.errors import ReviewError
    from calibrated_agent_evaluation.review.pairwise import comparisons

    objective_only = atom_set_from_dict({"criterion": "c", "objective": [{"name": "x"}]})
    with pytest.raises(ReviewError, match="no subjective dimension"):
        comparisons(["a", "b"], objective_only, rater_id=RATER)


def test_the_queue_order_and_sides_are_fixed_by_the_seed() -> None:
    from calibrated_agent_evaluation.review.pairwise import comparisons

    atoms = atom_set_from_dict(ATOMS)
    ids = [f"t{i}" for i in range(6)]
    first = comparisons(ids, atoms, rater_id=RATER, seed=7)
    assert first == comparisons(list(reversed(ids)), atoms, rater_id=RATER, seed=7)
    assert first != comparisons(ids, atoms, rater_id=RATER, seed=8)
    lefts = sum(c.trace_id_a < c.trace_id_b for c in first)
    assert 0 < lefts < len(first)  # neither side is always the lower id


def _label(root: Path, *argv: str) -> int:
    return main(["--root", str(root), "label", *argv])


def test_pairwise_cli_refusals_before_serving(root, tmp_path, capsys) -> None:
    atoms = tmp_path / "atoms.json"
    atoms.write_text(json.dumps(ATOMS))
    assert _label(root, "--pairwise", "--atoms", str(atoms)) == EXIT_USAGE
    assert "--pairwise is a mode of --serve" in capsys.readouterr().err
    assert _label(root, "--serve", "--pairwise", "--rater", RATER) == EXIT_USAGE
    assert "--pairwise needs --atoms" in capsys.readouterr().err
    assert (
        _label(
            root,
            "--serve",
            "--pairwise",
            "--atoms",
            str(atoms),
            "--rater",
            RATER,
            "--host",
            "0.0.0.0",
        )
        == EXIT_DOMAIN_ERROR
    )
    assert "refusing to serve" in capsys.readouterr().err
    assert (
        _label(
            root,
            "--serve",
            "--pairwise",
            "--atoms",
            str(atoms),
            "--rater",
            RATER,
            "--trace",
            "CLM-4471",
        )
        == EXIT_USAGE
    )
    assert "leaves nothing" in capsys.readouterr().err


def test_the_cli_store_reports_a_clashing_record_in_the_store_error_family(root) -> None:
    from calibrated_agent_evaluation._model import replace
    from calibrated_agent_evaluation.cli.stages.label_pairwise import preference_store

    app, client = _client(root, store=preference_store(root))
    _choose(client, app, "a")
    (stored,) = load_records_of(root, "preference")
    with pytest.raises(SubjectiveStoreError, match="already exists"):
        preference_store(root)([replace(stored, winner="b")])
