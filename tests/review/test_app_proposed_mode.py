# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The picker's "new mode" escape: a FAIL saves with a proposed mode, flagged for the owner.

The proposals go through the CLI's real record store under a temporary local root, so a second
app on the same root stands in for another labeler's session.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.cli.proposals import RecordsProposalStore
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.review.record import CodebookMode

MODES = (
    CodebookMode(name="refund_outside_window", description="refunded past the return window"),
    CodebookMode(name="wrong_tool"),
)


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "store"


def _client(make_client: Callable[..., Any], backend: Any, root: Path, **kw: Any) -> Any:
    kw.setdefault("codebook_version", "2")
    kw.setdefault("failure_modes", MODES)
    client, _app = make_client(backend, proposals=RecordsProposalStore(root), **kw)
    return client


def _propose(client: Any, trace_id: str, name: str, note: str = "", **extra: str) -> Any:
    data = {
        "trace_id": trace_id,
        "outcome": "FAIL",
        "critique": "told the customer someone would call back; nobody was asked to",
        "new_mode_name": name,
        "new_mode_note": note,
        **extra,
    }
    return client.post("/label", data=data)


def test_the_form_offers_a_new_mode_with_the_names_it_checks_against(
    make_client: Callable[..., Any], backend: Any, store_root: Path
) -> None:
    html = _client(make_client, backend, store_root).get("/").text
    assert 'id="new_mode_name"' in html
    assert 'id="new_mode_note"' in html
    assert "data-mode-names=" in html
    assert "refund_outside_window" in html
    assert "propose a new failure mode" in html  # the legend lists N


def test_a_proposed_mode_saves_the_label_and_flags_it_for_the_codebook_owner(
    make_client: Callable[..., Any], backend: Any, store_root: Path
) -> None:
    client = _client(make_client, backend, store_root)
    response = _propose(client, "ORD-1001", "Promised  callback", "a callback promised, none set")
    assert response.headers["HD-Label-Status"] == "recorded"
    assert "Proposed failure mode 'Promised callback' recorded for the codebook owner" in (
        response.text
    )
    [label] = backend.get_scores("ORD-1001")
    assert (label.failure_mode, label.codebook_version) == ("Promised callback", "2")
    [proposal] = load_records_of(store_root, "proposed-mode")
    assert (proposal.name, proposal.codebook_version, proposal.proposed_by) == (
        "Promised callback",
        "2",
        label.rater_id,
    )
    assert (proposal.trace_id, proposal.note) == ("ORD-1001", "a callback promised, none set")
    # The next trace's picker offers it, marked as proposed.
    assert "Promised callback (proposed)" in response.text


def test_another_session_sees_the_proposal_and_reuses_it_instead_of_proposing_again(
    make_client: Callable[..., Any], backend: Any, store_root: Path
) -> None:
    first = _client(make_client, backend, store_root)
    _propose(first, "ORD-1001", "Promised callback", "a callback promised, none set")
    second = _client(make_client, backend, store_root, rater_id="bob")
    assert "Promised callback (proposed)" in second.get("/").text
    response = _propose(second, "ORD-1002", "promised   CALLBACK", "same thing")
    assert "already in the picker as 'Promised callback'" in response.text
    [label] = [lb for lb in backend.get_scores("ORD-1002") if lb.rater_id == "bob"]
    assert label.failure_mode == "Promised callback"
    assert len(load_records_of(store_root, "proposed-mode")) == 1


def test_a_new_name_that_is_a_listed_mode_uses_the_listed_mode(
    make_client: Callable[..., Any], backend: Any, store_root: Path
) -> None:
    client = _client(make_client, backend, store_root)
    response = _propose(client, "ORD-1001", "Wrong_Tool")
    assert response.headers["HD-Label-Status"] == "recorded"
    [label] = backend.get_scores("ORD-1001")
    assert label.failure_mode == "wrong_tool"
    assert load_records_of(store_root, "proposed-mode") == []


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({}, "needs a note"),
        ({"new_mode_note": "x", "failure_mode": "wrong_tool"}, "not both"),
        ({"new_mode_note": "x", "outcome": "PASS"}, "only a FAIL"),
    ],
)
def test_a_proposal_that_cannot_stand_is_refused_and_nothing_is_written(
    make_client: Callable[..., Any],
    backend: Any,
    store_root: Path,
    extra: dict[str, str],
    message: str,
) -> None:
    client = _client(make_client, backend, store_root)
    note = extra.pop("new_mode_note", "")
    response = _propose(client, "ORD-1001", "Promised callback", note, **extra)
    assert response.headers["HD-Label-Status"] == "refused"
    assert message in response.text
    assert "Promised callback" in response.text  # what was typed comes back
    assert backend.get_scores("ORD-1001") == []
    assert load_records_of(store_root, "proposed-mode") == []


def test_without_a_codebook_list_there_is_no_new_mode_and_a_posted_one_is_refused(
    make_client: Callable[..., Any], backend: Any, store_root: Path
) -> None:
    client = _client(make_client, backend, store_root, failure_modes=(), codebook_version=None)
    assert 'id="new_mode_name"' not in client.get("/").text
    response = _propose(client, "ORD-1001", "Promised callback", "note")
    assert response.headers["HD-Label-Status"] == "refused"
    assert "no codebook version and failure-mode list" in response.text
    assert backend.get_scores("ORD-1001") == []


def test_a_proposal_that_cannot_be_stored_leaves_the_trace_unlabeled(
    make_client: Callable[..., Any], backend: Any, tmp_path: Path
) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("")
    client = _client(make_client, backend, blocked)
    response = _propose(client, "ORD-1001", "Promised callback", "note")
    assert response.headers["HD-Label-Status"] == "error"
    assert "the proposed failure mode was not stored, so the label was not either" in response.text
    assert backend.get_scores("ORD-1001") == []
