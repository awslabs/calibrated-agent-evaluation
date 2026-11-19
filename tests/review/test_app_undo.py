# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Undo retracts the stored label, not just the session's memory of it.

says undo retracts the most recently submitted label. ``/undo`` used to pop the session's
``done`` stack and re-queue the trace, never touching the backend, so a corrected verdict left both
the retracted and the new verdict in the store and ``calibrated-agent-evaluation run --ci`` voted on
both. These tests assert on the store (a real ``LocalBackend``), which is the evidence the old
acceptance row lacked.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.schema import BackendError, Label, Trace


def _seeded(cls: type[LocalBackend], root: Path, traces: list[Trace]) -> LocalBackend:
    backend = cls(root)
    # Starlette's TestClient drives the app from a portal thread, and LocalBackend's connection is
    # thread-affine (the served app keeps all access on one thread). The
    # test reopens the same file without the affinity check so the real store sits under the app.
    backend._conn.close()
    backend._conn = sqlite3.connect(root / "store.sqlite3", check_same_thread=False)
    for trace in traces:
        backend.put_trace(trace)
    return backend


@pytest.fixture
def store(corpus_traces: list[Trace], tmp_path: Path) -> LocalBackend:
    return _seeded(LocalBackend, tmp_path / "store", corpus_traces)


def test_undo_removes_the_label_from_the_store(
    make_client: Callable[..., Any], store: LocalBackend
) -> None:
    client, app = make_client(store)
    client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": ""})
    assert len(store.get_scores("ORD-1001")) == 1

    client.post("/undo")

    assert store.get_scores("ORD-1001") == []
    assert app.state.review_session.current() == "ORD-1001"


def test_a_corrected_verdict_leaves_only_the_correction_in_the_store(
    make_client: Callable[..., Any], store: LocalBackend
) -> None:
    """The ticket's failure scenario: PASS, notice the misread, undo, FAIL with a critique."""
    client, _app = make_client(store)
    client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": ""})
    client.post("/undo")
    client.post(
        "/label",
        data={"trace_id": "ORD-1001", "outcome": "FAIL", "critique": "refund skipped policy"},
    )

    [only] = store.get_scores("ORD-1001")
    assert only.outcome == "FAIL"
    assert only.critique == "refund skipped policy"


def test_undo_also_retracts_the_span_defect_label_written_with_the_verdict(
    make_client: Callable[..., Any], store: LocalBackend
) -> None:
    client, _app = make_client(store)
    client.post(
        "/label",
        data={
            "trace_id": "CLM-4471",
            "outcome": "FAIL",
            "critique": "intake dropped occupancy",
            "defect_span": "CLM-4471/intake",
        },
    )
    assert len(store.get_scores("CLM-4471")) == 2  # trace-level + span-level defect

    client.post("/undo")

    assert store.get_scores("CLM-4471") == []


def test_undo_with_nothing_labeled_changes_nothing(
    make_client: Callable[..., Any], store: LocalBackend
) -> None:
    client, app = make_client(store)
    before = list(app.state.review_session.pending)
    response = client.post("/undo")
    assert response.status_code == 200
    assert app.state.review_session.pending == before


class _RetractFails(LocalBackend):
    def retract_score(self, label: Label) -> None:
        raise BackendError("store unreachable")


def test_a_failed_retraction_is_surfaced_and_the_label_stays_labeled(
    make_client: Callable[..., Any], corpus_traces: list[Trace], tmp_path: Path
) -> None:
    """No silent failure: if the store kept the label, the session must not pretend it didn't."""
    store = _seeded(_RetractFails, tmp_path / "store", corpus_traces)
    client, app = make_client(store)
    client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": ""})

    response = client.post("/undo")

    assert "Undo failed" in response.text
    assert response.headers.get("HD-Label-Status") == "error"
    assert len(store.get_scores("ORD-1001")) == 1
    assert "ORD-1001" not in app.state.review_session.pending  # still shown as labeled, truthfully


_CREW_FAIL = {
    "trace_id": "CLM-4471",
    "outcome": "FAIL",
    "critique": "intake dropped occupancy",
    "defect_span": "CLM-4471/intake",
}


def test_a_failed_timing_log_write_on_undo_leaves_no_span_label_behind(
    make_client: Callable[..., Any], store: LocalBackend, tmp_path: Path
) -> None:
    log_path = tmp_path / "sessions.jsonl"
    client, _app = make_client(store, session_log=log_path)
    client.post("/label", data=_CREW_FAIL)
    log_path.chmod(0o400)  # the undo's log line cannot be appended
    try:
        response = client.post("/undo")
    finally:
        log_path.chmod(0o600)

    assert "timing log was not written" in response.text
    assert store.get_scores("CLM-4471") == []


class _VerdictRetractFails(LocalBackend):
    def retract_score(self, label: Label) -> None:
        if label.span_id is None:
            raise BackendError("store unreachable")
        super().retract_score(label)


def test_a_failed_verdict_retraction_puts_its_span_labels_back(
    make_client: Callable[..., Any], corpus_traces: list[Trace], tmp_path: Path
) -> None:
    store = _seeded(_VerdictRetractFails, tmp_path / "store", corpus_traces)
    client, _app = make_client(store)
    client.post("/label", data=_CREW_FAIL)

    response = client.post("/undo")

    assert "Undo failed" in response.text
    stored = store.get_scores("CLM-4471")
    assert {label.span_id for label in stored} == {None, "CLM-4471/intake"}
