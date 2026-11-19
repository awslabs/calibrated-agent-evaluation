# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP coverage for the session-flow routes.

`/defer`, `/undo`, `/summary` and `/trace/{id}` had no HTTP test, and the one undo test never
looked at a backend. `/undo` and `/trace/{id}` are now covered in test_app_undo.py and
test_review_checklist.py; this file covers `/defer` and `/summary` against the store.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibrated_agent_evaluation.review import ReviewSession

from .conftest import FakeBackend


class _Manual:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_defer_writes_nothing_and_moves_the_trace_to_the_end(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    client, app = make_client(backend)
    response = client.post("/defer")
    assert response.status_code == 200
    assert backend.labels == []
    assert app.state.review_session.pending == ["ORD-1002", "CLM-4471", "ORD-1001"]
    assert 'value="ORD-1002" name="trace_id"' in response.text  # the next trace is shown


def test_defer_of_the_last_trace_keeps_it_reachable(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    client, app = make_client(backend)
    for trace_id in ("ORD-1001", "ORD-1002"):
        client.post("/label", data={"trace_id": trace_id, "outcome": "PASS", "critique": ""})
    response = client.post("/defer")
    assert app.state.review_session.pending == ["CLM-4471"]
    assert 'value="CLM-4471" name="trace_id"' in response.text


def test_summary_reports_what_the_store_and_session_hold(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    clock = _Manual()
    session = ReviewSession.from_ids(["ORD-1001", "ORD-1002", "CLM-4471"], clock=clock)
    client, _app = make_client(backend, session=session)
    client.get("/")
    clock.t = 6.0
    client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": ""})
    clock.t = 8.0
    client.post("/defer")
    summary = client.get("/summary").json()
    assert summary["labeled_count"] == len(backend.labels) == 1
    assert summary["labeled_seconds"] == 6.0
    assert summary["elapsed_seconds"] >= 8.0  # the deferred trace's 2s count too
