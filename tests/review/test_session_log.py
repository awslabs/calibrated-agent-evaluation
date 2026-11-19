# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-label timing is persisted as it happens."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.review import ReviewError, ReviewSession
from calibrated_agent_evaluation.review.session_log import SessionLog

from .conftest import GOLD_RATER, FakeBackend


class _Manual:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_each_verdict_appends_one_line_with_its_time_on_trace(
    make_client: Callable[..., Any], backend: FakeBackend, tmp_path: Path
) -> None:
    log = tmp_path / "sessions" / "review.jsonl"
    clock = _Manual()
    session = ReviewSession.from_ids(["ORD-1001", "ORD-1002", "CLM-4471"], clock=clock)
    client, _app = make_client(backend, session=session, session_log=log)
    client.get("/")
    clock.t = 4.0
    client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": ""})
    # written per submission, so a crash after this point keeps it
    [first] = _lines(log)
    assert first["event"] == "label"
    assert first["trace_id"] == "ORD-1001"
    assert first["rater_id"] == GOLD_RATER
    assert first["seconds_on_trace"] == 4.0
    assert first["outcome"] == "PASS"
    assert first["recorded_at"].endswith("+00:00")

    client.post(
        "/label",
        data={
            "trace_id": "CLM-4471",
            "outcome": "FAIL",
            "critique": "intake dropped occupancy",
            "defect_span": "CLM-4471/intake",
        },
    )
    assert _lines(log)[-1]["span_labels"] == ["CLM-4471/intake"]


def test_an_undo_is_logged_so_a_reader_can_drop_the_retracted_verdict(
    make_client: Callable[..., Any], backend: FakeBackend, tmp_path: Path
) -> None:
    log = tmp_path / "review.jsonl"
    client, _app = make_client(backend, session_log=log)
    client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": ""})
    client.post("/undo")
    assert [(r["event"], r["trace_id"]) for r in _lines(log)] == [
        ("label", "ORD-1001"),
        ("undo", "ORD-1001"),
    ]


def test_an_unwritable_log_is_refused_at_startup(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(ReviewError, match="session log"):
        SessionLog(blocker / "review.jsonl")


def test_a_log_write_that_fails_after_the_verdict_is_stored_is_reported(
    make_client: Callable[..., Any], backend: FakeBackend, tmp_path: Path
) -> None:
    log = tmp_path / "review.jsonl"
    client, _app = make_client(backend, session_log=log)
    log.unlink()
    log.mkdir()  # appending to a directory fails
    response = client.post(
        "/label",
        data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": ""},
        headers={"HD-Optimistic": "1"},
    )
    assert response.headers.get("HD-Label-Status") == "error"  # so the client shows the notice
    assert "session timing log was not written" in response.text
    assert len(backend.get_scores("ORD-1001")) == 1  # the verdict itself is stored
