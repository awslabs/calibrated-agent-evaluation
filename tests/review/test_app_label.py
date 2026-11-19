# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Label route: record + advance, invalid-outcome refusal, submit-failure reconciliation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibrated_agent_evaluation.schema import BackendError


def test_label_records_outcome_and_advances(make_client: Callable[..., Any], backend: Any) -> None:
    client, app = make_client(backend)
    response = client.post(
        "/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": "ok"}
    )
    assert response.status_code == 200
    scores = backend.get_scores("ORD-1001")
    assert len(scores) == 1
    assert scores[0].outcome == "PASS"
    assert scores[0].kind == "human"
    assert "ORD-1001" not in app.state.review_session.pending  # advanced past the labeled trace


def test_invalid_outcome_is_rejected(make_client: Callable[..., Any], backend: Any) -> None:
    client, _app = make_client(backend)
    response = client.post(
        "/label", data={"trace_id": "ORD-1001", "outcome": "MAYBE", "critique": "x"}
    )
    assert "Not recorded" in response.text
    assert backend.get_scores("ORD-1001") == []


def test_submit_failure_leaves_trace_unlabeled(
    make_client: Callable[..., Any], make_backend: Callable[..., Any]
) -> None:
    backend = make_backend(submit_error=BackendError("langfuse down"))
    client, app = make_client(backend)
    response = client.post(
        "/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": "ok"}
    )
    assert "Submit failed" in response.text
    assert backend.labels == []  # nothing persisted
    assert "ORD-1001" in app.state.review_session.pending  # not advanced — not presented as labeled


def test_double_submit_is_surfaced_not_a_crash(
    make_client: Callable[..., Any], backend: Any
) -> None:
    # The optimistic-advance client can re-POST a trace already advanced past; recording it a
    # second time must be surfaced (guarded before the write), never a 500.
    client, _app = make_client(backend)
    client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": "ok"})
    response = client.post(
        "/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": "again"}
    )
    assert response.status_code == 200
    assert "Not advanced" in response.text


def test_double_submit_does_not_write_a_second_label(
    make_client: Callable[..., Any], backend: Any
) -> None:
    # The double-submit guard runs BEFORE the write, so a re-POST of an already-labeled trace
    # persists NO second label (no duplicate in the store) — the optimistic client can safely retry.
    client, _app = make_client(backend)
    client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": "ok"})
    client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": "again"})
    assert len(backend.get_scores("ORD-1001")) == 1  # exactly one label, not two


def test_label_response_signals_recorded_via_header(
    make_client: Callable[..., Any], backend: Any
) -> None:
    # The optimistic client reads HD-Label-Status to decide whether to keep its advance (recorded)
    # or reconcile (refused/error). A successful write signals `recorded`.
    client, _app = make_client(backend)
    response = client.post(
        "/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": "ok"}
    )
    assert response.headers.get("HD-Label-Status") == "recorded"


def test_failed_write_signals_error_via_header_so_client_reconciles(
    make_client: Callable[..., Any], make_backend: Callable[..., Any]
) -> None:
    backend = make_backend(submit_error=BackendError("langfuse down"))
    client, _app = make_client(backend)
    response = client.post(
        "/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": "ok"}
    )
    # `error` tells the optimistic client the write did not land — it must swap the view back so
    # the trace is never presented as labeled.
    assert response.headers.get("HD-Label-Status") == "error"
    assert "Submit failed" in response.text


def test_a_stale_tab_posting_a_different_verdict_gets_noop_and_writes_nothing(
    make_client: Callable[..., Any], backend: Any
) -> None:
    """Tab A PASSes X; tab B, still on X, FAILs it with a critique.

    The server keeps A's verdict and answers B with `noop`. hotkeys.js now swaps in the returned
    fragment on any status but `recorded` (tests/js/hotkeys.test.js), so B is told rather than
    shown a false advance.
    """
    client, _app = make_client(backend)
    client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": ""})
    response = client.post(
        "/label",
        data={"trace_id": "ORD-1001", "outcome": "FAIL", "critique": "tab B's careful critique"},
    )
    assert response.headers.get("HD-Label-Status") == "noop"
    assert "Not advanced" in response.text
    [only] = backend.get_scores("ORD-1001")
    assert only.outcome == "PASS"


def test_a_label_for_a_trace_past_the_head_is_accepted(
    make_client: Callable[..., Any], backend: Any
) -> None:
    """The optimistic client's second post names the NEXT trace."""
    client, app = make_client(backend)
    response = client.post(
        "/label", data={"trace_id": "ORD-1002", "outcome": "PASS", "critique": ""}
    )
    assert response.headers.get("HD-Label-Status") == "recorded"
    assert [lbl.trace_id for lbl in backend.labels] == ["ORD-1002"]
    assert app.state.review_session.pending == ["ORD-1001", "CLM-4471"]
