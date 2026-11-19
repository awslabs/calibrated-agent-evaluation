# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live check: the review app over a REAL self-hosted Langfuse.

Drives `build_app` with a `LangfuseBackend` through Starlette's TestClient: label a crew trace
with attribution and a span defect, undo it, label it again, and read back what Langfuse holds.
The trace is the corpus CLM-4471 remapped onto a unique id, so the persistent dev instance is not
polluted with labels on the shared corpus ids. Skips when the stack is not reachable.

    just langfuse-up
    just test-integration tests/integration/test_review_langfuse_live.py
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import pytest
from test_langfuse_live import _load_trace, _remap_trace

from _helpers import langfuse_reachable
from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend
from calibrated_agent_evaluation.schema.types import Label

pytestmark = pytest.mark.integration


@pytest.fixture
def live_backend() -> LangfuseBackend:
    pytest.importorskip("langfuse")
    pytest.importorskip("fasthtml")
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    backend = LangfuseBackend.from_credentials(
        os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-dev"),
        os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-calibrated-agent-evaluation-dev"),
        host,
    )
    reachable = langfuse_reachable(backend._client)
    if not reachable:
        pytest.skip(f"no reachable Langfuse at {host} (run `just langfuse-up`)")
    return backend


def _settled(backend: LangfuseBackend, trace_id: str, want: int) -> list[Label]:
    labels: list[Label] = []
    for _ in range(30):
        labels = backend.get_scores(trace_id)
        if len(labels) == want:
            time.sleep(3)  # and stays there: a late duplicate or a late tombstone would show
            labels = backend.get_scores(trace_id)
            if len(labels) == want:
                return labels
        time.sleep(1)
    return labels


def test_review_app_label_undo_relabel_round_trips_through_live_langfuse(
    live_backend: LangfuseBackend, tmp_path: Any
) -> None:
    from starlette.testclient import TestClient

    from calibrated_agent_evaluation.review.app import build_app
    from calibrated_agent_evaluation.review.session import WorkScope

    run_id = f"CLM-4471-ui-{uuid.uuid4().hex[:8]}"
    live_backend.put_trace(_remap_trace(_load_trace("CLM-4471"), run_id))
    for _ in range(30):  # trace ingestion is asynchronous
        try:
            live_backend.get_trace(run_id)
            break
        except Exception:
            time.sleep(1)

    app = build_app(
        live_backend,
        rater_id="acceptance-rater",
        scope=WorkScope(trace_id=run_id),
        session_log=tmp_path / "review.jsonl",
    )
    client = TestClient(app, base_url="http://127.0.0.1", headers={"HX-Request": "true"})
    form = {
        "trace_id": run_id,
        "outcome": "FAIL",
        "critique": "walk-back lands on intake",
        "culprit_span": f"{run_id}/intake",
        "manifestation_span": f"{run_id}/payout",
        "handoff_implicated": "intake->policy-checker",
        "defect_span": f"{run_id}/intake",
    }
    assert client.post("/label", data=form).headers["HD-Label-Status"] == "recorded"
    assert client.post("/undo").headers["HD-Label-Status"] == "recorded"  # immediately, no wait
    relabel = {**form, "critique": "intake dropped the vacancy signal"}
    assert client.post("/label", data=relabel).headers["HD-Label-Status"] == "recorded"

    labels = _settled(live_backend, run_id, 2)
    assert {(lbl.span_id, lbl.critique) for lbl in labels} == {
        (None, "intake dropped the vacancy signal"),
        (f"{run_id}/intake", "intake dropped the vacancy signal"),
    }
    trace_level = next(lbl for lbl in labels if lbl.span_id is None)
    assert trace_level.culprit_span == f"{run_id}/intake"
    assert trace_level.manifestation_span == f"{run_id}/payout"
    assert trace_level.rater_id == "acceptance-rater"
    assert all(lbl.kind == "human" for lbl in labels)
