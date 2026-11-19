# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefetch route decoupling: advancing the view must not write a label."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def test_next_prefetch_writes_no_label(make_client: Callable[..., Any], backend: Any) -> None:
    client, app = make_client(backend)
    response = client.get("/next")
    assert response.status_code == 200
    assert backend.labels == []
    assert app.state.review_session.summary()["labeled_count"] == 0


def test_labeling_the_last_trace_after_navigating_prefetches_the_trace_still_pending(
    make_client: Callable[..., Any], backend: Any
) -> None:
    client, app = make_client(backend)
    client.get("/")
    client.post("/nav", params={"to": "next"})
    client.post("/nav", params={"to": "next"})  # the view is on the last pending trace
    last = app.state.review_session.pending[-1]
    prefetched = client.get("/next", params={"after": last}).text
    assert "All traces labeled" not in prefetched

    response = client.post(
        "/label", data={"trace_id": last, "outcome": "PASS"}, headers={"HD-Optimistic": "1"}
    )
    assert response.headers["HD-Label-Status"] == "recorded"
    shown = app.state.review_session.pending[-1]
    assert f'value="{shown}"' in prefetched
    assert f'value="{shown}"' in client.get("/view").text


def test_labeling_the_only_pending_trace_prefetches_the_completion_view(
    make_client: Callable[..., Any], make_backend: Callable[..., Any], corpus_traces: list[Any]
) -> None:
    client, _ = make_client(make_backend(corpus_traces[:1]))
    only = corpus_traces[0].trace_id
    assert "All traces labeled" in client.get("/next", params={"after": only}).text
