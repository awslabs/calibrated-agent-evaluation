# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend reads on the labeling hot path.

Each verdict used to cost a `put_score` plus a `get_trace` for the response fragment (which the
optimistic client throws away) and another for the `/next` prefetch, on top of a startup probe
that had already read every trace. On Langfuse each read is an HTTP round trip.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibrated_agent_evaluation.review.session import WorkScope
from calibrated_agent_evaluation.schema import Trace

from .conftest import FakeBackend


class _Counting(FakeBackend):
    def __init__(self, traces: list[Trace]) -> None:
        super().__init__(traces)
        self.trace_reads: list[str] = []
        self.queries = 0

    def get_trace(self, trace_id: str) -> Trace:
        self.trace_reads.append(trace_id)
        return super().get_trace(trace_id)

    def query(self, *args: Any, **kwargs: Any) -> list[Trace]:
        self.queries += 1
        return super().query(*args, **kwargs)


def test_a_session_reads_each_trace_from_the_backend_at_most_once(
    make_client: Callable[..., Any], corpus_traces: list[Trace]
) -> None:
    backend = _Counting(corpus_traces)
    client, _app = make_client(backend)
    client.get("/")
    for trace_id in ("ORD-1001", "ORD-1002"):
        client.get("/next", params={"after": trace_id})
        client.post(
            "/label",
            data={"trace_id": trace_id, "outcome": "PASS", "critique": ""},
            headers={"HD-Optimistic": "1"},
        )
    client.post("/nav", params={"to": "next"})
    client.post("/nav", params={"to": "prev"})
    # every trace came back with the startup query, so the hot path read none again
    assert backend.trace_reads == []
    assert backend.queries == 1


def test_an_optimistic_recorded_response_has_no_body(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    client, _app = make_client(backend)
    response = client.post(
        "/label",
        data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": ""},
        headers={"HD-Optimistic": "1"},
    )
    assert response.headers.get("HD-Label-Status") == "recorded"
    assert response.text == ""


def test_a_blocking_recorded_response_still_carries_the_next_trace(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    """The fallback client (no prefetch ready) swaps this body in, so it must be the next trace."""
    client, _app = make_client(backend)
    response = client.post(
        "/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": ""}
    )
    assert 'value="ORD-1002" name="trace_id"' in response.text


def test_a_single_trace_scope_reads_one_trace_and_queries_nothing(
    make_client: Callable[..., Any], corpus_traces: list[Trace]
) -> None:
    backend = _Counting(corpus_traces)
    client, _app = make_client(backend, scope=WorkScope(trace_id="CLM-4471"))
    client.get("/")
    assert backend.queries == 0
    assert backend.trace_reads == ["CLM-4471"]
