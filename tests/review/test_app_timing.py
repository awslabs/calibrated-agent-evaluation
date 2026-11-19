# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-trace timing through the routes.

The view renders pending[view index] but the timer used to start on pending[0]; after
navigation the interval was charged to the wrong trace, and defer threw its time away.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibrated_agent_evaluation.review import ReviewSession


class _Manual:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _client(make_client: Callable[..., Any], backend: Any) -> tuple[Any, ReviewSession, _Manual]:
    clock = _Manual()
    session = ReviewSession.from_ids(["ORD-1001", "ORD-1002", "CLM-4471"], clock=clock)
    client, _app = make_client(backend, session=session)
    return client, session, clock


def test_time_after_navigation_is_charged_to_the_trace_labeled(
    make_client: Callable[..., Any], backend: Any
) -> None:
    client, session, clock = _client(make_client, backend)
    client.get("/")  # ORD-1001 on screen at t=0
    clock.t = 3.0
    client.post("/nav", params={"to": "next"})  # ORD-1002 on screen at t=3
    clock.t = 10.0
    client.post("/label", data={"trace_id": "ORD-1002", "outcome": "PASS", "critique": ""})
    [entry] = session.done
    assert entry.trace_id == "ORD-1002"
    assert entry.seconds_on_trace == 7.0
    assert session.summary()["elapsed_seconds"] == 10.0


def test_defer_moves_the_trace_on_screen_and_keeps_its_time(
    make_client: Callable[..., Any], backend: Any
) -> None:
    client, session, clock = _client(make_client, backend)
    client.get("/")
    client.post("/nav", params={"to": "next"})  # ORD-1002 on screen
    clock.t = 20.0
    client.post("/defer")
    assert session.pending == ["ORD-1001", "CLM-4471", "ORD-1002"]
    assert session.summary()["elapsed_seconds"] == 20.0


def test_after_labeling_a_navigated_trace_the_next_one_is_shown_and_timed(
    make_client: Callable[..., Any], backend: Any
) -> None:
    """The server's view after a label matches what /next served the optimistic client."""
    client, session, clock = _client(make_client, backend)
    client.get("/")
    client.post("/nav", params={"to": "next"})  # ORD-1002
    response = client.post(
        "/label", data={"trace_id": "ORD-1002", "outcome": "PASS", "critique": ""}
    )
    assert 'value="CLM-4471" name="trace_id"' in response.text
    clock.t = 4.0
    client.post("/label", data={"trace_id": "CLM-4471", "outcome": "PASS", "critique": ""})
    assert session.done[-1].seconds_on_trace == 4.0
