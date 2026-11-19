# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pacing signals in the labeling view.

The only count anywhere was on the completion screen, after the work-list was empty, and only two
of the hotkeys were discoverable from the button labels.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibrated_agent_evaluation.review.renderers.trace_view import HOTKEYS

from .conftest import FakeBackend


def test_the_view_shows_position_and_total(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    client, _app = make_client(backend)
    assert "Trace 1 of 3" in client.get("/").text
    client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": ""})
    assert "Trace 2 of 3" in client.get("/view").text


def test_a_prefetched_trace_shows_the_position_it_will_have(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    """The optimistic client displays the prefetch after the label lands: same number either way."""
    client, _app = make_client(backend)
    prefetched = client.get("/next", params={"after": "ORD-1001"}).text
    client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": ""})
    assert "Trace 2 of 3" in prefetched
    assert "Trace 2 of 3" in client.get("/view").text


def test_the_view_shows_elapsed_on_trace_time(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    client, _app = make_client(backend)
    page = client.get("/").text
    assert "on traces" in page
    assert "data-elapsed-seconds" in page


def test_the_page_carries_a_legend_of_every_hotkey(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    client, _app = make_client(backend)
    page = client.get("/").text
    assert 'id="hd-legend"' in page
    for keys, _action in HOTKEYS:
        assert keys in page
    for binding in ("1", "2", "D", "U", "?"):
        assert any(keys == binding for keys, _ in HOTKEYS)
