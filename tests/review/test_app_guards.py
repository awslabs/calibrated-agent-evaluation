# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The review surface refuses cross-site writes and foreign Host names."""

from __future__ import annotations

import socket
from collections.abc import Callable
from typing import Any

import pytest

from .conftest import GOLD_RATER, FakeBackend

pytest.importorskip("fasthtml")


def _raw_client(backend: FakeBackend, **kwargs: Any) -> Any:
    """A client with no page headers: what another site's form or script can send."""
    from starlette.testclient import TestClient

    from calibrated_agent_evaluation.review.app import build_app

    app = build_app(backend, rater_id=GOLD_RATER, **kwargs)
    return TestClient(app, base_url="http://127.0.0.1"), app


def _label_once(make_client: Callable[..., Any], backend: FakeBackend) -> None:
    client, _ = make_client(backend)
    response = client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS"})
    assert response.headers["HD-Label-Status"] == "recorded"


def test_a_cross_origin_undo_is_refused_and_the_label_stays(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    client, _ = make_client(backend)
    client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS"})
    stored = backend.get_scores("ORD-1001")
    response = client.post("/undo", headers={"Origin": "http://evil.example", "HX-Request": "true"})
    assert response.status_code == 403
    assert backend.get_scores("ORD-1001") == stored


def test_a_form_post_from_another_site_cannot_forge_a_ratified_label(
    backend: FakeBackend,
) -> None:
    client, _ = _raw_client(backend)
    response = client.post(
        "/label",
        data={"trace_id": "ORD-1001", "outcome": "PASS", "ratified": "1"},
        headers={"Origin": "http://evil.example"},
    )
    assert response.status_code == 403
    assert backend.get_scores("ORD-1001") == []


def test_a_write_without_the_page_header_is_refused(backend: FakeBackend) -> None:
    client, _ = _raw_client(backend)
    response = client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS"})
    assert response.status_code == 403
    assert "HX-Request" in response.text
    assert backend.get_scores("ORD-1001") == []


def test_a_write_linked_from_another_site_is_refused(backend: FakeBackend) -> None:
    client, _ = _raw_client(backend)
    response = client.post(
        "/defer", headers={"Referer": "http://evil.example/page", "HX-Request": "true"}
    )
    assert response.status_code == 403


def test_a_same_origin_write_from_the_page_is_accepted(backend: FakeBackend) -> None:
    client, _ = _raw_client(backend)
    response = client.post(
        "/label",
        data={"trace_id": "ORD-1001", "outcome": "PASS"},
        headers={"Origin": "http://127.0.0.1", "HX-Request": "true"},
    )
    assert response.headers["HD-Label-Status"] == "recorded"


def test_a_foreign_host_name_is_refused_before_any_read(backend: FakeBackend) -> None:
    from starlette.testclient import TestClient

    from calibrated_agent_evaluation.review.app import build_app

    app = build_app(backend, rater_id=GOLD_RATER)
    rebound = TestClient(app, base_url="http://attacker.example")
    response = rebound.get("/trace/ORD-1001")
    assert response.status_code == 400
    assert "ORD-1001" not in response.text


def test_navigation_is_a_write_and_not_reachable_by_get(backend: FakeBackend) -> None:
    client, _ = _raw_client(backend)
    assert client.get("/nav", params={"to": "next"}).status_code == 405


def test_the_allowed_hosts_follow_the_bind_address() -> None:
    from calibrated_agent_evaluation.review.guards import DEFAULT_ALLOWED_HOSTS, allowed_hosts_for

    assert allowed_hosts_for(None) == DEFAULT_ALLOWED_HOSTS
    assert allowed_hosts_for("10.0.0.5")[-1] == "10.0.0.5"
    assert socket.gethostname() in allowed_hosts_for("0.0.0.0")


def test_a_labeled_session_still_works_through_the_page_client(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    _label_once(make_client, backend)
