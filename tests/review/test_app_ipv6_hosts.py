# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The review surface answers on the IPv6 loopback, and Host headers parse with brackets."""

from __future__ import annotations

import pytest

from .conftest import GOLD_RATER, FakeBackend

pytest.importorskip("fasthtml")

PAGE = {"HX-Request": "true"}


def test_the_ipv6_loopback_host_header_is_accepted(backend: FakeBackend) -> None:
    from starlette.testclient import TestClient

    from calibrated_agent_evaluation.review.app import build_app
    from calibrated_agent_evaluation.review.guards import allowed_hosts_for

    app = build_app(backend, rater_id=GOLD_RATER, allowed_hosts=allowed_hosts_for("::1"))
    # The test client cannot take an IPv6 base URL; send the Host header a browser would.
    client = TestClient(app, base_url="http://127.0.0.1:5001")
    browser = {"Host": "[::1]:5001"}
    assert client.get("/", headers=browser).status_code == 200
    labeled = client.post(
        "/label",
        data={"trace_id": "ORD-1001", "outcome": "PASS"},
        headers={**PAGE, **browser, "Origin": "http://[::1]:5001"},
    )
    assert labeled.headers["HD-Label-Status"] == "recorded"


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("127.0.0.1:5001", "127.0.0.1"),
        ("[::1]:5001", "::1"),
        ("[::1]", "::1"),
        ("LocalHost", "localhost"),
        ("[::1", ""),
        ("[::1]x", ""),
    ],
)
def test_the_host_header_is_parsed_with_ipv6_brackets(header: str, expected: str) -> None:
    from calibrated_agent_evaluation.review.addresses import host_of

    assert host_of(header) == expected


def test_an_ipv6_bind_prints_a_bracketed_url_and_counts_as_loopback() -> None:
    from calibrated_agent_evaluation.review.addresses import is_loopback, surface_url

    assert surface_url("::1", 5001) == "http://[::1]:5001"
    assert surface_url("127.0.0.1", 5001) == "http://127.0.0.1:5001"
    assert all(is_loopback(h) for h in ("::1", "[::1]", "127.0.0.2", "localhost"))
    assert not any(is_loopback(h) for h in ("0.0.0.0", "::", "10.0.0.5", "laptop.example"))
