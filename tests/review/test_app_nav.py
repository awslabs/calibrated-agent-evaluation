# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Navigation route: moving the view records no verdict."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def test_nav_records_no_label(make_client: Callable[..., Any], backend: Any) -> None:
    client, app = make_client(backend)
    before = list(app.state.review_session.pending)
    response = client.post("/nav", params={"to": "next"})
    assert response.status_code == 200
    assert backend.labels == []
    assert app.state.review_session.summary()["labeled_count"] == 0
    assert app.state.review_session.pending == before  # nav doesn't mutate the work-list
