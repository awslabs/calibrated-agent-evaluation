# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completion state: an empty work-list shows completion, not a blank trace view."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def test_empty_worklist_shows_completion_state(
    make_client: Callable[..., Any], make_backend: Callable[..., Any]
) -> None:
    client, _app = make_client(make_backend(traces=[]))
    response = client.get("/")
    assert response.status_code == 200
    assert "All traces labeled" in response.text
