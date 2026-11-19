# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract: the app drives the backend through normalized types only (no vendor type).

The ``FakeBackend`` implements *only* the ``TraceStoreBackend``/``LabelBackend`` protocol
surface (normalized ``Trace``/``Label``). That the whole label flow works against it proves the
review app needs no vendor type to read traces or write labels.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibrated_agent_evaluation.schema import Label


def test_app_persists_normalized_label_through_protocol_only(
    make_client: Callable[..., Any], backend: Any
) -> None:
    client, _app = make_client(backend)
    client.post("/label", data={"trace_id": "ORD-1002", "outcome": "PASS", "critique": "ok"})
    assert backend.labels
    assert isinstance(backend.labels[0], Label)  # a normalized Label, not a vendor type
