# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The review app stamps the rater in session on every label it writes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from calibrated_agent_evaluation.review import ReviewError


def test_every_label_the_app_writes_carries_the_session_rater(
    make_client: Callable[..., Any], backend: Any
) -> None:
    client, _app = make_client(backend, rater_id="alice")
    client.post(
        "/label",
        data={
            "trace_id": "ORD-1001",
            "outcome": "FAIL",
            "critique": "refunded without checking policy",
        },
    )
    client.post(
        "/label",
        data={"trace_id": "ORD-1002", "outcome": "PASS", "critique": ""},
    )
    client.post(
        "/label",
        data={
            "trace_id": "CLM-4471",
            "outcome": "FAIL",
            "critique": "intake dropped occupancy",
            "defect_span": "CLM-4471/intake",
        },
    )
    # three trace-level labels plus the span-level defect label, all attributed
    assert len(backend.labels) == 4
    assert {lbl.rater_id for lbl in backend.labels} == {"alice"}


@pytest.mark.parametrize("bad", ["", "   "])
def test_the_app_refuses_to_open_a_session_with_no_rater(
    make_client: Callable[..., Any], backend: Any, bad: str
) -> None:
    with pytest.raises(ReviewError, match="rater_id"):
        make_client(backend, rater_id=bad)
