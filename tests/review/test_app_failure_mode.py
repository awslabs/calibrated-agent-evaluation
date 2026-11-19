# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The failure-mode picker: a FAIL carries a codebook mode, every label carries the version."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from calibrated_agent_evaluation.review.errors import ReviewError
from calibrated_agent_evaluation.review.labeling import build_label
from calibrated_agent_evaluation.review.record import CodebookMode

MODES = (
    CodebookMode(name="refund_outside_window", description="refunded past the return window"),
    CodebookMode(name="wrong_tool"),
)


def _client(make_client: Callable[..., Any], backend: Any) -> tuple[Any, Any]:
    return make_client(backend, codebook_version="2", failure_modes=MODES)


def test_the_form_offers_the_codebook_modes_after_the_critique(
    make_client: Callable[..., Any], backend: Any
) -> None:
    client, _app = _client(make_client, backend)
    html = client.get("/").text
    assert 'id="failure_mode"' in html
    assert "refund_outside_window" in html
    assert "wrong_tool" in html
    assert html.index('name="critique"') < html.index('id="failure_mode"')
    assert html.index('id="failure_mode"') < html.index('value="PASS"')
    assert 'data-modes-required="1"' in html


def test_a_fail_with_a_listed_mode_is_stored_with_the_mode_and_version(
    make_client: Callable[..., Any], backend: Any
) -> None:
    client, _app = _client(make_client, backend)
    client.post(
        "/label",
        data={
            "trace_id": "ORD-1001",
            "outcome": "FAIL",
            "critique": "refunded 41 days after delivery",
            "failure_mode": "refund_outside_window",
        },
    )
    [label] = backend.get_scores("ORD-1001")
    assert label.failure_mode == "refund_outside_window"
    assert label.codebook_version == "2"


def test_a_fail_without_a_mode_is_refused_and_keeps_what_was_typed(
    make_client: Callable[..., Any], backend: Any
) -> None:
    client, app = _client(make_client, backend)
    response = client.post(
        "/label",
        data={"trace_id": "ORD-1001", "outcome": "FAIL", "critique": "refunded late"},
    )
    assert response.headers["HD-Label-Status"] == "refused"
    assert "needs a failure mode" in response.text
    assert "refunded late" in response.text
    assert backend.get_scores("ORD-1001") == []
    assert "ORD-1001" in app.state.review_session.pending


def test_a_mode_outside_the_codebook_is_refused(
    make_client: Callable[..., Any], backend: Any
) -> None:
    client, _app = _client(make_client, backend)
    response = client.post(
        "/label",
        data={
            "trace_id": "ORD-1001",
            "outcome": "FAIL",
            "critique": "refunded late",
            "failure_mode": "refund_late",
        },
    )
    assert response.headers["HD-Label-Status"] == "refused"
    assert "not in the codebook" in response.text
    assert backend.get_scores("ORD-1001") == []


def test_a_pass_is_stored_with_the_version_and_no_mode(
    make_client: Callable[..., Any], backend: Any
) -> None:
    client, _app = _client(make_client, backend)
    client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": ""})
    [label] = backend.get_scores("ORD-1001")
    assert label.failure_mode is None
    assert label.codebook_version == "2"


def test_a_span_defect_label_carries_the_same_mode_and_version(
    make_client: Callable[..., Any], backend: Any, corpus_traces: list[Any]
) -> None:
    crew = next(t for t in corpus_traces if len(t.spans) > 1)
    client, _app = _client(make_client, backend)
    client.post(
        "/label",
        data={
            "trace_id": crew.trace_id,
            "outcome": "FAIL",
            "critique": "the policy checker ignored the exclusion",
            "failure_mode": "wrong_tool",
            "defect_span": crew.spans[1].span_id,
        },
    )
    labels = backend.get_scores(crew.trace_id)
    assert len(labels) == 2
    assert {(lb.failure_mode, lb.codebook_version) for lb in labels} == {("wrong_tool", "2")}


def test_without_a_codebook_there_is_no_picker_and_labels_are_unversioned(
    make_client: Callable[..., Any], backend: Any
) -> None:
    client, _app = make_client(backend)
    assert 'id="failure_mode"' not in client.get("/").text
    client.post(
        "/label", data={"trace_id": "ORD-1001", "outcome": "FAIL", "critique": "refunded late"}
    )
    [label] = backend.get_scores("ORD-1001")
    assert label.failure_mode is None
    assert label.codebook_version is None


def test_build_label_refuses_a_pass_that_names_a_mode() -> None:
    with pytest.raises(ReviewError, match="PASS carries no failure mode"):
        build_label(
            trace_id="T",
            outcome="PASS",
            critique="",
            failure_mode="wrong_tool",
            failure_modes={"wrong_tool"},
        )
