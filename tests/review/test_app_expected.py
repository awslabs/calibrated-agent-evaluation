# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Objective correct-output capture through the UI.

The review form lets a labeler record the objective ``expected`` output for an objective FAIL,
so a label produced *through the UI* reproduces the gold ``expected`` field (previously only
``build_label`` callers could set it). ``expected`` never substitutes for the critique.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibrated_agent_evaluation.schema import Label


def test_expected_from_the_form_lands_on_the_written_label(
    make_client: Callable[..., Any], backend: Any
) -> None:
    client, _app = make_client(backend)
    client.post(
        "/label",
        data={
            "trace_id": "ORD-1001",
            "outcome": "FAIL",
            "critique": "over-refunded out of window",
            "expected": '{"action": "REFUND_DENIED", "amount_usd": 0}',
        },
    )
    labels = backend.get_scores("ORD-1001")
    assert len(labels) == 1
    assert labels[0].expected == {"action": "REFUND_DENIED", "amount_usd": 0}


def test_ui_label_reproduces_the_gold_expected_field(
    make_client: Callable[..., Any], backend: Any, gold_labels: list[Label]
) -> None:
    # The oracle: a label produced through the route now matches the gold `expected` row.
    import json

    gold = next(g for g in gold_labels if g.trace_id == "ORD-1001" and g.span_id is None)
    assert gold.expected is not None  # sanity: the ORD-1001 gold row carries an objective answer
    client, _app = make_client(backend)
    client.post(
        "/label",
        data={
            "trace_id": "ORD-1001",
            "outcome": "FAIL",
            "critique": "human note",
            "expected": json.dumps(gold.expected),
        },
    )
    assert backend.get_scores("ORD-1001")[0].expected == gold.expected


def test_blank_expected_is_none_not_an_error(make_client: Callable[..., Any], backend: Any) -> None:
    # PASS / subjective FAIL: leaving `expected` blank is the norm — it must be None, not a refusal.
    client, _app = make_client(backend)
    response = client.post(
        "/label",
        data={"trace_id": "ORD-1002", "outcome": "PASS", "critique": "correct", "expected": ""},
    )
    assert response.headers.get("HD-Label-Status") == "recorded"
    assert backend.get_scores("ORD-1002")[0].expected is None


def test_malformed_expected_is_refused_not_silently_dropped(
    make_client: Callable[..., Any], backend: Any
) -> None:
    # A non-JSON `expected` is a typed refusal (no silent drop): nothing is written and the client
    # is told to reconcile.
    client, _app = make_client(backend)
    response = client.post(
        "/label",
        data={
            "trace_id": "ORD-1001",
            "outcome": "FAIL",
            "critique": "the why",
            "expected": "REFUND_DENIED",  # not a JSON object
        },
    )
    assert response.headers.get("HD-Label-Status") == "refused"
    assert "expected" in response.text.lower()
    assert backend.get_scores("ORD-1001") == []  # nothing persisted


def test_expected_is_offered_on_single_agent_traces_too(
    make_client: Callable[..., Any], backend: Any
) -> None:
    # ORD-1001 is a one-span trace; the `expected` input is not gated on the crew branch, so the
    # form still offers it (objective answers apply to single-agent traces).
    client, _app = make_client(backend)
    html = client.get("/").text
    assert 'name="expected"' in html
