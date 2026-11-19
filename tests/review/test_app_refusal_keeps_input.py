# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A refused or failed submission re-renders what the labeler typed.

Every refusal path used to rebuild the form from scratch, seeding the critique only from an LLM
suggestion and the attribution and ``expected`` fields from nothing. One stray brace in
``expected`` threw away a paragraph of critique. These tests post real content, force each refusal
path, and assert the content comes back in the returned fragment.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any

from calibrated_agent_evaluation.schema import BackendError, Label

from .conftest import FakeBackend

_CRITIQUE = "Intake dropped the vacancy signal; policy-checker never saw occupancy & paid out."
_BAD_EXPECTED = '{"decision": "DENIED",'  # a missing closing brace


def _crew_fail(**over: str) -> dict[str, str]:
    data = {
        "trace_id": "CLM-4471",
        "outcome": "FAIL",
        "critique": _CRITIQUE,
        "culprit_span": "CLM-4471/intake",
        "manifestation_span": "CLM-4471/payout",
        "handoff_implicated": "intake->policy-checker",
        "defect_span": "CLM-4471/intake",
        "expected": '{"decision": "DENIED"}',
    }
    data.update(over)
    return data


def _assert_input_survives(text: str, data: dict[str, str]) -> None:
    page = html.unescape(text)
    assert data["critique"] in page
    assert data["expected"] in page
    assert data["handoff_implicated"] in page
    # select-style fields re-render with the submitted option chosen
    for name in ("culprit_span", "manifestation_span", "defect_span"):
        assert f'name="{name}"' in page
    assert page.count("selected") >= 3


def test_a_malformed_expected_refusal_keeps_everything_typed(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    client, _app = make_client(backend)
    data = _crew_fail(expected=_BAD_EXPECTED)
    response = client.post("/label", data=data)
    assert response.headers.get("HD-Label-Status") == "refused"
    assert backend.labels == []
    _assert_input_survives(response.text, data)


def test_a_failed_write_keeps_everything_typed(
    make_client: Callable[..., Any], make_backend: Callable[..., Any]
) -> None:
    backend = make_backend(submit_error=BackendError("langfuse down"))
    client, _app = make_client(backend)
    data = _crew_fail()
    response = client.post("/label", data=data)
    assert response.headers.get("HD-Label-Status") == "error"
    _assert_input_survives(response.text, data)


def test_the_refused_trace_is_the_one_re_rendered(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    """The view returns to the trace whose write did not land, not to the head of the queue."""
    client, _app = make_client(backend)
    response = client.post("/label", data=_crew_fail(expected=_BAD_EXPECTED))
    assert 'value="CLM-4471" name="trace_id"' in response.text


class _SpanWriteFails(FakeBackend):
    def put_score(self, label: Label) -> None:
        if label.span_id is not None:
            raise BackendError("span write failed")
        super().put_score(label)


def test_a_failed_span_defect_write_rolls_back_the_verdict_and_keeps_the_input(
    make_client: Callable[..., Any], corpus_traces: list[Any]
) -> None:
    """The verdict and its span-defect label land together or not at all."""
    backend = _SpanWriteFails(list(corpus_traces))
    client, app = make_client(backend)
    data = _crew_fail()
    response = client.post("/label", data=data)
    assert response.headers.get("HD-Label-Status") == "error"
    assert backend.get_scores("CLM-4471") == []  # the trace-level label was rolled back
    assert "CLM-4471" in app.state.review_session.pending
    _assert_input_survives(response.text, data)
