# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ratification guard is reachable from a real submission.

`/label` passed `ratified=from_suggestion`, so `from_suggestion and not ratified` could never be
true: any verdict keypress on a prefilled trace wrote the LLM's critique verbatim as a
`kind="human"` gold label. Ratification is now its own signal: an explicit accept, or an
override the server can see (a different verdict or an edited critique).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from calibrated_agent_evaluation.review import Suggestion
from calibrated_agent_evaluation.schema import Trace

from .conftest import FakeBackend

_SUGGESTED = Suggestion(outcome="PASS", critique="in-window refund, policy checked")


class _StubPrefill:
    """Suggests on ORD-1002 only (a test double for the opt-in LLM seam)."""

    def suggest(self, trace: Trace) -> Suggestion | None:
        return _SUGGESTED if trace.trace_id == "ORD-1002" else None


def _client(make_client: Callable[..., Any], backend: FakeBackend) -> Any:
    client, _app = make_client(backend, prefill=_StubPrefill())
    client.get("/next", params={"after": "ORD-1001"})  # ORD-1002's suggestion is shown
    return client


def _post(client: Any, **data: str) -> Any:
    return client.post("/label", data={"trace_id": "ORD-1002", **data})


def test_a_verdict_key_on_an_unedited_suggestion_is_refused(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    client = _client(make_client, backend)
    response = _post(client, outcome="PASS", critique=_SUGGESTED.critique, from_suggestion="1")
    assert response.headers.get("HD-Label-Status") == "refused"
    assert "has not been ratified" in response.text
    assert backend.labels == []


def test_dropping_the_hidden_field_does_not_bypass_the_guard(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    """The server knows what it suggested; it does not take the client's word for it."""
    client = _client(make_client, backend)
    response = _post(client, outcome="PASS", critique=_SUGGESTED.critique)
    assert response.headers.get("HD-Label-Status") == "refused"
    assert backend.labels == []


def test_an_explicit_accept_records_a_human_label(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    client = _client(make_client, backend)
    response = _post(
        client, outcome="PASS", critique=_SUGGESTED.critique, from_suggestion="1", ratified="1"
    )
    assert response.headers.get("HD-Label-Status") == "recorded"
    [label] = backend.labels
    assert label.kind == "human"
    assert label.critique == _SUGGESTED.critique


@pytest.mark.parametrize(
    ("outcome", "critique"),
    [
        ("PASS", "in-window refund; I checked the policy tool output myself"),  # edited critique
        ("FAIL", "the refund amount is wrong"),  # the other verdict
    ],
)
def test_an_override_is_a_ratification(
    make_client: Callable[..., Any], backend: FakeBackend, outcome: str, critique: str
) -> None:
    client = _client(make_client, backend)
    response = _post(client, outcome=outcome, critique=critique, from_suggestion="1")
    assert response.headers.get("HD-Label-Status") == "recorded"
    assert backend.labels[0].critique == critique


def test_a_claimed_suggestion_the_server_never_made_is_not_one(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    client, _app = make_client(backend, prefill=_StubPrefill())
    response = client.post(
        "/label",
        data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": "", "from_suggestion": "1"},
    )
    assert response.headers.get("HD-Label-Status") == "recorded"


def test_a_prefilled_form_offers_an_explicit_accept(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    client = _client(make_client, backend)
    html = client.get("/next", params={"after": "ORD-1001"}).text
    assert 'data-suggested-outcome="PASS"' in html
    assert "Accept suggestion (A)" in html
