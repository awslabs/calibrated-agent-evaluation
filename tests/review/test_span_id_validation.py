# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attribution span ids must name real spans.

The culprit/manifestation/defect fields were free text with a placeholder hint, and nothing
checked the value against ``trace.spans``. ``calibrated-agent-evaluation run --ci`` derives the
gated agent from ``culprit_span``, so one mistyped character minted a failure mode for an agent that
does not exist.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest

from calibrated_agent_evaluation.review import ReviewError, build_label
from calibrated_agent_evaluation.review.renderers.trace_view import labeling_form
from calibrated_agent_evaluation.schema import Trace

from .conftest import FakeBackend


def _clm(corpus_traces: list[Trace]) -> Trace:
    return next(t for t in corpus_traces if t.trace_id == "CLM-4471")


@pytest.mark.parametrize("field", ["culprit_span", "manifestation_span", "defect_span"])
def test_a_span_id_not_in_the_trace_is_refused_and_nothing_is_written(
    make_client: Callable[..., Any], backend: FakeBackend, field: str
) -> None:
    client, app = make_client(backend)
    response = client.post(
        "/label",
        data={
            "trace_id": "CLM-4471",
            "outcome": "FAIL",
            "critique": "intake dropped occupancy",
            field: "CLM-4471/policy-checkr",  # one character short
        },
    )
    assert response.headers.get("HD-Label-Status") == "refused"
    assert "is not a span of trace" in response.text
    assert backend.labels == []
    assert "CLM-4471" in app.state.review_session.pending


def test_build_label_refuses_an_unknown_culprit_when_given_the_trace_spans() -> None:
    with pytest.raises(ReviewError, match="culprit_span"):
        build_label(
            trace_id="T",
            outcome="FAIL",
            critique="x",
            culprit_span="T/ghost",
            span_ids={"T/a", "T/b"},
        )


def test_the_attribution_fields_are_selects_over_exactly_the_trace_spans(
    corpus_traces: list[Trace],
) -> None:
    from fasthtml.common import to_xml

    trace = _clm(corpus_traces)
    page = to_xml(labeling_form(trace))
    want = ["", *(s.span_id for s in trace.spans)]
    for name in ("culprit_span", "manifestation_span", "defect_span"):
        block = re.search(rf'<select name="{name}".*?</select>', page, re.S)
        assert block is not None, f"{name} is not a <select>"
        assert re.findall(r'<option value="([^"]*)"', block.group(0)) == want
