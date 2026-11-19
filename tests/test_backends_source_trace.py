# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A SourceTrace carries exactly one of a trace, an OTLP document or a read error."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.backends import SourceTrace
from calibrated_agent_evaluation.schema.errors import BackendError
from calibrated_agent_evaluation.schema.types import Trace


def _trace() -> Trace:
    return Trace(
        trace_id="T1",
        frame="discovery",
        ownership="probed",
        input={"q": "hi"},
        output={"a": "hello"},
        taxonomy_cell={},
    )


@pytest.mark.parametrize(
    "form",
    [{"trace": _trace()}, {"otlp": {"resourceSpans": []}}, {"error": "unreadable"}],
    ids=["trace", "otlp", "error"],
)
def test_each_single_form_is_accepted(form: dict[str, object]) -> None:
    st = SourceTrace(source_id="s1", **form)  # type: ignore[arg-type]
    assert st.source_id == "s1"


def test_no_form_is_refused_naming_the_three() -> None:
    with pytest.raises(BackendError, match="exactly one of trace, otlp or error; got none"):
        SourceTrace(source_id="s1")


def test_two_forms_are_refused_naming_both() -> None:
    with pytest.raises(BackendError, match="got trace, error"):
        SourceTrace(source_id="s1", trace=_trace(), error="also broken")


def test_a_session_id_is_accepted_only_with_an_otlp_document() -> None:
    with_otlp = SourceTrace(source_id="t-1", otlp={"resourceSpans": []}, session_id="s-1")
    assert with_otlp.session_id == "s-1"
    with pytest.raises(BackendError, match="'t-1' carries a blank session_id"):
        SourceTrace(source_id="t-1", otlp={"resourceSpans": []}, session_id="  ")
    for form in ({"error": "unreadable"}, {"trace": _trace()}):
        with pytest.raises(BackendError, match="'t-1' carries a session_id"):
            SourceTrace(source_id="t-1", session_id="s-1", **form)  # type: ignore[arg-type]
