# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Critique-first form layout: the critique field precedes the verdict control."""

from __future__ import annotations

from fasthtml.common import to_xml

from calibrated_agent_evaluation.review.renderers.trace_view import labeling_form
from calibrated_agent_evaluation.schema import Trace


def test_critique_field_precedes_verdict_control(corpus_traces: list[Trace]) -> None:
    ord1001 = next(t for t in corpus_traces if t.trace_id == "ORD-1001")
    html = to_xml(labeling_form(ord1001))
    assert 'name="critique"' in html
    assert 'value="PASS"' in html
    # the critique (the "why") is entered before the pass/fail verdict.
    assert html.index('name="critique"') < html.index('value="PASS"')


def test_holds_for_a_crew_trace_too(corpus_traces: list[Trace]) -> None:
    clm = next(t for t in corpus_traces if t.trace_id == "CLM-4471")
    html = to_xml(labeling_form(clm))
    assert html.index('name="critique"') < html.index('value="FAIL"')
