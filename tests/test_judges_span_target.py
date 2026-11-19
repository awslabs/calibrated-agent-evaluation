# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a span-level judge reads, rendered from a real ClaimsCrew trace.

The judged span in full under its ancestor path from the root, each ancestor with its input and
tool calls but not its output, and nothing from its siblings or descendants; the
target is keyed on ``(trace_id, span_id)``.
"""

from __future__ import annotations

import pytest

from _helpers import load_corpus_trace
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.judges import (
    JudgeLevelError,
    ancestor_path,
    render_span,
    span_target,
)
from calibrated_agent_evaluation.schema.types import ToolCall


@pytest.fixture
def crew():
    return load_corpus_trace("CLM-4471")


def _span(trace, agent):
    return next(s for s in trace.spans if s.agent == agent)


def test_a_specialist_is_rendered_under_its_supervisor(crew):
    intake = _span(crew, "intake")
    supervisor = _span(crew, "supervisor")
    text = render_span(crew, intake)
    assert text.startswith(f"Customer request: {crew.input}\n")
    assert "Steps leading to the step being judged, from the root (outputs withheld):" in text
    assert "  [1] agent supervisor (supervisor)\n" in text
    assert f"      received: {supervisor.input}\n" in text
    assert "Step being judged: agent intake (specialist)\n" in text
    assert f"  received: {intake.input}\n" in text
    (tool,) = intake.tool_calls
    assert f"    tool {tool.name}(args={tool.args}) -> {tool.result}\n" in text
    assert text.endswith(f"  emitted: {intake.output}\n")


def test_siblings_and_their_tool_calls_are_not_shown(crew):
    text = render_span(crew, _span(crew, "intake"))
    for sibling in ("policy-checker", "fraud-scorer", "payout"):
        assert f"agent {sibling}" not in text
    assert "lookup_policy_exclusions" not in text
    assert "score_fraud_risk" not in text


def test_the_root_has_no_path(crew):
    text = render_span(crew, _span(crew, "supervisor"))
    assert "The step being judged is the root of the run." in text
    assert "Steps leading" not in text
    assert ancestor_path(crew, _span(crew, "supervisor")) == []


def test_the_target_is_keyed_on_the_span(crew):
    target = span_target(crew, _span(crew, "payout"))
    assert (target.trace_id, target.span_id) == ("CLM-4471", "CLM-4471/payout")
    assert target.content == render_span(crew, _span(crew, "payout"))


def test_an_ancestor_the_trace_does_not_hold_is_refused(crew):
    orphan = replace(_span(crew, "intake"), parent="CLM-4471/missing")
    broken = replace(crew, spans=[s for s in crew.spans if s.agent != "intake"] + [orphan])
    with pytest.raises(JudgeLevelError, match="partial path"):
        render_span(broken, orphan)


def test_a_span_from_another_trace_is_refused(crew):
    other = load_corpus_trace("ORD-1001")
    with pytest.raises(JudgeLevelError, match="not in trace"):
        render_span(crew, other.spans[0])


def test_a_parent_cycle_is_refused(crew):
    supervisor = replace(_span(crew, "supervisor"), parent="CLM-4471/intake")
    looped = replace(crew, spans=[supervisor if s.agent == "supervisor" else s for s in crew.spans])
    with pytest.raises(JudgeLevelError, match="cycle"):
        render_span(looped, _span(looped, "intake"))


# --- an ancestor's output is withheld ------------------------------------------


def test_an_ancestors_output_is_withheld(crew):
    """The supervisor writes its answer after its children ran, and it summarizes intake's result.

    Shown, it hands a judge of intake the answer. Withheld, the judge reads only what led up to the
    step: each ancestor's input and tool calls, then the judged span in full.
    """
    supervisor = _span(crew, "supervisor")
    text = render_span(crew, _span(crew, "intake"))
    assert "STAGE 1: INTAKE" in str(supervisor.output)  # the leak this guards against
    assert "STAGE 1: INTAKE" not in text
    assert "Steps leading to the step being judged, from the root (outputs withheld):" in text
    lines = text.splitlines()
    start = lines.index("  [1] agent supervisor (supervisor)")
    end = lines.index("Step being judged: agent intake (specialist)")
    assert lines[start:end] == [
        "  [1] agent supervisor (supervisor)",
        f"      received: {supervisor.input}",
        "      tool calls:",
        "        (none)",
    ]


def test_an_ancestors_tool_calls_are_shown(crew):
    call = ToolCall(name="route_claim", args={"claim": "CLM-4471"}, result={"next": "intake"})
    supervisor = replace(_span(crew, "supervisor"), tool_calls=[call])
    routed = replace(crew, spans=[supervisor if s.agent == "supervisor" else s for s in crew.spans])
    text = render_span(routed, _span(routed, "intake"))
    assert "        tool route_claim(args={'claim': 'CLM-4471'}) -> {'next': 'intake'}\n" in text
    assert "STAGE 1: INTAKE" not in text


def test_the_judged_span_is_shown_in_full_with_its_output(crew):
    intake = _span(crew, "intake")
    assert render_span(crew, intake).endswith(f"  emitted: {intake.output}\n")
    root = _span(crew, "supervisor")
    assert render_span(crew, root).endswith(f"  emitted: {root.output}\n")
