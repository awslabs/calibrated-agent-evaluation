# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The convention registry: custom adapters + unknown-kind refusal.

A user convention adapter registered under a new kind is dispatched to without touching
the built-ins; an unregistered kind raises a typed error naming what IS registered, never
a silent no-op or partial Trace.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.normalize import SessionSpans, normalize
from calibrated_agent_evaluation.normalize.conventions import registry
from calibrated_agent_evaluation.normalize.errors import NormalizationError
from calibrated_agent_evaluation.normalize.tree import AgentNode, Intermediate, ToolInvocation
from calibrated_agent_evaluation.schema.validate import validate


def _custom_adapter(session: SessionSpans) -> Intermediate:
    """A trivial user convention: one agent + one tool call (proves dispatch reached us)."""
    node = AgentNode(
        agent="custom-agent",
        role="specialist",
        parent=None,
        input={"message": "hi"},
        output={"answer": "done"},
        tool_calls=[ToolInvocation(name="custom_tool", args={"a": 1}, result={"ok": True})],
        order=0,
    )
    return Intermediate(
        trace_id=session.trace_id or "custom-trace",
        frame=session.frame,
        taxonomy_cell=dict(session.taxonomy_cell),
        trace_input={"message": "hi"},
        trace_output={"answer": "done"},
        nodes=[node],
    )


def test_registered_custom_convention_is_dispatched_to() -> None:
    kind = "test/custom-convention"
    if kind not in registry.registered_kinds():
        registry.register_convention(kind, _custom_adapter)

    trace = normalize(SessionSpans(spans=[{"attributes": {}}], semconv=kind, trace_id="CUST-1"))  # type: ignore[arg-type]
    validate(trace)
    assert trace.ownership == "owned"
    assert [c.name for s in trace.spans for c in s.tool_calls] == ["custom_tool"]
    # Built-ins are untouched — the three shipped conventions still resolve.
    assert {"gen_ai/messages", "openllmetry", "openinference"} <= registry.registered_kinds()


def test_unregistered_kind_raises_naming_registered_conventions() -> None:
    with pytest.raises(NormalizationError) as exc:
        registry.get_adapter("no/such-convention")
    msg = str(exc.value)
    assert "no/such-convention" in msg
    assert "gen_ai/messages" in msg  # names what IS registered


def test_registering_over_a_builtin_is_refused() -> None:
    # A user may not silently shadow a shipped adapter — that would be a footgun, not a feature.
    with pytest.raises(NormalizationError):
        registry.register_convention("gen_ai/messages", _custom_adapter)
