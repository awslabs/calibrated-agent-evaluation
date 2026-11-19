# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent-boundary registry and Traceloop's LangGraph node rule.

The registry contract is tested here on its own terms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.normalize import agent_boundaries
from calibrated_agent_evaluation.normalize.conventions import openllmetry_adapter
from calibrated_agent_evaluation.normalize.errors import NormalizationError


def _span(**attributes: str) -> dict[str, Any]:
    """An OTLP-JSON span carrying only string attributes, the shape the rule reads."""
    return {
        "attributes": [{"key": k, "value": {"stringValue": v}} for k, v in attributes.items()],
    }


_NODE = "traceloop.association.properties.langgraph_node"
_ENTITY = "traceloop.entity.name"


def test_a_langgraph_node_span_is_an_agent_named_by_its_node() -> None:
    span = _span(**{"gen_ai.operation.name": "execute_task", _NODE: "planner", _ENTITY: "planner"})

    assert openllmetry_adapter._langgraph_node(span) == "planner"
    assert agent_boundaries.boundary_name(span) == "planner"


def test_a_runnable_inside_a_node_is_not_an_agent() -> None:
    """Inner runnables inherit the node's `langgraph_node` property with their own entity name.

    The two real cases from the capture: the section subgraph wrapper (entity `LangGraph`, node
    `researcher`) and an anonymous lambda inside `human` (entity `unknown`).
    """
    wrapper = _span(
        **{"gen_ai.operation.name": "execute_task", _NODE: "researcher", _ENTITY: "LangGraph"}
    )
    inner = _span(**{"gen_ai.operation.name": "execute_task", _NODE: "human", _ENTITY: "unknown"})

    assert agent_boundaries.boundary_name(wrapper) is None
    assert agent_boundaries.boundary_name(inner) is None


def test_an_llm_span_inside_a_node_is_not_an_agent() -> None:
    """A chat span carries the node property too; only `execute_task` spans can be nodes."""
    chat = _span(**{"gen_ai.operation.name": "chat", _NODE: "browser", _ENTITY: "browser"})

    assert agent_boundaries.boundary_name(chat) is None


def test_a_second_registration_under_one_name_is_refused() -> None:
    with pytest.raises(NormalizationError, match="already registered"):
        agent_boundaries.register_agent_boundary("traceloop-langgraph", lambda _span: None)


def test_no_langgraph_key_appears_in_the_convention_neutral_reader() -> None:
    """The rule's keys live with the dialect that owns them, never in the generic reader."""
    reader = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "calibrated_agent_evaluation"
        / "normalize"
        / "otel_adapter.py"
    ).read_text(encoding="utf-8")

    for vendor_key in ("langgraph_node", "traceloop.", "association.properties"):
        assert vendor_key not in reader, f"{vendor_key!r} leaked into the generic reader"
