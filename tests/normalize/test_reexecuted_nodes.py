# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A node that executes twice normalizes.

The normalized span id was `f"{trace_id}/{agent}"` and the `seq` was assigned into a dict keyed by
agent name. Both are unique only while every agent runs at most once — and a cyclic topology breaks
that by design: a repair or retry edge re-runs one node, emitting sibling spans that are perfectly
legitimate OTel. The collision surfaced as `validate` refusing a *correct* trace, first on
`duplicate span_id` and, once that was worked around, on `duplicate seq`.

The producer cannot fix it: the only escape is encoding loop bookkeeping into `gen_ai.agent.name`
(`generator#2`), which breaks counting agents by name.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.normalize.tree import AgentNode, Intermediate, build_spans
from calibrated_agent_evaluation.schema.errors import SchemaError
from calibrated_agent_evaluation.schema.types import Span, Trace
from calibrated_agent_evaluation.schema.validate import validate

TRACE = "t"


def _inter(*nodes: AgentNode) -> Intermediate:
    return Intermediate(
        trace_id=TRACE,
        frame="production-representative",
        taxonomy_cell={},
        trace_input={"prompt": "how many controls?"},
        trace_output={"text": "42"},
        nodes=list(nodes),
    )


def _repair_run() -> Intermediate:
    """The shape a generate-and-validate repair loop produces: generator and validator run twice."""
    order = [
        ("supervisor", None),
        ("schema_context", "supervisor"),
        ("sql_generator", "supervisor"),
        ("sql_validator_repair", "sql_generator"),
        ("sql_generator", "supervisor"),
        ("sql_validator_repair", "sql_generator"),
        ("visualization", "supervisor"),
    ]
    return _inter(
        *[
            AgentNode(
                agent=agent,
                role="supervisor" if parent is None else "specialist",
                parent=parent,
                order=i,
            )
            for i, (agent, parent) in enumerate(order)
        ]
    )


def _as_trace(spans: list[Span]) -> Trace:
    return Trace(
        trace_id=TRACE,
        frame="production-representative",
        ownership="owned",
        input={"prompt": "how many controls?"},
        output={"text": "42"},
        taxonomy_cell={},
        spans=spans,
    )


def test_a_repair_run_normalizes_and_validates() -> None:
    trace = _as_trace(build_spans(_repair_run()))
    validate(trace)  # used to raise: duplicate span_id values within trace
    assert len(trace.spans) == 7


def test_the_re_execution_gets_its_own_id_and_the_first_keeps_the_historical_one() -> None:
    """The first execution's id is unchanged, so every previously normalized artifact still
    matches byte for byte and only a re-execution introduces something new."""
    ids = [s.span_id for s in build_spans(_repair_run())]
    assert ids.count("t/sql_generator") == 1
    assert "t/sql_generator#2" in ids
    assert len(set(ids)) == len(ids)


def test_seq_is_unique_and_contiguous_across_re_executions() -> None:
    """The dict keyed by agent name let a re-executed node overwrite its own earlier entry, so two
    nodes came out sharing one seq and the trace died on `duplicate seq` instead."""
    spans = build_spans(_repair_run())
    seqs = sorted(s.seq for s in spans)
    assert seqs == list(range(len(spans)))


def test_occurrence_says_which_attempt_without_parsing_the_id() -> None:
    spans = {(s.agent, s.occurrence): s for s in build_spans(_repair_run())}
    assert ("sql_generator", 1) in spans
    assert ("sql_generator", 2) in spans
    assert spans[("supervisor", 1)].occurrence == 1


def test_a_child_points_at_the_nearest_preceding_execution_of_its_parent() -> None:
    """For a repair loop this is the whole point: the second validator hangs off the second
    generator, not the first, or the tree claims a causal edge that never happened."""
    spans = build_spans(_repair_run())
    validators = [s for s in spans if s.agent == "sql_validator_repair"]
    assert [v.parent for v in validators] == ["t/sql_generator", "t/sql_generator#2"]


def test_a_single_execution_topology_is_completely_unchanged() -> None:
    """The regression guard for every existing fixture: no suffix appears where nothing repeats."""
    spans = build_spans(
        _inter(
            AgentNode(agent="supervisor", role="supervisor", parent=None, order=0),
            AgentNode(agent="worker", role="specialist", parent="supervisor", order=1),
        )
    )
    assert [s.span_id for s in spans] == ["t/supervisor", "t/worker"]
    assert [s.occurrence for s in spans] == [1, 1]
    assert spans[1].parent == "t/supervisor"


def test_a_parent_naming_an_absent_agent_still_dangles_for_the_validator_to_catch() -> None:
    """Negative path: the builder never silently drops or repoints an unresolvable edge."""
    spans = build_spans(
        _inter(
            AgentNode(agent="supervisor", role="supervisor", parent=None, order=0),
            AgentNode(agent="worker", role="specialist", parent="ghost", order=1),
        )
    )
    assert spans[1].parent == "ghost"
    with pytest.raises(SchemaError):
        validate(_as_trace(spans))


def test_an_adapter_pinned_span_id_still_wins() -> None:
    """`AgentNode.span_id` is the adapter's override; occurrence derivation must not steal it."""
    spans = build_spans(
        _inter(
            AgentNode(agent="a", role="supervisor", parent=None, order=0, span_id="pinned-0"),
            AgentNode(agent="a", role="specialist", parent="a", order=1, span_id="pinned-1"),
        )
    )
    assert [s.span_id for s in spans] == ["pinned-0", "pinned-1"]
    assert spans[1].parent == "pinned-0"
