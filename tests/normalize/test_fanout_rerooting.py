# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The crew fan-out re-rooting policy, and its parity across adapters.

A Graph or Swarm emits its agent spans as siblings under a multiagent root that is not itself an
agent span, so every agent resolves to *no* agent parent. The Strands adapter re-rooted them under
the entry agent; the generic OTel adapter did not, and left N parentless roots — which
``validate()`` refuses, because every root takes ``seq`` 0. The consequence was that the identical
real capture normalized when read as a Strands session and was **unrepresentable** when the very
same spans were read as generic OTLP, with the failure surfacing as ``duplicate seq values`` rather
than as anything a caller could act on.

The policy now lives once in ``tree.reroot_siblings`` (policy belongs to ``tree``, parsing to the
adapters) and both adapters apply it, so these tests assert the shared behaviour and the parity.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from calibrated_agent_evaluation.normalize import OTelSpans, StrandsSession, normalize
from calibrated_agent_evaluation.normalize.tree import reroot_siblings
from calibrated_agent_evaluation.schema.validate import validate


def test_generic_otel_adapter_normalizes_a_real_crew_fanout(raw_dir: Path) -> None:
    """The real ClaimsCrew Graph read as *generic* OTLP builds a tree instead of being refused."""
    trace = normalize(
        OTelSpans(raw_dir / "strands" / "CLM-4471.otlp.json", semconv="gen_ai/latest")
    )
    validate(trace)

    assert trace.ownership == "owned"
    assert len(trace.spans) == 5
    roots = [span for span in trace.spans if span.parent is None]
    assert len(roots) == 1, "a fan-out must normalize to one root, not N"
    assert roots[0].agent == "supervisor"
    assert sorted(span.seq for span in trace.spans) == [0, 1, 2, 3, 4]
    assert roots[0].role == "supervisor"


def test_both_adapters_agree_on_the_tree_of_one_capture(raw_dir: Path) -> None:
    """Parity: one capture, two readers, same topology.

    Content recovery legitimately differs (each adapter implements its own convention's grammar);
    the *tree* must not, or which adapter a caller happened to pick would change the crew.
    """
    path = raw_dir / "strands" / "CLM-4471.otlp.json"
    strands = normalize(StrandsSession(path, trace_id="CLM-4471"))
    generic = normalize(OTelSpans(path, semconv="gen_ai/latest", trace_id="CLM-4471"))

    def shape(trace: object) -> set[tuple[str, str | None]]:
        return {(s.agent, s.parent) for s in trace.spans}  # type: ignore[attr-defined]

    assert shape(strands) == shape(generic)


def test_reroot_siblings_promotes_the_first_parentless_agent() -> None:
    """Two or more parentless agents: the first becomes the root, the rest its children."""
    parents = reroot_siblings([None, None, None], ["supervisor", "intake", "payout"])

    assert parents == [None, "supervisor", "supervisor"]


def test_reroot_siblings_leaves_a_single_root_untouched() -> None:
    """Negative: one parentless agent is already a valid root, so nothing is promoted."""
    parents = reroot_siblings([None, "solo"], ["solo", "child"])

    assert parents == [None, "solo"]


def test_reroot_siblings_never_repoints_a_nested_parent() -> None:
    """Negative: an Agent-as-Tools nesting is real parentage and must survive re-rooting.

    Re-rooting exists to give a *parentless* fan-out a root; a resolved parent is evidence, and
    overwriting it would invent a topology the capture does not show.
    """
    parents = reroot_siblings([None, None, "other"], ["entry", "other", "nested"])

    assert parents == [None, "entry", "other"]


def test_a_re_executed_entry_agent_does_not_become_a_second_root() -> None:
    """The SelectorGroupChat shape: planner, worker, planner, all parentless siblings.

    Keyed by name, both planner executions collapsed into one map entry, the second stayed a root,
    and the trace was refused on duplicate seq. Found on AutoGen 0.7.5's own telemetry.
    """
    parents = reroot_siblings([None, None, None], ["planner", "searcher", "planner"])

    assert parents == [None, "planner", "planner"]


def test_an_agent_run_under_two_parents_keeps_each_parent() -> None:
    """The silent half of the same defect: a name-keyed map kept only the last parent it saw."""
    parents = reroot_siblings([None, "x", None, "z"], ["x", "y", "z", "y"])

    assert parents == [None, "x", "x", "z"]


def test_reroot_siblings_refuses_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="2 parents for 3 names"):
        reroot_siblings([None, None], ["a", "b", "c"])


def _with_supervisor_re_executed(source: Path, target: Path) -> None:
    """The real ClaimsCrew Graph capture, with the supervisor's agent span run a second time.

    The copy gets a fresh span id and is moved after every other span, so it is the planner,
    worker, planner shape. Nothing else in the capture changes.
    """
    payload = json.loads(source.read_text(encoding="utf-8"))
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    supervisor = next(
        s
        for s in spans
        if any(
            a["key"] == "gen_ai.agent.name" and a["value"].get("stringValue") == "supervisor"
            for a in s.get("attributes", [])
        )
        and s["name"].startswith("invoke_agent")
    )
    last_end = max(int(s["endTimeUnixNano"]) for s in spans)
    again = copy.deepcopy(supervisor)
    again["spanId"] = "c3VwZXJ2aXNvcjI="  # base64 of b"supervisor2", an 8+ byte id
    duration = int(supervisor["endTimeUnixNano"]) - int(supervisor["startTimeUnixNano"])
    again["startTimeUnixNano"] = str(last_end + 1)
    again["endTimeUnixNano"] = str(last_end + 1 + duration)
    spans.append(again)
    target.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("reader", ["generic", "strands"])
def test_a_re_executed_supervisor_normalizes_to_one_root(
    raw_dir: Path, tmp_path: Path, reader: str
) -> None:
    path = tmp_path / "CLM-4471.otlp.json"
    _with_supervisor_re_executed(raw_dir / "strands" / "CLM-4471.otlp.json", path)
    source = (
        OTelSpans(path, semconv="gen_ai/latest", trace_id="CLM-4471")
        if reader == "generic"
        else StrandsSession(path, trace_id="CLM-4471")
    )

    trace = normalize(source)  # used to raise: duplicate seq values within trace
    validate(trace)

    roots = [s for s in trace.spans if s.parent is None]
    assert [r.agent for r in roots] == ["supervisor"]
    [again] = [s for s in trace.spans if s.agent == "supervisor" and s.occurrence == 2]
    assert again.parent == roots[0].span_id
    assert sorted(s.seq for s in trace.spans) == list(range(len(trace.spans)))
