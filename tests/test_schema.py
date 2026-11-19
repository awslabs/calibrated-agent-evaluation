# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the normalized schema (Trace/Span/Label) and its validate gate.

Tests assert behavior at the schema boundary (validation accept/reject, field preservation), not
dataclass getters (rubric rule 4). Invalid-literal cases pass values the type system would
reject (``# type: ignore``) precisely because ``validate`` — not the type checker — is the
runtime gate under test.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.schema import (
    Label,
    SchemaError,
    Span,
    ToolCall,
    Trace,
    validate,
)

# --- shared valid-artifact builders (a valid baseline each test mutates) ------------------


def _span(**over: object) -> Span:
    base = {
        "span_id": "T-1/root",
        "trace_id": "T-1",
        "agent": "orderbot",
        "role": "specialist",
        "parent": None,
        "seq": 0,
        "status": "completed",
        "input": {"message": "hi"},
        "tool_calls": [ToolCall(name="lookup", args={"id": "A"}, result={"ok": True})],
        "output": {"action": "DONE"},
    }
    base.update(over)
    return Span(**base)  # type: ignore[arg-type]


def _owned_trace(**over: object) -> Trace:
    base = {
        "trace_id": "T-1",
        "frame": "discovery",
        "ownership": "owned",
        "input": {"message": "hi"},
        "output": {"action": "DONE"},
        "taxonomy_cell": {"feature": "refund"},
        "spans": [_span()],
    }
    base.update(over)
    return Trace(**base)  # type: ignore[arg-type]


def _label(**over: object) -> Label:
    base = {
        "trace_id": "T-1",
        "span_id": None,
        "kind": "human",
        "outcome": "FAIL",
        "critique": "the agent ignored the policy tool result",
    }
    base.update(over)
    return Label(**base)  # type: ignore[arg-type]


# --- Trace owned/probed shape + invalid literals ---------------------------


def test_owned_trace_carries_populated_span_tree() -> None:
    trace = _owned_trace()
    validate(trace)  # does not raise
    assert trace.ownership == "owned"
    assert len(trace.spans) == 1


def test_probed_trace_has_no_spans_and_keeps_fingerprint() -> None:
    probed = _owned_trace(ownership="probed", spans=[], endpoint_fingerprint="sha256:abc123")
    validate(probed)  # does not raise
    assert probed.spans == []
    assert probed.endpoint_fingerprint == "sha256:abc123"


def test_owned_trace_with_no_spans_is_rejected() -> None:
    with pytest.raises(SchemaError, match="non-empty span tree"):
        validate(_owned_trace(spans=[]))


def test_probed_trace_with_spans_is_rejected() -> None:
    with pytest.raises(SchemaError, match="spans == \\[\\]"):
        validate(_owned_trace(ownership="probed"))


def test_invalid_frame_literal_is_rejected() -> None:
    with pytest.raises(SchemaError, match="frame"):
        validate(_owned_trace(frame="weekly"))  # type: ignore[arg-type]


def test_invalid_ownership_literal_is_rejected() -> None:
    with pytest.raises(SchemaError, match="ownership"):
        validate(_owned_trace(ownership="borrowed"))  # type: ignore[arg-type]


# --- Span preserves the step + ordering ------------------------------------


def test_span_preserves_input_tool_calls_and_output() -> None:
    span = _span(
        tool_calls=[
            ToolCall(name="lookup_order", args={"order_id": "A-7781"}, result={"amount": 89.99}),
            ToolCall(name="issue_refund", args={"order_id": "A-7781"}, result={"status": "issued"}),
        ]
    )
    trace = _owned_trace(spans=[span])
    validate(trace)
    [got] = trace.spans
    assert got.input == {"message": "hi"}
    assert got.output == {"action": "DONE"}
    assert [tc.name for tc in got.tool_calls] == ["lookup_order", "issue_refund"]
    assert got.tool_calls[0].args == {"order_id": "A-7781"}
    assert got.tool_calls[0].result == {"amount": 89.99}


def test_parent_and_seq_reconstruct_tree_ordering() -> None:
    root = _span(span_id="T-1/sup", agent="supervisor", role="supervisor", seq=0, parent=None)
    child_b = _span(span_id="T-1/b", parent="T-1/sup", seq=2)
    child_a = _span(span_id="T-1/a", parent="T-1/sup", seq=1)
    trace = _owned_trace(spans=[root, child_b, child_a])
    validate(trace)
    ordered = sorted(trace.spans, key=lambda s: s.seq)
    assert [s.span_id for s in ordered] == ["T-1/sup", "T-1/a", "T-1/b"]
    assert all(s.parent in (None, "T-1/sup") for s in trace.spans)


def test_span_with_non_integer_seq_is_rejected() -> None:
    with pytest.raises(SchemaError, match=r"seq.*integer"):
        validate(_owned_trace(spans=[_span(seq=None)]))  # type: ignore[arg-type]


def test_boolean_seq_is_rejected_even_though_bool_is_an_int() -> None:
    # bool is a subclass of int; an explicit ordering key of True/False is meaningless.
    with pytest.raises(SchemaError, match=r"seq.*integer"):
        validate(_owned_trace(spans=[_span(seq=True)]))


def test_duplicate_seq_within_a_trace_is_rejected() -> None:
    a = _span(span_id="T-1/a", parent=None, seq=0)
    b = _span(span_id="T-1/b", parent=None, seq=0)
    with pytest.raises(SchemaError, match="duplicate seq"):
        validate(_owned_trace(spans=[a, b]))


def test_duplicate_span_id_within_a_trace_is_rejected() -> None:
    # Two spans sharing a span_id (but distinct seq) would silently collapse to one in any
    # {span_id: span} index — the walk-back ancestry map and process-judge set-difference
    # both build one, so a real span would vanish from attribution. Refuse it at validation.
    a = _span(span_id="T-1/dup", agent="supervisor", role="supervisor", parent=None, seq=0)
    b = _span(span_id="T-1/dup", agent="intake", parent=None, seq=1)
    with pytest.raises(SchemaError, match="duplicate span_id"):
        validate(_owned_trace(spans=[a, b]))


def test_tool_call_missing_result_shape_is_rejected() -> None:
    with pytest.raises(SchemaError, match=r"result.*(object|dictionary)"):
        _span(tool_calls=[ToolCall(name="t", args={}, result="oops")])  # type: ignore[arg-type]


def test_unresolvable_parent_is_rejected() -> None:
    orphan = _span(span_id="T-1/x", parent="T-1/ghost", seq=1)
    root = _span(span_id="T-1/root", parent=None, seq=0)
    with pytest.raises(SchemaError, match="not a span in this trace"):
        validate(_owned_trace(spans=[root, orphan]))


def test_two_span_parent_cycle_is_rejected() -> None:
    # A→B→A: both parents resolve within the trace, so the membership check passes; only the
    # acyclicity check catches it. Without it the tree is not a tree (traversal never terminates).
    a = _span(span_id="T-1/a", parent="T-1/b", seq=0)
    b = _span(span_id="T-1/b", parent="T-1/a", seq=1)
    with pytest.raises(SchemaError, match="cycle"):
        validate(_owned_trace(spans=[a, b]))


def test_self_parented_span_is_rejected() -> None:
    loop = _span(span_id="T-1/loop", parent="T-1/loop", seq=0)
    with pytest.raises(SchemaError, match="cycle"):
        validate(_owned_trace(spans=[loop]))


# --- Label binary outcome + non-empty critique -----------------------------


def test_label_requires_binary_outcome_and_nonempty_critique() -> None:
    validate(_label(outcome="PASS", critique="correct in-window refund"))
    validate(_label(outcome="FAIL", critique="refund issued out of window"))


def test_empty_critique_is_rejected() -> None:
    with pytest.raises(SchemaError, match="critique must be non-empty"):
        validate(_label(critique="   "))


def test_non_binary_outcome_is_rejected() -> None:
    with pytest.raises(SchemaError, match="outcome"):
        validate(_label(outcome="MAYBE"))  # type: ignore[arg-type]


def test_invalid_kind_is_rejected() -> None:
    with pytest.raises(SchemaError, match="kind"):
        validate(_label(kind="oracle"))  # type: ignore[arg-type]


# --- trace_id mandatory on every artifact type -----------------------------


def test_validate_accepts_artifacts_bearing_a_trace_id() -> None:
    validate(_owned_trace())
    validate(_span())
    validate(_label())


@pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
def test_empty_or_whitespace_trace_id_is_rejected_on_trace(bad: str) -> None:
    with pytest.raises(SchemaError, match="trace_id"):
        validate(_owned_trace(trace_id=bad, spans=[_span(trace_id=bad)]))


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_trace_id_is_rejected_on_span(bad: str) -> None:
    with pytest.raises(SchemaError, match="trace_id"):
        validate(_span(trace_id=bad))


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_trace_id_is_rejected_on_label(bad: str) -> None:
    with pytest.raises(SchemaError, match="trace_id"):
        validate(_label(trace_id=bad))


# --- tool_calls on Span only; single-agent = one-span tree -----------------


def test_trace_exposes_no_tool_calls_attribute() -> None:
    # tool_calls live ONLY on Span; the methodology plane must read from exactly one place.
    assert not hasattr(_owned_trace(), "tool_calls")


def test_single_agent_owned_trace_is_a_one_span_tree_holding_tool_calls() -> None:
    root = _span(
        span_id="ORD-1001/orderbot",
        trace_id="ORD-1001",
        parent=None,
        seq=0,
        tool_calls=[ToolCall(name="issue_refund", args={}, result={"status": "issued"})],
    )
    trace = _owned_trace(trace_id="ORD-1001", spans=[root])
    validate(trace)
    assert len(trace.spans) == 1
    assert trace.spans[0].parent is None
    assert [tc.name for tc in trace.spans[0].tool_calls] == ["issue_refund"]


def test_owned_trace_with_zero_spans_is_rejected() -> None:
    with pytest.raises(SchemaError, match="span tree"):
        validate(_owned_trace(spans=[]))


# --- optional Label.expected (happy + default-null) ----------------


def test_label_with_expected_and_critique_validates_and_keeps_both() -> None:
    label = _label(
        outcome="FAIL",
        critique="refund issued out of window",
        expected={"action": "REFUND_DENIED", "amount_usd": 0},
    )
    validate(label)  # happy companion: expected + non-empty critique is valid
    assert label.expected == {"action": "REFUND_DENIED", "amount_usd": 0}
    assert label.critique


def test_label_omitting_expected_defaults_it_to_null() -> None:
    pass_label = _label(outcome="PASS", critique="correct in-window refund")
    validate(pass_label)
    assert pass_label.expected is None


def test_non_dict_expected_is_rejected() -> None:
    with pytest.raises(SchemaError, match=r"expected.*(object|dictionary)"):
        validate(_label(expected="should-be-an-object"))  # type: ignore[arg-type]


# --- expected never substitutes for the critique ---------------------------


def test_expected_set_with_empty_critique_is_rejected() -> None:
    with pytest.raises(SchemaError):
        validate(_label(critique="", expected={"action": "DENY"}))


def test_critique_is_required_even_when_expected_is_null() -> None:
    # Removing the critique is rejected regardless of whether expected is present.
    with pytest.raises(SchemaError, match="critique must be non-empty"):
        validate(replace(_label(), critique=""))


# --- the session a trace belongs to ----------------------------------------------------------


def test_a_trace_written_before_sessions_loads_with_no_session() -> None:
    from calibrated_agent_evaluation.schema.serde import to_dict, trace_from_dict
    from calibrated_agent_evaluation.schema.types import SCHEMA_VERSION

    doc = to_dict(_owned_trace())
    # A trace with no session is written without the key, as it was before the field existed.
    assert "session_id" not in doc
    loaded = trace_from_dict(doc)
    assert loaded.session_id is None
    assert loaded.schema_version == SCHEMA_VERSION == "1"


def test_the_committed_corpus_still_loads_with_no_session() -> None:
    import json
    from pathlib import Path

    from calibrated_agent_evaluation.schema.serde import to_dict, trace_from_dict

    paths = sorted(
        (Path(__file__).resolve().parent.parent / "fixtures" / "normalized").glob("*.json")
    )
    assert len(paths) == 6
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        trace = trace_from_dict(doc)
        validate(trace)
        assert trace.session_id is None, path.name
        # Written back, it gains no session key: the field adds nothing to a trace without one.
        assert "session_id" not in to_dict(trace), path.name


def test_a_trace_session_round_trips_through_serde() -> None:
    from calibrated_agent_evaluation.schema.serde import to_dict, trace_from_dict

    trace = _owned_trace(session_id="calibrated-agent-evaluation-0123456789abcdef0123456789abcdef")
    assert trace_from_dict(to_dict(trace)) == trace


def test_a_session_id_that_is_not_a_string_is_refused() -> None:
    from calibrated_agent_evaluation.schema.serde import to_dict, trace_from_dict

    doc = to_dict(_owned_trace())
    doc["session_id"] = 42
    with pytest.raises(SchemaError, match="session_id"):
        trace_from_dict(doc)


@pytest.mark.parametrize("blank", ["", "   "])
def test_an_empty_session_id_is_refused_and_nothing_is_stored(blank: str) -> None:
    from calibrated_agent_evaluation.backends.local import LocalBackend

    trace = _owned_trace(session_id=blank)
    with pytest.raises(SchemaError, match=r"trace T-1: session_id is blank"):
        validate(trace)
    with LocalBackend(":memory:") as store:
        with pytest.raises(SchemaError):
            store.put_trace(trace)
        assert store.query() == []
