# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural validation gate for the normalized schema.

``validate(artifact)`` is the in-code twin of ``fixtures/validate_corpus.py``: it enforces
the schema contract on a constructed dataclass, raising :class:`SchemaError` on any
violation. Backends call it on every ``put`` so invalid data never reaches the
store. The dataclasses themselves stay pure data holders; the ``Literal`` annotations
document intent but do not enforce at runtime, so the checks live here.

Enforced here:
  * non-empty ``trace_id`` on ``Trace``, ``Span``, and ``Label``.
  * valid ``frame``/``ownership`` literals; ``owned`` ⇒ populated span tree,
    ``probed`` ⇒ ``spans == []``.
  * valid ``role``/``status`` literals and an integer ``seq`` on every span;
    unique ``seq`` and resolvable ``parent`` within a trace.
  * binary ``outcome`` and a non-empty ``critique`` on every label.
  * a single-/multi-agent owned trace is a non-empty span tree (the one read path).
  * a trace's ``session_id`` is a non-blank string or ``None``.
  * a label's ``rater_id`` is a non-empty string or ``None``.
  * a label with a non-null ``expected`` must still carry a non-empty ``critique``
    (``expected`` may never substitute for the human critique).
"""

from __future__ import annotations

from typing import get_args

from calibrated_agent_evaluation.schema.errors import SchemaError
from calibrated_agent_evaluation.schema.types import (
    Frame,
    Label,
    LabelKind,
    Outcome,
    Ownership,
    Role,
    Span,
    SpanStatus,
    ToolCall,
    Trace,
)

_FRAMES = frozenset(get_args(Frame))
_OWNERSHIPS = frozenset(get_args(Ownership))
_OUTCOMES = frozenset(get_args(Outcome))
_ROLES = frozenset(get_args(Role))
_KINDS = frozenset(get_args(LabelKind))
_STATUSES = frozenset(get_args(SpanStatus))


def _require_trace_id(value: object, ctx: str) -> None:
    # Every artifact carries a non-empty (non-whitespace) trace_id.
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{ctx}: missing/empty trace_id")


def validate(artifact: Trace | Span | Label) -> None:
    """Validate a normalized artifact. Raises :class:`SchemaError` on any violation.

    Args:
        artifact: The trace, span or label to check.

    Raises:
        SchemaError: The artifact breaks a schema rule.
    """
    if isinstance(artifact, Trace):
        _validate_trace(artifact)
    elif isinstance(artifact, Span):
        _validate_span(artifact, expected_trace_id=artifact.trace_id)
    elif isinstance(artifact, Label):
        _validate_label(artifact)
    else:  # pragma: no cover - defensive: validate is only called on schema types
        raise SchemaError(f"validate: unsupported artifact type {type(artifact).__name__!r}")


def _validate_tool_call(tc: ToolCall, ctx: str) -> None:
    # Each tool call carries name, args, result.
    if not isinstance(tc.name, str) or not tc.name:
        raise SchemaError(f"{ctx}: tool_call missing 'name'")
    if not isinstance(tc.args, dict):
        raise SchemaError(f"{ctx}: tool_call {tc.name!r} 'args' must be an object")
    if not isinstance(tc.result, dict):
        raise SchemaError(f"{ctx}: tool_call {tc.name!r} 'result' must be an object")


def _validate_span(span: Span, *, expected_trace_id: str) -> None:
    ctx = f"span {span.span_id or '<unknown>'}"
    _require_trace_id(span.trace_id, ctx)
    if span.trace_id != expected_trace_id:
        raise SchemaError(f"{ctx}: span.trace_id does not match its parent trace")
    if not isinstance(span.span_id, str) or not span.span_id:
        raise SchemaError(f"{ctx}: missing span_id")
    if not isinstance(span.agent, str) or not span.agent:
        raise SchemaError(f"{ctx}: missing agent")
    if span.role not in _ROLES:
        raise SchemaError(f"{ctx}: role {span.role!r} not in {sorted(_ROLES)}")
    if span.parent is not None and not isinstance(span.parent, str):
        raise SchemaError(f"{ctx}: parent must be a span_id string or None")
    # Explicit integer seq (Langfuse has no native ordering). bool is an int; reject it.
    if not isinstance(span.seq, int) or isinstance(span.seq, bool):
        raise SchemaError(f"{ctx}: seq must be an integer (explicit ordering)")
    if span.status not in _STATUSES:
        raise SchemaError(f"{ctx}: status {span.status!r} not in {sorted(_STATUSES)}")
    if not isinstance(span.input, dict):
        raise SchemaError(f"{ctx}: span.input (handoff received) must be an object")
    if not isinstance(span.output, dict):
        raise SchemaError(f"{ctx}: span.output (handoff emitted) must be an object")
    if not isinstance(span.tool_calls, list):
        raise SchemaError(f"{ctx}: tool_calls must be a list")
    for tc in span.tool_calls:
        _validate_tool_call(tc, ctx)


def _validate_trace(trace: Trace) -> None:
    _require_trace_id(trace.trace_id, "trace")
    ctx = f"trace {trace.trace_id}"
    if trace.frame not in _FRAMES:
        raise SchemaError(f"{ctx}: frame {trace.frame!r} not in {sorted(_FRAMES)}")
    if trace.ownership not in _OWNERSHIPS:
        raise SchemaError(f"{ctx}: ownership {trace.ownership!r} not in {sorted(_OWNERSHIPS)}")
    if not isinstance(trace.input, dict):
        raise SchemaError(f"{ctx}: trace.input must be an object")
    if not isinstance(trace.output, dict):
        raise SchemaError(f"{ctx}: trace.output must be an object")
    if not isinstance(trace.taxonomy_cell, dict):
        raise SchemaError(f"{ctx}: taxonomy_cell must be an object")
    if not isinstance(trace.spans, list):
        raise SchemaError(f"{ctx}: spans must be a list")
    # A blank session would group every blank-session trace together as one session.
    if trace.session_id is not None and not trace.session_id.strip():
        raise SchemaError(
            f"{ctx}: session_id is blank; leave it unset when the source recorded no session"
        )

    # Owned ⇒ populated span tree (the single tool-call read path);
    # probed ⇒ outcome-only with no spans.
    if trace.ownership == "owned" and not trace.spans:
        raise SchemaError(f"{ctx}: owned trace must have a non-empty span tree")
    if trace.ownership == "probed" and trace.spans:
        raise SchemaError(f"{ctx}: probed trace must have spans == [] (outcome-only)")

    span_ids: set[str] = set()
    seqs: list[int] = []
    parent_of: dict[str, str | None] = {}
    for span in trace.spans:
        _validate_span(span, expected_trace_id=trace.trace_id)
        span_ids.add(span.span_id)
        seqs.append(span.seq)
        parent_of[span.span_id] = span.parent
    # Tree integrity: every non-null parent must reference a span in this trace.
    for span in trace.spans:
        if isinstance(span.parent, str) and span.parent not in span_ids:
            raise SchemaError(
                f"span {span.span_id}: parent {span.parent!r} is not a span in this trace"
            )
    # span_id values are unique within a trace. A duplicate id silently collapses two spans
    # into one in any {span.span_id: span} index (e.g. the walk-back ancestry map and the
    # process judge's set-difference), so a real span would vanish from attribution; refuse it.
    if len(span_ids) != len(trace.spans):
        raise SchemaError(f"{ctx}: duplicate span_id values within trace")
    # The span tree must be acyclic; a parent chain that loops (A→B→A, or a span
    # parented to itself) is not a tree and would make ordering/traversal non-terminating.
    _require_acyclic(parent_of, ctx)
    # seq values are unique within a trace (so ordering is total).
    if len(seqs) != len(set(seqs)):
        raise SchemaError(f"{ctx}: duplicate seq values within trace")


def _require_acyclic(parent_of: dict[str, str | None], ctx: str) -> None:
    # Walk each span's parent chain; a repeated node on one chain is a cycle. Parents are already
    # known to resolve within the trace, so every chain ends at a root (None) unless it loops.
    for start in parent_of:
        seen: set[str] = set()
        node: str | None = start
        while node is not None:
            if node in seen:
                raise SchemaError(f"{ctx}: span parent cycle involving {node!r}")
            seen.add(node)
            node = parent_of.get(node)


def _validate_label(label: Label) -> None:
    _require_trace_id(label.trace_id, "label")
    ctx = f"label for {label.trace_id}"
    if label.outcome not in _OUTCOMES:
        raise SchemaError(f"{ctx}: outcome {label.outcome!r} not in {sorted(_OUTCOMES)}")
    # Non-empty critique; the critique-before-verdict asset. Always required.
    if not isinstance(label.critique, str) or not label.critique.strip():
        raise SchemaError(f"{ctx}: critique must be non-empty")
    if label.kind not in _KINDS:
        raise SchemaError(f"{ctx}: kind {label.kind!r} not in {sorted(_KINDS)}")
    if not isinstance(label.span_defect, bool):
        raise SchemaError(f"{ctx}: span_defect must be a boolean")
    if label.span_id is not None and not isinstance(label.span_id, str):
        raise SchemaError(f"{ctx}: span_id must be a span_id string or None")
    # rater_id names a rater or is absent. An empty string would read as attributed while naming
    # nobody, which is worse than None for an agreement statistic keyed on it.
    if label.rater_id is not None and (
        not isinstance(label.rater_id, str) or not label.rater_id.strip()
    ):
        raise SchemaError(f"{ctx}: rater_id must be a non-empty string or None")
    # expected is an object or null.
    if label.expected is not None and not isinstance(label.expected, dict):
        raise SchemaError(f"{ctx}: expected must be an object or None")
    # expected may never substitute for the human critique (re-checked explicitly so the
    # rule is local to expected even if the critique check above is ever relaxed).
    if label.expected is not None and (
        not isinstance(label.critique, str) or not label.critique.strip()
    ):
        raise SchemaError(
            f"{ctx}: a Label with 'expected' set must still carry a non-empty critique"
        )
