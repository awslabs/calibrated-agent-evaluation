# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized data-model contract.

The vendor-neutral ``Trace`` / ``Span`` / ``Label`` models every plane serializes to. They are
frozen pydantic models: construction checks each field's type and literal values (a bad ``frame``
or a boolean ``seq`` raises :class:`~calibrated_agent_evaluation.schema.errors.SchemaError`), while
the rules that span several fields (the span tree, critique before verdict, ownership) live in
:mod:`calibrated_agent_evaluation.schema.validate`, which backends call on ``put``. No vendor SDK
type appears here; the methodology plane reads only these.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import ConfigDict, Field, StrictInt

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.schema.errors import SchemaError

# Stamped on every persisted Trace; `load` refuses on mismatch (no silent coercion).
SCHEMA_VERSION = "1"

Ownership = Literal["owned", "probed"]
Frame = Literal["discovery", "production-representative"]
Outcome = Literal["PASS", "FAIL"]
Role = Literal["specialist", "supervisor", "tool"]
# "policy-engine": a policy engine denied a call the agent attempted; never gold, never a judge.
LabelKind = Literal["human", "llm-judge", "code", "policy-engine"]
SpanStatus = Literal["completed", "errored"]


class ToolCall(CalibratedAgentEvaluationModel):
    """One tool invocation inside a span. Reconstructed by the normalizer."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = SchemaError

    name: str
    args: dict[str, Any]
    result: dict[str, Any]


class TokenUsage(CalibratedAgentEvaluationModel):
    """Tokens a single agent step consumed, as the source reported them.

    Every field is optional and independently so, because a source may report a total without its
    parts (or the reverse), and a normalizer must never compute one from the others; a derived
    total is indistinguishable from a reported one once it is stored, and the two disagree whenever
    a provider counts cached or reasoning tokens outside the pair. Absent means *not reported*,
    never zero.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = SchemaError

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class Span(CalibratedAgentEvaluationModel):
    """One agent step in an owned trace.

    Preserves the handoff ``input`` received, the ``tool_calls`` made, the ``output``
    emitted, and explicit tree ordering via ``parent`` + ``seq`` (Langfuse has no native
    ordering, so ``seq`` is carried explicitly).

    ``usage`` carries the per-step token counts when the source reported them. It exists because
    normalizing used to *discard* them: OTel sources carry ``gen_ai.usage.*`` per agent span, the
    schema had nowhere to put it, and so a normalized ``Trace`` could describe a run's topology
    while being silent about what the run cost; which is much of why a team ingests traces at all.
    ``None`` means the source reported nothing, and is deliberately different from a
    :class:`TokenUsage` whose fields are all ``None``: the second says a usage record existed and
    was empty.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = SchemaError

    span_id: str
    trace_id: str  # every artifact references a trace_id
    agent: str
    role: Role
    parent: str | None  # tree edge: None for the root span
    seq: StrictInt  # explicit ordering; a bool is refused, not read as 0 or 1
    status: SpanStatus
    input: dict[str, Any]  # handoff payload RECEIVED
    tool_calls: list[ToolCall] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)  # handoff payload EMITTED
    started_at: str | None = None
    ended_at: str | None = None
    usage: TokenUsage | None = None
    #: Which execution of ``agent`` this span is: 1 for a node that ran once, 2 for the second
    #: execution of the same agent, and so on. A cyclic topology (a repair or retry edge) re-runs a
    #: node, so ``agent`` alone does not identify a step; and reading the count off the ``span_id``
    #: suffix would make an identifier carry a record, which is what collapsed here in the first
    #: place. Ordering still comes from ``seq``; this answers "which attempt", not "when".
    occurrence: StrictInt = 1


class Trace(CalibratedAgentEvaluationModel):
    """The central object.

    An ``owned`` trace carries a populated ``spans`` tree; a ``probed`` trace is
    input/output only with ``spans == []`` (and an ``endpoint_fingerprint``). ``frame``,
    ``ownership``, and ``taxonomy_cell`` are first-class queryable dimensions.

    ``tool_calls`` live ONLY on ``Span``; ``Trace`` deliberately has no
    ``tool_calls`` field. A single-agent owned trace is therefore represented as a
    *one-span tree*: a single root ``Span`` (``parent=None``, ``seq=0``) that holds the
    agent's ``tool_calls`` (see ``fixtures/normalized/ORD-1001.json``). The methodology
    plane reads tool calls from exactly one place (``span.tool_calls``) for both single-
    and multi-agent traces; no branching on agent architecture (target-agnosticism).
    The ``Trace``-level ``input``/``output`` remain the terminal request/decision; the
    root span's ``input``/``output`` mirror them for a single-agent trace.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = SchemaError

    trace_id: str
    frame: Frame
    ownership: Ownership  # owned ⇒ spans populated; probed ⇒ spans == []
    input: dict[str, Any]
    output: dict[str, Any]
    taxonomy_cell: dict[str, str]
    spans: list[Span] = Field(default_factory=list)
    captured_at: str | None = None
    endpoint_fingerprint: str | None = None  # probed only
    #: The session the trace belongs to (a conversation, or one agent run), as its source named
    #: it: the runtime session an agent was invoked under, or the OpenTelemetry ``session.id`` its
    #: spans carried. Traces with one ``session_id`` are turns of one session. ``None`` means the
    #: source recorded no session; Calibrated Agent Evaluation never makes one up, so an unset value
    # stays visible. : An unset value is left out when the trace is serialized, so a trace with no
    # session is : written exactly as it was before the field existed (same JSON, same hash).
    session_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    schema_version: str = SCHEMA_VERSION


class Label(CalibratedAgentEvaluationModel):
    """The universal eval-result object, mapped to a Langfuse ``Score`` on write.

    Binary ``outcome`` and a required, non-empty ``critique`` are structural:
    a non-binary or critique-less label cannot be represented. Carries the crew-attribution
    fields (``culprit_span`` / ``manifestation_span`` / ``handoff_implicated`` / ``span_defect``)
    and ``codebook_version``; each references a ``trace_id`` and an optional ``span_id``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = SchemaError

    trace_id: str
    span_id: str | None  # set for span-level gold; None for trace-level
    kind: LabelKind
    outcome: Outcome  # binary
    critique: str  # non-empty, ordered before verdict
    failure_mode: str | None = None
    culprit_span: str | None = None  # crew attribution
    manifestation_span: str | None = None
    handoff_implicated: str | None = None
    span_defect: bool = False  # true ⇒ non-gating span-fail
    codebook_version: str | None = None
    # A human's correct output for an objective FAIL (an action as {"action": "..."});
    # null for subjective/PASS. NEVER substitutes for the required critique.
    expected: dict[str, Any] | None = None
    #: Who applied the label: the rater's stable id for a human label (the adjudicator's, for an
    #: adjudication row). ``kind`` separates a human from a judge, not one human from another, so
    #: without this two raters double-coding one trace write rows nobody can tell apart and
    #: inter-rater agreement is unrepresentable. ``None`` means *not recorded* (a label written
    #: before the field existed, or by a non-human path), never "anonymous".
    rater_id: str | None = None
