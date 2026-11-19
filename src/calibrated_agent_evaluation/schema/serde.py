# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless (de)serialization between normalized models and plain JSON-able dicts.

One canonical mapping used by *both* the LocalBackend (SQLite JSON columns) and the
dump/load codecs (CSV/JSON), so a ``put``→``get`` and a ``dump``→``load`` reconstruct an
artifact identically; field-for-field. Keeping this in ``schema/`` (not a
backend) means the mapping is vendor-free and shared, never duplicated per backend.

``to_dict`` produces a JSON-serializable dict; ``trace_from_dict`` / ``span_from_dict`` /
``label_from_dict`` rebuild the frozen model. Round-trip equality is the contract these
guarantee and the property test exercises.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from calibrated_agent_evaluation.schema.errors import SchemaError
from calibrated_agent_evaluation.schema.types import (
    SCHEMA_VERSION,
    Label,
    Span,
    TokenUsage,
    ToolCall,
    Trace,
)


def to_dict(artifact: Trace | Span | Label | ToolCall) -> dict[str, Any]:
    """Return a JSON-serializable dict for a normalized artifact (nested objects included).

    Args:
        artifact: The artifact to convert.

    Returns:
        The artifact as plain dicts and lists.
    """
    return artifact.model_dump(mode="json")


def tool_call_from_dict(data: dict[str, Any]) -> ToolCall:
    """Rebuild a ``ToolCall`` from the dict :func:`to_dict` produced."""
    return ToolCall.model_validate(data)


def token_usage_from_dict(data: dict[str, Any] | None) -> TokenUsage | None:
    """Rebuild a ``TokenUsage``, keeping an absent record absent.

    ``None`` stays ``None`` rather than becoming an empty record, because "the source reported
    nothing" and "the source reported no numbers" are different facts and `to_dict` round-trips
    both.
    """
    return None if data is None else TokenUsage.model_validate(data)


def span_from_dict(data: dict[str, Any]) -> Span:
    """Rebuild a ``Span`` from the dict :func:`to_dict` produced."""
    return Span.model_validate(data)


def trace_from_dict(data: dict[str, Any]) -> Trace:
    """Rebuild a ``Trace`` from the dict :func:`to_dict` produced.

    Args:
        data: The dict form of a trace.

    Returns:
        The trace.

    Raises:
        SchemaError: The dict carries no ``schema_version``.
    """
    _require_schema_version(data)
    return Trace.model_validate(data)


def _require_schema_version(data: dict[str, Any]) -> str:
    """Return the dict's ``schema_version``, refusing when it carries none.

    ``data.get("schema_version", SCHEMA_VERSION)`` stamped the *running* version onto a dict that
    had none, contradicting ``types.SCHEMA_VERSION``'s own stated contract ("`load` refuses on
    mismatch (no silent coercion)"): a dict with no version was treated as if it had already
    matched. Every writer in the repo stamps it -- ``to_dict`` on a ``Trace`` always emits it, and
    every frozen corpus fixture carries it -- so an absent one means the document did not come from
    this schema, and guessing is the one thing that must not happen.
    """
    version = data.get("schema_version")
    if version is None:
        raise SchemaError(
            f"trace {data.get('trace_id')!r} carries no schema_version; refusing to assume it is "
            f"the running {SCHEMA_VERSION!r} (no silent coercion). Re-dump it from a Calibrated "
            "Agent Evaluation "
            "backend, or add the field explicitly if you are migrating by hand"
        )
    return str(version)


def label_identity(label: Label) -> str:
    """The identity two writes of "the same label" share: a canonical encoding of every field.

    Both score-write paths are append-only -- ``LocalBackend`` inserts against an autoincrement
    primary key, and Langfuse's ``create_score`` API has no update -- so a re-``put_score`` of one
    label leaves two rows, and every rate computed over them moves toward whichever verdict was
    written twice.

    **Full content, not a key like ``(trace_id, span_id, kind)``.** A coarser key is tempting and
    wrong: it silently discards labels that genuinely differ. Measured while writing this --
    ``test_get_scores_pages_through_all_results_no_silent_truncation`` writes 120 human labels
    on one trace with distinct critiques, and a ``(trace_id, span_id, kind, codebook_version)`` key
    collapsed all 120 to one. Trading a double-count for a silent drop is not a fix.

    It also carries the dual-coding path. ``rater_id`` is part of the content, so two raters who
    double-code the same span stay two labels even when their verdict *and* critique match (two
    PASSes with the canonical default critique differ only in the rater), while one rater
    re-putting one label still collapses to one.
    """
    return json.dumps(to_dict(label), sort_keys=True, default=str)


def dedup_labels(labels: Iterable[Label]) -> list[Label]:
    """Collapse byte-identical labels, keeping the last written, preserving first-seen order.

    Read-side rather than write-side, which is the pattern ``LangfuseBackend._reconstruct_trace``
    already established for the same root cause: Langfuse's write path is append-only and its delete
    is asynchronous across Postgres+ClickHouse (verified unreliable even after a 25s wait), so a
    correct read is the only fix available there. Applying it to both backends keeps one behaviour
    instead of two, and needs no migration of an existing local store.
    """
    by_identity: dict[str, Label] = {}
    for label in labels:
        by_identity[label_identity(label)] = label
    return list(by_identity.values())


_MACHINE_KINDS = frozenset({"llm-judge", "code"})


def latest_verdicts(labels: Iterable[Label]) -> list[Label]:
    """Keep one machine verdict per target, failure mode and rater: the newest. Order is kept.

    ``labels`` must be oldest first, which is the order ``get_scores`` returns. A judge re-run
    writes a new verdict for every target it already scored, and a live model words its critique
    differently each time, so :func:`dedup_labels` keeps both rows. Counting both doubles n with
    no new evidence, which narrows every interval and can turn a failing gate green.

    Only LLM-judge and code-scorer verdicts with a ``rater_id`` are collapsed, keyed on
    ``(trace_id, span_id, kind, failure_mode, rater_id)``: that rater id names the judge and its
    exact version, so two such rows are the same judgment made twice. Human labels and verdicts
    with no recorded rater pass through unchanged, since two of them on one target may be
    different people or different judges.

    Args:
        labels: Labels in write order, oldest first.

    Returns:
        The labels with every superseded machine verdict removed.
    """
    ordered = list(labels)
    newest: dict[tuple[object, ...], int] = {}
    for index, label in enumerate(ordered):
        if label.kind in _MACHINE_KINDS and label.rater_id is not None:
            key = (label.trace_id, label.span_id, label.kind, label.failure_mode, label.rater_id)
            newest[key] = index
    keep = set(newest.values())
    return [
        label
        for index, label in enumerate(ordered)
        if index in keep or label.kind not in _MACHINE_KINDS or label.rater_id is None
    ]


def label_from_dict(data: dict[str, Any]) -> Label:
    """Rebuild a ``Label`` from the dict :func:`to_dict` produced.

    A document written before ``rater_id`` existed is unattributed, not malformed: the field
    defaults to ``None``.
    """
    return Label.model_validate(data)
