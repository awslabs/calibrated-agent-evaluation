# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trajectory normalizer: ``normalize(source) -> Trace``.

The single seam that turns a vendor-shaped trace (Inspect ``.eval``, generic OTLP, a Strands
session) into the one normalized :class:`~calibrated_agent_evaluation.schema.types.Trace`, so the
methodology plane is written once against ``Trace``/``Span`` and never sees a vendor type.
It dispatches on the source descriptor, runs the chosen adapter to a vendor-free intermediate,
applies the ownership gate (owned vs degraded ``probed``), validates against the schema,
and returns the ``Trace``.

This package is the second member of the methodology-plane vendor allowlist (alongside
``backends/``, ``execute/``): the adapters import ``inspect_ai`` / ``opentelemetry.proto``;
lazily, inside their modules; so importing ``calibrated_agent_evaluation.normalize`` needs no opt-in
extra, and a missing extra surfaces as a typed :class:`NormalizationError` naming the installable
string, never a bare ``ImportError``.
"""

from __future__ import annotations

from collections import Counter

from calibrated_agent_evaluation.normalize.errors import NormalizationError
from calibrated_agent_evaluation.normalize.guidance import tag_span
from calibrated_agent_evaluation.normalize.ownership import Recovery, resolve_with_recovery
from calibrated_agent_evaluation.normalize.sources import (
    InspectEvalLog,
    NormalizerSource,
    OTelSpans,
    SessionSpans,
    StrandsSession,
)
from calibrated_agent_evaluation.normalize.tree import Intermediate
from calibrated_agent_evaluation.schema.errors import SchemaError
from calibrated_agent_evaluation.schema.types import Trace
from calibrated_agent_evaluation.schema.validate import validate

__all__ = [
    "InspectEvalLog",
    "NormalizationError",
    "NormalizerSource",
    "OTelSpans",
    "Recovery",
    "SessionSpans",
    "StrandsSession",
    "normalize",
    "normalize_all",
    "normalize_with_recovery",
    "tag_span",
]


def _to_intermediate(source: NormalizerSource | SessionSpans) -> Intermediate:
    """Dispatch on the source descriptor and run its adapter to the vendor-free intermediate.

    Adapters are imported lazily so the package imports with zero extras and a missing extra is
    caught as a typed error inside the adapter. An unrecognized descriptor raises
    ``NormalizationError``; no partial ``Trace`` is ever returned.
    """
    if isinstance(source, InspectEvalLog):
        from calibrated_agent_evaluation.normalize.inspect_adapter import to_intermediate

        return to_intermediate(source)
    if isinstance(source, OTelSpans):
        from calibrated_agent_evaluation.normalize.otel_adapter import to_intermediate

        return to_intermediate(source)
    if isinstance(source, StrandsSession):
        from calibrated_agent_evaluation.normalize.strands_adapter import to_intermediate

        return to_intermediate(source)
    if isinstance(source, SessionSpans):
        # A raw in-memory span list (from a dataset row or a caller). Pick the convention
        # (auto-detect or the declared one) and run its adapter via the shared convention layer.
        from calibrated_agent_evaluation.normalize.conventions.dispatch import to_intermediate

        return to_intermediate(source)
    # Unrecognized source descriptor; refuse loudly, drop nothing.
    raise NormalizationError(
        f"unrecognized normalizer source of type {type(source).__name__!r}; expected one of "
        f"InspectEvalLog / OTelSpans / StrandsSession / SessionSpans; for LangGraph, CrewAI or "
        "AutoGen, export the framework's OpenTelemetry spans and read them with OTelSpans"
    )


def normalize(source: NormalizerSource | SessionSpans) -> Trace:
    """Normalize one raw trace source into a validated :class:`Trace`.

    Returns an ``owned`` ``Trace`` when a span tree with per-step content was recovered, or a
    degraded ``probed`` ``Trace`` (``spans == []``, terminal I/O preserved) when it was not.
    Raises :class:`NormalizationError` (chained via ``from``) on an
    unrecognized/unsupported source, a missing opt-in extra, a vendor/parser/
    filesystem read failure, or a mapped ``Trace`` that fails ``validate()``.
    It never returns a partially-populated ``Trace``.

    Args:
        source: The raw trace to read: a file source such as ``OTelSpans`` or ``InspectEvalLog``,
            or in-memory ``SessionSpans``.

    Returns:
        The validated trace, ``owned`` or ``probed``.

    Raises:
        NormalizationError: The source can't be read, isn't recognized, needs a missing extra, or
            maps to a trace that fails validation.
    """
    trace, _recovery = normalize_with_recovery(source)
    return trace


def normalize_all(source: NormalizerSource | SessionSpans) -> list[Trace]:
    """Normalize a source into EVERY validated ``Trace`` it holds: one per unit of evaluation.

    For an :class:`InspectEvalLog` that unit is a **sample**, and a real eval log holds one per
    dataset row: the adapter used to read the first sample and drain the rest away, so an eval over
    N rows produced 1 trace and every downstream rate, count and gate decision came from 1/N of the
    data with no error anywhere. Every other source is one trace by construction, so this returns a
    single-element list for them and callers need no special-casing.

    Args:
        source: The raw trace source to read.

    Returns:
        One validated trace per unit of evaluation in the source.

    Raises:
        NormalizationError: As for :func:`normalize`.
    """
    if isinstance(source, InspectEvalLog):
        from calibrated_agent_evaluation.normalize.inspect_adapter import to_intermediates

        traces = [_validated(trace) for trace in _traces_from(to_intermediates(source))]
        _refuse_duplicate_ids(source, traces)
        return traces
    return [normalize(source)]


def _refuse_duplicate_ids(source: InspectEvalLog, traces: list[Trace]) -> None:
    """Refuse a log whose samples resolve to repeated trace ids; one would replace another."""
    counts = Counter(trace.trace_id for trace in traces)
    repeated = sorted(trace_id for trace_id, n in counts.items() if n > 1)
    if repeated:
        raise NormalizationError(
            f"Inspect log {str(source.path)!r} has several samples resolving to trace id(s) "
            f"{', '.join(map(repr, repeated))}; a store keyed on trace id would keep only one. "
            "Give each sample a distinct metadata trace_id"
        )


def _traces_from(intermediates: list[Intermediate]) -> list[Trace]:
    return [resolve_with_recovery(inter)[0] for inter in intermediates]


def _validated(trace: Trace) -> Trace:
    try:
        validate(trace)
    except SchemaError as exc:
        raise NormalizationError(
            f"normalized trace {trace.trace_id!r} failed schema validation: {exc}"
        ) from exc
    return trace


def normalize_with_recovery(
    source: NormalizerSource | SessionSpans,
) -> tuple[Trace, Recovery]:
    """:func:`normalize`, additionally returning what was recovered before the ownership collapse.

    For a caller that has to distinguish "the adapter recovered nothing" from "the adapter recovered
    a tree whose every node was empty"; the ingest gates, which refuse the first and degrade the
    second. Both produce a ``probed`` trace with ``spans == []``, so the finished ``Trace`` cannot
    tell them apart on its own.
    """
    intermediate = _to_intermediate(source)
    trace, recovery = resolve_with_recovery(intermediate)
    try:
        # Only schema-valid Traces leave the normalizer. A mapping bug (duplicate seq,
        # dangling parent, ...) surfaces as SchemaError → wrapped typed, never a raw SchemaError.
        validate(trace)
    except SchemaError as exc:
        raise NormalizationError(
            f"normalized trace {trace.trace_id!r} failed schema validation: {exc}"
        ) from exc
    return trace, recovery
