# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset ingestion: one normalized ``Trace`` per session/record.

``load_traces(source, *, limit=None, mapping=None, frame="discovery")`` opens the reader for a
dataset source
(streaming parquet, JSONL, or an opt-in HuggingFace-hub resolver),
turns each record into a vendor-free
:class:`~calibrated_agent_evaluation.normalize.sources.SessionSpans`, and yields
``normalize(SessionSpans(...))``; so any captured-trace dataset flows through the same convention
layer + tree builder + schema validation as a single OTLP file.

This is the **only** package here permitted to import a loader SDK (``pyarrow`` for parquet,
``huggingface_hub`` for remote resolve); imported lazily inside the reader modules, so importing
``calibrated_agent_evaluation.normalize.datasets`` needs no extra and a missing extra surfaces as a
typed :class:`~calibrated_agent_evaluation.normalize.errors.NormalizationError` naming the
installable string, never a bare ``ImportError``. JSONL uses the stdlib ``json`` (zero dependency).
The public surface returns only normalized-schema ``Trace`` objects and vendor-free descriptors.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from calibrated_agent_evaluation.normalize import normalize
from calibrated_agent_evaluation.normalize.errors import NormalizationError
from calibrated_agent_evaluation.normalize.sources import (
    DatasetMapping,
    DatasetSource,
    HFDataset,
    JSONLDataset,
    ParquetDataset,
    SessionSpans,
)
from calibrated_agent_evaluation.schema.types import Frame, Trace

__all__ = [
    "DatasetMapping",
    "DatasetSource",
    "HFDataset",
    "JSONLDataset",
    "ParquetDataset",
    "SessionSpans",
    "iter_sessions",
    "load_traces",
]


def load_traces(
    source: DatasetSource,
    *,
    limit: int | None = None,
    mapping: DatasetMapping | None = None,
    frame: Frame = "discovery",
) -> Iterator[Trace]:
    """Yield one normalized ``Trace`` per record of a captured-trace dataset.

    Streams records from the source (parquet / JSONL / HuggingFace-hub resolve), turns each into
    a :class:`SessionSpans` (via ``mapping``; the source's own mapping is the default), and
    yields ``normalize(SessionSpans(...))``. Honors ``limit`` by stopping after N records
    (bounded working set). Raises a typed :class:`NormalizationError` on a
    missing extra, an unreadable source / bad mapping, or an unrecognized
    source type; never a partial or silently-dropped result.

    ``frame`` defaults to ``"discovery"``, deliberately NOT to ``SessionSpans.frame``'s own default.
    ``frame`` is an analyst's claim about where data came from, and
    ``"production-representative"`` is the one value that authorizes a headline rate
    (``sampling.assert_production_frame``); so inheriting it by omission meant a public benchmark
    slice arrived pre-authorized to make a production claim nobody had made. A caller who has not
    said where the data came from gets the frame that unlocks nothing.
    """
    if limit is not None and limit < 0:
        raise NormalizationError(f"limit must be non-negative, got {limit}")
    effective_mapping = mapping or getattr(source, "mapping", DatasetMapping())
    semconv = getattr(source, "semconv", "auto")
    for count, record in enumerate(_iter_records(source)):
        if limit is not None and count >= limit:
            break
        yield normalize(_record_to_session(record, effective_mapping, semconv, frame))


def iter_sessions(
    source: DatasetSource,
    *,
    mapping: DatasetMapping | None = None,
    frame: Frame = "discovery",
) -> Iterator[SessionSpans | NormalizationError]:
    """Yield each record of a dataset as its :class:`SessionSpans`, or the error mapping it.

    For a caller that needs each record before normalization, as
    ``calibrated-agent-evaluation ingest`` does to measure the source side of its no-silent-loss
    gates, and that reports a record which cannot be mapped on its own line instead of losing the
    rest of the dataset to it: a record the mapping cannot read is yielded as its
    :class:`NormalizationError`, not raised. A failure of the reader itself (an unreadable file, a
    malformed JSONL line) is raised, because the stream cannot continue. Same reader, mapping and
    ``frame`` default as :func:`load_traces`.
    """
    effective_mapping = mapping or getattr(source, "mapping", DatasetMapping())
    semconv = getattr(source, "semconv", "auto")
    for record in _iter_records(source):
        try:
            yield _record_to_session(record, effective_mapping, semconv, frame)
        except NormalizationError as exc:
            yield exc


def _iter_records(source: DatasetSource) -> Iterator[dict[str, Any]]:
    """Open the reader for the source kind and yield raw record dicts."""
    if isinstance(source, ParquetDataset):
        from calibrated_agent_evaluation.normalize.datasets import parquet

        return parquet.iter_rows(source.path)
    if isinstance(source, JSONLDataset):
        from calibrated_agent_evaluation.normalize.datasets import jsonl

        return jsonl.iter_rows(source.path)
    if isinstance(source, HFDataset):
        from calibrated_agent_evaluation.normalize.datasets import hf

        return hf.iter_rows(source)
    raise NormalizationError(
        f"unrecognized dataset source of type {type(source).__name__!r}; "
        f"expected ParquetDataset / JSONLDataset / HFDataset"
    )


def _record_to_session(
    record: dict[str, Any], mapping: DatasetMapping, semconv: str, frame: Frame = "discovery"
) -> SessionSpans:
    """Map one dataset record to a vendor-free SessionSpans via the column mapping."""
    try:
        spans = record[mapping.spans_column]
    except KeyError as exc:
        raise NormalizationError(
            f"dataset record has no spans column {mapping.spans_column!r}; "
            f"available columns: {sorted(record)}"
        ) from exc
    if not isinstance(spans, list):
        raise NormalizationError(
            f"spans column {mapping.spans_column!r} is not a list (got {type(spans).__name__})"
        )
    trace_id = record.get(mapping.session_id_column)
    taxonomy = {
        col: str(record[col]) for col in mapping.taxonomy_columns if record.get(col) is not None
    }
    return SessionSpans(
        spans=spans,
        semconv=semconv,  # type: ignore[arg-type]  # validated by the convention dispatch
        trace_id=str(trace_id) if trace_id is not None else None,
        taxonomy_cell=taxonomy,
        frame=frame,
    )
