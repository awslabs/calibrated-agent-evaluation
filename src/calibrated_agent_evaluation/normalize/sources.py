# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vendor-free source descriptors: the public input surface of ``normalize``.

Each descriptor names *one* raw trace to normalize and, where the wire cannot carry them,
lets the caller stamp the trace-level header fields (``trace_id`` / ``frame`` /
``taxonomy_cell``). These headers resolve in a layered order:

  1. a Calibrated Agent Evaluation **convention attribute** on the wire
     (``calibrated_agent_evaluation.trace_id`` / ``calibrated_agent_evaluation.frame`` /
     ``calibrated_agent_evaluation.taxonomy.<key>``), or the Inspect ``.eval`` ``sample.metadata``
     which carries ``trace_id`` + ``taxonomy_cell`` natively;
  2. the **descriptor override** below (an explicit one-off ingest);
  3. a **wire/default fallback**; ``trace_id`` from the OTel ``traceId`` hex, ``frame``
     defaulting to ``"production-representative"``, ``taxonomy_cell`` to ``{}``.

``frame`` is an analyst-assigned sampling label (why the trace was captured), not a runtime
property an agent emits; so the caller, not the trace, is its authoritative source
(:mod:`calibrated_agent_evaluation.normalize.guidance` documents the convention and helpers). No
vendor SDK type appears here; these are frozen dataclasses over stdlib types only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from calibrated_agent_evaluation.schema.types import Frame

# OTel semantic-convention modes the adapter branches on. "gen_ai/legacy" and
# "gen_ai/latest" are the two OTel GenAI convention generations; "openllmetry" is read by the
# convention layer.
Semconv = Literal["gen_ai/legacy", "gen_ai/latest", "openllmetry"]

# The convention set the shared convention layer reconstructs. "auto" runs detection (the default
# for the new sources); the named values force one adapter. `gen_ai/messages` (Claude Code /
# current OpenLLMetry), `openllmetry` (legacy dotted-index + latest-JSON fall-through),
# `openinference` (Arize/Phoenix) are the convention-layer conventions; the two built-in grammars
# stay in `otel_adapter` under `Semconv`.
Convention = Literal["gen_ai/messages", "openllmetry", "openinference", "auto"]
# Where message/tool content resides on a span: span attributes or span events.
ContentLocation = Literal["attributes", "events"]


@dataclass(frozen=True)
class InspectEvalLog:
    """An Inspect ``.eval`` (or ``.json``) log to normalize.

    ``trace_id``/``taxonomy_cell`` are read from the sample metadata when present; ``frame``
    is analyst-assigned and defaults to ``"production-representative"`` (override per capture).

    An Inspect log holds **one sample per dataset row**, so a real eval log is normally
    multi-sample. Use :func:`calibrated_agent_evaluation.normalize.normalize_all` for one ``Trace``
    per sample; on a multi-sample log refuses unless ``sample_id`` names which one to read, rather
    than silently evaluating the log at 1/N of its data.

    Attributes:
        path: The ``.eval`` or ``.json`` log file.
        trace_id: The trace id; read from the sample metadata when omitted.
        frame: The sampling frame.
        taxonomy_cell: The taxonomy cell; read from the sample metadata when omitted.
        sample_id: The sample to read from a multi-sample log.
    """

    path: Path
    trace_id: str | None = None
    frame: Frame = "production-representative"
    taxonomy_cell: dict[str, str] = field(default_factory=dict)
    sample_id: str | None = None  # select ONE sample from a multi-sample log


@dataclass(frozen=True)
class OTelSpans:
    """A generic OpenTelemetry source.

    ``path`` is a frozen OTLP-JSON file (read with stdlib ``json``) or a binary ``.otlp``
    protobuf (needs the ``[otel]`` extra). ``semconv`` selects the convention branch.
    ``content_location`` is a **best-effort preference hint**, not a hard gate: the reader tries
    the named placement first, then falls back to the other, because real traces mix span
    attributes and span events; falling back is what prevents the silent-drop of
    event-located content. The header fields resolve in the layered order (convention attribute →
    this override → OTel-``traceId`` hex / defaults).

    Attributes:
        path: The OTLP-JSON or binary ``.otlp`` file.
        semconv: The OTel GenAI convention to read with, or ``"auto"``.
        content_location: Where to look for message content first.
        trace_id: The trace id, when the spans don't carry one.
        frame: The sampling frame.
        taxonomy_cell: The taxonomy cell, when the spans don't carry one.
    """

    path: Path
    # Default to "auto" so pointing at ANY OTLP file just works; detection picks the
    # convention (the convention-layer conventions + the two built-in grammars), and
    # raises a typed error only when the signal is absent/ambiguous (never a silent wrong guess).
    # A caller may still force a grammar by naming it (fixtures declare gen_ai/latest).
    semconv: Semconv | Convention = "auto"
    content_location: ContentLocation = "events"
    trace_id: str | None = None
    frame: Frame = "production-representative"
    taxonomy_cell: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StrandsSession:
    """An OTel-native Strands export.

    Strands is pattern-agnostic (Graph / Swarm / Agent-as-Tools all share one span-name
    grammar); ``mode`` selects the legacy-vs-latest GenAI convention. Headers resolve in the
    layered order.

    Attributes:
        path: The exported OTLP-JSON file.
        mode: The GenAI convention generation the export uses.
        trace_id: The trace id, when the spans don't carry one.
        frame: The sampling frame.
        taxonomy_cell: The taxonomy cell, when the spans don't carry one.
    """

    path: Path
    mode: Literal["legacy", "latest"] = "latest"
    trace_id: str | None = None
    frame: Frame = "production-representative"
    taxonomy_cell: dict[str, str] = field(default_factory=dict)


NormalizerSource = InspectEvalLog | OTelSpans | StrandsSession


# --- dataset ingestion + in-memory session descriptors -----------------------------
# These extend the normalizer's public input surface to captured-trace datasets and raw span
# lists. All vendor-free (stdlib types only): pyarrow/huggingface_hub live behind
# `normalize.datasets`, never on this surface. `SessionSpans` is the seam the dataset loaders and
# the shared convention layer both hand to the tree builder.


@dataclass(frozen=True)
class SessionSpans:
    """One session's raw OTLP-JSON spans, in memory, to normalize via the convention layer.

    ``spans`` is the list of span dicts (canonical OTLP-JSON shape) for a single session/record.
    ``semconv="auto"`` runs detection; a named :data:`Convention` forces one adapter.
    The header fields resolve in the layered order (convention attribute → this override →
    wire/default).
    This is what a dataset row becomes before the convention adapter turns it into a ``Trace``.

    Attributes:
        spans: The session's spans as OTLP-JSON dicts.
        semconv: The convention to read with, or ``"auto"``.
        trace_id: The trace id, when the spans don't carry one.
        frame: The sampling frame.
        taxonomy_cell: The taxonomy cell, when the spans don't carry one.
    """

    spans: list[dict[str, Any]]
    semconv: Convention = "auto"
    trace_id: str | None = None
    frame: Frame = "production-representative"
    taxonomy_cell: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetMapping:
    """Which columns a dataset keeps its session id, span list, and taxonomy dims in.

    Defaults match the ``Exgentic/agent-llm-traces`` shape. A differently-shaped dataset supplies
    its own column names so the loader reads it without a code change. ``taxonomy_columns``
    are copied into each ``Trace``'s ``taxonomy_cell`` as queryable dims (a benchmark's own
    ``success``/``score`` may be listed here, but are NEVER turned into a gold ``Label``).
    """

    spans_column: str = "spans"
    session_id_column: str = "session_id"
    taxonomy_columns: tuple[str, ...] = ("harness", "benchmark")


@dataclass(frozen=True)
class ParquetDataset:
    """A local parquet dataset: one ``Trace`` per row. Needs the ``[datasets]`` extra."""

    path: Path
    semconv: Convention = "auto"
    mapping: DatasetMapping = field(default_factory=DatasetMapping)


@dataclass(frozen=True)
class JSONLDataset:
    """A local JSON-lines dataset: one ``Trace`` per line. Read with the stdlib json."""

    path: Path
    semconv: Convention = "auto"
    mapping: DatasetMapping = field(default_factory=DatasetMapping)


@dataclass(frozen=True)
class HFDataset:
    """A HuggingFace-hub dataset resolved by repo id. Needs the ``[hf]`` extra.

    Resolves the repo's parquet shards to local files, then normalizes through the same path as
    :class:`ParquetDataset`. ``split`` selects the config split (default ``"train"``).
    """

    repo_id: str
    split: str = "train"
    semconv: Convention = "auto"
    mapping: DatasetMapping = field(default_factory=DatasetMapping)


DatasetSource = ParquetDataset | JSONLDataset | HFDataset
