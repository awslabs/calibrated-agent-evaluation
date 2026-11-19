# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-format detection for ``calibrated-agent-evaluation ingest``.

Given a trace file, pick the normalizer source descriptor that recovers it faithfully:
``StrandsSession`` for a Strands OTLP-JSON (its tools nest under
``execute_event_loop_cycle → chat``, which the generic OTel adapter would drop; routing here is the
front-line defense the gate backstops), ``OTelSpans`` for a generic gen_ai OTLP-JSON, and
``InspectEvalLog`` for an Inspect ``.eval``. The trace id defaults to the file name's first dotted
segment (``CLM-4471.otlp.json → CLM-4471``), so ingest is deterministic and reproduces the frozen
oracle ids. Two files whose stems collide are disambiguated by :func:`wire_trace_id` instead --
see ``ingest.core.resolve_batch_trace_ids``.

Vendor-free: reads OTLP-JSON with the stdlib ``json`` module only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from calibrated_agent_evaluation.ingest.errors import IngestError
from calibrated_agent_evaluation.normalize import (
    InspectEvalLog,
    NormalizerSource,
    OTelSpans,
    StrandsSession,
    guidance,
    otlp_json,
)
from calibrated_agent_evaluation.normalize.sources import (
    Convention,
    DatasetSource,
    HFDataset,
    JSONLDataset,
    ParquetDataset,
    Semconv,
)
from calibrated_agent_evaluation.schema.types import Frame

_SUPPORTED = ".otlp.json / .json / .otlp / .eval / .jsonl / .parquet"

# Every value `--format` accepts. The trace formats resolve to one normalizer descriptor per file;
# the dataset formats to a reader that yields one trace per record.
TRACE_FORMATS: tuple[str, ...] = ("strands", "otel", "inspect")
DATASET_FORMATS: tuple[str, ...] = ("jsonl", "parquet", "hf")
FORMATS: tuple[str, ...] = ("auto", *TRACE_FORMATS, *DATASET_FORMATS)


def _trace_id_for(path: Path) -> str:
    """The file name's first dotted segment (``OR1.events.otlp.json → OR1``)."""
    return path.name.split(".")[0]


def _load_otlp(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IngestError(f"cannot read OTLP-JSON {path.name!r}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise IngestError(f"{path.name!r} is not an OTLP-JSON object")
    return loaded


def _looks_strands(otlp: dict[str, Any]) -> bool:
    """True if the trace was emitted by the Strands SDK.

    Two independent signals, either sufficient: the instrumentation scope name contains ``strands``
    (Strands emits under ``strands.telemetry.tracer``), or a span carries the Strands-specific
    ``execute_event_loop_cycle`` / ``invoke_graph`` nesting layer that defeats the generic adapter.
    """
    for resource_spans in otlp.get("resourceSpans", []) or []:
        for scope_spans in resource_spans.get("scopeSpans", []) or []:
            scope_name = str((scope_spans.get("scope") or {}).get("name") or "")
            if "strands" in scope_name.lower():
                return True
            for span in scope_spans.get("spans", []) or []:
                if str(span.get("name", "")).startswith(
                    ("execute_event_loop_cycle", "invoke_graph")
                ):
                    return True
    return False


def wire_trace_id(path: Path) -> str | None:
    """The trace id the SOURCE carries, or ``None`` when it carries none.

    Prefers the ``calibrated_agent_evaluation.trace_id`` convention attribute (the same first layer
    the adapters read), then the OTLP ``traceId`` rendered as stable hex. Used only to break a
    filename-stem collision: the stem stays the default because it is what makes ingest reproduce
    the frozen oracle ids, and an id that changes with the transport would make every fixture id a
    hex blob.

    Never raises. An unreadable or non-OTLP file simply has no wire id, and the caller is about to
    report that file's real problem through the normal detect/normalize path.
    """
    if path.suffix == ".eval":
        return None  # an Inspect log carries its own sample id; nothing OTel-shaped to read
    try:
        otlp = _load_otlp(path)
    except IngestError:
        return None
    for span in otlp_json.iter_spans(otlp):
        declared = otlp_json.attributes(span).get(guidance.ATTR_TRACE_ID)
        if declared:
            return str(declared)
    for span in otlp_json.iter_spans(otlp):
        raw = span.get("traceId")
        if isinstance(raw, str) and raw:
            return otlp_json.trace_id_hex(raw)
    return None


# Conventions a dataset record can be read with: the shared convention layer's (the built-in
# `gen_ai/latest` / `gen_ai/legacy` grammars read OTLP files only).
_DATASET_CONVENTIONS = ("auto", "gen_ai/messages", "openllmetry", "openinference")


def detect_dataset(
    path: Path, fmt: str = "auto", *, convention: str = "auto"
) -> DatasetSource | None:
    """Return the dataset reader for ``path``, or ``None`` when it is a single-trace source.

    ``jsonl`` / ``parquet`` force that reader; ``hf`` reads ``path`` as a HuggingFace repo id (the
    ``[hf]`` extra resolves it); ``auto`` picks by suffix (``.jsonl`` / ``.parquet``). A dataset
    is one trace per record, so it is ingested record by record rather than as one descriptor.
    """
    if fmt == "jsonl" or (fmt == "auto" and path.suffix == ".jsonl"):
        return JSONLDataset(path=path, semconv=_dataset_convention(convention))
    if fmt == "parquet" or (fmt == "auto" and path.suffix == ".parquet"):
        return ParquetDataset(path=path, semconv=_dataset_convention(convention))
    if fmt == "hf":
        return HFDataset(repo_id=path.as_posix(), semconv=_dataset_convention(convention))
    return None


def _dataset_convention(convention: str) -> Convention:
    if convention not in _DATASET_CONVENTIONS:
        raise IngestError(
            f"--convention {convention} reads OTLP files only; a dataset record is read with "
            f"one of {' | '.join(_DATASET_CONVENTIONS)}"
        )
    return cast("Convention", convention)


def detect_source(
    path: Path,
    fmt: str = "auto",
    *,
    trace_id: str | None = None,
    convention: str = "auto",
    frame: Frame = "discovery",
) -> NormalizerSource:
    """Return the normalizer source descriptor for ``path``.

    ``fmt`` forces a format (``strands`` / ``otel`` / ``inspect``); ``auto`` (default) detects it.
    ``trace_id`` overrides the filename-stem default (the batch collision resolver passes one).
    Raises :class:`IngestError` for an unrecognized shape; the caller records that as a skipped
    source, never a silent drop.

    ``frame`` applies to a source whose spans carry no ``calibrated_agent_evaluation.frame``; it
    defaults to ``"discovery"`` because a file says nothing about how its traces were sampled.
    """
    tid = trace_id or _trace_id_for(path)
    if convention != "auto":
        # A declared convention (`calibrated-agent-evaluation ingest --convention`) is the answer to
        # detect's ambiguity refusal, so it means the OTel reader with that grammar, skipping the
        # Strands sniff.
        if fmt not in ("auto", "otel"):
            raise IngestError(f"--convention applies to OTLP sources, not --format {fmt}")
        return OTelSpans(
            path=path,
            trace_id=tid,
            semconv=cast("Semconv | Convention", convention),
            frame=frame,
        )
    if fmt == "strands":
        return StrandsSession(path=path, trace_id=tid, frame=frame)
    if fmt == "otel":
        return OTelSpans(path=path, trace_id=tid, frame=frame)
    if fmt == "inspect":
        return InspectEvalLog(path=path, trace_id=tid, frame=frame)
    if fmt != "auto":
        raise IngestError(f"unknown --format {fmt!r}; expected {' | '.join(FORMATS)}")

    if path.suffix == ".eval":
        return InspectEvalLog(path=path, trace_id=tid, frame=frame)
    if path.suffix == ".otlp":
        # Binary protobuf; the generic adapter decodes it (the [otel] extra). Strands is detected
        # from OTLP-JSON only, so a binary Strands export needs --format strands.
        return OTelSpans(path=path, trace_id=tid, frame=frame)
    if path.name.endswith(".otlp.json") or path.suffix == ".json":
        otlp = _load_otlp(path)
        if _looks_strands(otlp):
            return StrandsSession(path=path, trace_id=tid, frame=frame)
        return OTelSpans(path=path, trace_id=tid, frame=frame)
    raise IngestError(f"unrecognized trace source {path.name!r}; expected one of {_SUPPORTED}")
