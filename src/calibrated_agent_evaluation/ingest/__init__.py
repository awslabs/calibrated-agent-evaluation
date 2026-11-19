# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace ingestion & capture.

The first-class input boundary: turn an agent run you captured, a directory of trace dumps, or a
Langfuse project into normalized Traces in the store; with no bespoke glue and no silent loss.

Public surface is re-exported here so callers write
``from calibrated_agent_evaluation.ingest import run_ingest``. Vendor SDKs (OpenTelemetry, Langfuse)
are reached only through the allowlisted ``execute/`` and ``backends/`` planes; nothing under
``ingest/`` imports a vendor type.
"""

from __future__ import annotations

from calibrated_agent_evaluation.ingest.core import (
    IngestReport,
    IngestResult,
    IngestStatus,
    capture_and_ingest,
    contentless_node_count,
    ingest_langfuse,
    probed_refusal_error,
    recovered_content_count,
    run_ingest,
    source_content_span_count,
    source_signals,
    verify_no_silent_loss,
)
from calibrated_agent_evaluation.ingest.detect import detect_source
from calibrated_agent_evaluation.ingest.errors import IngestAbortedError, IngestError
from calibrated_agent_evaluation.ingest.sample import (
    SampleRecord,
    draw_sample,
    dump_samples,
    load_samples,
)

__all__ = [
    "IngestAbortedError",
    "IngestError",
    "IngestReport",
    "IngestResult",
    "IngestStatus",
    "SampleRecord",
    "capture_and_ingest",
    "contentless_node_count",
    "detect_source",
    "draw_sample",
    "dump_samples",
    "ingest_langfuse",
    "load_samples",
    "probed_refusal_error",
    "recovered_content_count",
    "run_ingest",
    "source_content_span_count",
    "source_signals",
    "verify_no_silent_loss",
]
