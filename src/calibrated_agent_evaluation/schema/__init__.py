# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized-schema contract for the library's data plane.

The vendor-neutral ``Trace`` / ``Span`` / ``Label`` types every plane serializes to,
the ``validate`` gate that enforces the schema invariants, the typed
error hierarchy, and the ``SCHEMA_VERSION`` constant. Nothing here imports a vendor SDK
type; this package is the firewall the methodology plane reads through.
"""

from __future__ import annotations

from calibrated_agent_evaluation.schema.errors import (
    BackendError,
    CalibratedAgentEvaluationDataError,
    SchemaError,
    SchemaVersionError,
    TraceNotFoundError,
    UnregisteredArtifactError,
)
from calibrated_agent_evaluation.schema.serde import (
    label_from_dict,
    latest_verdicts,
    span_from_dict,
    to_dict,
    trace_from_dict,
)
from calibrated_agent_evaluation.schema.types import (
    SCHEMA_VERSION,
    Frame,
    Label,
    LabelKind,
    Outcome,
    Ownership,
    Role,
    Span,
    SpanStatus,
    TokenUsage,
    ToolCall,
    Trace,
)
from calibrated_agent_evaluation.schema.validate import validate

__all__ = [
    "SCHEMA_VERSION",
    "BackendError",
    "CalibratedAgentEvaluationDataError",
    "Frame",
    "Label",
    "LabelKind",
    "Outcome",
    "Ownership",
    "Role",
    "SchemaError",
    "SchemaVersionError",
    "Span",
    "SpanStatus",
    "TokenUsage",
    "ToolCall",
    "Trace",
    "TraceNotFoundError",
    "UnregisteredArtifactError",
    "label_from_dict",
    "latest_verdicts",
    "span_from_dict",
    "to_dict",
    "trace_from_dict",
    "validate",
]
