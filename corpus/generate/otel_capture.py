# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTel span capture — re-exports from the canonical location in
calibrated_agent_evaluation.execute.strands_capture.

``SpanCapture`` was promoted to ``src/calibrated_agent_evaluation/execute/strands_capture.py`` so a
pip-installed user can run ``StrandsExecutor`` without the ``corpus/`` tree. This module re-exports
the full public surface so existing corpus scripts and reference agents continue to work unchanged.
"""

from calibrated_agent_evaluation.execute.strands_capture import (
    SpanCapture,
    capture_spans,
    spans_to_otlp_json,
    write_otlp_json,
)

__all__ = [
    "SpanCapture",
    "capture_spans",
    "spans_to_otlp_json",
    "write_otlp_json",
]
