# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""User guidance for making OTel traces cleaner to ingest.

Most sources are generic OpenTelemetry, and the normalized ``Trace`` needs three trace-level
"header" fields the raw wire does not uniformly carry: ``trace_id``, ``frame`` (an
analyst-assigned sampling label; *why* the trace was captured, not a runtime property), and
``taxonomy_cell``. The normalizer resolves them in a layered order (convention attribute on the
wire → descriptor override → wire/default fallback), so a plain OTel trace still normalizes;
but a trace tagged with the Calibrated Agent Evaluation convention normalizes with friendly ids for
free.

This module is the single source of that convention: the attribute keys, a helper to stamp them
onto a span at instrumentation time, and a message builder the degradation/fallback path uses to
tell the user *which* header it had to synthesize and *how* to set it; so the guidance is
actionable, not just "missing". It imports no vendor SDK (it takes a ``set_attribute`` callable),
so it stays inside the methodology-plane boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# The Calibrated Agent Evaluation OTel convention: set these on the ROOT span (or the resource) so
# the normalizer recovers friendly headers instead of synthesizing them from the opaque OTel
# traceId.
ATTR_TRACE_ID = "calibrated_agent_evaluation.trace_id"
ATTR_FRAME = "calibrated_agent_evaluation.frame"
# e.g. calibrated_agent_evaluation.taxonomy.feature = "refund_handling"
ATTR_TAXONOMY_PREFIX = "calibrated_agent_evaluation.taxonomy."


def tag_span(
    set_attribute: Callable[[str, Any], object],
    *,
    trace_id: str,
    frame: str,
    taxonomy_cell: dict[str, str],
) -> None:
    """Stamp the library's header convention onto an OTel span at instrumentation time.

    ``set_attribute`` is the span's ``set_attribute(key, value)`` method (passed in so this
    helper carries no ``opentelemetry`` import; the boundary stays clean). Call it on the root
    ``invoke_agent`` span so the normalizer reads ``trace_id``/``frame``/``taxonomy_cell`` from
    the wire (the first layer of header resolution) rather than falling back to synthesized values.
    """
    set_attribute(ATTR_TRACE_ID, trace_id)
    set_attribute(ATTR_FRAME, frame)
    for key, value in taxonomy_cell.items():
        set_attribute(f"{ATTR_TAXONOMY_PREFIX}{key}", value)


def synthesized_header_hint(field: str, synthesized_value: str) -> str:
    """Return an actionable hint naming a header the normalizer had to synthesize.

    Used in the fallback/degradation path so the user learns which header was missing and the
    exact convention attribute that would fix it (to teach better instrumentation).
    """
    attr = {
        "trace_id": ATTR_TRACE_ID,
        "frame": ATTR_FRAME,
        "taxonomy_cell": f"{ATTR_TAXONOMY_PREFIX}<key>",
    }.get(field, f"calibrated_agent_evaluation.{field}")
    return (
        f"{field} was not on the trace; synthesized {synthesized_value!r}. "
        f"Set the {attr!r} span attribute at instrumentation time "
        f"(see calibrated_agent_evaluation.normalize.guidance.tag_span) for a stable value."
    )
