# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OTel-ingestion guidance surface.

The layered header resolution's convention layer: an OTLP trace tagged with the
``calibrated_agent_evaluation.*`` convention (via ``guidance.tag_span``) normalizes with the
friendly ``trace_id``/``frame``/ ``taxonomy_cell`` recovered from the wire, rather than the
synthesized OTel-hex fallback — the "teach better instrumentation" path most OTel users take.
"""

from __future__ import annotations

import json
from pathlib import Path

from calibrated_agent_evaluation.normalize import OTelSpans, guidance, normalize
from calibrated_agent_evaluation.schema.validate import validate


def test_tag_span_writes_the_documented_convention_attributes() -> None:
    written: dict[str, object] = {}
    guidance.tag_span(
        lambda k, v: written.__setitem__(k, v),
        trace_id="ORD-9",
        frame="discovery",
        taxonomy_cell={"feature": "refund_handling", "persona": "P1"},
    )
    assert written[guidance.ATTR_TRACE_ID] == "ORD-9"
    assert written[guidance.ATTR_FRAME] == "discovery"
    assert written[f"{guidance.ATTR_TAXONOMY_PREFIX}feature"] == "refund_handling"
    assert written[f"{guidance.ATTR_TAXONOMY_PREFIX}persona"] == "P1"


def test_convention_attributes_on_the_wire_are_recovered_as_headers(
    raw_dir: Path, tmp_path: Path
) -> None:
    # Tag a real capture's root span with the calibrated_agent_evaluation.* convention and confirm
    # the adapter reads the friendly headers from the wire (the first resolution layer), not the
    # OTel-hex fallback, and not the
    # descriptor override (which we deliberately leave unset here).
    otlp = json.loads((raw_dir / "otel" / "ORD-1001.events.otlp.json").read_text(encoding="utf-8"))
    for rs in otlp["resourceSpans"]:
        for ss in rs["scopeSpans"]:
            for span in ss["spans"]:
                if span["name"].startswith("invoke_agent"):
                    span["attributes"].extend(
                        [
                            {"key": guidance.ATTR_TRACE_ID, "value": {"stringValue": "ORD-1001"}},
                            {"key": guidance.ATTR_FRAME, "value": {"stringValue": "discovery"}},
                            {
                                "key": f"{guidance.ATTR_TAXONOMY_PREFIX}feature",
                                "value": {"stringValue": "refund_handling"},
                            },
                        ]
                    )
    tagged = tmp_path / "tagged.otlp.json"
    tagged.write_text(json.dumps(otlp), encoding="utf-8")

    # No descriptor trace_id/frame — the wire convention must supply them.
    trace = normalize(OTelSpans(tagged))
    validate(trace)
    assert trace.trace_id == "ORD-1001"  # friendly id from the wire, not the OTel-hex fallback
    assert trace.frame == "discovery"
    assert trace.taxonomy_cell.get("feature") == "refund_handling"


def test_synthesized_header_hint_names_the_fixing_attribute() -> None:
    hint = guidance.synthesized_header_hint("trace_id", "abc123hex")
    assert "abc123hex" in hint
    assert guidance.ATTR_TRACE_ID in hint  # tells the user exactly which attribute to set
