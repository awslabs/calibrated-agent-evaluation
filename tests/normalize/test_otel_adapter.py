# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic OTel adapter tests.

Against the real ``fixtures/raw/otel/*.otlp.json`` corpus: content recovered from span **events**
, the same content in **attributes** also recovered (proving both locations handled), the
event-located-content-not-dropped guard (the silent-drop anti-pattern), and the
multi-container walk (spans split across two ``ResourceSpans`` are all read — the silent-truncation
guard).
"""

from __future__ import annotations

import json
from pathlib import Path

from calibrated_agent_evaluation.normalize import OTelSpans, normalize
from calibrated_agent_evaluation.schema.validate import validate


def _otel(raw_dir: Path, name: str, **over: object) -> OTelSpans:
    return OTelSpans(raw_dir / "otel" / name, trace_id="ORD-1001", **over)  # type: ignore[arg-type]


def test_content_in_events_is_recovered_non_empty(raw_dir: Path) -> None:
    trace = normalize(_otel(raw_dir, "ORD-1001.events.otlp.json", content_location="events"))
    validate(trace)
    assert trace.ownership == "owned"
    root = trace.spans[0]
    # Per-step input/output recovered from gen_ai.*.message events (non-empty).
    assert root.input
    assert root.output
    # tool_calls recovered with args + result from gen_ai.tool.message events.
    assert len(root.tool_calls) == 3
    for tc in root.tool_calls:
        assert tc.args, f"{tc.name} args empty"
        assert tc.result, f"{tc.name} result empty"


def test_same_content_in_attributes_also_recovered_non_empty(raw_dir: Path) -> None:
    # The SAME logical run with content flattened onto span ATTRIBUTES normalizes non-empty too —
    # proving both content-locations are handled, not just one.
    trace = normalize(_otel(raw_dir, "ORD-1001.attrs.otlp.json", content_location="attributes"))
    validate(trace)
    root = trace.spans[0]
    assert len(root.tool_calls) == 3
    crp = next(tc for tc in root.tool_calls if tc.name == "check_refund_policy")
    assert crp.result["within_window"] is True
    assert crp.result["days_since_delivery"] == 51


def test_event_located_content_is_not_dropped(raw_dir: Path) -> None:
    # Negative (the silent-drop anti-pattern): an attributes-only reader would drop the
    # events-variant content. Assert the adapter recovers the event content — the tool args/results
    # are present precisely because events are read, not only attributes.
    events_file = json.loads(
        (raw_dir / "otel" / "ORD-1001.events.otlp.json").read_text(encoding="utf-8")
    )
    # Confirm the fixture genuinely has EMPTY span attributes for content (content is in events),
    # so recovering it proves events were read.
    spans = [
        s for rs in events_file["resourceSpans"] for ss in rs["scopeSpans"] for s in ss["spans"]
    ]
    tool_span = next(s for s in spans if s["name"] == "execute_tool check_refund_policy")
    attr_keys = {a["key"] for a in tool_span.get("attributes", [])}
    assert "gen_ai.tool.output" not in attr_keys  # content is NOT on attributes here
    assert tool_span.get("events")  # it is in events

    trace = normalize(_otel(raw_dir, "ORD-1001.events.otlp.json", content_location="events"))
    crp = next(tc for tc in trace.spans[0].tool_calls if tc.name == "check_refund_policy")
    assert crp.result, "event-located tool result was dropped (silent-drop anti-pattern)"


def test_content_location_falls_back_so_content_is_never_dropped(raw_dir: Path) -> None:
    # content_location is a best-effort PREFERENCE, not a hard gate: reading the events-variant with
    # the "attributes"-preference must still recover the content (fall back to events), never drop
    # it. This is what makes the field honest — it orders the lookup, it doesn't silently gate it.
    trace = normalize(_otel(raw_dir, "ORD-1001.events.otlp.json", content_location="attributes"))
    validate(trace)
    root = trace.spans[0]
    assert len(root.tool_calls) == 3
    # content lived in events; the attributes-preference read still recovered it (fallback).
    for tc in root.tool_calls:
        assert tc.args
        assert tc.result


def test_spans_split_across_two_resource_spans_are_all_read(raw_dir: Path, tmp_path: Path) -> None:
    # Negative/edge (derived by mutating a real capture): split the spans across TWO ResourceSpans
    # (and two ScopeSpans). The adapter must read spans from EVERY container — reading only the
    # first would silently truncate the trace (the silent-truncation failure mode).
    original = json.loads(
        (raw_dir / "otel" / "ORD-1001.events.otlp.json").read_text(encoding="utf-8")
    )
    all_spans = [
        s for rs in original["resourceSpans"] for ss in rs["scopeSpans"] for s in ss["spans"]
    ]
    assert len(all_spans) == 4
    resource = original["resourceSpans"][0]["resource"]
    # Put the agent span in the FIRST container and its 3 tool spans in the SECOND.
    agent = [s for s in all_spans if s["name"].startswith("invoke_agent")]
    tools = [s for s in all_spans if s["name"].startswith("execute_tool")]
    split = {
        "resourceSpans": [
            {"resource": resource, "scopeSpans": [{"scope": {"name": "a"}, "spans": agent}]},
            {"resource": resource, "scopeSpans": [{"scope": {"name": "b"}, "spans": tools}]},
        ]
    }
    split_file = tmp_path / "split.otlp.json"
    split_file.write_text(json.dumps(split), encoding="utf-8")

    trace = normalize(OTelSpans(split_file, trace_id="ORD-1001"))
    validate(trace)
    # All 3 tool calls (from the SECOND container) are attached to the agent (from the FIRST).
    assert len(trace.spans) == 1
    assert len(trace.spans[0].tool_calls) == 3
