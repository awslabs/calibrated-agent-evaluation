# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-placement independence for the Strands adapter.

Strands emits the same run's content in one of two placements depending on
``OTEL_SEMCONV_STABILITY_OPT_IN``: legacy span *events*, or the latest-experimental
``gen_ai.*.messages`` / ``gen_ai.tool.call.*`` payloads on span *attributes*. The adapter must
recover the
attribute placement be read rather than dropped, and the adapter only read events — so a pipeline
that keeps attributes but discards span events (AWS CloudWatch Transaction Search does exactly
that) yielded a correctly-shaped tree with every step's content empty, and no error.

These tests take the **real** ``CLM-4471`` capture and re-emit it under the latest convention with
its events removed, so the assertion is placement-independence over genuinely captured content
rather than over a hand-shaped stub: the same run must normalize to the same ``Trace``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.normalize import NormalizationError, StrandsSession, normalize
from calibrated_agent_evaluation.schema.validate import validate

_TEXT_PART = '[{{"role": "{role}", "parts": [{{"type": "text", "content": {content}}}]}}]'


def _attrs(carrier: dict[str, Any]) -> dict[str, Any]:
    raw = carrier.get("attributes")
    if isinstance(raw, dict):
        return dict(raw)
    out: dict[str, Any] = {}
    for kv in raw or []:
        value = kv.get("value", {})
        out[kv["key"]] = next(iter(value.values()), None) if value else None
    return out


def _set_attr(span: dict[str, Any], key: str, value: str) -> None:
    raw = span.get("attributes")
    if isinstance(raw, dict):
        raw[key] = value
        return
    span["attributes"] = [*(raw or []), {"key": key, "value": {"stringValue": value}}]


def _messages(role: str, content: str) -> str:
    """Wrap one payload as a latest-convention single-text-part message list."""
    return _TEXT_PART.format(role=role, content=json.dumps(content))


def _first_event(span: dict[str, Any], name: str, field: str) -> str | None:
    for event in span.get("events") or []:
        if event.get("name") == name:
            value = _attrs(event).get(field)
            if isinstance(value, str):
                return value
    return None


def _last_event(span: dict[str, Any], name: str, field: str) -> str | None:
    found = None
    for event in span.get("events") or []:
        if event.get("name") == name:
            value = _attrs(event).get(field)
            if isinstance(value, str):
                found = value
    return found


def to_latest_convention(otlp: dict[str, Any]) -> dict[str, Any]:
    """Re-emit a legacy Strands OTLP payload in the latest convention, span-attributes-only.

    Mirrors what the SDK does under ``gen_ai_latest_experimental,gen_ai_span_attributes_only``, and
    mirrors the legacy reader's own selection rules (first ``user.message``, last ``choice``) so the
    two placements carry byte-identical payloads. Events are then deleted, which is the part that
    reproduces a pipeline that keeps only attributes.
    """
    payload = json.loads(json.dumps(otlp))
    for resource_spans in payload.get("resourceSpans", []):
        for scope_spans in resource_spans.get("scopeSpans", []):
            for span in scope_spans.get("spans", []):
                op = _attrs(span).get("gen_ai.operation.name")
                user = _first_event(span, "gen_ai.user.message", "content")
                choice = _last_event(span, "gen_ai.choice", "message")
                tool_in = _first_event(span, "gen_ai.tool.message", "content")
                if op == "invoke_agent":
                    if user is not None:
                        _set_attr(span, "gen_ai.input.messages", _messages("user", user))
                    if choice is not None:
                        _set_attr(span, "gen_ai.output.messages", _messages("assistant", choice))
                elif op == "execute_tool":
                    if tool_in is not None:
                        _set_attr(span, "gen_ai.tool.call.arguments", tool_in)
                    if choice is not None:
                        _set_attr(span, "gen_ai.tool.call.result", choice)
                span.pop("events", None)
    return payload


@pytest.fixture
def latest_capture(raw_dir: Path, tmp_path: Path) -> Path:
    """The real CLM-4471 capture, re-emitted latest-convention with every span event removed."""
    source = json.loads((raw_dir / "strands" / "CLM-4471.otlp.json").read_text(encoding="utf-8"))
    out = tmp_path / "CLM-4471.otlp.json"
    out.write_text(json.dumps(to_latest_convention(source)), encoding="utf-8")
    return out


def test_attribute_located_content_normalizes_identically_to_events(
    raw_dir: Path, latest_capture: Path
) -> None:
    """The same run in either placement yields the same Trace — content, tools and tree."""
    legacy = normalize(
        StrandsSession(raw_dir / "strands" / "CLM-4471.otlp.json", trace_id="CLM-4471")
    )
    latest = normalize(StrandsSession(latest_capture, mode="latest", trace_id="CLM-4471"))
    validate(latest)

    assert latest == legacy


def test_attribute_located_content_is_recovered_not_merely_shaped(
    latest_capture: Path,
) -> None:
    """Negative: the failure being fixed was a right-shaped tree with every step empty.

    Asserted independently of the equality test above, because equality would also hold if BOTH
    sides came back empty — the shape of a check that passes when nothing was recovered.
    """
    trace = normalize(StrandsSession(latest_capture, mode="latest", trace_id="CLM-4471"))

    assert len(trace.spans) == 5
    assert all(span.input for span in trace.spans)
    assert trace.input
    assert trace.output
    extract = next(
        call
        for span in trace.spans
        for call in span.tool_calls
        if call.name == "extract_claim_fields"
    )
    assert extract.args, "tool arguments were dropped"
    assert "occupancy_status" in extract.result


def test_declared_mode_orders_placements_and_never_gates_them(
    raw_dir: Path, latest_capture: Path
) -> None:
    """``mode`` is a hint. Either declared mode recovers either placement.

    A mode that *gated* would be worse than no mode at all: a caller who cannot know which
    placement their exporter preserved would get silence instead of content.
    """
    latest_read_as_legacy = normalize(StrandsSession(latest_capture, mode="legacy"))
    legacy_read_as_latest = normalize(
        StrandsSession(raw_dir / "strands" / "CLM-4471.otlp.json", mode="latest")
    )

    assert all(span.input for span in latest_read_as_legacy.spans)
    assert all(span.input for span in legacy_read_as_latest.spans)


def test_malformed_messages_payload_is_a_typed_error_not_silence(
    latest_capture: Path, tmp_path: Path
) -> None:
    """Unreadable content that is *present* must never normalize as "no content"."""
    payload = json.loads(latest_capture.read_text(encoding="utf-8"))
    for resource_spans in payload["resourceSpans"]:
        for scope_spans in resource_spans["scopeSpans"]:
            for span in scope_spans["spans"]:
                if _attrs(span).get("gen_ai.input.messages"):
                    _set_attr(span, "gen_ai.input.messages", "{not json")
    broken = tmp_path / "broken.otlp.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(NormalizationError, match="not valid JSON"):
        normalize(StrandsSession(broken, mode="latest"))


def test_span_status_and_timings_survive_normalization(raw_dir: Path, tmp_path: Path) -> None:
    """An ERROR source span becomes ``status="errored"``; wire timings reach ``started_at``.

    Before this, ``tree.build_spans`` hard-coded ``"completed"``, so ``SpanStatus="errored"`` was
    unreachable from every adapter — the standard OTel failure signal was silently discarded and
    a failed step was indistinguishable from a clean one.
    """
    payload = json.loads((raw_dir / "strands" / "CLM-4471.otlp.json").read_text(encoding="utf-8"))
    marked = None
    for resource_spans in payload["resourceSpans"]:
        for scope_spans in resource_spans["scopeSpans"]:
            for span in scope_spans["spans"]:
                if _attrs(span).get("gen_ai.operation.name") == "invoke_agent" and marked is None:
                    marked = _attrs(span).get("gen_ai.agent.name")
                    span["status"] = {"code": "STATUS_CODE_ERROR", "message": "boom"}
    path = tmp_path / "errored.otlp.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    trace = normalize(StrandsSession(path, trace_id="CLM-4471"))
    validate(trace)

    by_agent = {span.agent: span for span in trace.spans}
    assert by_agent[str(marked)].status == "errored"
    assert all(span.status == "completed" for name, span in by_agent.items() if name != marked)
    assert all(span.started_at and span.ended_at for span in trace.spans)


def test_a_clean_capture_reports_no_errored_span(raw_dir: Path) -> None:
    """The status contract's negative half: nothing is marked errored without a source signal."""
    trace = normalize(
        StrandsSession(raw_dir / "strands" / "CLM-4471.otlp.json", trace_id="CLM-4471")
    )

    assert [span.status for span in trace.spans] == ["completed"] * 5
