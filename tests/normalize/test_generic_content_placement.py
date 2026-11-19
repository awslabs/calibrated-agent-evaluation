# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-placement independence for the GENERIC OTel reader.

The Strands adapter learned this lesson first (see ``test_strands_content_placement``); the generic
``gen_ai/latest`` reader never did. It read agent content from ``gen_ai.input``/``output.content``
and tool content from ``gen_ai.tool.input``/``output``, and nowhere else. So a trace from a real
instrumentor using the current convention's ``gen_ai.*.messages`` / ``gen_ai.tool.call.*`` keys — or
Traceloop's own ``traceloop.entity.*`` / ``gen_ai.task.*`` keys, which is what
``opentelemetry-instrumentation-langchain`` emits and therefore what gpt-researcher emits —
normalized to a structurally perfect multi-agent tree with ``{}`` in every content field, resolved
``owned`` and reported ``ingested``. No error anywhere.

Method: take the **real** ``ORD-1001`` attributes capture and *relocate* its content to each key
family, changing nothing but the key. Relocation rather than re-synthesis is deliberate: it isolates
the one variable under test, so an assertion of Trace equality is a claim about placement and about
nothing else. The content stays the genuinely captured content (repo rule: fixtures are real vendor
output, never hand-shaped).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.normalize import NormalizationError, OTelSpans, normalize
from calibrated_agent_evaluation.normalize import content_placements as cp

_RAW = Path(__file__).resolve().parent.parent.parent / "fixtures" / "raw" / "otel"
_SOURCE = _RAW / "ORD-1001.attrs.otlp.json"
_TRACE_ID = "ORD-1001"

# The legacy keys the fixture actually carries, per role.
_BASELINE = {
    "agent_input": "gen_ai.input.content",
    "agent_output": "gen_ai.output.content",
    "tool_args": "gen_ai.tool.input",
    "tool_result": "gen_ai.tool.output",
}

# The current GenAI convention's spelling for the same four roles.
_LATEST = {
    "agent_input": "gen_ai.input.messages",
    "agent_output": "gen_ai.output.messages",
    "tool_args": "gen_ai.tool.call.arguments",
    "tool_result": "gen_ai.tool.call.result",
}

# Traceloop's, registered by the openllmetry convention rather than known to the generic reader.
_TRACELOOP = {
    "agent_input": "traceloop.entity.input",
    "agent_output": "traceloop.entity.output",
    "tool_args": "traceloop.entity.input",
    "tool_result": "traceloop.entity.output",
}

# A family nobody registered. The reader must not find these.
_UNREGISTERED = {
    "agent_input": "acme.agent.prompt",
    "agent_output": "acme.agent.completion",
    "tool_args": "acme.tool.args",
    "tool_result": "acme.tool.result",
}


def _attrs(span: dict[str, Any]) -> dict[str, Any]:
    raw = span.get("attributes")
    if isinstance(raw, dict):
        return dict(raw)
    out: dict[str, Any] = {}
    for kv in raw or []:
        value = kv.get("value", {})
        out[kv["key"]] = next(iter(value.values()), None) if value else None
    return out


def _relocate(otlp: dict[str, Any], target: dict[str, str]) -> dict[str, Any]:
    """Move every content value off its baseline key onto ``target``'s key for the same role.

    The value is copied verbatim and the old key removed, so the reader has exactly one place to
    find
    it. Removing the old key is the part that makes the test capable of failing: leaving both in
    place would let a reader that still knows only the legacy key pass.
    """
    payload = json.loads(json.dumps(otlp))
    for resource_spans in payload.get("resourceSpans", []):
        for scope_spans in resource_spans.get("scopeSpans", []):
            for span in scope_spans.get("spans", []):
                present = _attrs(span)
                kept = [
                    kv
                    for kv in span.get("attributes", [])
                    if kv["key"] not in set(_BASELINE.values())
                ]
                for role, baseline_key in _BASELINE.items():
                    if baseline_key not in present:
                        continue
                    kept.append(
                        {
                            "key": target[role],
                            "value": {"stringValue": str(present[baseline_key])},
                        }
                    )
                span["attributes"] = kept
    return payload


def _normalize_relocated(tmp_path: Path, target: dict[str, str], name: str) -> Any:
    source = json.loads(_SOURCE.read_text(encoding="utf-8"))
    out = tmp_path / f"{name}.otlp.json"
    out.write_text(json.dumps(_relocate(source, target)), encoding="utf-8")
    return normalize(OTelSpans(path=out, semconv="gen_ai/latest", trace_id=_TRACE_ID))


def _content(trace: Any) -> list[tuple[str, Any, Any, list[tuple[str, Any, Any]]]]:
    """Every node's recovered content, as the comparable shape these tests assert on."""
    return [
        (
            span.agent,
            span.input,
            span.output,
            [(tc.name, tc.args, tc.result) for tc in span.tool_calls],
        )
        for span in trace.spans
    ]


def _baseline(tmp_path: Path) -> Any:
    """The fixture normalized through its own (legacy) keys — the content that must be preserved."""
    return _normalize_relocated(tmp_path, _BASELINE, "baseline")


def test_baseline_fixture_has_content_to_lose(tmp_path: Path) -> None:
    """Guard on the guard: the assertions below are worthless if the baseline is already empty."""
    trace = _baseline(tmp_path)
    nodes = _content(trace)
    assert nodes, "fixture produced no nodes"
    assert any(inp or out for _, inp, out, _ in nodes)
    assert any(args or result for _, _, _, tools in nodes for _, args, result in tools)


def test_latest_convention_keys_recover_the_same_content(tmp_path: Path) -> None:
    """Relocating content to `gen_ai.*.messages` / `gen_ai.tool.call.*` loses nothing.

    Before the placement registry this returned a correct tree with `{}` everywhere.
    """
    assert _content(_normalize_relocated(tmp_path, _LATEST, "latest")) == _content(
        _baseline(tmp_path)
    )


def test_vendor_registered_keys_recover_the_same_content(tmp_path: Path) -> None:
    """Traceloop's keys work because the openllmetry convention registered them.

    This is the gpt-researcher shape: a correct `invoke_agent` tree whose content sits only on
    `traceloop.entity.*`. The generic reader recovers it without ever naming Traceloop.
    """
    assert _content(_normalize_relocated(tmp_path, _TRACELOOP, "traceloop")) == _content(
        _baseline(tmp_path)
    )


def test_unregistered_keys_recover_nothing_and_the_trace_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Negative: the REGISTRY recovers content, not a permanently wider key list.

    An unregistered family recovers no content at all, which pins the boundary: the fix is
    extensible by registration, not by the reader having been taught every key in the world.

    Since contentless trees started degrading, that outcome is also *announced* — the trace
    degrades to `probed` with a
    reason instead of presenting an intact-looking tree with `{}` everywhere. This test previously
    asserted the tree survived (its old name said "but keep the tree"), which was the original bug
    being reproduced faithfully and then accepted. Reproducing it is still the point; accepting it
    quietly is what changed.
    """
    with caplog.at_level("WARNING", logger="calibrated_agent_evaluation.normalize"):
        trace = _normalize_relocated(tmp_path, _UNREGISTERED, "unregistered")
    baseline = _baseline(tmp_path)

    assert baseline.ownership == "owned", "the registered form recovers content and stays owned"
    assert trace.ownership == "probed", "the unregistered form recovers none and must say so"
    assert trace.spans == []
    assert "empty content" in caplog.text
    assert "no registered convention reads" in caplog.text


def test_no_vendor_key_appears_in_the_convention_neutral_reader() -> None:
    """Vendor isolation is the reason this is a registry.

    A vendor attribute key hardcoded into `otel_adapter` would pass every content test above and
    still be the coupling the constitution forbids, so it is asserted directly.
    """
    reader = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "calibrated_agent_evaluation"
        / "normalize"
        / "otel_adapter.py"
    ).read_text(encoding="utf-8")
    for vendor_key in ("traceloop.", "gen_ai.task.", "openinference.", "llm.input_messages"):
        assert vendor_key not in reader, f"{vendor_key!r} leaked into the generic reader"


def test_registry_refuses_a_builtin_shadow_and_an_unknown_role() -> None:
    """Negative: a bad registration fails loudly rather than registering dead keys.

    A typo'd role would otherwise register keys no reader consults — silent content loss dressed as
    a successful registration.
    """
    with pytest.raises(NormalizationError, match="built-in family"):
        cp.register_content_placements("gen_ai", {"agent_input": ("x",)})
    with pytest.raises(NormalizationError, match="unknown role"):
        cp.register_content_placements("acme-typo", {"agent_inputs": ("x",)})
    with pytest.raises(NormalizationError, match="unknown content role"):
        cp.placement_keys("agent_inputs")  # type: ignore[arg-type]
