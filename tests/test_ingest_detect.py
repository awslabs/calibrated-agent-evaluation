# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detection tests: routing strands vs generic-OTel vs inspect (unknown → error); routing a
Strands trace to its adapter recovers tools the generic adapter drops; a legacy-semconv Strands
trace recovers the oracle's tool calls through detection; and boundary ops take precedence over
an attribute-only grammar instead of being ambiguous."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.ingest import IngestError, detect_source
from calibrated_agent_evaluation.normalize import (
    InspectEvalLog,
    NormalizationError,
    OTelSpans,
    StrandsSession,
    normalize,
    otlp_json,
)
from calibrated_agent_evaluation.normalize.conventions.detect import detect
from calibrated_agent_evaluation.schema.serde import to_dict
from calibrated_agent_evaluation.schema.types import Trace

_RAW = Path(__file__).resolve().parent.parent / "fixtures" / "raw"


def _tool_names(trace: object) -> list[str]:
    return sorted(tc.name for s in trace.spans for tc in s.tool_calls)  # type: ignore[attr-defined]


def test_detects_strands_generic_otel_and_inspect() -> None:
    """Auto-detection picks the adapter from the trace shape and derives the trace id."""
    strands = detect_source(_RAW / "strands" / "CLM-4471.otlp.json")
    assert isinstance(strands, StrandsSession)
    assert strands.trace_id == "CLM-4471"

    generic = detect_source(_RAW / "otel" / "ORD-1002.events.otlp.json")
    assert isinstance(generic, OTelSpans)
    assert generic.trace_id == "ORD-1002"

    inspect = detect_source(_RAW / "inspect" / "ORD-1001.eval")
    assert isinstance(inspect, InspectEvalLog)


def test_unrecognized_shape_raises_ingest_error(tmp_path: Path) -> None:
    """Negative: a file matching no adapter is an IngestError (→ skipped in a run)."""
    junk = tmp_path / "notes.txt"
    junk.write_text("not a trace", encoding="utf-8")
    with pytest.raises(IngestError, match="unrecognized"):
        detect_source(junk)


def test_strands_routing_recovers_tools_the_generic_adapter_drops() -> None:
    """The same Strands file recovers tools via the Strands adapter but drops them
    via the generic OTel adapter (the BUG-1 nesting) — so detection routing is load-bearing."""
    src = _RAW / "strands" / "CLM-4471.otlp.json"
    strands = normalize(StrandsSession(path=src, trace_id="CLM-4471"))
    n_strands = len(_tool_names(strands))
    try:
        generic = normalize(OTelSpans(path=src, trace_id="CLM-4471"))
        n_generic = len(_tool_names(generic))
    except NormalizationError:
        n_generic = 0  # the generic adapter may degrade/refuse the nested shape outright
    assert n_strands > 0
    assert n_generic < n_strands
    assert isinstance(detect_source(src), StrandsSession)


def _tools_by_agent(spans: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {s["agent"]: [tc["name"] for tc in s["tool_calls"]] for s in spans}


def test_legacy_strands_trace_recovers_the_oracle_tool_calls() -> None:
    """Continuity: a legacy-semconv Strands trace ingested through detection
    recovers every agent's tool calls, by count, name and order, as the hand-authored oracle
    records them, and as ``normalize(StrandsSession(mode="legacy"))`` does. The oracle is
    independent of the code, so a dropped tail or a lost agent fails here."""
    src = _RAW / "strands" / "CLM-4471.otlp.json"
    oracle = json.loads((_RAW.parent / "normalized" / "CLM-4471.json").read_text(encoding="utf-8"))
    expected = _tools_by_agent(oracle["spans"])

    via_detect = to_dict(normalize(detect_source(src)))
    legacy = to_dict(normalize(StrandsSession(path=src, trace_id="CLM-4471", mode="legacy")))

    assert sum(len(names) for names in expected.values()) == 4
    assert _tools_by_agent(via_detect["spans"]) == expected
    assert _tools_by_agent(legacy["spans"]) == expected


def test_attribute_located_tool_io_is_recovered_whatever_location_is_declared() -> None:
    """A trace whose tool I/O sits in span attributes recovers every tool
    call's arguments and result through detection, equal to the oracle. Declaring the other
    location is a hint that orders the lookup and never gates it, so it recovers the same."""
    src = _RAW / "otel" / "ORD-1001.attrs.otlp.json"
    oracle = json.loads((_RAW.parent / "normalized" / "ORD-1001.json").read_text(encoding="utf-8"))
    expected = [(tc["name"], tc["args"], tc["result"]) for tc in oracle["spans"][0]["tool_calls"]]

    def tool_io(trace: Trace) -> list[tuple[str, Any, Any]]:
        return [(tc.name, tc.args, tc.result) for s in trace.spans for tc in s.tool_calls]

    detected = detect_source(src)
    assert isinstance(detected, OTelSpans)
    assert tool_io(normalize(detected)) == expected
    declared_events = OTelSpans(path=src, trace_id="ORD-1001", content_location="events")
    assert tool_io(normalize(declared_events)) == expected


def _spans(path: Path) -> list[dict[str, Any]]:
    return otlp_json.iter_spans(json.loads(path.read_text(encoding="utf-8")))


def _with_messages_attrs(path: Path, out: Path) -> Path:
    """Add the latest convention's message keys to a real boundary-op trace, keeping the spans.

    This is the commonest real shape rather than a contrivance: a Traceloop-instrumented crew emits
    ``invoke_agent`` boundary spans AND ``gen_ai.*.messages``, which is precisely the pair that made
    ``detect`` report the source ambiguous and made ``calibrated-agent-evaluation ingest`` skip the
    file.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    for resource_spans in payload["resourceSpans"]:
        for scope_spans in resource_spans["scopeSpans"]:
            for span in scope_spans["spans"]:
                attrs = otlp_json.attributes(span)
                if attrs.get("gen_ai.operation.name") != "invoke_agent":
                    continue
                for src_key, dst_key in (
                    ("gen_ai.input.content", "gen_ai.input.messages"),
                    ("gen_ai.output.content", "gen_ai.output.messages"),
                ):
                    if src_key in attrs:
                        span["attributes"].append(
                            {"key": dst_key, "value": {"stringValue": str(attrs[src_key])}}
                        )
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def test_boundary_ops_win_over_an_attribute_only_grammar(tmp_path: Path) -> None:
    """A trace carrying both `invoke_agent` spans and `gen_ai.*.messages` resolves to the
    grammar with the tree, and ingests — where it previously raised "ambiguous" and was skipped.

    Asserted at the boundary: not just detect's return value, but that `semconv="auto"` now produces
    the same multi-node tree the declared grammar does.
    """
    mixed = _with_messages_attrs(
        _RAW / "otel" / "ORD-1001.attrs.otlp.json", tmp_path / "mixed.otlp.json"
    )
    assert detect(_spans(mixed)) == "gen_ai/latest"

    auto = normalize(OTelSpans(path=mixed, semconv="auto", trace_id="ORD-1001"))
    declared = normalize(OTelSpans(path=mixed, semconv="gen_ai/latest", trace_id="ORD-1001"))
    assert [s.agent for s in auto.spans] == [s.agent for s in declared.spans]
    assert _tool_names(auto) == _tool_names(declared) != []


def test_attribute_only_traces_keep_their_own_grammar() -> None:
    """Negative: the precedence rule must not swallow traces that have no boundary ops.

    Without this, "prefer gen_ai/latest" would degenerate into "always gen_ai/latest", and every
    attribute-only convention would collapse to an empty tree. Each of the three real convention
    fixtures still detects as itself.
    """
    for relative, expected in (
        ("messages/swebench_tool_call.otlp.json", "gen_ai/messages"),
        ("openinference/weather_tool_call.otlp.json", "openinference"),
        ("openllmetry/weather_tool_call.otlp.json", "openllmetry"),
    ):
        assert detect(_spans(_RAW / "otel" / relative)) == expected


def test_ambiguity_among_attribute_only_grammars_still_refuses() -> None:
    """Negative: the refusal is narrowed, not removed.

    Two attribute-only grammars with no boundary ops to break the tie is still a case where guessing
    would silently mis-read the trace, so it must keep raising.
    """
    spans = _spans(_RAW / "otel" / "messages" / "swebench_tool_call.otlp.json")
    spans[0].setdefault("attributes", []).append(
        {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}}
    )
    with pytest.raises(NormalizationError, match="ambiguous"):
        detect(spans)
