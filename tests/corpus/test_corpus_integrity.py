# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the frozen ``fixtures/raw/`` corpus and the capture harness against silent rot.

These tests assert the captured corpus is well-formed in the wire shapes the normalizer is
coded against (canonical OTLP-JSON, parseable ``.eval``), and that the committed cassettes let the
agents **replay offline** (the determinism guarantee the executors reuse). They read only committed
files
and replay recorded model calls — no network — but they do import the corpus-generation packages, so
they skip cleanly when the ``corpus`` dependency group is not installed (a contributor running just
the default suite).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RAW = _REPO_ROOT / "fixtures" / "raw"

# Skip the whole module unless the corpus deps (strands/anthropic/mcp/inspect/otel-sdk) are present.
_CORPUS_DEPS = ("strands", "anthropic", "mcp", "inspect_ai", "opentelemetry.sdk")
_have_corpus = all(importlib.util.find_spec(m) is not None for m in _CORPUS_DEPS)
pytestmark = pytest.mark.skipif(
    not _have_corpus, reason="corpus dependency group not installed (uv sync --group corpus)"
)


def _spans(otlp: dict) -> list[dict]:
    return [
        span
        for resource in otlp.get("resourceSpans", [])
        for scope in resource.get("scopeSpans", [])
        for span in scope.get("spans", [])
    ]


_OTEL_FILES = {
    "ORD-1001.events.otlp.json": 4,
    "ORD-1001.attrs.otlp.json": 4,
    "ORD-1001.build2.otlp.json": 4,
    "ORD-1002.events.otlp.json": 4,
    "ORD-DEGRADED.otlp.json": 1,
}


@pytest.mark.parametrize("name", sorted(_OTEL_FILES))
def test_otel_files_are_canonical_otlp_json(name: str) -> None:
    """Each OTel capture is canonical OTLP-JSON with the expected span count (stdlib-readable)."""
    otlp = json.loads((_RAW / "otel" / name).read_text(encoding="utf-8"))
    assert "resourceSpans" in otlp, f"{name}: not canonical OTLP-JSON (missing resourceSpans)"
    spans = _spans(otlp)
    assert len(spans) == _OTEL_FILES[name], f"{name}: expected {_OTEL_FILES[name]} spans"


def test_degraded_trace_has_only_a_single_outer_span() -> None:
    """ORD-DEGRADED is the under-instrumented case: one span, no per-step tool children."""
    otlp = json.loads((_RAW / "otel" / "ORD-DEGRADED.otlp.json").read_text(encoding="utf-8"))
    spans = _spans(otlp)
    assert len(spans) == 1
    assert not spans[0].get("events"), "degraded trace must carry no per-step content events"


def test_orderbot_over_refund_present_in_events_capture() -> None:
    """The headline ORD-1001 failure is real in the capture: the buggy tool said within_window=true
    and the agent issued the refund (an `execute_tool issue_refund` span exists)."""
    otlp = json.loads((_RAW / "otel" / "ORD-1001.events.otlp.json").read_text(encoding="utf-8"))
    names = {s["name"] for s in _spans(otlp)}
    assert "execute_tool issue_refund" in names, "over-refund not captured (no issue_refund span)"
    assert "execute_tool check_refund_policy" in names


def test_strands_capture_has_verified_span_names() -> None:
    """The frozen Strands export carries the real strands-native span names ."""
    otlp = json.loads((_RAW / "strands" / "CLM-4471.otlp.json").read_text(encoding="utf-8"))
    spans = _spans(otlp)
    names = {s["name"] for s in spans}
    assert "invoke_graph" in names
    assert any(n.startswith("invoke_agent ") for n in names)
    assert any(n.startswith("execute_tool ") for n in names)
    systems = {
        a["value"].get("stringValue")
        for s in spans
        for a in s.get("attributes", [])
        if a["key"] == "gen_ai.system"
    }
    assert systems == {"strands-agents"}, f"unexpected gen_ai.system values: {systems}"


@pytest.mark.parametrize(
    ("trace_id", "signal"),
    [
        # attribution contrast/trap crew traces — each a real Strands Graph whose gold-bearing
        # defect is injected via its oracle tool results (fixtures/README.md).
        ("CLM-4473", "vacant_35_days"),  # policy-checker's lookup returns a broken COVERED verdict
        ("CLM-4475", "12000"),  # intake misread $1,200 -> 12000 (outcome-pass / span-fail)
        ("CLM-4477", "vacant_30_days"),  # occupancy at the exact W-7 >30 boundary (disputed)
    ],
)
def test_attribution_captures_carry_their_injected_signal(trace_id: str, signal: str) -> None:
    """Each crew capture is a real strands-native Graph carrying its injected gold signal.

    Proves the contrast/trap traces are genuinely emitted (not hand-shaped): a five-agent Graph with
    the strands span-name grammar, and the defect the attribution gold hangs on present in the wire.
    """
    otlp = json.loads((_RAW / "strands" / f"{trace_id}.otlp.json").read_text(encoding="utf-8"))
    spans = _spans(otlp)
    names = {s["name"] for s in spans}
    assert "invoke_graph" in names, f"{trace_id}: not a Strands Graph capture"
    agents = {n.removeprefix("invoke_agent ") for n in names if n.startswith("invoke_agent ")}
    assert {"supervisor", "intake", "policy-checker", "fraud-scorer", "payout"} <= agents
    assert signal in json.dumps(otlp), f"{trace_id}: injected signal {signal!r} absent from capture"


def test_intake_dropped_vacancy_in_strands_capture() -> None:
    """CLM-4471's worked example: intake's extractor genuinely emitted occupancy_status=null.

    The structured tool result rides as a JSON string inside the intake ``execute_tool`` span event
    (``gen_ai.choice``), so we parse that event's content rather than search the (escaped) blob.
    """
    otlp = json.loads((_RAW / "strands" / "CLM-4471.otlp.json").read_text(encoding="utf-8"))
    intake = next(s for s in _spans(otlp) if s["name"] == "execute_tool extract_claim_fields")
    payloads: list[str] = [
        attr["value"].get("stringValue", "")
        for event in intake.get("events", [])
        for attr in event.get("attributes", [])
    ]
    # The tool result is JSON nested (and escaped) inside the event payload; find the innermost
    # JSON object and assert the field is genuinely null — the naive-extractor bug.
    joined = " ".join(payloads)
    assert "occupancy_status" in joined, "intake span carries no occupancy_status field"
    normalized = joined.replace("\\", "").replace(" ", "")
    assert '"occupancy_status":null' in normalized, "intake did not emit occupancy_status=null"


def test_swarm_capture_has_handoff_tool_span() -> None:
    """The Strands Swarm export proves the strands_adapter Swarm path against real output.

    A Swarm coordinates via the framework's registered ``handoff_to_agent`` *tool*, so the handoff
    surfaces as an ``execute_tool handoff_to_agent`` span under an ``invoke_swarm`` root — never a
    top-level ``handoff_to_agent`` span primitive. This is the shape the normalizer must land as a
    ``ToolCall``, and the empirical refutation of the old `handoff_to_agent`-as-span spec.
    """
    otlp = json.loads((_RAW / "strands" / "CLM-7720-swarm.otlp.json").read_text(encoding="utf-8"))
    spans = _spans(otlp)
    names = {s["name"] for s in spans}
    assert "invoke_swarm" in names, "swarm capture missing the invoke_swarm root"
    assert "execute_tool handoff_to_agent" in names, "handoff is not an execute_tool span"
    # The handoff is a tool span, NOT a top-level span named bare 'handoff_to_agent'.
    assert "handoff_to_agent" not in names, "handoff_to_agent must be a tool, not a span primitive"
    assert {"invoke_agent triage", "invoke_agent adjuster"} <= names


def test_agent_as_tool_capture_nests_subagent_under_tool_span() -> None:
    """The Agent-as-Tools export proves the discriminator against real output.

    Calling a sub-agent exposed via ``Agent.as_tool()`` nests the sub-agent's ``invoke_agent`` span
    UNDER the parent's ``execute_tool {child}`` span — so the normalizer must promote it to a child
    ``Span`` (agent boundary), while a plain leaf tool (``execute_tool issue_refund``) stays a
    ``ToolCall``. The fixture carries both so the discriminator has a real example.
    """
    otlp = json.loads((_RAW / "strands" / "ORD-3310-as-tool.otlp.json").read_text(encoding="utf-8"))
    spans = _spans(otlp)
    by_id = {s["spanId"]: s for s in spans}
    tool_pa = next((s for s in spans if s["name"] == "execute_tool policy_advisor"), None)
    agent_pa = next((s for s in spans if s["name"] == "invoke_agent policy_advisor"), None)
    assert tool_pa is not None, "missing execute_tool policy_advisor span"
    assert agent_pa is not None, "missing nested invoke_agent policy_advisor span"
    # The agent-as-tool sub-agent is nested directly under the parent's execute_tool span.
    assert agent_pa.get("parentSpanId") == tool_pa["spanId"], "sub-agent not nested under its tool"
    # The parent's leaf tool is a plain execute_tool with no invoke_agent child (-> a ToolCall).
    leaf = next((s for s in spans if s["name"] == "execute_tool issue_refund"), None)
    assert leaf is not None, "missing leaf execute_tool issue_refund span"
    leaf_children = [s for s in spans if s.get("parentSpanId") == leaf["spanId"]]
    assert not any(c["name"].startswith("invoke_agent") for c in leaf_children), (
        "leaf tool must not wrap an agent"
    )
    assert by_id  # tree is self-consistent (ids resolvable)


@pytest.mark.parametrize("name", ["ORD-1001.eval", "ORD-1002.eval", "CLM-4471.eval"])
def test_inspect_eval_logs_parse_via_event_tree(name: str) -> None:
    """Each .eval parses through Inspect's public reader + event_tree (the inspect_adapter path)."""
    from inspect_ai.event import event_tree
    from inspect_ai.log import read_eval_log_samples

    samples = list(read_eval_log_samples(str(_RAW / "inspect" / name), resolve_attachments="full"))
    assert len(samples) == 1
    tree = event_tree(samples[0].events)
    assert tree, f"{name}: event_tree produced no nodes"


def test_multiple_strands_targets_capture_in_one_process() -> None:
    """Two Strands captures in ONE process must each land their spans (singleton-reset regression).

    Strands memoizes a process-global tracer bound to the provider that was global at first use, so
    without ``SpanCapture.register_global`` resetting that singleton, the second target would emit
    zero spans (the bug that froze a 3-byte ``{}`` export). Replays offline — no Bedrock.
    """
    from corpus.agents import claimscrew_swarm, orderbot_agent_as_tool
    from corpus.generate.cassette import Cassette, Mode
    from corpus.generate.otel_capture import SpanCapture, spans_to_otlp_json

    def _capture(module: object, cassette_name: str, service: str) -> int:
        cassette = Cassette(_RAW / "cassettes" / f"{cassette_name}.cassette.json", mode=Mode.REPLAY)
        with SpanCapture(service) as cap:
            module.run(cassette, cap)  # type: ignore[attr-defined]
        return len(_spans(spans_to_otlp_json(cap.finished_spans)))

    first = _capture(claimscrew_swarm, "claimscrew_swarm", "claims-swarm")
    second = _capture(orderbot_agent_as_tool, "orderbot_agent_as_tool", "orderbot-as-tool")
    assert first > 0, "first Strands target captured no spans"
    assert second > 0, "second Strands target captured no spans (tracer singleton not reset)"


def test_corpus_replays_offline_deterministically() -> None:
    """Replaying the committed cassettes reproduces an OrderBot OTel capture with no network.

    This is the guarantee the executors reuse: a fresh clone re-freezes the corpus offline. We
    replay one
    agent (fast) rather than the whole harness and assert the over-refund is reproduced.
    """
    from corpus.agents import orderbot_otel
    from corpus.agents.otel_emit import ContentLocation
    from corpus.generate.cassette import Cassette, Mode
    from corpus.generate.otel_capture import SpanCapture, spans_to_otlp_json
    from corpus.generate.scenarios import ORDERBOT_FAIL, load_scenario

    cassette = Cassette(_RAW / "cassettes" / "orderbot_otel.cassette.json", mode=Mode.REPLAY)
    from corpus.generate.bedrock_client import BedrockConverseClient

    client = BedrockConverseClient(cassette)
    scenario = load_scenario(ORDERBOT_FAIL)
    with SpanCapture("orderbot") as capture:
        record = orderbot_otel.run(
            client, scenario, capture, content_location=ContentLocation.EVENTS
        )
    names = {s["name"] for s in _spans(spans_to_otlp_json(capture.finished_spans))}
    assert "execute_tool issue_refund" in names
    assert any(tc.name == "issue_refund" for tc in record.tool_calls)
