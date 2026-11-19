# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-silent-loss gate tests: tool spans present but none recovered → refuse (a genuine zero-tool
agent is allowed); content on the source spans but none recovered → refuse; duplicate span ids →
refuse; and the contentless-node count, which is reported whether or not it degrades."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.ingest import (
    IngestError,
    IngestResult,
    contentless_node_count,
    run_ingest,
    source_content_span_count,
    source_signals,
    verify_no_silent_loss,
)
from calibrated_agent_evaluation.normalize import OTelSpans, normalize, otlp_json
from calibrated_agent_evaluation.schema.errors import TraceNotFoundError
from calibrated_agent_evaluation.schema.types import Frame, Label, Ownership, Span, ToolCall, Trace

_RAW = Path(__file__).resolve().parent.parent / "fixtures" / "raw"
_RAW_OTEL = _RAW / "otel"
_RAW_STRANDS = _RAW / "strands"


class _FakeBackend:
    """A minimal in-memory ``TraceStoreBackend`` for the end-to-end contentless-count assertion."""

    def __init__(self) -> None:
        self.stored: dict[str, Trace] = {}

    def put_trace(self, trace: Trace) -> None:
        self.stored[trace.trace_id] = trace

    def get_trace(self, trace_id: str) -> Trace:
        if trace_id not in self.stored:
            raise TraceNotFoundError(trace_id)
        return self.stored[trace_id]

    def query(
        self,
        frame: Frame | None = None,
        cell: dict[str, str] | None = None,
        ownership: Ownership | None = None,
    ) -> list[Trace]:
        return list(self.stored.values())

    def put_score(self, label: Label) -> None:  # pragma: no cover - unused by ingest
        raise NotImplementedError

    def get_scores(self, trace_id: str, span_id: str | None = None) -> list[Label]:
        return []

    def retract_score(self, label: Label) -> None:  # pragma: no cover - unused here
        raise NotImplementedError


def _trace(
    *,
    tool_calls: list[ToolCall],
    step_input: dict[str, Any] | None = None,
    step_output: dict[str, Any] | None = None,
) -> Trace:
    span = Span(
        span_id="s0",
        trace_id="t0",
        agent="orchestrator",
        role="supervisor",
        parent=None,
        seq=0,
        status="completed",
        input=step_input or {},
        tool_calls=tool_calls,
        output=step_output or {},
    )
    return Trace(
        trace_id="t0",
        frame="production-representative",
        ownership="owned",
        input={},
        output={},
        taxonomy_cell={},
        spans=[span],
    )


def _otlp(spans: list[dict[str, Any]]) -> dict[str, Any]:
    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}


def _span(span_id: str, op: str, *, agent: str | None = None, tool: str | None = None) -> dict:
    attrs: list[dict[str, Any]] = [{"key": "gen_ai.operation.name", "value": {"stringValue": op}}]
    if agent is not None:
        attrs.append({"key": "gen_ai.agent.name", "value": {"stringValue": agent}})
    if tool is not None:
        attrs.append({"key": "gen_ai.tool.name", "value": {"stringValue": tool}})
    return {"spanId": span_id, "name": op, "attributes": attrs}


def test_refuses_when_source_has_tool_spans_but_trace_recovered_none() -> None:
    """Source carried tool-execution spans, trace recovered 0 tool calls → IngestError."""
    trace = _trace(tool_calls=[])
    with pytest.raises(IngestError, match="0 tool calls"):
        verify_no_silent_loss(source_tool_span_count=3, dup_span_names=[], trace=trace)


def test_allows_genuine_zero_tool_agent() -> None:
    """Edge: a source with no tool spans is NOT refused for having no tool calls."""
    trace = _trace(tool_calls=[])
    verify_no_silent_loss(source_tool_span_count=0, dup_span_names=[], trace=trace)  # no raise


def test_allows_faithful_trace_with_recovered_tools() -> None:
    """Tool spans present and recovered → no refusal."""
    trace = _trace(tool_calls=[ToolCall(name="foo", args={}, result={})])
    verify_no_silent_loss(source_tool_span_count=1, dup_span_names=[], trace=trace)  # no raise


def test_refuses_duplicate_span_ids_and_names_the_fix() -> None:
    """Duplicate span ids → IngestError naming the offending agent and the name= fix."""
    trace = _trace(tool_calls=[ToolCall(name="foo", args={}, result={})])
    with pytest.raises(IngestError, match="Strands Agents"):
        verify_no_silent_loss(
            source_tool_span_count=1, dup_span_names=["Strands Agents"], trace=trace
        )


def test_source_signals_counts_tool_spans() -> None:
    """source_signals counts execute_tool / gen_ai.tool.name spans, ignores agent spans."""
    otlp = _otlp(
        [
            _span("a", "invoke_agent", agent="orchestrator"),
            _span("b", "execute_tool", tool="lookup"),
            _span("c", "chat"),
            _span("d", "execute_tool", tool="write"),
        ]
    )
    tool_spans, dups = source_signals(otlp)
    assert tool_spans == 2
    assert dups == []


def test_source_signals_flags_duplicate_span_ids() -> None:
    """Two spans sharing a span id are reported by their agent name."""
    otlp = _otlp(
        [
            _span("dup", "invoke_agent", agent="Strands Agents"),
            _span("dup", "invoke_agent", agent="Strands Agents"),
        ]
    )
    _tool_spans, dups = source_signals(otlp)
    assert dups == ["Strands Agents"]


# --- the content half -------------------------------------------------------------------
# A tool-call *count* cannot see the loss a placement mismatch actually causes: every step is
# recovered and every tool is named, while every input, output, arg and result is empty. That
# trace passes the count gate and has lost everything a judge would read.


def _with_attr(span: dict[str, Any], key: str, value: str) -> dict[str, Any]:
    span["attributes"] = [*span["attributes"], {"key": key, "value": {"stringValue": value}}]
    return span


def test_refuses_when_source_had_content_but_none_was_recovered() -> None:
    """A right-shaped trace whose every step is empty is a refusal, not a pass."""
    named_only = _trace(tool_calls=[ToolCall(name="lookup", args={}, result={})])
    with pytest.raises(IngestError, match="placement"):
        verify_no_silent_loss(
            source_tool_span_count=1,
            dup_span_names=[],
            trace=named_only,
            source_content_spans=4,
        )


def test_allows_a_source_that_genuinely_carried_no_content() -> None:
    """Edge: content absent from the wire is not the adapter dropping it.

    The distinction the gate exists to make. A pipeline that strips content upstream of ingest has
    lost it, but ingest cannot be the thing that reports a loss it did not cause — and refusing
    every content-free capture would refuse the legitimate ones.
    """
    named_only = _trace(tool_calls=[ToolCall(name="lookup", args={}, result={})])
    verify_no_silent_loss(
        source_tool_span_count=1, dup_span_names=[], trace=named_only, source_content_spans=0
    )  # no raise


def test_allows_a_trace_whose_content_partly_survived() -> None:
    """Any recovered content clears the gate.

    Scoped to a total loss on purpose. A partial recovery is a real signal but not a decidable
    one — the source count is spans-with-content, not fields — and refusing on it would refuse
    faithful traces whose deterministic steps have no model I/O to recover.
    """
    partly = _trace(tool_calls=[ToolCall(name="lookup", args={}, result={})], step_output={"a": 1})
    verify_no_silent_loss(
        source_tool_span_count=1, dup_span_names=[], trace=partly, source_content_spans=4
    )  # no raise


def test_source_content_span_count_sees_every_placement() -> None:
    """Counted placement-agnostically: legacy events, latest attributes, and the details event.

    Whichever placement it rode in, content that was on the wire has to be comparable against what
    came back — a per-placement count would go blind exactly when the placement was the problem.
    """
    legacy = _span("a", "invoke_agent", agent="orchestrator")
    legacy["events"] = [
        {
            "name": "gen_ai.user.message",
            "attributes": [{"key": "content", "value": {"stringValue": "hi"}}],
        }
    ]
    latest_attr = _with_attr(
        _span("b", "invoke_agent", agent="worker"),
        "gen_ai.output.messages",
        '[{"role": "assistant", "parts": [{"type": "text", "content": "done"}]}]',
    )
    details_event = _span("c", "execute_tool", tool="lookup")
    details_event["events"] = [
        {
            "name": "gen_ai.client.inference.operation.details",
            "attributes": [
                {"key": "gen_ai.tool.call.result", "value": {"stringValue": '{"ok": true}'}}
            ],
        }
    ]
    bare = _span("d", "chat")

    assert source_content_span_count(_otlp([legacy, latest_attr, details_event, bare])) == 3


def test_source_content_span_count_is_zero_for_a_stripped_capture() -> None:
    """Negative: names and structure but no content counts zero, so the gate stays off."""
    otlp = _otlp(
        [
            _span("a", "invoke_agent", agent="orchestrator"),
            _span("b", "execute_tool", tool="lookup"),
        ]
    )

    assert source_content_span_count(otlp) == 0


def test_source_content_span_count_ignores_empty_payloads() -> None:
    """An attribute holding an empty list or object is the key's presence, not content.

    Regression. Counting `gen_ai.input.messages = "[]"` as content made the gate refuse a
    genuinely contentless source (`degraded_v2.otlp.json`) with a message blaming a placement the
    adapter did not read, when there was nothing in any placement to read.
    """
    empties = [
        _with_attr(_span("a", "chat"), "gen_ai.input.messages", "[]"),
        _with_attr(_span("b", "chat"), "gen_ai.output.messages", "{}"),
        _with_attr(_span("c", "execute_tool", tool="lookup"), "gen_ai.tool.call.result", " [ ] "),
    ]

    assert source_content_span_count(_otlp(empties)) == 0


def test_source_content_span_count_reads_registered_vendor_placements() -> None:
    """Content on a vendor family's registered keys counts, so the gate sees what the reader reads.

    Without it, a Traceloop-shaped source whose content the reader then failed to recover would
    compare 0 against 0 and pass, the silent emptying the content gate exists to prevent.
    """
    traceloop = _with_attr(
        _span("a", "invoke_agent", agent="writer"), "traceloop.entity.input", '{"q": "x"}'
    )

    assert source_content_span_count(_otlp([traceloop])) == 1


def test_contentless_messages_source_degrades_rather_than_refusing() -> None:
    """`calibrated-agent-evaluation ingest` agrees with `normalize()` on `degraded_v2`: probed with
    a reason, no refusal.

    Found re-running the acceptance checklist: the file was the batch's one refusal, so a fixture
    whose whole purpose is to show an honest degrade made
    `calibrated-agent-evaluation ingest fixtures/raw/otel` exit 1.
    """
    report = run_ingest([_RAW_OTEL / "degraded_v2.otlp.json"], backend=_FakeBackend())
    (result,) = report.results

    assert result.status == "ingested", result.reason
    assert result.ownership == "probed"


# --- contentless nodes ---------------------------------------------------------
#
# degrades a trace whose EVERY node is contentless; the count is reported whether or not
# it degrades. The gap they close, measured against the installed tree before the fix was written:
# `ownership._has_recovered_content` treats the mere PRESENCE of a `tool_calls` entry as recovered
# content, so a trace with correctly-named tool calls and `{}` in every args, result, input and
# output field resolves to `owned`. `test_FR017_all_contentless_tree_is_currently_owned` pins that
# live, so the degrade's arrival is a visible change rather than a claim.
#
# Content is stripped at the SOURCE, not post-normalize, so the content gate does not fire:
# the source genuinely has none there, which is a different claim from "the adapter dropped it".

_CONTENT_KEYS = frozenset(
    {
        "gen_ai.input.content",
        "gen_ai.output.content",
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.tool.input",
        "gen_ai.tool.output",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
    }
)

# One real agent with three real linked tool calls — the shape that exposes the `owned` gap. The
# 5-agent CLM-4471 fan-out cannot be used for it: read as generic OTLP its tool spans do not link to
# their agent at all (the BUG-1 nesting, see tests/normalize/test_fanout_rerooting.py), so it has no
# recovered tool content to remove.
_ONE_AGENT = _RAW_OTEL / "ORD-1001.attrs.otlp.json"
# The 5-agent fan-out, used only for the partial case: several nodes, no tools, is the point there.
_CREW = _RAW_STRANDS / "CLM-4471.otlp.json"


def _strip_content(
    otlp: dict[str, Any],
    *,
    ops: set[str] | None = None,
    agents: set[str] | None = None,
) -> dict[str, Any]:
    """Remove every content placement from the matching spans.

    ``ops`` selects by ``gen_ai.operation.name`` (``invoke_agent`` / ``execute_tool``) and
    ``agents`` by ``gen_ai.agent.name``; either omitted means "don't filter on that".
    Selecting by operation name rather than by agent name is the reliable form: a mistyped agent
    name silently matches nothing, and the test then passes against an unmodified fixture.
    """
    payload = json.loads(json.dumps(otlp))
    for resource_spans in payload.get("resourceSpans", []):
        for scope_spans in resource_spans.get("scopeSpans", []):
            for span in scope_spans.get("spans", []):
                attrs = otlp_json.attributes(span)
                if ops is not None and attrs.get("gen_ai.operation.name") not in ops:
                    continue
                if agents is not None and attrs.get("gen_ai.agent.name") not in agents:
                    continue
                span["attributes"] = [
                    kv for kv in span.get("attributes", []) if kv["key"] not in _CONTENT_KEYS
                ]
                span.pop("events", None)
    return payload


def _normalize_generic(tmp_path: Path, payload: dict[str, Any], name: str, trace_id: str) -> Trace:
    out = tmp_path / f"{name}.otlp.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return normalize(OTelSpans(path=out, semconv="gen_ai/latest", trace_id=trace_id))


def test_intact_traces_report_zero_contentless_nodes(tmp_path: Path) -> None:
    """Baseline: both fixtures have content on every node, so the count is 0.

    Without this the assertions below could pass against a fixture that was empty to begin with.
    """
    one = _normalize_generic(
        tmp_path, json.loads(_ONE_AGENT.read_text(encoding="utf-8")), "intact1", "ORD-1001"
    )
    assert len(one.spans) == 1
    assert sum(len(s.tool_calls) for s in one.spans) == 3
    assert contentless_node_count(one) == 0

    crew = _normalize_generic(
        tmp_path, json.loads(_CREW.read_text(encoding="utf-8")), "intact5", "CLM-4471"
    )
    assert len(crew.spans) == 5
    assert contentless_node_count(crew) == 0


def test_all_contentless_tree_degrades_to_probed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Every content field empty ⇒ `probed` with a reason that names empty content.

    This is the flipped form of `test_FR017_all_contentless_tree_is_currently_owned`. The
    gap it used to pin was in `ownership._has_recovered_content`, which counted a `tool_calls`
    entry's *existence* as content without looking inside it — so a placement mismatch that
    recovered the tree and every tool's name while emptying every payload shipped as `owned`, and a
    judge reading it read nothing.

    The tree is discarded, not kept alongside `probed`. That is forced, not chosen: `validate`
    holds `probed ⇒ spans == []` (`schema/validate.py:125`), so there is no representation for
    "structure recovered, content absent". The structure is still visible in the *reason*, which
    names the node and tool counts.
    """
    stripped = _strip_content(json.loads(_ONE_AGENT.read_text(encoding="utf-8")))
    with caplog.at_level("WARNING", logger="calibrated_agent_evaluation.normalize"):
        trace = _normalize_generic(tmp_path, stripped, "empty", "ORD-1001")

    assert trace.ownership == "probed"
    assert trace.spans == [], "the schema has no owned-shaped probed trace"
    assert contentless_node_count(trace) == 0, "no nodes left to count once it degrades"

    reason = caplog.text
    assert "empty content" in reason, "the reason must name WHY, not just that it degraded"
    assert "1 recovered node(s)" in reason, "and how much structure was there"
    assert "3 named tool call(s)" in reason, "including the tools whose payloads were empty"


def test_a_tool_call_with_only_a_name_is_not_content(tmp_path: Path) -> None:
    """The precise boundary the contentless-degrade rule moved: a named tool call with no args and
    no result.

    Negative-path companion to the test above. Before that rule this single fact — `tool_calls`
    being
    non-empty — was enough to certify a trace as `owned`. A tool name records that something
    happened, not what; the distinction is the whole requirement, so it is asserted on its own.
    """
    stripped = _strip_content(json.loads(_ONE_AGENT.read_text(encoding="utf-8")))
    trace = _normalize_generic(tmp_path, stripped, "namesonly", "ORD-1001")
    assert trace.ownership == "probed"

    # ... and the same source WITH tool payloads intact stays owned, so the degrade is caused by the
    # empty payloads rather than by anything else the stripping did.
    agents_only = _strip_content(
        json.loads(_ONE_AGENT.read_text(encoding="utf-8")), ops={"invoke_agent"}
    )
    kept = _normalize_generic(tmp_path, agents_only, "toolsintact", "ORD-1001")
    assert kept.ownership == "owned"
    assert any(tc.args or tc.result for s in kept.spans for tc in s.tool_calls)


def test_partial_emptiness_is_counted_and_does_not_degrade(tmp_path: Path) -> None:
    """Stripping content from SOME agents keeps the trace `owned` and reports the count.

    This is the no-false-degrade boundary. An agent that legitimately carries no content of its own
    (a
    purely-delegating supervisor, a `create_agent` span) must not degrade the whole trace — so the
    count is what stops that leniency from being silence.
    """
    stripped = _strip_content(
        json.loads(_CREW.read_text(encoding="utf-8")), agents={"intake", "payout"}
    )
    trace = _normalize_generic(tmp_path, stripped, "partial", "CLM-4471")

    assert len(trace.spans) == 5
    assert contentless_node_count(trace) == 2
    assert trace.ownership == "owned", "partial content loss must NOT degrade the trace"


def _ingest_crew(tmp_path: Path, payload: dict[str, Any], name: str) -> IngestResult:
    """Push a (possibly stripped) CLM-4471 payload through `run_ingest` on the Strands adapter.

    The Strands adapter rather than the generic reader: read as generic OTLP this fixture's tool
    spans do not link to their agent at all (the BUG-1 nesting, see
    tests/normalize/test_fanout_rerooting.py) and the tool gate refuses it before any count
    is reached — the gate working correctly, and the wrong path for these assertions.
    """
    path = tmp_path / f"{name}.otlp.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = run_ingest([path], backend=_FakeBackend(), fmt="strands")
    (result,) = report.results
    return result


def test_ingest_result_carries_the_count_end_to_end(tmp_path: Path) -> None:
    """Boundary: the figure reaches `IngestResult`, not just the helper.

    A count computed and never surfaced would be the same defect one layer up, so this asserts it on
    the object the recovery summary renders. Partial emptiness is the case that keeps the count
    meaningful: an all-empty trace now degrades and has no nodes left to count, so a count
    only ever reported on a degraded trace would always read 0.
    """
    stripped = _strip_content(
        json.loads(_CREW.read_text(encoding="utf-8")), agents={"intake", "payout"}
    )
    result = _ingest_crew(tmp_path, stripped, "CLM-4471")

    assert result.status == "ingested", result.reason
    assert result.ownership == "owned", "partial emptiness must not degrade"
    assert result.span_count == 5
    assert result.contentless_node_count == 1
    assert 0 < result.contentless_node_count < result.span_count, "the partial case, end to end"


def test_an_all_empty_trace_ingests_rather_than_being_refused(tmp_path: Path) -> None:
    """The "shall not refuse", asserted through `run_ingest` rather than at the gate.

    The regression risk this covers is a composition, not a single function: the
    contentless-degrade rule collapses the
    tree to `spans == []`, and the tool gate then sees a source with 4 `execute_tool` spans
    and a trace with 0 tool calls, which is its exact refusal condition. It refused until the gate
    was taught to measure what the ADAPTER linked (`Recovery.recovered_tool_calls`) rather than what
    survived the collapse. Both requirements are satisfiable at once; nothing asserts that but this.
    """
    stripped = _strip_content(json.loads(_CREW.read_text(encoding="utf-8")))
    result = _ingest_crew(tmp_path, stripped, "CLM-4471")

    assert result.status == "ingested", result.reason
    assert result.ownership == "probed"
    assert result.span_count == 0
    assert result.tool_call_count == 0
    assert result.contentless_node_count == 0


def test_still_refuses_tools_the_adapter_never_linked(tmp_path: Path) -> None:
    """The other half of that composition: gating on the adapter must not blunt the real refusal.

    Same fixture, read through the generic OTel adapter, where the tool spans genuinely do not link.
    `Recovery.recovered_tool_calls` is 0 because nothing was recovered — not because a collapse
    discarded it — so the content gate must still fire. If this passes as `ingested`, the fix went
    too far.
    """
    path = tmp_path / "CLM-4471.otlp.json"
    path.write_text(_CREW.read_text(encoding="utf-8"), encoding="utf-8")
    report = run_ingest([path], backend=_FakeBackend(), fmt="otel")
    (result,) = report.results

    assert result.status == "refused"
    assert result.reason is not None
    assert "recovered 0 tool calls" in result.reason


# --- the content gate vs a declared degradation ------------------------------


def test_content_gate_allows_a_terminal_only_source_that_degraded(tmp_path: Path) -> None:
    """A legitimately-probed trace is not a content loss, and must not be refused as one.

    Regression. The content gate compared `source_content_span_count` (>= 1 here: the single
    agent span carries `gen_ai.input/output.content`) against `recovered_content_count`, which walks
    `trace.spans` — and a probed trace has none, by definition, because its content lives in the
    terminal envelope. So every terminal-I/O-only OTLP source was refused with a message blaming a
    placement mismatch, when in fact nothing had been lost and there was no placement to go and fix.
    """
    otlp = json.loads((_RAW_OTEL / "ORD-DEGRADED.otlp.json").read_text(encoding="utf-8"))
    trace = normalize(OTelSpans(_RAW_OTEL / "ORD-DEGRADED.otlp.json"))

    assert trace.ownership == "probed"
    assert trace.spans == []
    assert trace.input, "the envelope is what a probed trace recovers"
    assert trace.output
    assert source_content_span_count(otlp) == 1

    verify_no_silent_loss(0, [], trace, source_content_spans=1)  # must not raise


def test_content_gate_still_refuses_a_degradation_that_dropped_per_step_content() -> None:
    """The other side of that boundary: several content-carrying source spans collapsing to one.

    Content on exactly one source span *is* the terminal envelope. Content on several was per-step,
    and a degradation that keeps only the terminal step has silently dropped the rest — so relaxing
    the gate for `probed` must not relax it here.
    """
    probed = Trace(
        trace_id="T-COLLAPSED",
        frame="discovery",
        ownership="probed",
        input={"text": "start"},
        output={"text": "done"},
        taxonomy_cell={},
        spans=[],
    )
    with pytest.raises(IngestError, match="recovered none"):
        verify_no_silent_loss(0, [], probed, source_content_spans=4)


def test_content_gate_refuses_a_probed_trace_with_an_empty_envelope() -> None:
    """A probed trace that kept nothing at all is still a loss, however few spans the source had."""
    empty = Trace(
        trace_id="T-EMPTY",
        frame="discovery",
        ownership="probed",
        input={},
        output={},
        taxonomy_cell={},
        spans=[],
    )
    with pytest.raises(IngestError, match="recovered none"):
        verify_no_silent_loss(0, [], empty, source_content_spans=1)


def test_tool_gate_is_not_relaxed_for_a_probed_trace() -> None:
    """A source with tool-execution spans has step-level data: degrading is loss, not degrading."""
    probed = Trace(
        trace_id="T-TOOLS-LOST",
        frame="discovery",
        ownership="probed",
        input={"text": "start"},
        output={"text": "done"},
        taxonomy_cell={},
        spans=[],
    )
    with pytest.raises(IngestError, match="recovered 0 tool calls"):
        verify_no_silent_loss(3, [], probed)


# --- the committed planted-bad inputs the acceptance checklist runs against ---

_PLANTED = _RAW / "planted"


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("_bad_unlinked_tools.otlp.json", "recovered 0 tool calls"),
        ("_bad_dup_span.otlp.json", "give each Agent(...) a distinct name="),
    ],
)
def test_calibrated_agent_evaluation_ingest_refuses_the_planted_bad_inputs(
    name: str,
    reason: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each planted file is refused by name through the CLI, exit 1, nothing persisted.

    The files are real captures with one fault each (`fixtures/raw/planted/PROVENANCE.md`): tool
    spans whose parent is absent, and two agents sharing a span id.
    """
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.cli.dispatch import main

    monkeypatch.chdir(tmp_path)
    code = main(["ingest", str(_PLANTED / name)])
    out = capsys.readouterr()

    assert code == 1
    assert "1 refused" in out.out + out.err
    assert reason in out.out + out.err
    assert LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).query() == []
