# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`trace.output` is the final answer, not whichever span started last.

Both adapters derived the trace-level output from `sorted(..., key=start_time)[-1]` — the last span
to *start*. In a supervisor/agent-as-tools topology the supervisor starts first and finishes last,
so that expression returned a **sub-agent's intermediate result** as the run's answer.
Nothing anywhere said so: the trace validated, resolved `owned`, and ingested clean, so every
downstream judge prompt, scorer comparison and review-UI panel read the wrong text as the outcome.

Method: the real `ORD-3310-as-tool` capture (a genuine two-agent Strands run where `orderbot`
delegates to `policy_advisor` and composes the reply) with only its **timestamps** transformed. The
content stays the captured content, so an assertion about which output is returned is a claim about
ordering and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.normalize import OTelSpans, StrandsSession, normalize

_RAW = Path(__file__).resolve().parent.parent.parent / "fixtures" / "raw"
_AS_TOOL = _RAW / "strands" / "ORD-3310-as-tool.otlp.json"

_SUPERVISOR = "orderbot"
_SUB_AGENT = "policy_advisor"


def _load() -> dict[str, Any]:
    return json.loads(_AS_TOOL.read_text(encoding="utf-8"))


def _agent_name(span: dict[str, Any]) -> str | None:
    for kv in span.get("attributes") or []:
        if kv.get("key") == "gen_ai.agent.name":
            return next(iter(kv.get("value", {}).values()), None)
    return None


def _write(tmp_path: Path, payload: dict[str, Any], name: str) -> Path:
    path = tmp_path / f"{name}.otlp.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _spans(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        span
        for rs in payload.get("resourceSpans", [])
        for ss in rs.get("scopeSpans", [])
        for span in ss.get("spans", [])
    ]


def _make_sub_agent_start_last(payload: dict[str, Any]) -> dict[str, Any]:
    """Push the sub-agent's START past the supervisor's, leaving both END times untouched.

    The concurrency/fan-out shape the old code got wrong, reproduced from a real capture: the
    supervisor still finishes last (it composes the answer) but is no longer the last to start.
    """
    out = json.loads(json.dumps(payload))
    ends = [int(s["endTimeUnixNano"]) for s in _spans(out)]
    latest_end = max(ends)
    for span in _spans(out):
        if _agent_name(span) == _SUB_AGENT:
            # Start after every other span started, but keep the captured (earlier) end time.
            span["startTimeUnixNano"] = str(latest_end - 1)
    return out


def _captured_outputs() -> tuple[str, str]:
    """The two real answers, so the assertions compare against captured text, not a guess."""
    trace = normalize(StrandsSession(_AS_TOOL))
    by_agent = {span.agent: span.output for span in trace.spans}
    return json.dumps(by_agent[_SUPERVISOR]), json.dumps(by_agent[_SUB_AGENT])


def test_strands_trace_output_is_the_supervisors_answer() -> None:
    """Baseline on the untransformed capture: the composed reply, not the delegate's."""
    trace = normalize(StrandsSession(_AS_TOOL))
    supervisor_output, sub_agent_output = _captured_outputs()

    assert json.dumps(trace.output) == supervisor_output
    assert json.dumps(trace.output) != sub_agent_output
    assert "refund policy assessment" not in json.dumps(trace.output), (
        "that phrase is the delegate's intermediate assessment, not the customer-facing answer"
    )


def test_strands_trace_output_survives_the_sub_agent_starting_last(tmp_path: Path) -> None:
    """The regression, with the sub-agent invoked last: still the supervisor's answer.

    Under the old `ordered[-1]` (order assigned by start time) this returned `policy_advisor`'s
    intermediate assessment.
    """
    path = _write(tmp_path, _make_sub_agent_start_last(_load()), "reordered")
    trace = normalize(StrandsSession(path))
    supervisor_output, sub_agent_output = _captured_outputs()

    assert json.dumps(trace.output) == supervisor_output
    assert json.dumps(trace.output) != sub_agent_output


def _strip_agent_output(payload: dict[str, Any], agent: str) -> dict[str, Any]:
    """Remove one agent's own output content, leaving its tool calls and the rest untouched."""
    for span in _spans(payload):
        if _agent_name(span) == agent:
            span["attributes"] = [
                kv
                for kv in span.get("attributes", [])
                if kv["key"] not in {"gen_ai.output.messages", "gen_ai.output.content"}
            ]
            span["events"] = [
                ev for ev in span.get("events", []) or [] if ev.get("name") != "gen_ai.choice"
            ]
    return payload


def test_strands_falls_back_to_the_last_ENDED_agent_when_the_root_composed_nothing(
    tmp_path: Path,
) -> None:
    """The handoff-terminal topology: no root answer, so the terminal agent's output is the answer.

    With the supervisor's own output stripped, the fallback picks by time — and the transform makes
    the sub-agent both end last AND start first, so start-time ordering and end-time ordering
    disagree. Start-time ordering picks the supervisor (which has nothing); end-time ordering picks
    the sub-agent (which has the only remaining answer). Only the latter is right.
    """
    payload = _strip_agent_output(_load(), _SUPERVISOR)
    ends = [int(s["endTimeUnixNano"]) for s in _spans(payload)]
    starts = [int(s["startTimeUnixNano"]) for s in _spans(payload)]
    for span in _spans(payload):
        if _agent_name(span) == _SUB_AGENT:
            span["startTimeUnixNano"] = str(min(starts) - 1000)  # starts FIRST
            span["endTimeUnixNano"] = str(max(ends) + 1000)  # ends LAST

    trace = normalize(StrandsSession(_write(tmp_path, payload, "no-root-output")))
    by_agent = {span.agent: span.output for span in trace.spans}

    assert not by_agent[_SUPERVISOR], "precondition: the supervisor's own output was stripped"
    assert by_agent[_SUB_AGENT], "precondition: the sub-agent still carries its captured output"
    assert trace.output == by_agent[_SUB_AGENT], "the last agent to END supplies the answer"


def test_generic_otel_adapter_prefers_the_root_agents_output(tmp_path: Path) -> None:
    """The same fix on the generic reader — both adapters carried the identical expression.

    This capture keeps its content in `gen_ai.choice` **events**, which the generic `gen_ai/latest`
    attribute reader does not look at, so the content is relocated onto `gen_ai.output.content`.
    Values stay the captured ones; only the placement moves (the same relocation method as
    `test_generic_content_placement`). The sub-agent is transformed to start last, which is what the
    old start-time expression got wrong.
    """
    payload = _make_sub_agent_start_last(_load())
    relocated = 0
    for span in _spans(payload):
        if _agent_name(span) is None:
            continue
        for event in span.get("events", []) or []:
            if event.get("name") != "gen_ai.choice":
                continue
            for kv in event.get("attributes", []) or []:
                if kv.get("key") == "message":
                    span["attributes"].append(
                        {"key": "gen_ai.output.content", "value": dict(kv["value"])}
                    )
                    relocated += 1
    assert relocated == 2, "precondition: both agents' outputs relocated onto the generic key"

    trace = normalize(OTelSpans(_write(tmp_path, payload, "generic"), semconv="gen_ai/latest"))
    by_agent = {span.agent: span.output for span in trace.spans}

    assert by_agent[_SUPERVISOR], "precondition: the root recovered its own output"
    assert by_agent[_SUB_AGENT], "precondition: the sub-agent recovered its own output"
    assert by_agent[_SUPERVISOR] != by_agent[_SUB_AGENT], "precondition: the two differ"
    assert trace.output == by_agent[_SUPERVISOR], "the root's composed answer wins"
