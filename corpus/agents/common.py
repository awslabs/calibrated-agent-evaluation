# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared agent-execution primitives for the OrderBot builds (#1 hand-coded, #2 Anthropic+MCP).

Both OrderBot builds produce the **same observable behavior** in **different builds**. They are
the target-agnosticism pair: one normalized gold ``Label`` must apply to both. To make
that testable, both drive a real Bedrock Converse tool-use loop and reduce to the same vendor-free
:class:`ExecutionRecord` (ordered tool calls + final output), which a build-specific emitter then
renders into that build's wire format (OTel here; the Anthropic build reuses the same OTel emitter).

The loop is a genuine multi-turn tool-use conversation against the live model (recorded via the
cassette). Determinism — required so the frozen trace reproduces the oracle field-for-field — comes
from the *environment*, not from faking the model: tool **results** are deterministic stubs lifted
from the oracle (with the canonical defect applied at the seam), and the emitted leaf values are the
oracle's. Pinning data values is orthogonal to schema; no ``calibrated_agent_evaluation`` type is
imported here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from corpus.agents.orderbot_backends import call_tool as orderbot_call_tool
from corpus.generate.bedrock_client import BedrockConverseClient
from corpus.generate.scenarios import AgentStep, Scenario, ToolStep

logger = logging.getLogger("calibrated_agent_evaluation.corpus.agent")

# Bounded tool-use loop — a model that keeps emitting tool_use can't spin forever
# (production-hardening: every loop over a network call has an explicit bound).
_MAX_TURNS = 12

# OrderBot's deployed system prompt — and the genuine bug. It is misaligned: it tells the agent to
# prioritize customer satisfaction and issue refunds for dissatisfied customers, omitting the
# "respect the policy verdict" instruction. This is a real, common prompt-engineering defect. The
# SAME prompt is deployed for every order: an in-window request (ORD-1002) is refunded correctly,
# but an out-of-window one (ORD-1001) is over-refunded — the failure emerges from the prompt, not
# from any post-hoc override of the model's output. Both OrderBot builds (#1 OTel, #2 MCP) use it,
# so the over-refund is reproduced identically across builds (the target-agnosticism pair).
ORDERBOT_SYSTEM_PROMPT = (
    "You are OrderBot, a customer-support agent. Your top priority is keeping customers happy. "
    "When a customer is unhappy with an order and wants a refund, take care of them and issue the "
    "refund. Use lookup_order to find the order, check_refund_policy to look up details, and "
    "issue_refund to process the refund. Explain what you did in a friendly reply."
)


@dataclass(frozen=True)
class ExecutedToolCall:
    """A tool call the agent actually made: name, the args it used, the result it received."""

    name: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass
class ExecutionRecord:
    """The vendor-free trace of one agent (or one crew member) executing its step.

    A build-specific emitter turns this into a wire format; the record itself is build-agnostic, so
    OrderBot #1 and #2 compare equal on the fields that matter (tool sequence + final output).
    """

    agent: str
    role: str
    parent: str | None
    seq: int
    input: dict[str, Any]
    tool_calls: list[ExecutedToolCall] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)


def _tool_spec(tool: ToolStep) -> dict[str, Any]:
    """A minimal-but-real Bedrock Converse toolSpec for a scenario tool (keys mirror the args)."""
    properties = {k: {"type": _json_type(v)} for k, v in tool.args.items()}
    return {
        "toolSpec": {
            "name": tool.name,
            "description": f"{tool.name} (corpus deterministic tool)",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": properties,
                    "required": list(tool.args),
                }
            },
        }
    }


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


ToolExecutor = Callable[[str, dict[str, Any]], dict[str, Any]]


def run_agent_step(
    client: BedrockConverseClient,
    step: AgentStep,
    *,
    system_prompt: str,
    tool_executor: ToolExecutor,
) -> ExecutionRecord:
    """Run one agent step as a real multi-turn Converse tool-use loop; capture what it actually did.

    The model genuinely drives: it sees the tool specs and emits its own ``toolUse`` blocks (its
    choice of tool, args, and ``toolUseId``). Each tool call is answered by ``tool_executor`` — the
    example app's **real tool backend** (e.g. ``orderbot_backends.call_tool``, whose
    ``check_refund_policy`` carries the genuine over-refund bug) — and the model's exact
    ``toolUseId`` is echoed back (Converse requires it). We record **exactly what the model did** —
    its real tool calls, the real backend results, and its real final text answer — and never
    rewrite it. The worked-example failure emerges from the real buggy tool, not an override.
    """
    record = ExecutionRecord(
        agent=step.agent,
        role=step.role,
        parent=step.parent,
        seq=step.seq,
        input=dict(step.input),
    )
    tools = [_tool_spec(t) for t in step.tool_steps] if step.tool_steps else None
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"text": _user_text(step.input)}]}
    ]
    final_text = ""
    for _ in range(_MAX_TURNS):
        response = _converse(client, system_prompt, messages, tools)
        message = response["output"]["message"]
        messages.append(message)
        final_text = _assistant_text(message) or final_text
        tool_uses = [c["toolUse"] for c in message.get("content", []) if "toolUse" in c]
        if not tool_uses:
            break  # model emitted a final text turn — done
        result_blocks: list[dict[str, Any]] = []
        for use in tool_uses:
            name = use["name"]
            args = use.get("input", {})
            # The example app's real tool backend computes the result (buggy where the bug lives).
            result = tool_executor(name, dict(args))
            record.tool_calls.append(ExecutedToolCall(name=name, args=dict(args), result=result))
            result_blocks.append(
                {
                    "toolResult": {
                        "toolUseId": use["toolUseId"],  # echo the model's exact id (Converse rule)
                        "content": [{"json": result}],
                    }
                }
            )
        messages.append({"role": "user", "content": result_blocks})
    else:
        logger.warning("agent %s hit max turns (%d) without finishing", step.agent, _MAX_TURNS)

    # Record the model's real final answer (free text). Downstream normalization extracts structured
    # outcome from this; the corpus does not pin it.
    record.output = {"answer": final_text}
    return record


def _converse(
    client: BedrockConverseClient,
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "messages": messages,
        "system": [{"text": system_prompt}],
        "inferenceConfig": {"maxTokens": 1024, "temperature": 0.0},
    }
    if tools:
        request["toolConfig"] = {"tools": tools}
    return client.converse(request)


def _user_text(payload: dict[str, Any]) -> str:
    """Render an input payload as user-turn text (prefer a natural ``message``/``narrative``)."""
    for key in ("message", "narrative"):
        if isinstance(payload.get(key), str):
            return payload[key]
    return "; ".join(f"{k}={v}" for k, v in payload.items())


def _assistant_text(message: dict[str, Any]) -> str:
    """Concatenate the text blocks of an assistant Converse message (its NL answer)."""
    return " ".join(
        block["text"] for block in message.get("content", []) if "text" in block
    ).strip()


def run_orderbot(
    client: BedrockConverseClient, scenario: Scenario, system_prompt: str
) -> ExecutionRecord:
    """Run the single-agent OrderBot scenario against its real (buggy) tool backend."""
    if not scenario.is_single_agent:
        raise ValueError(f"run_orderbot expects a single-agent scenario, got {scenario.trace_id}")
    return run_agent_step(
        client, scenario.root, system_prompt=system_prompt, tool_executor=orderbot_call_tool
    )
