# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OrderBot build #2 — Anthropic Messages SDK + stdio-MCP tools + a skill, emitting OTel.

The target-agnosticism twin of build #1 (``corpus-companion.md`` deliverable #2). Same observable
behavior (``lookup_order -> check_refund_policy -> issue_refund``, same over-refund), but a
genuinely different build:

  * tools come from a **real stdio MCP server** (``orderbot_mcp_server``) spawned as a subprocess
    and called over the MCP protocol — not in-process stubs;
  * the agent loop uses the **Anthropic Messages API** (a different wire shape than build #1's
    Bedrock Converse), routed through the cassette so it records once and replays offline;
  * a **skill** is pulled from the MCP server as a prompt (``refund_policy_skill``).

Because both builds reduce to the same :class:`~corpus.agents.common.ExecutionRecord` and emit
via the same OTel emitter, the corpus can prove one gold ``Label`` applies to both. The
model genuinely drives tool use; the MCP tools return deterministic oracle results (defect seam
applied), and the record is reconciled to the oracle so the frozen build-#2 trace matches it.

This path is async (the MCP stdio client is async-only).
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from corpus.agents.common import ORDERBOT_SYSTEM_PROMPT, ExecutedToolCall, ExecutionRecord
from corpus.agents.otel_emit import emit_trace
from corpus.generate.cassette import Cassette
from corpus.generate.otel_capture import SpanCapture
from corpus.generate.scenarios import Scenario

logger = logging.getLogger("calibrated_agent_evaluation.corpus.orderbot_mcp")

# Anthropic-on-Bedrock model id (Messages API). Mirrors build #1's model so the pair is comparable.
_ANTHROPIC_BEDROCK_MODEL = "global.anthropic.claude-sonnet-4-6"
_MAX_TURNS = 12


async def run(
    cassette: Cassette,
    scenario: Scenario,
    capture: SpanCapture,
    *,
    region: str = "us-west-2",
) -> ExecutionRecord:
    """Run OrderBot build #2 end-to-end and emit its OTel span tree into ``capture``.

    Spawns the stdio MCP server for ``scenario``, runs a real Anthropic Messages tool-use loop
    (through the cassette), and emits OTel for exactly what the agent did. Returns the
    build-agnostic record for the equivalence comparison against build #1.
    """
    record = await _run_loop(cassette, scenario, region=region)
    tracer = capture.tracer("calibrated_agent_evaluation.orderbot.mcp")
    emit_trace(tracer, [record], trace_input=scenario.trace_input)
    return record


async def _run_loop(cassette: Cassette, scenario: Scenario, *, region: str) -> ExecutionRecord:
    step = scenario.root
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "corpus.agents.orderbot_mcp_server", scenario.trace_id],
    )
    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        mcp_tools = (await session.list_tools()).tools
        anthropic_tools = [_to_anthropic_tool(t) for t in mcp_tools]
        # Pull the skill (an MCP prompt) and fold it into the system prompt — a real skill use.
        skill = await session.get_prompt("refund_policy_skill")
        system_prompt = _system_prompt(skill)

        client = _anthropic_client(region)
        messages: list[dict[str, Any]] = [{"role": "user", "content": _user_text(step.input)}]
        rec = ExecutionRecord(
            agent=step.agent,
            role=step.role,
            parent=step.parent,
            seq=step.seq,
            input=dict(step.input),
        )
        final_text = ""
        for _ in range(_MAX_TURNS):
            response = await _messages_create(
                cassette, client, system_prompt, messages, anthropic_tools
            )
            messages.append({"role": "assistant", "content": response["content"]})
            final_text = _content_text(response["content"]) or final_text
            tool_uses = [b for b in response["content"] if b.get("type") == "tool_use"]
            if not tool_uses:
                break
            result_blocks: list[dict[str, Any]] = []
            for use in tool_uses:
                # Call the REAL MCP tool over stdio; it returns the deterministic example backend.
                mcp_result = await session.call_tool(use["name"], use.get("input", {}))
                payload = _mcp_result_payload(mcp_result)
                # Record exactly what the model called + what the backend returned (no rewrite).
                rec.tool_calls.append(
                    ExecutedToolCall(
                        name=use["name"], args=dict(use.get("input", {})), result=payload
                    )
                )
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": use["id"],
                        "content": json.dumps(payload),
                    }
                )
            messages.append({"role": "user", "content": result_blocks})
        else:
            logger.warning("orderbot-mcp hit max turns (%d)", _MAX_TURNS)

    # Record the model's real final answer; the failure (if any) is the model's, from the prompt.
    rec.output = {"answer": final_text}
    return rec


def _anthropic_client(region: str) -> Any:
    # AnthropicBedrock is re-exported from the package root but not in __all__; the import is real
    # (verified against anthropic==0.114.0), so silence pyright's private-import heuristic only.
    from anthropic import AnthropicBedrock  # pyright: ignore[reportPrivateImportUsage]

    # aws_profile/aws_region honored from env/args; creds stay in the process (never logged/stored).
    return AnthropicBedrock(aws_region=region)


async def _messages_create(
    cassette: Cassette,
    client: Any,
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """One Anthropic Messages call through the cassette; returns a plain dict (content blocks)."""
    request = {
        "model": _ANTHROPIC_BEDROCK_MODEL,
        "max_tokens": 1024,
        "temperature": 0.0,
        "system": system_prompt,
        "messages": messages,
        "tools": tools,
    }

    async def live_call() -> dict[str, Any]:
        msg = await _to_thread_create(client, request)
        # Normalize the SDK Message to a content-block dict (what we replay).
        return {
            "content": [_block_to_dict(b) for b in msg.content],
            "stop_reason": msg.stop_reason,
        }

    return await cassette.aplay(request, live_call)


async def _to_thread_create(client: Any, request: dict[str, Any]) -> Any:
    from anyio.to_thread import run_sync

    # AnthropicBedrock (sync client) does a blocking HTTP call; run it off the event loop so we
    # don't block the MCP stdio transport (ASYNC-safe).
    return await run_sync(lambda: client.messages.create(**request))


def _block_to_dict(block: Any) -> dict[str, Any]:
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": block.text}
    if btype == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    # Any other block type: serialize via the SDK's own dumper so nothing is silently dropped.
    return block.model_dump(mode="json") if hasattr(block, "model_dump") else {"type": btype}


def _content_text(content: list[dict[str, Any]]) -> str:
    """Concatenate the text blocks of an assistant message (its natural-language answer)."""
    return " ".join(b["text"] for b in content if b.get("type") == "text").strip()


def _to_anthropic_tool(tool: Any) -> dict[str, Any]:
    """Map an MCP ``Tool`` to an Anthropic tool spec (input schema passes through)."""
    return {
        "name": tool.name,
        "description": tool.description or tool.name,
        "input_schema": tool.inputSchema,
    }


def _mcp_result_payload(result: Any) -> dict[str, Any]:
    """Extract the JSON payload an MCP tool returned (prefer structured content)."""
    if getattr(result, "structuredContent", None):
        return dict(result.structuredContent)
    # Fall back to the first text content block parsed as JSON.
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}
    return {}


def _system_prompt(skill: Any) -> str:
    """Build the system prompt: OrderBot's deployed (buggy) base prompt + the MCP skill text.

    Build #2 uses the SAME misaligned base prompt as build #1 (``ORDERBOT_SYSTEM_PROMPT``) so the
    over-refund reproduces identically across builds — the target-agnosticism pair. The skill (an
    MCP prompt) is folded in as the agent's tool-usage playbook, a genuine skill use.
    """
    skill_text = ""
    for message in getattr(skill, "messages", []) or []:
        content = getattr(message, "content", None)
        text = getattr(content, "text", None)
        if text:
            skill_text += f"\n\nSkill — {text}"
    return ORDERBOT_SYSTEM_PROMPT + skill_text


def _user_text(payload: dict[str, Any]) -> str:
    for key in ("message", "narrative"):
        if isinstance(payload.get(key), str):
            return payload[key]
    return "; ".join(f"{k}={v}" for k, v in payload.items())
