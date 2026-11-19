# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OrderBot + ClaimsCrew as Inspect AI tasks — mints the frozen ``.eval`` logs.

`corpus-companion.md` §3 / deliverable #2: the corpus needs real Inspect ``.eval`` logs
(``fixtures/raw/inspect/ORD-1001.eval``, ``CLM-4471.eval``, plus the ORD-1002 clean control). These
are produced by a genuine Inspect ``eval()`` run — a real ``Task`` with real ``@tool``s solved by
the built-in ``react`` agent — so the frozen ``.eval`` carries the authentic
``ModelEvent``/``ToolEvent`` transcript the trajectory normalizer's ``inspect_adapter`` reads via
``event_tree`` (this is also the executor's ``.eval`` path).

The model runs through the ``cassette/...`` Inspect provider (``inspect_cassette_provider``) so the
eval records once against live Bedrock and replays offline/deterministically. Tools are
deterministic example backends (the same naive-extractor bug for intake); the transcript is genuine.

Run via ``corpus/generate/record.py`` (which configures the cassette + log dir). Importing this
module registers the tasks and the ``cassette`` provider with Inspect.
"""

from __future__ import annotations

from typing import Any

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.tool import Tool, ToolDef

# Importing the provider module registers the `cassette/...` Inspect model provider via @modelapi
# (side-effect import — the decorator runs at import time).
import corpus.agents.inspect_cassette_provider  # noqa: F401
from corpus.agents import orderbot_backends
from corpus.agents.common import ORDERBOT_SYSTEM_PROMPT
from corpus.generate.scenarios import AgentStep, Scenario, load_scenario

# ---------------------------------------------------------------------------------------------
# Tools — the example apps' real backends. OrderBot tools delegate to `orderbot_backends` (whose
# check_refund_policy carries the genuine over-refund bug), shared with the OTel/MCP builds. The
# ClaimsCrew tools serve scenario backend data (intake's extractor genuinely returns
# occupancy_status=null — the naive-extractor bug). Both are real example tool implementations.
# ---------------------------------------------------------------------------------------------


def _scenario_tool(step: AgentStep, tool_name: str) -> Tool:
    """Build an Inspect tool for ``tool_name`` backed by the example app's real tool implementation.

    Uses ``ToolDef`` with an explicit description + per-parameter descriptions (Inspect requires
    both), so the tool exposes the real parameter names and the model calls it naturally. OrderBot
    tools call ``orderbot_backends`` (the shared buggy backend); ClaimsCrew tools return the
    scenario's backend data (where intake's ``occupancy_status=null`` bug lives).
    """
    tool_step = next(t for t in step.tool_steps if t.name == tool_name)
    param_names = list(tool_step.args)
    is_orderbot = tool_name in orderbot_backends.ORDERBOT_TOOLS
    scenario_result = dict(tool_step.result)

    async def execute(**kwargs: Any) -> dict[str, Any]:
        if is_orderbot:
            return orderbot_backends.call_tool(tool_name, kwargs)
        return scenario_result

    execute.__signature__ = _signature_for(param_names)  # type: ignore[attr-defined]
    return ToolDef(
        execute,
        name=tool_name,
        description=f"{tool_name} for the {step.agent} agent",
        parameters={name: f"the {name} argument" for name in param_names},
    ).as_tool()


def _signature_for(param_names: list[str]) -> Any:
    import inspect

    params = [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=str)
        for name in param_names
    ]
    return inspect.Signature(params, return_annotation=dict)


def _build_task(trace_id: str, instructions: str) -> Task:
    scenario = load_scenario(trace_id)
    sample = Sample(
        input=_sample_input(scenario),
        metadata={"trace_id": trace_id, "taxonomy_cell": scenario.taxonomy_cell},
    )
    agent = react(
        name="orderbot" if trace_id.startswith("ORD") else "claims",
        prompt=instructions,
        tools=_all_tools(scenario),
    )
    return Task(dataset=[sample], solver=agent)


def _all_tools(scenario: Scenario) -> list[Any]:
    # OrderBot: the single agent's three tools. ClaimsCrew: every specialist's tool (the react agent
    # drives them in one transcript — the .eval exercises multi-tool reconstruction either way).
    # Dedupe by tool name: Bedrock Converse rejects duplicate tool names in toolConfig.
    tools: list[Any] = []
    seen: set[str] = set()
    for step in scenario.steps:
        for t in step.tool_steps:
            if t.name not in seen:
                seen.add(t.name)
                tools.append(_scenario_tool(step, t.name))
    return tools


def _sample_input(scenario: Scenario) -> str:
    payload = scenario.trace_input
    for key in ("message", "narrative"):
        if isinstance(payload.get(key), str):
            return payload[key]
    return "; ".join(f"{k}={v}" for k, v in payload.items())


# OrderBot's deployed (buggy, permissive) prompt — the same misalignment as the OTel/MCP builds, so
# the over-refund reproduces here too. Shared from common.py to keep one source of the bug.
_ORDERBOT_PROMPT = ORDERBOT_SYSTEM_PROMPT
_CLAIMS_PROMPT = (
    "You are an insurance claims adjudicator. Use extract_claim_fields, lookup_policy_exclusions, "
    "score_fraud_risk, and issue_payment as needed, then state the final decision."
)


@task
def orderbot_fail() -> Task:
    """ORD-1001 — out-of-window over-refund (the headline single-agent failure)."""
    return _build_task("ORD-1001", _ORDERBOT_PROMPT)


@task
def orderbot_clean() -> Task:
    """ORD-1002 — in-window refund, clean PASS (the control)."""
    return _build_task("ORD-1002", _ORDERBOT_PROMPT)


@task
def claimscrew_fail() -> Task:
    """CLM-4471 — claim wrongly paid; intake drops the vacancy signal."""
    return _build_task("CLM-4471", _CLAIMS_PROMPT)
