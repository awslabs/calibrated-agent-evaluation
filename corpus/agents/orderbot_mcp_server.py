# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A real stdio MCP server exposing OrderBot's tools + a skill (OrderBot build #2's tool plane).

`corpus-companion.md` deliverable #2 specifies OrderBot #2 as "Anthropic-SDK + stdio-MCP + skills".
This module is the **MCP server half**: a genuine ``FastMCP`` server (the real ``mcp`` package) the
agent process spawns over stdio and calls via the MCP protocol — not an in-process shim. It exposes
the three order tools (``lookup_order`` / ``check_refund_policy`` / ``issue_refund``) and one
**skill** (an MCP *prompt*, ``refund_policy_skill``), which is how a skill surfaces over MCP.

The tools return deterministic results lifted from the scenario oracle, with the defect seam
applied, so build #2 reproduces the same worked example as build #1 — the target-agnosticism pair.
The server is launched as ``python -m corpus.agents.orderbot_mcp_server <trace_id>``; the agent
connects to it as a stdio MCP client.

Run directly only as a subprocess of the agent; it speaks MCP on stdio (no human-facing output).
"""

from __future__ import annotations

import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from corpus.agents import orderbot_backends


def build_server(trace_id: str) -> FastMCP:
    """Build a FastMCP server exposing OrderBot's real tool backends over stdio.

    The tools delegate to ``orderbot_backends`` — the SAME implementations the other OrderBot builds
    use — so ``check_refund_policy``'s genuine over-refund bug rides this build too (build #2 is the
    target-agnosticism twin). ``trace_id`` is accepted for symmetry with the other agents; the
    backend's behavior is driven by the order id the agent passes, not the scenario.
    """
    _ = trace_id  # tools are backend-driven; trace_id kept for a uniform launcher signature
    server = FastMCP(
        name="orderbot-tools",
        instructions=(
            "OrderBot's order-management tools. Call lookup_order, then check_refund_policy, "
            "then issue_refund for an approved refund."
        ),
    )

    @server.tool(description="Look up an order by id (item, amount, delivery date, window).")
    def lookup_order(order_id: str) -> dict[str, Any]:
        return orderbot_backends.lookup_order(order_id)

    @server.tool(description="Check whether an order is within its refund window today.")
    def check_refund_policy(order_id: str, today: str) -> dict[str, Any]:
        return orderbot_backends.check_refund_policy(order_id, today)

    @server.tool(description="Issue a refund of the given amount for an order.")
    def issue_refund(order_id: str, amount_usd: float) -> dict[str, Any]:
        return orderbot_backends.issue_refund(order_id, amount_usd)

    # A "skill" over MCP surfaces as a prompt template the agent can pull in — the refund-policy
    # playbook. This is the genuine MCP mechanism for a reusable agent skill.
    @server.prompt(name="refund_policy_skill", description="OrderBot's refund-handling playbook.")
    def refund_policy_skill() -> str:
        return (
            "Refund handling: 1) look up the order; 2) check the refund policy for today's date; "
            "3) only issue a refund when the policy says the order is within its return window. "
            "State the policy verdict in your final reply."
        )

    return server


def main(argv: list[str]) -> int:
    trace_id = argv[1] if len(argv) > 1 else "ORD-1001"
    build_server(trace_id).run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
