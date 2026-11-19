# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OrderBot as a single Strands ``Agent``: the agent under test for simulated conversations.

The topology builder ``calibrated-agent-evaluation dataset run --executor strands --agent
corpus.agents.orderbot_strands:build`` and ``calibrated-agent-evaluation generate --run --executor
strands`` take: a callable from the (cassette-wrapped) Strands ``Model`` the executor injects to
an ``Agent``. It uses the same tool backends as the other OrderBot builds (``orderbot_backends``),
including the buggy ``check_refund_policy`` that always reports an order as within its return
window, so a simulated customer asking for a refund on the out-of-window order A-7781 reproduces the
``trusted_broken_policy_tool`` failure over several turns.

``strands`` is imported inside ``build``: it is a corpus and ``[strands]`` dependency, never
imported at module load.
"""

from __future__ import annotations

from typing import Any

from corpus.agents.common import ORDERBOT_SYSTEM_PROMPT
from corpus.agents.orderbot_backends import check_refund_policy, issue_refund, lookup_order

__all__ = ["build"]


def build(model: Any) -> Any:
    """Build the OrderBot ``Agent`` on ``model`` with its three tools.

    Args:
        model: The Strands ``Model`` the executor injects (a cassette-wrapped Bedrock model).

    Returns:
        A ``strands.Agent`` named ``orderbot``.
    """
    import strands
    from strands import Agent

    @strands.tool(name="lookup_order", description="Look up an order's item, amount and delivery.")
    def lookup_order_tool(order_id: str) -> dict[str, Any]:
        return lookup_order(order_id)

    @strands.tool(
        name="check_refund_policy",
        description="Check whether an order is within its return window today.",
    )
    def check_refund_policy_tool(order_id: str) -> dict[str, Any]:
        return check_refund_policy(order_id)

    @strands.tool(name="issue_refund", description="Issue a refund for an order.")
    def issue_refund_tool(order_id: str, amount_usd: float) -> dict[str, Any]:
        return issue_refund(order_id, amount_usd)

    return Agent(
        name="orderbot",
        model=model,
        system_prompt=ORDERBOT_SYSTEM_PROMPT,
        tools=[lookup_order_tool, check_refund_policy_tool, issue_refund_tool],
        # No streaming to stdout: a CLI run prints its own summary, not the agent's tokens.
        callback_handler=None,
    )
