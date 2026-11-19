# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OrderBot build #3c — a Strands **Agent-as-Tools** composition, emitting Strands-native OTel.

``corpus-companion.md`` deliverable #3c. Strands lets one agent expose another as a tool via
``Agent.as_tool()`` (verified against ``strands-agents==1.45.0`` ``agent/agent.py`` /
``agent/_agent_as_tool.py``). When the parent calls that tool, the sub-agent runs **inside** the
parent's tool execution, so the captured trace nests an ``invoke_agent {child}`` span **under** the
parent's ``execute_tool {child}`` span.

This is the corpus' proof for the trajectory normalizer's ``strands_adapter`` **Agent-as-Tools**
path and its sub-agent discriminator: an ``execute_tool`` span that *wraps an ``invoke_agent``*
is an **agent boundary** -> a child ``Span``, **not** a flattened ``ToolCall``.
A plain leaf tool (no nested agent) stays a ``ToolCall``; this fixture has both, so the normalizer's
discriminator has a real example to be verified against.

Build shape (a genuine Strands application — the agent conforms to Strands, not to Calibrated Agent
Evaluation):
  * a sub-agent ``policy_advisor`` with a real ``@strands.tool`` (``check_refund_policy``), exposed
    to the parent via ``policy_advisor.as_tool(name="policy_advisor")``;
  * a parent ``orderbot`` whose tools are the **sub-agent-as-tool** plus a real leaf tool
    (``issue_refund``) — so the capture carries one ``execute_tool policy_advisor`` (nesting an
    ``invoke_agent policy_advisor``) and one leaf ``execute_tool issue_refund``;
  * the live model calls are recorded/replayed by the custom ``CassetteModel`` **Strands provider**.

Determinism comes from cassette replay + ``temperature=0`` + deterministic tool backends.
"""

from __future__ import annotations

import logging
from typing import Any

from corpus.generate.cassette import Cassette
from corpus.generate.otel_capture import SpanCapture

logger = logging.getLogger("calibrated_agent_evaluation.corpus.orderbot_agent_as_tool")

_MODEL_ID = "global.anthropic.claude-sonnet-4-6"

# A self-contained worked example; the normalized topology (which execute_tool nests an agent) is
# observed FROM the real capture, never assumed before it (corpus-companion "assume nothing").
_TASK = (
    "Customer C-91002 wants a refund for order A-3310 (a $129.00 espresso grinder). "
    "Decide whether to refund and, if so, issue it."
)


def run(
    cassette: Cassette,
    capture: SpanCapture,
    *,
    region: str = "us-west-2",
) -> dict[str, Any]:
    """Run the agent-as-tools OrderBot; Strands emits OTel into ``capture``.

    Returns the parent agent's final answer text. Spans are captured via the global provider
    ``capture`` installs; the caller serializes them to OTLP-JSON.
    """
    # Local imports: strands is a corpus-only dep, never importable from
    # src/calibrated_agent_evaluation.
    import strands
    from strands import Agent
    from strands.models.bedrock import BedrockModel

    from corpus.agents.strands_cassette_model import CassetteModel

    capture.register_global()

    inner = BedrockModel(model_id=_MODEL_ID, region_name=region, temperature=0.0)
    model = CassetteModel(inner, cassette)

    @strands.tool(name="check_refund_policy", description="Check whether an order is refundable.")
    def check_refund_policy(order_id: str) -> dict[str, Any]:
        # Deterministic example backend: this order is genuinely in-window, so the advisor allows
        # the refund (a clean example — the as-tool nesting is what this fixture exercises).
        return {
            "order_id": order_id,
            "days_since_delivery": 6,
            "return_window_days": 30,
            "within_window": True,
            "policy_verdict": "WITHIN_WINDOW_REFUND_ALLOWED",
        }

    @strands.tool(name="issue_refund", description="Issue a refund for an order.")
    def issue_refund(order_id: str, amount_usd: float) -> dict[str, Any]:
        return {
            "refund_id": "RF-44190",
            "order_id": order_id,
            "amount_usd": amount_usd,
            "status": "issued",
        }

    # The sub-agent: a focused policy advisor with its own leaf tool.
    policy_advisor = Agent(
        name="policy_advisor",
        model=model,
        description="Advises whether an order qualifies for a refund under policy.",
        system_prompt=(
            "You are a refund-policy advisor. Use the check_refund_policy tool for the order in "
            "question and report whether a refund is allowed and why."
        ),
        tools=[check_refund_policy],
    )

    # The parent: uses the sub-agent AS A TOOL, plus a real leaf tool. Calling policy_advisor runs
    # the sub-agent inside the parent's tool execution -> nested invoke_agent under execute_tool.
    orderbot = Agent(
        name="orderbot",
        model=model,
        system_prompt=(
            "You are OrderBot, a customer-service refund agent. First consult the policy_advisor "
            "tool to determine whether the order qualifies for a refund. If it does, use the "
            "issue_refund tool to issue it. Then reply to the customer."
        ),
        tools=[policy_advisor.as_tool(name="policy_advisor"), issue_refund],
    )

    result = orderbot(_TASK)
    answer = str(result)
    logger.info("agent-as-tools orderbot finished (%d chars)", len(answer))
    return {"task": _TASK, "answer": answer}
