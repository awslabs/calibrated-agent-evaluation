# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from opentelemetry import trace

tracer = trace.get_tracer("orderbot")


def lookup_order(order_id: str) -> dict:
    return {"order_id": order_id, "status": "delivered", "days_since_delivery": 41}


def run_once() -> None:
    request = {"request": "Refund order A-1001, it arrived broken."}
    with tracer.start_as_current_span(
        "invoke_agent orderbot",
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "orderbot",
            "gen_ai.input.content": json.dumps(request),
        },
    ) as agent_span:
        args = {"order_id": "A-1001"}
        with tracer.start_as_current_span(
            "execute_tool lookup_order",
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "lookup_order",
                "gen_ai.tool.call.arguments": json.dumps(args),
            },
        ) as tool_span:
            result = lookup_order(**args)
            tool_span.set_attribute("gen_ai.tool.call.result", json.dumps(result))
        answer = {"decision": "refund_approved", "amount": 59.0}
        agent_span.set_attribute("gen_ai.output.content", json.dumps(answer))
