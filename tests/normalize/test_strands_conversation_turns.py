# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Strands agent run over a conversation keeps every user turn, in order.

Strands emits one ``gen_ai.user.message`` event per user turn of the history the agent was called
with (verified against a live ``strands-agents==1.45.0`` run on Bedrock: a three-message history
came back as user / assistant / user events on the ``invoke_agent`` span). The adapter read the
first event only, so a conversation normalized to its opening line with everything the customer
said afterwards dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.normalize import StrandsSession, normalize


def _event(name: str, text: str) -> dict[str, Any]:
    return {
        "name": name,
        "attributes": [
            {"key": "content", "value": {"stringValue": json.dumps([{"text": text}])}},
        ],
    }


def _conversation_otlp(user_turns: list[str], answer: str) -> dict[str, Any]:
    """One agent span whose events carry a whole conversation, the way a live run emits it."""
    events: list[dict[str, Any]] = [_event("gen_ai.system.message", "You are support.")]
    for index, turn in enumerate(user_turns):
        events.append(_event("gen_ai.user.message", turn))
        if index < len(user_turns) - 1:
            events.append(_event("gen_ai.assistant.message", f"reply {index}"))
    events.append(
        {
            "name": "gen_ai.choice",
            "attributes": [
                {"key": "message", "value": {"stringValue": answer}},
                {"key": "finish_reason", "value": {"stringValue": "end_turn"}},
            ],
        }
    )
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "calibrated_agent_evaluation.trace_id",
                            "value": {"stringValue": "CONV-1"},
                        },
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "name": "invoke_agent support",
                                "spanId": "aa",
                                "traceId": "bb",
                                "startTimeUnixNano": "1",
                                "endTimeUnixNano": "2",
                                "attributes": [
                                    {
                                        "key": "gen_ai.operation.name",
                                        "value": {"stringValue": "invoke_agent"},
                                    },
                                    {
                                        "key": "gen_ai.agent.name",
                                        "value": {"stringValue": "support"},
                                    },
                                ],
                                "events": events,
                            }
                        ]
                    }
                ],
            }
        ]
    }


def _normalize(tmp_path: Path, otlp: dict[str, Any]) -> Any:
    path = tmp_path / "conversation.otlp.json"
    path.write_text(json.dumps(otlp), encoding="utf-8")
    return normalize(StrandsSession(path=path, mode="legacy"))


def test_every_user_turn_of_a_conversation_is_kept_in_order(tmp_path: Path) -> None:
    turns = ["my refund never arrived", "it is order A-7781", "so can you refund it?"]
    trace = _normalize(tmp_path, _conversation_otlp(turns, "Refunded."))

    [agent] = trace.spans
    rendered = json.dumps(agent.input)
    for turn in turns:
        assert turn in rendered, rendered
    # In order, not just present.
    positions = [rendered.index(turn) for turn in turns]
    assert positions == sorted(positions)


def test_a_single_turn_run_is_unchanged(tmp_path: Path) -> None:
    trace = _normalize(tmp_path, _conversation_otlp(["just the one message"], "Done."))
    [agent] = trace.spans
    assert agent.input == {"answer": "just the one message"}
