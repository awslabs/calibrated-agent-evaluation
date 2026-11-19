# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agents under test for the `calibrated-agent-evaluation generate --run` tests, importable as
`cli_agents:<name>`.

`echo` is a real Inspect solver (one model call per sample) that the tests run against Inspect's
offline `mockllm/model`, so the CLI's executor construction path runs end to end with no network.

`support_agent` is a Strands topology builder for the `--multi-turn` tests: the executor hands it
the model it owns and it returns a single real `strands.Agent`, which is the smallest agent that
can hold a conversation. Strands is imported inside the function, so the default offline suite
imports this module without the `[strands]` extra.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.solver import generate

echo = generate()

_SUPPORT_PROMPT = (
    "You are a retail support agent. Answer in at most two sentences. Ask for the order number "
    "if you do not have it, and say plainly whether you can help."
)


def support_agent(model: Any) -> Any:
    """One real Strands agent on the executor's model; the agent under test for a conversation."""
    from strands import Agent

    return Agent(model=model, system_prompt=_SUPPORT_PROMPT, callback_handler=None)
