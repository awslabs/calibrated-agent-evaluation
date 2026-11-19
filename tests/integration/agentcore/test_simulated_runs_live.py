# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A simulated conversation against a real SigV4 AgentCore runtime, judged by GoalSuccessRate.

Like every test here, it runs only with ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE=1`` and as the
live-test role (see ``conftest.py``), and every client it builds carries the run's billed-call
budget. It needs the runtime variables the runtime executor's live test reads, and is skipped,
naming the unset ones, without them:

- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_RUNTIME_ARN``: a runtime that accepts AWS IAM (SigV4)
  calls; its ``DEFAULT`` endpoint is invoked twice, two billed calls.
- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_PAYLOAD_FORMAT``: the named payload format the runtime
  reads.
- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_QUESTION``: the conversation's opening message.

The customer's second message is scripted: the live-test role grants no Bedrock model access, and
what this test proves against the real service is the conversation session (both turns on one
runtime session, stopped once) and the session-level judge (one ``Evaluate`` call, a billed call,
whose one result maps to the conversation). The simulated users are tested offline against their
installed SDK surfaces. No ARN or account ID is written here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.agentcore]

RUNTIME_ARN = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_RUNTIME_ARN"
PAYLOAD_FORMAT = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_PAYLOAD_FORMAT"
QUESTION = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_QUESTION"
_QUALIFIER = "DEFAULT"
_FOLLOW_UP = "Thanks. Can you explain that answer more simply, in one sentence?"
_JUDGE_FILE = (
    Path(__file__).resolve().parents[3] / "fixtures" / "judge-files" / "agentcore-goal-success.json"
)


def _required(*names: str) -> list[str]:
    values = [os.environ.get(name, "").strip() for name in names]
    missing = [name for name, value in zip(names, values, strict=True) if not value]
    if missing:
        pytest.skip(f"set {', '.join(missing)} to run this live test")
    return values


@pytest.fixture(autouse=True)
def _no_cassette_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", raising=False)


class _ScriptedUser:
    """Sends one scripted follow-up, then stops with the goal met."""

    def __init__(self) -> None:
        self.calls = 0

    def act(self, agent_message: str) -> Any:
        from calibrated_agent_evaluation.synth.conversation import UserTurn

        self.calls += 1
        if self.calls == 1:
            return UserTurn(message=_FOLLOW_UP, stop=False)
        return UserTurn(message=None, stop=True, stop_reason="goal_completed")


class _OperationCount:
    """A before-call handler that counts the runtime invocations and session stops sent.

    A before-call handler sees the serialized request, where the session id is a header, so the
    sessions themselves are read from the executor's run records instead.
    """

    def __init__(self) -> None:
        self.invocations = 0
        self.stops = 0

    def __call__(self, **kwargs: Any) -> None:
        name = kwargs["model"].name
        if name == "InvokeAgentRuntime":
            self.invocations += 1
        elif name == "StopRuntimeSession":
            self.stops += 1


def test_a_live_conversation_runs_on_one_session_and_is_judged_as_one(
    live_budget: Any, tmp_path: Path
) -> None:
    runtime_arn, payload_format, question = _required(RUNTIME_ARN, PAYLOAD_FORMAT, QUESTION)
    import calibrated_agent_evaluation.agentcore as agentcore
    from calibrated_agent_evaluation.agentcore import (
        AgentCoreRuntimeExecutor,
        AgentCoreRuntimeTask,
        AwsClients,
        RuntimeTarget,
    )
    from calibrated_agent_evaluation.agentcore.evaluate import run_agentcore_judge
    from calibrated_agent_evaluation.agentcore.judge import (
        build_agentcore_judge,
        load_agentcore_judge_file,
    )
    from calibrated_agent_evaluation.synth.conversation import (
        assemble_conversation_trace,
        run_simulated_conversation,
    )

    reader_type = getattr(agentcore, "AgentCoreSpanReader", None)
    if reader_type is None:
        pytest.skip(
            "needs the AgentCore span reader, "
            "calibrated_agent_evaluation.agentcore.AgentCoreSpanReader"
        )
    target = RuntimeTarget(
        runtime_arn=runtime_arn, qualifier=_QUALIFIER, payload_format=payload_format
    )
    calls = _OperationCount()
    conversation_id = "live-simulated-run"
    stop_results: list[Any] = []

    with (
        AwsClients.from_profile(region=target.region, handlers=[live_budget, calls]) as clients,
        AgentCoreRuntimeExecutor(
            target, span_reader=reader_type(clients), clients=clients, log_dir=tmp_path
        ) as executor,
    ):

        def agent_turn(turn: Any, history: list[dict[str, Any]]) -> Any:
            prompt = history[-1]["content"][0]["text"]
            [trace] = executor.run(AgentCoreRuntimeTask(prompt=prompt, conversation=turn))
            return trace

        run = run_simulated_conversation(
            _ScriptedUser(),
            agent_turn,
            conversation_id=conversation_id,
            opening=question,
            max_turns=3,
            on_end=lambda: stop_results.append(executor.end_conversation(conversation_id)),
        )
        conversation = assemble_conversation_trace(
            run,
            scenario_id=conversation_id,
            goal=f"Get a clear answer to: {question}",
            assertions=["The agent answers the user's question"],
            simulator="strands",
            frame="discovery",
            taxonomy_cell={},
        )
        judge = build_agentcore_judge(load_agentcore_judge_file(_JUDGE_FILE), clients=clients)
        labels = run_agentcore_judge(
            judge, [conversation], clients=clients, confirm=lambda _estimate: True
        )
        records = list(executor.run_records)

    # Both turns went to one runtime session, held open between them and stopped once at the end.
    assert calls.invocations == 2
    assert len(records) == 2
    [session] = {record.session_id for record in records}
    assert session.startswith("calibrated-agent-evaluation-")
    assert [record.session_stop for record in records] == ["held_open", "held_open"]
    assert stop_results == ["stopped"]
    assert calls.stops == 1
    assert run.stop_reason == "goal_completed"
    assert {span.span_id.split(":", 1)[0] for span in conversation.spans} == {"t1", "t2"}
    # One session-level verdict, filed under the conversation's trace.
    [label] = labels
    assert label.trace_id == conversation.trace_id
    assert label.span_id is None
    assert label.outcome in ("PASS", "FAIL")
    assert f"[fingerprint={judge.fingerprint}]" in label.critique
