# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The simulated-conversation engine: stop rules, the conversation trace, and failures."""

from __future__ import annotations

from typing import Any

import pytest

from calibrated_agent_evaluation.schema.types import Span, Trace
from calibrated_agent_evaluation.synth.conversation import (
    ConversationTurn,
    UserTurn,
    assemble_conversation_trace,
    message_text,
    run_simulated_conversation,
)
from calibrated_agent_evaluation.synth.errors import ConversationError, SynthError

OPENING = "I want a refund for order A-7781, the blender just isn't for me."


class ScriptedUser:
    """A simulated user that plays a fixed script of moves and records what it was told."""

    def __init__(self, moves: list[UserTurn]) -> None:
        self._moves = list(moves)
        self.heard: list[str] = []

    def act(self, agent_message: str) -> UserTurn:
        self.heard.append(agent_message)
        if not self._moves:
            return UserTurn(message="still here", stop=False)
        return self._moves.pop(0)


class RaisingUser:
    def __init__(self, on_call: int) -> None:
        self._calls = 0
        self._on_call = on_call

    def act(self, agent_message: str) -> UserTurn:
        self._calls += 1
        if self._calls == self._on_call:
            raise ValueError("actor model throttled")
        return UserTurn(message="go on", stop=False)


class SessionService:
    """Stands in for a runtime that keeps one session per conversation open until it is ended."""

    def __init__(self) -> None:
        self.open_sessions: set[str] = set()

    def end(self, conversation_id: str) -> None:
        self.open_sessions.discard(conversation_id)


class SpanAgent:
    """An agent under test whose every turn is an owned trace with the same two span ids."""

    def __init__(self, service: SessionService | None = None, fail_on: int | None = None) -> None:
        self.service = service
        self.fail_on = fail_on
        self.histories: list[list[dict[str, Any]]] = []

    def __call__(self, turn: ConversationTurn, history: list[dict[str, Any]]) -> Trace:
        if self.service is not None:
            self.service.open_sessions.add(turn.conversation_id)
        self.histories.append(history)
        if turn.turn == self.fail_on:
            raise SynthError("agent run failed: InvokeAgentRuntime returned 500")
        last_user = message_text(history[-1])
        trace_id = f"turn-{turn.turn}"
        return Trace(
            trace_id=trace_id,
            frame="discovery",
            ownership="owned",
            input={"prompt": last_user},
            output={"text": f"reply {turn.turn} to: {last_user}"},
            taxonomy_cell={},
            spans=[
                Span(
                    span_id="root",
                    trace_id=trace_id,
                    agent="orderbot",
                    role="supervisor",
                    parent=None,
                    seq=0,
                    status="completed",
                    input={"prompt": last_user},
                ),
                Span(
                    span_id="policy",
                    trace_id=trace_id,
                    agent="policy_advisor",
                    role="specialist",
                    parent="root",
                    seq=1,
                    status="completed",
                    input={},
                ),
            ],
            captured_at=f"2026-11-05T10:00:0{turn.turn}+00:00",
        )


def _run(user: Any, agent: Any, *, max_turns: int = 4, on_end: Any = None) -> Any:
    return run_simulated_conversation(
        user,
        agent,
        conversation_id="ORD-1001-sim",
        opening=OPENING,
        max_turns=max_turns,
        on_end=on_end,
    )


def _assemble(run: Any, **overrides: Any) -> Trace:
    options: dict[str, Any] = {
        "scenario_id": "ORD-1001-sim",
        "goal": "Get a refund for order A-7781.",
        "assertions": ["Agent denies a refund outside the return window"],
        "simulator": "strands",
        "frame": "discovery",
        "taxonomy_cell": {"feature": "returns"},
    }
    options.update(overrides)
    return assemble_conversation_trace(run, **options)


def test_conversation_ends_when_the_simulated_user_reaches_the_goal() -> None:
    user = ScriptedUser(
        [
            UserTurn(message="But I only used it once, can you make an exception?", stop=False),
            UserTurn(message=None, stop=True, stop_reason="goal_completed"),
        ]
    )
    run = _run(user, SpanAgent())
    assert run.stop_reason == "goal_completed"
    assert [m["role"] for m in run.transcript] == ["user", "assistant", "user", "assistant"]
    assert run.transcript[0]["text"] == OPENING
    assert run.transcript[2]["text"] == "But I only used it once, can you make an exception?"
    assert len(run.turn_traces) == 2
    # The user answered each agent reply, and the agent saw the whole history each turn.
    assert user.heard == [run.transcript[1]["text"], run.transcript[3]["text"]]


def test_conversation_stops_at_max_turns_without_asking_the_user_again() -> None:
    user = ScriptedUser([])
    run = _run(user, SpanAgent(), max_turns=1)
    assert run.stop_reason == "max_turns"
    assert user.heard == []
    assert len(run.turn_traces) == 1
    # A user that never stops is cut off at exactly max_turns agent turns.
    endless = ScriptedUser([])
    run = _run(endless, SpanAgent(), max_turns=3)
    assert run.stop_reason == "max_turns"
    assert sum(1 for m in run.transcript if m["role"] == "assistant") == 3
    assert len(endless.heard) == 2


def test_an_empty_user_message_ends_the_conversation() -> None:
    run = _run(ScriptedUser([UserTurn(message="  ", stop=False)]), SpanAgent())
    assert run.stop_reason == "actor_stopped"
    assert len(run.turn_traces) == 1
    # A stop with no reason is also actor_stopped.
    run = _run(ScriptedUser([UserTurn(message=None, stop=True)]), SpanAgent())
    assert run.stop_reason == "actor_stopped"


def test_a_conversation_becomes_one_trace_with_every_turns_spans() -> None:
    user = ScriptedUser(
        [
            UserTurn(message="Can you make an exception?", stop=False),
            UserTurn(message=None, stop=True, stop_reason="goal_completed"),
        ]
    )
    run = _run(user, SpanAgent())
    trace = _assemble(run, dataset={"dataset_id": "orderbot_sim-abc", "version": "2"})
    assert trace.input["conversation"] == [dict(m) for m in run.transcript]
    assert trace.input["scenario_id"] == "ORD-1001-sim"
    assert trace.input["goal"] == "Get a refund for order A-7781."
    assert trace.input["assertions"] == ["Agent denies a refund outside the return window"]
    assert trace.input["stop_reason"] == "goal_completed"
    assert trace.input["turns"] == 2
    assert trace.input["dataset"] == {"dataset_id": "orderbot_sim-abc", "version": "2"}
    assert trace.output == {"text": run.transcript[-1]["text"]}
    assert [s.span_id for s in trace.spans] == ["t1:root", "t1:policy", "t2:root", "t2:policy"]
    seqs = [s.seq for s in trace.spans]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
    assert {s.trace_id for s in trace.spans} == {trace.trace_id}
    assert trace.captured_at == "2026-11-05T10:00:01+00:00"


def test_span_ids_stay_unique_across_turns() -> None:
    run = _run(ScriptedUser([UserTurn(message="again", stop=False)]), SpanAgent(), max_turns=2)
    trace = _assemble(run)
    assert len({s.span_id for s in trace.spans}) == 4
    parents = {s.span_id: s.parent for s in trace.spans}
    assert parents["t1:policy"] == "t1:root"
    assert parents["t2:policy"] == "t2:root"
    # An unpublished slice adds no dataset key.
    assert "dataset" not in trace.input


def test_a_conversation_with_no_agent_turn_is_refused() -> None:
    run = _run(ScriptedUser([]), SpanAgent(), max_turns=1)
    empty = type(run)(
        conversation_id=run.conversation_id,
        turn_traces=(),
        transcript=run.transcript[:1],
        stop_reason="actor_stopped",
    )
    with pytest.raises(ConversationError, match="no agent turn"):
        _assemble(empty)


def test_a_conversation_trace_id_follows_its_content() -> None:
    first = _assemble(_run(ScriptedUser([]), SpanAgent(), max_turns=2))
    again = _assemble(_run(ScriptedUser([]), SpanAgent(), max_turns=2))
    assert first.trace_id == again.trace_id
    assert first.trace_id.startswith("sim-ORD-1001-sim-")
    other = _assemble(
        _run(ScriptedUser([UserTurn(message="different", stop=False)]), SpanAgent(), max_turns=2)
    )
    assert other.trace_id != first.trace_id


def test_a_failure_mid_conversation_names_the_turn_and_closes_the_session() -> None:
    service = SessionService()
    with pytest.raises(ConversationError) as user_failed:
        _run(RaisingUser(on_call=1), SpanAgent(service), on_end=lambda: service.end("ORD-1001-sim"))
    assert (user_failed.value.side, user_failed.value.turn) == ("user", 2)
    assert user_failed.value.scenario_id == "ORD-1001-sim"
    assert isinstance(user_failed.value.__cause__, ValueError)
    assert service.open_sessions == set()

    with pytest.raises(ConversationError, match="InvokeAgentRuntime returned 500") as agent_failed:
        _run(
            ScriptedUser([]),
            SpanAgent(service, fail_on=3),
            on_end=lambda: service.end("ORD-1001-sim"),
        )
    assert (agent_failed.value.side, agent_failed.value.turn) == ("agent", 3)
    assert service.open_sessions == set()


def test_a_conversation_needs_an_opening_and_a_turn_limit() -> None:
    with pytest.raises(SynthError, match="max_turns of at least 1"):
        _run(ScriptedUser([]), SpanAgent(), max_turns=0)
    with pytest.raises(SynthError, match="empty opening"):
        run_simulated_conversation(
            ScriptedUser([]), SpanAgent(), conversation_id="c", opening=" ", max_turns=2
        )


def test_a_users_goodbye_as_it_stops_is_kept_with_no_agent_turn_after_it() -> None:
    user = ScriptedUser(
        [UserTurn(message="Thanks, that's all.", stop=True, stop_reason="goal_completed")]
    )
    run = _run(user, SpanAgent())
    assert run.stop_reason == "goal_completed"
    assert [m["role"] for m in run.transcript] == ["user", "assistant", "user"]
    assert run.transcript[-1]["text"] == "Thanks, that's all."
    assert len(run.turn_traces) == 1
    trace = _assemble(run)
    assert trace.input["turns"] == 1
    # The trace's output is the agent's last reply, not the customer's goodbye.
    assert trace.output == {"text": run.transcript[1]["text"]}


def test_a_failing_end_hook_does_not_hide_the_conversations_own_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def end() -> None:
        raise RuntimeError("StopRuntimeSession timed out")

    with pytest.raises(ConversationError, match="InvokeAgentRuntime returned 500") as failed:
        _run(ScriptedUser([]), SpanAgent(fail_on=2), on_end=end)
    assert failed.value.side == "agent"
    notes = getattr(failed.value, "__notes__", [])
    assert any("StopRuntimeSession timed out" in note for note in notes)
    assert "StopRuntimeSession timed out" in caplog.text


def test_a_failing_end_hook_after_a_finished_conversation_keeps_the_conversation() -> None:
    ended: list[str] = []

    def end() -> None:
        ended.append("tried")
        raise RuntimeError("could not build the client to stop the session")

    run = _run(ScriptedUser([UserTurn(message=None, stop=True)]), SpanAgent(), on_end=end)
    # The hook ran once, and the finished conversation came back whole despite its failure.
    assert ended == ["tried"]
    assert run.stop_reason == "actor_stopped"
    assert len(run.turn_traces) == 1
    assert _assemble(run).input["turns"] == 1
