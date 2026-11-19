# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-turn tests."""

from __future__ import annotations

import builtins
import re
from typing import Any

import pytest

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.execute.task_builders import StrandsTaskBuilder
from calibrated_agent_evaluation.sampling.types import FrameResult, FrameRow
from calibrated_agent_evaluation.schema.types import Trace
from calibrated_agent_evaluation.synth import SynthError, generate_and_run
from calibrated_agent_evaluation.synth.borrow.simulator import run_conversation
from calibrated_agent_evaluation.synth.types import SynthInput
from test_synth_helpers import (
    ContentlessConversationExecutor,
    FakeActorSimulator,
    FakeRenderer,
    RaisingActorSimulator,
    RaisingExecutor,
    StructuredActorSimulator,
    StructuredActResult,
    StubConversationExecutor,
)


def _frame(cells: list[dict[str, str]], frame: str = "discovery") -> FrameResult:
    rows = [FrameRow(cell=c, frame=frame) for c in cells]  # type: ignore[arg-type]
    return FrameResult(rows=rows, frame=frame)  # type: ignore[arg-type]


def _agent_turn(user_message: str) -> str:
    """A stub 'real agent' — its replies are the agent turns (never authored by the simulator)."""
    return f"AGENT-REPLY to: {user_message[:30]}"


def _turn_task(item: SynthInput, history: list[dict[str, Any]]) -> Any:
    """Build one conversation turn the way the Strands builder does, without the vendor import."""
    return StrandsTaskBuilder(build_topology=lambda _model: None).build_conversation_turn(
        item, history
    )


class TestMultiTurnConversation:
    """User turns come from the simulator, agent turns from the real agent."""

    def test_alternating_turns_from_correct_sources(self) -> None:
        sim = FakeActorSimulator(initial_query="hello", n_turns=2)
        turns = run_conversation(
            actor_profile=None,
            initial_query="hello",
            agent_turn=_agent_turn,
            simulator_cls=lambda **kw: sim,  # inject the already-built fake
        )
        # ≥2 turns; strict alternation user/agent/user/agent...
        assert len(turns) >= 4
        roles = [t["role"] for t in turns]
        assert roles[0] == "user"
        assert roles[1] == "agent"
        # Every agent turn came from _agent_turn (carries the AGENT-REPLY marker).
        agent_texts = [t["text"] for t in turns if t["role"] == "agent"]
        assert all(t.startswith("AGENT-REPLY") for t in agent_texts)
        # The simulator saw the agent messages (proves route-through, not self-authoring).
        assert sim.received_agent_messages
        assert all(m.startswith("AGENT-REPLY") for m in sim.received_agent_messages)

    def test_first_user_turn_is_the_initial_query(self) -> None:
        turns = run_conversation(
            actor_profile=None,
            initial_query="my order broke",
            agent_turn=_agent_turn,
            simulator_cls=lambda **kw: FakeActorSimulator(n_turns=1),
        )
        assert turns[0] == {"role": "user", "text": "my order broke"}


class TestSimulatorReplyText:
    """The next user turn is the simulator's structured ``message``, not its whole document."""

    def test_the_structured_message_is_the_user_turn(self) -> None:
        turns = run_conversation(
            actor_profile=None,
            initial_query="my refund never arrived",
            agent_turn=_agent_turn,
            simulator_cls=lambda **kw: StructuredActorSimulator(n_turns=1),
        )
        user_turns = [t["text"] for t in turns if t["role"] == "user"]
        assert user_turns[0] == "my refund never arrived"
        assert user_turns[1].startswith("and also:")
        assert all("reasoning" not in text for text in user_turns)

    def test_a_stop_without_a_message_adds_no_user_turn(self) -> None:
        turns = run_conversation(
            actor_profile=None,
            initial_query="hello",
            agent_turn=_agent_turn,
            simulator_cls=lambda **kw: StructuredActorSimulator(n_turns=0),
        )
        assert [t["role"] for t in turns] == ["user", "agent"]
        assert str(StructuredActResult(None)) != ""  # the document stringifies; the turn does not


class TestGenerateAndRunMultiTurn:
    """Each agent turn is a real executor run over the conversation so far."""

    def _frame_with_cell(self) -> FrameResult:
        return _frame([{"feature": "returns", "persona": "consumer"}])

    def test_the_agent_answers_every_turn_through_the_executor(self) -> None:
        executor = StubConversationExecutor()
        traces = generate_and_run(
            self._frame_with_cell(),
            FakeRenderer(),
            executor,
            None,  # the single-shot task_builder is not used on the multi-turn path
            multi_turn=True,
            turn_task=_turn_task,
            actor_profile_for=lambda _i: None,
            simulator_cls=lambda **kw: FakeActorSimulator(n_turns=2),
        )

        # One stored conversation per input, answering with the LAST turn's reply.
        assert len(traces) == 1
        assert traces[0].trace_id.startswith("sim-gen-1-")
        assert traces[0].output == {"text": f"the agent's answer ({len(executor.histories)})"}
        assert traces[0].frame == "discovery"
        assert traces[0].taxonomy_cell == {"feature": "returns", "persona": "consumer"}

        # Every turn ran through the executor, each one seeing the whole conversation so far.
        assert len(executor.histories) >= 2
        assert [len(h) for h in executor.histories] == [
            2 * i + 1 for i in range(len(executor.histories))
        ]
        last = executor.histories[-1]
        assert [message["role"] for message in last[:3]] == ["user", "assistant", "user"]
        assert last[0]["content"][0]["text"].startswith("input for feature=returns")
        # The agent answered the newest user turn, so the history ends on one.
        assert last[-1]["role"] == "user"

    def test_the_stored_conversation_carries_every_turn_of_the_transcript(self) -> None:
        executor = StubConversationExecutor(reply="we refunded you")
        traces = generate_and_run(
            self._frame_with_cell(),
            FakeRenderer(),
            executor,
            None,
            multi_turn=True,
            turn_task=_turn_task,
            actor_profile_for=lambda _i: None,
            simulator_cls=lambda **kw: FakeActorSimulator(n_turns=2),
        )
        transcript = traces[0].input["conversation"]
        roles = [turn["role"] for turn in transcript]
        assert roles == ["user", "assistant"] * (len(roles) // 2)
        assert transcript[0]["text"].startswith("input for feature=returns")
        assert all("we refunded you" in t["text"] for t in transcript if t["role"] == "assistant")
        assert all(turn["text"].strip() for turn in transcript)

    def test_the_simulator_reacts_to_the_executors_answer(self) -> None:
        simulator = FakeActorSimulator(n_turns=1)
        executor = StubConversationExecutor(reply="we refunded you")
        generate_and_run(
            self._frame_with_cell(),
            FakeRenderer(),
            executor,
            None,
            multi_turn=True,
            turn_task=_turn_task,
            actor_profile_for=lambda _i: None,
            simulator_cls=lambda **kw: simulator,
        )
        assert simulator.received_agent_messages
        assert all("we refunded you" in m for m in simulator.received_agent_messages)

    def test_max_turns_bounds_the_conversation(self) -> None:
        executor = StubConversationExecutor()
        generate_and_run(
            self._frame_with_cell(),
            FakeRenderer(),
            executor,
            None,
            multi_turn=True,
            turn_task=_turn_task,
            actor_profile_for=lambda _i: None,
            simulator_cls=lambda **kw: FakeActorSimulator(n_turns=99),
            max_turns=3,
        )
        assert len(executor.histories) == 3

    def test_the_strands_simulator_gets_the_runs_turn_limit(self) -> None:
        built: list[dict[str, Any]] = []

        def simulator(**kw: Any) -> FakeActorSimulator:
            built.append(kw)
            return FakeActorSimulator(n_turns=99, **kw)

        generate_and_run(
            self._frame_with_cell(),
            FakeRenderer(),
            StubConversationExecutor(),
            None,
            multi_turn=True,
            turn_task=_turn_task,
            actor_profile_for=lambda _i: None,
            simulator_cls=simulator,
            max_turns=3,
        )
        assert [kw["max_turns"] for kw in built] == [3]

    def test_multiturn_without_its_injections_is_refused(self) -> None:
        with pytest.raises(SynthError, match="turn_task"):
            generate_and_run(
                self._frame_with_cell(),
                FakeRenderer(),
                StubConversationExecutor(),
                None,
                multi_turn=True,
                simulator_cls=lambda **kw: FakeActorSimulator(n_turns=1),
            )

    def test_multiturn_executor_failure_wrapped_and_chained(self) -> None:
        with pytest.raises(SynthError, match="agent run failed") as exc_info:
            generate_and_run(
                _frame([{"feature": "returns"}]),
                FakeRenderer(),
                RaisingExecutor(),
                None,
                multi_turn=True,
                turn_task=_turn_task,
                actor_profile_for=lambda _i: None,
                simulator_cls=lambda **kw: FakeActorSimulator(n_turns=1),
            )
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_a_turn_with_no_text_output_is_refused(self) -> None:
        with pytest.raises(SynthError, match="carried no text output"):
            generate_and_run(
                self._frame_with_cell(),
                FakeRenderer(),
                ContentlessConversationExecutor(),
                None,
                multi_turn=True,
                turn_task=_turn_task,
                actor_profile_for=lambda _i: None,
                simulator_cls=lambda **kw: FakeActorSimulator(n_turns=1),
            )


class TestMultiTurnExtraAbsent:
    """Multi-turn without the [synth] extra raises, naming the install string."""

    def test_missing_strands_evals_raises_naming_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = builtins.__import__

        def _blocked(name: str, *args: object, **kwargs: object) -> object:
            if name == "strands_evals" or name.startswith("strands_evals."):
                raise ImportError("No module named 'strands_evals'")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", _blocked)
        with pytest.raises(SynthError, match=re.escape(install_hint("synth"))):
            # No simulator_cls → forces the lazy real import, which is now blocked.
            run_conversation(actor_profile=None, initial_query="x", agent_turn=_agent_turn)


class TestMultiTurnVendorFailureWrapped:
    """A strands_evals exception is wrapped in SynthError (chained), never escapes raw."""

    def test_simulator_act_raise_wrapped_and_chained(self) -> None:
        with pytest.raises(SynthError, match="input simulator failed") as exc_info:
            run_conversation(
                actor_profile=None,
                initial_query="x",
                agent_turn=_agent_turn,
                simulator_cls=RaisingActorSimulator,
            )
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)


class _SpanConversationExecutor(StubConversationExecutor):
    """The stub conversation executor, with each turn's trace carrying a one-span tree."""

    def run(self, task: Any) -> list[Trace]:
        from calibrated_agent_evaluation._model import replace
        from calibrated_agent_evaluation.schema.types import Span

        [trace] = super().run(task)
        span = Span(
            span_id="agent",
            trace_id=trace.trace_id,
            agent="orderbot",
            role="supervisor",
            parent=None,
            seq=0,
            status="completed",
            input={"turn": len(self.histories)},
        )
        return [replace(trace, ownership="owned", spans=[span], endpoint_fingerprint=None)]


class _EmptyOnSecondTurnExecutor(StubConversationExecutor):
    def run(self, task: Any) -> list[Trace]:
        produced = super().run(task)
        return [] if len(self.histories) == 2 else produced


def test_a_generated_conversation_keeps_every_turns_spans() -> None:
    executor = _SpanConversationExecutor()
    [trace] = generate_and_run(
        _frame([{"feature": "returns"}]),
        FakeRenderer(),
        executor,
        None,
        multi_turn=True,
        turn_task=_turn_task,
        actor_profile_for=lambda _i: None,
        simulator_cls=lambda **kw: FakeActorSimulator(n_turns=5),
        max_turns=3,
    )
    assert len(executor.histories) == 3
    assert [s.span_id for s in trace.spans] == ["t1:agent", "t2:agent", "t3:agent"]
    assert [s.input for s in trace.spans] == [{"turn": 1}, {"turn": 2}, {"turn": 3}]
    assert trace.input["turns"] == 3
    assert trace.input["stop_reason"] == "max_turns"


def test_an_empty_turn_result_is_refused_naming_the_turn() -> None:
    from calibrated_agent_evaluation.synth.errors import ConversationError

    with pytest.raises(ConversationError, match="produced no trace") as exc_info:
        generate_and_run(
            _frame([{"feature": "returns"}]),
            FakeRenderer(),
            _EmptyOnSecondTurnExecutor(),
            None,
            multi_turn=True,
            turn_task=_turn_task,
            actor_profile_for=lambda _i: None,
            simulator_cls=lambda **kw: FakeActorSimulator(n_turns=5),
        )
    assert isinstance(exc_info.value, SynthError)
    assert (exc_info.value.side, exc_info.value.turn) == ("agent", 2)


class _Structured:
    def __init__(self, message: str | None, *, stop: bool, stop_reason: str | None) -> None:
        self.message = message
        self.stop = stop
        self.stop_reason = stop_reason


class _Result:
    def __init__(self, structured: _Structured) -> None:
        self.structured_output = structured


class _OneShotSimulator:
    """Mirrors ``ActorSimulator``: one structured result, then ``has_next()`` is false."""

    def __init__(self, result: _Result, **_kwargs: Any) -> None:
        self._result = result
        self._done = False

    def has_next(self) -> bool:
        return not self._done

    def act(self, agent_message: str, **_kwargs: Any) -> _Result:
        self._done = True
        return self._result


def test_the_strands_simulated_user_keeps_the_simulators_stop_reason() -> None:
    from calibrated_agent_evaluation.synth.borrow.simulator import StrandsSimulatedUser

    done = _Result(_Structured(None, stop=True, stop_reason="goal_completed"))
    user = StrandsSimulatedUser(None, "hi", simulator_cls=lambda **kw: _OneShotSimulator(done))
    move = user.act("Here is your refund.")
    assert (move.message, move.stop, move.stop_reason) == (None, True, "goal_completed")
    # Once the simulator has nothing next, the user stops without asking it again.
    assert user.act("Anything else?").stop


def test_an_unknown_simulator_stop_reason_is_read_as_actor_stopped() -> None:
    from calibrated_agent_evaluation.synth.borrow.simulator import StrandsSimulatedUser

    odd = _Result(_Structured(None, stop=True, stop_reason="tired"))
    user = StrandsSimulatedUser(None, "hi", simulator_cls=lambda **kw: _OneShotSimulator(odd))
    assert user.act("ok").stop_reason == "actor_stopped"


def test_a_turn_with_two_traces_is_refused() -> None:
    class TwoTraces(StubConversationExecutor):
        def run(self, task: Any) -> list[Trace]:
            return super().run(task) * 2

    with pytest.raises(SynthError, match="produced 2 traces"):
        generate_and_run(
            _frame([{"feature": "returns"}]),
            FakeRenderer(),
            TwoTraces(),
            None,
            multi_turn=True,
            turn_task=_turn_task,
            actor_profile_for=lambda _i: None,
            simulator_cls=lambda **kw: FakeActorSimulator(n_turns=1),
        )
