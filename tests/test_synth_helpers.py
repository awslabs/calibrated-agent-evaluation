# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test doubles for the synthesis suite.

Deterministic fakes that mirror the real injected surfaces (Renderer, Embedder, the
ExecutorBackend shape, ActorSimulator) so the tests assert behavior without network or an LLM.
"""

from __future__ import annotations

import hashlib
from typing import Any

from calibrated_agent_evaluation.schema.types import Trace
from calibrated_agent_evaluation.synth.types import SynthInput

# A sentinel the StubExecutor stamps into every Trace's output, so a test can prove the routed
# output is the EXECUTOR's, never the renderer's.
EXECUTOR_SENTINEL = {"answer": "REAL-AGENT-OUTPUT-FROM-EXECUTOR"}


class FakeRenderer:
    """Deterministic tuple→NL renderer: joins the cell values into a stable sentence."""

    def __init__(self, prefix: str = "input") -> None:
        self.prefix = prefix
        self.calls: list[dict[str, str]] = []

    def render(self, cell: dict[str, str]) -> str:
        self.calls.append(cell)
        parts = "; ".join(f"{k}={v}" for k, v in sorted(cell.items()))
        return f"{self.prefix} for {parts}"


class BlankRenderer:
    """Renderer that emits whitespace-only text — exercises the empty-input refusal."""

    def render(self, cell: dict[str, str]) -> str:
        return "   "


class DuplicateRenderer:
    """Renderer that always emits the SAME text regardless of cell — mode-collapse."""

    def render(self, cell: dict[str, str]) -> str:
        return "identical input every time"


class FakeEmbedder:
    """Deterministic hash-based embedder (mirrors the test embedder). 16-dim SHA-256 vector."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[:16]]


class StubTaskBuilder:
    """Builds a trivial 'task' (the input list itself) — the executor stub reads it back."""

    def __init__(self) -> None:
        self.built: list[list[SynthInput]] = []

    def build(self, inputs: list[SynthInput]) -> Any:
        self.built.append(inputs)
        return inputs


class StubExecutor:
    """An ExecutorBackend stub: turns each SynthInput into a frame-tagged Trace with a real output.

    Emulates the real contract — the OUTPUT is the executor's (EXECUTOR_SENTINEL), never the
    renderer text — and propagates the input's frame/cell onto the Trace.
    """

    def __init__(self) -> None:
        self.run_calls: list[Any] = []

    def run(self, task: Any) -> list[Trace]:
        self.run_calls.append(task)
        inputs = task if isinstance(task, list) else [task]
        return [self._trace(i, idx) for idx, i in enumerate(inputs)]

    @staticmethod
    def _trace(item: SynthInput, idx: int) -> Trace:
        return Trace(
            trace_id=f"SYN-{idx:04d}",
            frame=item.frame,
            ownership="probed",
            input={"query": item.text},
            output=dict(EXECUTOR_SENTINEL),
            taxonomy_cell=item.cell,
            endpoint_fingerprint="stub-executor",
        )


class RaisingExecutor:
    """An ExecutorBackend stub that always raises — exercises error wrapping."""

    def run(self, task: Any) -> list[Trace]:
        raise RuntimeError("boom from the agent")


class MislabelingExecutor:
    """Returns a Trace whose frame/cell DON'T match the input — exercises the frame-mismatch
    rejection."""

    def run(self, task: Any) -> list[Trace]:
        return [
            Trace(
                trace_id="SYN-BAD",
                frame="discovery",
                ownership="probed",
                input={"query": "x"},
                output=dict(EXECUTOR_SENTINEL),
                taxonomy_cell={"unexpected": "cell"},
                endpoint_fingerprint="stub",
            )
        ]


class SpyExecutor:
    """Records whether run was called — proves the generate-only tier never routes."""

    def __init__(self) -> None:
        self.called = False

    def run(self, task: Any) -> list[Trace]:
        self.called = True
        return []


class FakeActorSimulator:
    """Mirrors strands_evals.ActorSimulator: .act() returns the next user turn, .has_next() gates.

    Emits `n_turns` follow-up user messages, then stops. Records agent messages it received so a
    test can prove agent turns came from the real agent, not the simulator.
    """

    def __init__(
        self,
        actor_profile: Any = None,
        initial_query: str = "",
        n_turns: int = 2,
        **_kwargs: Any,
    ) -> None:
        self.actor_profile = actor_profile
        self.initial_query = initial_query
        self._remaining = n_turns
        self.received_agent_messages: list[str] = []

    def has_next(self) -> bool:
        return self._remaining > 0

    def act(self, agent_message: str, **_kwargs: Any) -> str:
        self.received_agent_messages.append(agent_message)
        self._remaining -= 1
        return f"follow-up user turn (saw: {agent_message[:20]})"


class RaisingActorSimulator:
    """ActorSimulator fake whose.act raises — exercises vendor-exception wrapping."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def has_next(self) -> bool:
        return True

    def act(self, agent_message: str, **_kwargs: Any) -> str:
        raise ValueError("simulator internal error")


class StubConversationExecutor:
    """Mirrors ``StrandsExecutor`` for the multi-turn path: one run per agent turn.

    ``run`` takes the executor's own task type (a ``StrandsTask``, whose ``input`` is the
    conversation so far) and returns exactly one frame-tagged Trace, like the real executor does.
    A task of the wrong type is refused, as the real one refuses it. ``histories`` records the
    message list each turn was given, so a test can prove the agent saw the whole conversation.
    """

    def __init__(self, reply: str = "the agent's answer") -> None:
        self.histories: list[list[dict[str, Any]]] = []
        self._reply = reply

    def run(self, task: Any) -> list[Trace]:
        from calibrated_agent_evaluation.execute.strands_executor import StrandsTask

        if not isinstance(task, StrandsTask):
            raise TypeError(f"expected a StrandsTask, got {type(task).__name__}")
        history: list[dict[str, Any]] = (
            [m for m in task.input if isinstance(m, dict)]
            if isinstance(task.input, list)
            else [{"role": "user", "content": [{"text": str(task.input)}]}]
        )
        self.histories.append(history)
        turn = len(self.histories)
        return [
            Trace(
                trace_id=f"CONV-{turn:04d}",
                frame="discovery",
                ownership="probed",
                input={"messages": str(history)},
                output={"answer": f"{self._reply} ({turn})"},
                taxonomy_cell=dict(task.taxonomy_cell),
                endpoint_fingerprint="stub-conversation-executor",
            )
        ]


class ContentlessConversationExecutor(StubConversationExecutor):
    """A conversation executor whose trace carries no text output at all."""

    def run(self, task: Any) -> list[Trace]:
        [trace] = super().run(task)
        return [
            Trace(
                trace_id=trace.trace_id,
                frame=trace.frame,
                ownership=trace.ownership,
                input=trace.input,
                output={},
                taxonomy_cell=trace.taxonomy_cell,
                endpoint_fingerprint=trace.endpoint_fingerprint,
            )
        ]


class _ActorResponseLike:
    """The fields ``ActorResponse`` carries that the borrow reads."""

    def __init__(self, message: str | None, reasoning: str) -> None:
        self.message = message
        self.reasoning = reasoning
        self.stop = message is None


class StructuredActResult:
    """Mirrors ``strands.AgentResult`` with structured output: the user turn is its ``message``.

    ``str()`` returns the whole structured document, the way the real ``AgentResult`` does, so a
    test can tell the difference between reading ``structured_output.message`` and stringifying.
    """

    def __init__(self, message: str | None, *, reasoning: str = "because") -> None:
        self.structured_output = _ActorResponseLike(message=message, reasoning=reasoning)

    def __str__(self) -> str:
        return (
            f'{{"reasoning": "{self.structured_output.reasoning}", '
            f'"message": "{self.structured_output.message}"}}'
        )


class StructuredActorSimulator:
    """``ActorSimulator`` fake returning structured results, stopping with a message-less one."""

    def __init__(
        self,
        actor_profile: Any = None,
        initial_query: str = "",
        n_turns: int = 1,
        **_kwargs: Any,
    ) -> None:
        self.actor_profile = actor_profile
        self.initial_query = initial_query
        self._remaining = n_turns
        self.stop = False

    def has_next(self) -> bool:
        return not self.stop

    def act(self, agent_message: str, **_kwargs: Any) -> StructuredActResult:
        if self._remaining <= 0:
            self.stop = True
            return StructuredActResult(None)
        self._remaining -= 1
        return StructuredActResult(f"and also: {agent_message[:20]}")
