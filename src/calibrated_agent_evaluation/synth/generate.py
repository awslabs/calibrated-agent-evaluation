# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two-step generation engine: the Generate stage.

``generate`` renders a taxonomy frame into frame-tagged :class:`SynthInput`s (universal; works
for any agent, generate-only tier). ``generate_and_run`` additionally routes the inputs
through a real agent via an injected ``ExecutorBackend`` + ``TaskBuilder`` and returns the
executor's validated, frame-tagged :class:`~calibrated_agent_evaluation.schema.types.Trace`s. The
engine is inputs-only: it never fabricates an agent output or a label; a
routed output is always the executor's.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any, Protocol, cast, runtime_checkable

from calibrated_agent_evaluation.sampling.types import FrameResult
from calibrated_agent_evaluation.schema.types import Trace
from calibrated_agent_evaluation.synth.conversation import ConversationHooks, ConversationTurn
from calibrated_agent_evaluation.synth.errors import RepresentativenessWarning, SynthError
from calibrated_agent_evaluation.synth.guards import (
    DEFAULT_DEDUP_THRESHOLD,
    dedup,
    enforce_diversity_floor,
)
from calibrated_agent_evaluation.synth.risk import DEFAULT_RISK_LEXICON, scan_risk_zones
from calibrated_agent_evaluation.synth.types import Embedder, Renderer, SynthInput, TaskBuilder


@runtime_checkable
class _Executor(Protocol):
    """Structural mirror of the ``ExecutorBackend`` (duck-typed to avoid importing execute/).

    We program against the shape, not the class, so ``synth/`` carries no dependency on
    ``execute/``; either the Inspect or the Strands executor satisfies it.
    """

    def run(self, task: Any) -> list[Trace]: ...


def generate(
    frame: FrameResult,
    renderer: Renderer,
    *,
    embedder: Embedder | None = None,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    diversity_floor: int = 0,
    max_regen: int = 5,
    risk_lexicon: frozenset[str] = DEFAULT_RISK_LEXICON,
) -> list[SynthInput]:
    """Render a frame into frame-tagged inputs.

    Two steps: (i) take each ``FrameRow`` cell, (ii) render it via ``renderer``. The
    produced ``SynthInput`` carries the row's ``cell``/``frame``/``weight``.
    Empty/whitespace renderer output is refused. Emits a ``RiskZoneWarning`` when a
    taxonomy value names a risk-zone domain. When an ``embedder`` is given, near-duplicate
    inputs are removed and (if ``diversity_floor > 1``) the per-cell floor is enforced.
    No agent is invoked in this tier.

    Dedup and the floor change how many inputs each cell holds. For a production-representative
    frame the weights are then recomputed so each cell keeps the total weight the frame gave it,
    and the weighted distribution still matches the declared one. If the guards empty a cell,
    no weighting can restore it: every input is retagged ``discovery`` and a
    :class:`~calibrated_agent_evaluation.synth.errors.RepresentativenessWarning` names the cells.
    """
    scan_risk_zones(frame, lexicon=risk_lexicon)

    inputs: list[SynthInput] = []
    for row in frame.rows:
        text = renderer.render(row.cell)
        if not text or not text.strip():
            raise SynthError(f"renderer produced an empty input for cell {row.cell}")
        inputs.append(SynthInput(text=text, cell=row.cell, frame=row.frame, weight=row.weight))

    if embedder is not None:
        rendered = inputs
        inputs = dedup(inputs, embedder, threshold=dedup_threshold)
        if diversity_floor > 1:
            inputs = enforce_diversity_floor(
                inputs,
                embedder,
                renderer,
                floor=diversity_floor,
                threshold=dedup_threshold,
                max_regen=max_regen,
            )
        inputs = _reweight_production(rendered, inputs)
    return inputs


def _cell_key(item: SynthInput) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(item.cell.items()))


def _reweight_production(rendered: list[SynthInput], kept: list[SynthInput]) -> list[SynthInput]:
    """Give each production cell back its declared total weight, or drop the representative tag."""
    declared: defaultdict[tuple[tuple[str, str], ...], float] = defaultdict(float)
    for item in rendered:
        if item.frame == "production-representative":
            declared[_cell_key(item)] += item.weight
    if not declared:
        return kept
    counts = Counter(_cell_key(i) for i in kept if i.frame == "production-representative")
    emptied = [dict(cell) for cell in declared if counts[cell] == 0]
    if emptied:
        warnings.warn(
            f"dedup and the diversity floor left {len(emptied)} production-representative cell(s) "
            f"with no input ({emptied[:5]}), so the inputs can no longer match the declared "
            "distribution; they are retagged 'discovery' and must not feed headline rates. Lower "
            "the dedup threshold or raise the regeneration budget to keep every cell",
            RepresentativenessWarning,
            stacklevel=3,
        )
        return [
            dataclasses.replace(i, frame="discovery")
            if i.frame == "production-representative"
            else i
            for i in kept
        ]
    return [
        dataclasses.replace(i, weight=declared[_cell_key(i)] / counts[_cell_key(i)])
        if i.frame == "production-representative"
        else i
        for i in kept
    ]


def generate_and_run(
    frame: FrameResult,
    renderer: Renderer,
    executor: _Executor | None,
    task_builder: TaskBuilder | None,
    *,
    multi_turn: bool = False,
    turn_task: Callable[[SynthInput, list[dict[str, Any]]], Any] | None = None,
    actor_profile_for: Callable[[SynthInput], Any] | None = None,
    simulator_cls: Callable[..., Any] | None = None,
    max_turns: int = 6,
    conversation_hooks: ConversationHooks | None = None,
    **generate_kwargs: Any,
) -> list[Trace]:
    """Render inputs and route them through the real agent, returning frame-tagged Traces.

    Single-shot: builds a task via ``task_builder`` and runs it through ``executor``; the returned
    traces are validated by the executor and their ``frame``/``taxonomy_cell`` are verified to match
    the originating rows; a mismatch is refused (never silently accepted). The output is
    always the executor's, never the renderer's.

    Multi-turn (``multi_turn=True``): each input drives a simulated conversation. The borrowed
    ``ActorSimulator`` writes the user turns; every agent turn is a fresh executor run over the
    conversation so far, built by ``turn_task(input, history)`` (the task type is the executor's
    concern, so the builder is injected the way ``task_builder`` is). Each input is stored as one
    conversation trace that keeps every turn's spans, and its frame and taxonomy cell are checked
    like any other. ``conversation_hooks`` plugs in an executor that holds a session per
    conversation (its turn task and end hook) and another simulated user.

    Raises :class:`SynthError` if ``executor`` is ``None`` (outputs are never fabricated),
    if the executor raises, or on any borrowed-simulator failure.
    """
    if executor is None:
        raise SynthError(
            "agent outputs require a real ExecutorBackend; Calibrated Agent Evaluation never "
            "fabricates outputs. Pass an executor, or use generate() for the inputs-only tier."
        )

    inputs = generate(frame, renderer, **generate_kwargs)

    if multi_turn:
        return _run_multi_turn(
            inputs,
            executor,
            turn_task=turn_task,
            actor_profile_for=actor_profile_for,
            simulator_cls=simulator_cls,
            max_turns=max_turns,
            hooks=conversation_hooks or ConversationHooks(),
        )

    if task_builder is None:
        raise SynthError("single-shot route-through requires a task_builder")

    # `build_all` lets the builder say how many executor calls its framework needs: one batched
    # Inspect Task holding N Samples, or N StrandsTasks each holding one input. Before this,
    # `generate_and_run` called `build(inputs)` once and ran the single task it got back; so a
    # one-input-per-task builder evaluated `inputs[0]` and silently discarded the rest, while the
    # frame/cell checks passed because the one trace it did return was individually valid.
    build_all = cast(
        "Callable[[list[SynthInput]], list[Any]] | None",
        getattr(task_builder, "build_all", None),
    )
    tasks: list[Any] = list(build_all(inputs)) if build_all else [task_builder.build(inputs)]

    traces: list[Trace] = []
    for task in tasks:
        try:
            traces.extend(executor.run(task))
        except Exception as exc:
            raise SynthError(f"agent run failed: {type(exc).__name__}: {exc}") from exc

    _verify_frame_propagation(traces, inputs)
    return traces


def _run_multi_turn(
    inputs: list[SynthInput],
    executor: _Executor,
    *,
    turn_task: Callable[[SynthInput, list[dict[str, Any]]], Any] | None,
    actor_profile_for: Callable[[SynthInput], Any] | None,
    simulator_cls: Callable[..., Any] | None,
    max_turns: int,
    hooks: ConversationHooks,
) -> list[Trace]:
    """Drive a multi-turn conversation per input; return one conversation trace for each."""
    if (turn_task is None and hooks.turn_task is None) or actor_profile_for is None:
        raise SynthError(
            "multi-turn synthesis requires turn_task (how one agent turn becomes an executor "
            "task) and actor_profile_for (the persona per input)"
        )

    traces: list[Trace] = []
    for index, item in enumerate(inputs, start=1):
        traces.append(
            _converse(
                item,
                executor,
                conversation_id=f"gen-{index}",
                turn_task=turn_task,
                actor_profile=actor_profile_for(item),
                simulator_cls=simulator_cls,
                max_turns=max_turns,
                hooks=hooks,
            )
        )
    _verify_frame_propagation(traces, inputs)
    return traces


def _converse(
    item: SynthInput,
    executor: _Executor,
    *,
    conversation_id: str,
    turn_task: Callable[[SynthInput, list[dict[str, Any]]], Any] | None,
    actor_profile: Any,
    simulator_cls: Callable[..., Any] | None,
    max_turns: int,
    hooks: ConversationHooks,
) -> Trace:
    """One conversation: the simulator's user turns, the executor's agent turns, one trace.

    The agent sees the whole history on every turn, because that is what a multi-turn agent gets in
    production; each turn is a separate executor run. The stored trace keeps every turn's spans in
    turn order and carries the transcript the loop drove as its input: every message in it was
    either written by the simulator or returned by the agent.
    """
    from calibrated_agent_evaluation.synth.borrow.simulator import StrandsSimulatedUser
    from calibrated_agent_evaluation.synth.conversation import (
        assemble_conversation_trace,
        run_simulated_conversation,
    )

    def agent_turn(turn: ConversationTurn, history: list[dict[str, Any]]) -> Trace:
        if hooks.turn_task is not None:
            task = hooks.turn_task(item, history, turn)
        elif turn_task is not None:
            task = turn_task(item, history)
        else:  # pragma: no cover - _run_multi_turn refuses a conversation with neither
            raise SynthError("a conversation needs a turn task")
        produced = executor.run(task)
        if not produced:
            raise SynthError(
                "the agent's turn produced no trace, so the conversation cannot be recorded"
            )
        if len(produced) > 1:
            # One task is one agent turn; keeping one of several traces would drop the others'
            # spans from the conversation without a word.
            raise SynthError(
                f"the agent's turn produced {len(produced)} traces, and a conversation turn must "
                "produce exactly one"
            )
        return produced[0]

    if hooks.user_for_conversation is not None:
        user = hooks.user_for_conversation(item, actor_profile, conversation_id)
    elif hooks.user_for is not None:
        user = hooks.user_for(item, actor_profile)
    else:
        user = StrandsSimulatedUser(
            actor_profile, item.text, max_turns=max_turns, simulator_cls=simulator_cls
        )
    end = hooks.end
    run = run_simulated_conversation(
        user,
        agent_turn,
        conversation_id=conversation_id,
        opening=item.text,
        max_turns=max_turns,
        on_end=None if end is None else (lambda: _call(end, conversation_id)),
    )
    return assemble_conversation_trace(
        run,
        scenario_id=conversation_id,
        goal=str(getattr(actor_profile, "actor_goal", "") or ""),
        assertions=(),
        simulator=hooks.simulator,
        frame=item.frame,
        taxonomy_cell=item.cell,
    )


def _verify_frame_propagation(traces: list[Trace], inputs: list[SynthInput]) -> None:
    """Refuse traces whose frame/cell don't match a synthesized input, or that are too few.

    The frame tag and taxonomy cell must survive the round-trip through the executor unchanged; a
    mismatch means the population was silently altered; refused rather than accepted.

    The count check is structural, not per-adapter: any builder/executor combination that drops
    inputs is caught here even though every trace it *did* return is individually valid, which is
    exactly how the Strands collapse passed every check. Fewer traces than inputs is a drop;
    **more** is not, because one task can legitimately produce several traces (the multi-turn path
    does), so the refusal is one-sided on purpose.
    """
    if len(traces) < len(inputs):
        raise SynthError(
            f"{len(inputs)} synthesized input(s) produced only {len(traces)} trace(s); "
            "the task builder or executor dropped inputs; every input must reach the agent "
            "(a builder whose framework takes one input per task needs build_all)"
        )
    valid_frames = {i.frame for i in inputs}
    valid_cells = [i.cell for i in inputs]
    for trace in traces:
        if trace.frame not in valid_frames:
            raise SynthError(
                f"executor returned a trace tagged frame={trace.frame!r}, not among the "
                f"synthesized frames {sorted(valid_frames)}; frame propagation violated"
            )
        if trace.taxonomy_cell not in valid_cells:
            raise SynthError(
                f"executor returned a trace with taxonomy_cell {trace.taxonomy_cell} "
                f"not among the synthesized cells; taxonomy propagation violated"
            )


def _call(end: Callable[[str], object], conversation_id: str) -> None:
    """Run a conversation's end hook; what it returns (a session stop's result) is not needed."""
    end(conversation_id)
