# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete ``TaskBuilder``s: the caller-side wiring between synthesis and the executor.

``synth.generate_and_run`` injects a ``TaskBuilder`` (``build(inputs) -> Any``) that turns
synthesized :class:`~calibrated_agent_evaluation.synth.types.SynthInput`s into a framework task the
``ExecutorBackend`` runs. The task type is the executor's vendor concern (an Inspect ``Task``, a
``StrandsTask``), so these builders live in ``execute/`` (the vendor-adapter exemption package), not
in ``synth/``.

A builder cannot invent the *agent under test*; the synthesized inputs are only the inputs. So each
builder takes the agent-side piece at construction (an Inspect ``solver``, or a Strands topology
``build`` callable) and assembles the task from the inputs, **threading each input's
``taxonomy_cell``
and ``frame`` through** so the produced Trace stays taxonomy-attributable; the contract between
synthesis and the executor. An empty inputs list is refused.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibrated_agent_evaluation.execute.strands_executor import StrandsTask
from calibrated_agent_evaluation.synth.errors import SynthError
from calibrated_agent_evaluation.synth.types import SynthInput

__all__ = ["InspectTaskBuilder", "StrandsTaskBuilder"]


class InspectTaskBuilder:
    """Assemble an Inspect ``Task`` from synthesized inputs.

    ``solver`` is the agent under test (an Inspect solver/agent). Each ``SynthInput`` becomes a
    ``Sample`` whose ``metadata`` carries the ``taxonomy_cell`` + ``frame`` so they survive
    execution + normalization. Inspect is imported lazily so this module loads without the
    ``[inspect]`` extra.
    """

    def __init__(self, solver: Any) -> None:
        self._solver = solver

    def build_all(self, inputs: list[SynthInput]) -> list[Any]:
        """A single batched task: an Inspect ``Task`` already holds one ``Sample`` per input.

        Batching is the point here (N separate evals would be slower and would scatter N logs) so
        this returns one task, which is what makes ``build_all`` the builder's decision rather than
        ``generate_and_run``'s.
        """
        return [self.build(inputs)]

    def build(self, inputs: list[SynthInput]) -> Any:
        """Build an Inspect ``Task`` with one ``Sample`` per input."""
        if not inputs:
            raise SynthError("InspectTaskBuilder.build received no inputs; refusing an empty task")
        from inspect_ai import Task
        from inspect_ai.dataset import Sample

        samples = [
            Sample(
                input=item.text,
                metadata={"taxonomy_cell": dict(item.cell), "frame": item.frame},
            )
            for item in inputs
        ]
        return Task(dataset=samples, solver=self._solver)


class StrandsTaskBuilder:
    """Assemble a :class:`StrandsTask` from synthesized inputs.

    ``build_topology`` is the Strands topology builder (``Callable[[model], Agent|Graph|Swarm]``)
    the ``StrandsExecutor`` injects the model into. The produced ``StrandsTask`` copies the first
    input's ``taxonomy_cell`` so the normalized Trace is taxonomy-attributable; the load-bearing
    contract between synthesis and the executor: ``synth._verify_frame_propagation`` rejects a
    Trace whose cell doesn't match a synthesized input.
    """

    def __init__(
        self,
        build_topology: Callable[[Any], Any],
        *,
        mode: str = "latest",
    ) -> None:
        self._build_topology = build_topology
        self._mode = mode

    def build_all(self, inputs: list[SynthInput]) -> list[StrandsTask]:
        """One ``StrandsTask`` per input: a StrandsTask carries exactly one ``input`` string.

        This is the fix for the collapse. ``build`` takes a list for protocol compatibility but can
        only represent the first element, so a caller that built once and ran once evaluated 1 of N
        inputs; with no error, because the single returned trace carried a valid frame and cell.
        """
        if not inputs:
            raise SynthError(
                "StrandsTaskBuilder.build_all received no inputs; refusing an empty run"
            )
        return [self.build([item]) for item in inputs]

    def build_conversation_turn(
        self, item: SynthInput, history: list[dict[str, Any]]
    ) -> StrandsTask:
        """Build the task for one agent turn of a multi-turn conversation.

        ``history`` is the conversation so far as Strands messages (``{"role", "content"}``), which
        is what ``Agent.__call__`` takes as a prior-message list (verified against
        ``strands-agents==1.45.0``), so the agent sees every earlier turn the way it would in
        production. The cell and frame ride along as they do single-shot, so the final turn's trace
        stays taxonomy-attributable.
        """
        if not history:
            raise SynthError("a conversation turn needs the history up to it, which was empty")
        return StrandsTask(
            build=self._build_topology,
            input=list(history),
            mode=self._mode,  # type: ignore[arg-type]
            taxonomy_cell=dict(item.cell),
        )

    def build(self, inputs: list[SynthInput]) -> StrandsTask:
        """Build a ``StrandsTask`` for ``inputs[0]``. Prefer :meth:`build_all` for a list.

        Kept single-task because that is what a ``StrandsTask`` is. It no longer silently ignores
        the tail: more than one input here means the caller bypassed ``build_all``, and one input
        would be evaluated while the rest vanished, so it refuses instead.
        """
        if len(inputs) > 1:
            raise SynthError(
                f"StrandsTaskBuilder.build received {len(inputs)} inputs but a StrandsTask "
                "carries exactly one; call build_all(inputs) to run them all"
            )
        if not inputs:
            raise SynthError("StrandsTaskBuilder.build received no inputs; refusing an empty task")
        first = inputs[0]
        return StrandsTask(
            build=self._build_topology,
            input=first.text,
            mode=self._mode,  # type: ignore[arg-type]
            taxonomy_cell=dict(first.cell),
        )
