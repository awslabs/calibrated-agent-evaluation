# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data model for the input-synthesis engine.

Pure frozen dataclasses + injected protocols; no vendor SDK type appears here (the
``synth/`` core is scanned by the methodology-plane boundary test). The vendor-touching
``ActorSimulator`` borrow lives behind ``synth/borrow/``; the guards reach embeddings through
the injected :class:`~calibrated_agent_evaluation.cluster.embed.Embedder` protocol; one embedding
contract shared across clustering + synthesis, no new dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# Re-export the Embedder so callers inject one embedder for both clustering and the synth
# guards. Importing it here does not couple synth to any vendor; Embedder is a Protocol.
from calibrated_agent_evaluation.cluster.embed import Embedder
from calibrated_agent_evaluation.schema.types import Frame

__all__ = ["Embedder", "Renderer", "SynthInput", "TaskBuilder"]


@dataclass(frozen=True)
class SynthInput:
    """One synthesized natural-language input for a taxonomy cell.

    ``text`` is the rendered NL input (non-empty). ``cell``/``frame``/``weight`` are
    propagated unchanged from the originating ``FrameRow`` so every synthesized artifact is
    frame-tagged and taxonomy-attributable. ``turns`` is populated only for multi-turn
    generation; ``None`` for single-shot.
    """

    text: str
    cell: dict[str, str]
    frame: Frame
    weight: float = 1.0
    turns: list[dict[str, str]] | None = None


@runtime_checkable
class Renderer(Protocol):
    """Injectable tuple-to-NL renderer: the render step (caller-provided; no vendor type).

    Mirrors the ``Embedder`` / ``Prefill`` injection pattern: the engine BUILDS the
    two-step pipeline + guards; the caller supplies the concrete NL renderer (a template renderer,
    an LLM renderer, ...). The engine ships only deterministic test fakes.
    """

    def render(self, cell: dict[str, str]) -> str:
        """Render a single taxonomy cell into a natural-language input."""
        ...


@runtime_checkable
class TaskBuilder(Protocol):
    """Injectable task-assembly seam (caller-provided; no vendor type in ``synth/``).

    Turns rendered inputs into a framework task the ``ExecutorBackend`` accepts. The return type
    is ``Any`` because the task type is the executor's vendor concern (an Inspect ``Task``, a
    Strands ``StrandsTask``), NOT synth's; keeping ``synth/`` framework-agnostic and vendor-free.
    The caller supplies the framework-specific builder.

    **Optional extension: ``build_all(inputs) -> list[Any]``.** ``generate_and_run`` calls it when
    present (via ``getattr``, which is why it is documented here rather than declared above;
    declaring it would make every existing builder non-conforming). It returns one task per executor
    call, so the batching decision belongs to the builder: an Inspect ``Task`` already holds one
    ``Sample`` per input and returns a single task, while a ``StrandsTask`` carries exactly one
    input and returns N. A builder whose framework is one-input-per-task and which omits it would
    evaluate only the first input (a defect that once shipped), so the count guard in
    ``_verify_frame_propagation`` refuses that outcome whatever the builder does.
    """

    def build(self, inputs: list[SynthInput]) -> Any:
        """Assemble a framework task from the synthesized inputs."""
        ...
