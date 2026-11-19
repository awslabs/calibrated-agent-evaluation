# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedup + diversity guards for the synthesis engine.

Pure computation over the injected :class:`~calibrated_agent_evaluation.cluster.embed.Embedder` (no
new dependency, no vendor type). ``dedup`` removes near-duplicate inputs (embedding cosine distance
below a threshold); ``enforce_diversity_floor`` guarantees a per-cell minimum of mutually-distinct
inputs, regenerating candidates within a bounded budget and warning explicitly on short-fall
rather than silently under-delivering (conventions: "no silent truncation").
"""

from __future__ import annotations

import math
import warnings

from calibrated_agent_evaluation.synth.errors import DiversityFloorWarning
from calibrated_agent_evaluation.synth.types import Embedder, Renderer, SynthInput

# Below this cosine distance two inputs count as near-duplicates. 0.0 = identical
# direction; callers tune per embedder geometry. Conservative default: only very-close pairs.
DEFAULT_DEDUP_THRESHOLD = 0.05


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance ``1 - cos`` in ``[0, 2]``; ``0`` for identical direction."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        # A zero vector has no direction; treat as maximally distant so it is never silently
        # merged away (no silent failure). Callers should not feed empty embeddings.
        return 1.0
    return 1.0 - (dot / (na * nb))


def dedup(
    inputs: list[SynthInput],
    embedder: Embedder,
    *,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> list[SynthInput]:
    """Remove near-duplicate inputs, keeping one representative per group.

    Two inputs are near-duplicates when their embedding cosine distance is ``< threshold``. The
    first occurrence in input order is the kept representative. ``threshold <= 0`` keeps every
    input (no pair is strictly below a non-positive threshold), so dedup is opt-out by config.
    """
    if not inputs:
        return []
    vectors = embedder.embed([i.text for i in inputs])
    kept: list[SynthInput] = []
    kept_vectors: list[list[float]] = []
    for item, vec in zip(inputs, vectors, strict=True):
        if any(_cosine_distance(vec, kv) < threshold for kv in kept_vectors):
            continue
        kept.append(item)
        kept_vectors.append(vec)
    return kept


def enforce_diversity_floor(
    inputs: list[SynthInput],
    embedder: Embedder,
    renderer: Renderer,
    *,
    floor: int,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    max_regen: int = 5,
) -> list[SynthInput]:
    """Guarantee ``floor`` mutually-distinct inputs per taxonomy cell.

    For each cell, keeps its deduped inputs and, while it holds fewer than ``floor`` distinct
    inputs, re-renders that cell up to ``max_regen`` times to add new distinct inputs. If the
    floor still cannot be met, emits a :class:`DiversityFloorWarning` naming the cell and the
    achieved count (never silently under-delivers). ``floor <= 1`` is a no-op guard.
    """
    if floor <= 1:
        return inputs

    by_cell: dict[tuple[tuple[str, str], ...], list[SynthInput]] = {}
    order: list[tuple[tuple[str, str], ...]] = []
    for item in inputs:
        key = tuple(sorted(item.cell.items()))
        if key not in by_cell:
            by_cell[key] = []
            order.append(key)
        by_cell[key].append(item)

    result: list[SynthInput] = []
    for key in order:
        cell_inputs = dedup(by_cell[key], embedder, threshold=threshold)
        attempts = 0
        while len(cell_inputs) < floor and attempts < max_regen:
            attempts += 1
            cell = dict(key)
            candidate_text = renderer.render(cell)
            template = by_cell[key][0]
            candidate = SynthInput(
                text=candidate_text,
                cell=template.cell,
                frame=template.frame,
                weight=template.weight,
            )
            cell_inputs = dedup([*cell_inputs, candidate], embedder, threshold=threshold)
        if len(cell_inputs) < floor:
            warnings.warn(
                f"cell {dict(key)} yielded {len(cell_inputs)} distinct input(s); "
                f"diversity floor is {floor} (exhausted {max_regen} regeneration attempts)",
                DiversityFloorWarning,
                stacklevel=2,
            )
        result.extend(cell_inputs)
    return result
