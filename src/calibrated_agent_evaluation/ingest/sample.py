# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seeded random sampling of an ingest population, and the record that makes it reproducible.

A headline rate may only be computed over a production-representative frame, and a file dump or a
Langfuse project is not one by itself: whoever exported it chose what went in. Drawing a seeded
simple random sample of the whole population, and recording the seed and the population size, is
what earns the tag. :func:`draw_sample` is the draw; :class:`SampleRecord` is the evidence, written
by ``calibrated-agent-evaluation ingest --sample``.

Sampling within one taxonomy cell at a time (``--cell``) is a stratified draw: each cell is its own
population with its own record, so a reader can see every stratum's size and seed.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict, Field

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._provenance import ProvenanceStamp
from calibrated_agent_evaluation._recordstore import RecordStore
from calibrated_agent_evaluation.ingest.errors import IngestError
from calibrated_agent_evaluation.schema.types import Frame

__all__ = [
    "SAMPLE_SCHEMA_VERSION",
    "SampleRecord",
    "draw_sample",
    "dump_samples",
    "load_samples",
]

SAMPLE_SCHEMA_VERSION = "1.0"


class SampleRecord(CalibratedAgentEvaluationModel):
    """How a sampled ingest drew its traces, so the draw can be repeated exactly.

    ``population_size`` is how many candidates the draw chose from, ``sample_size`` how many it
    kept, and the seed rides on ``provenance.seed``. ``drawn`` names the population members the
    draw selected (normalized trace ids for a store pull, source paths for files); ``trace_ids``
    names the traces that then landed in the store, which is shorter than ``drawn`` exactly when a
    sampled source was refused or skipped, and the ingest report says which. ``cell`` is the
    taxonomy cell the population was restricted to, empty when the whole population was in scope,
    and
    ``excluded_unknown_cell`` counts candidates left out of that stratum because their cell was
    unknown. ``frame`` is the frame the sampled traces were stored with: a seeded simple random
    sample of the population is production-representative, which is the point of drawing one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = IngestError

    sample_id: str
    source: str
    population_size: int
    sample_size: int
    drawn: tuple[str, ...]
    trace_ids: tuple[str, ...]
    frame: Frame = "production-representative"
    cell: dict[str, str] = Field(default_factory=dict)
    excluded_unknown_cell: int = 0
    schema_version: str = SAMPLE_SCHEMA_VERSION
    provenance: ProvenanceStamp


def draw_sample(population: Sequence[str], *, size: int, seed: int) -> list[str]:
    """Draw ``size`` ids from ``population`` without replacement, reproducibly.

    The population is sorted before the draw, so the same ids and seed give the same sample
    whatever order the store listed them in. The returned ids are in sorted order too.

    Args:
        population: Every candidate id, in any order; duplicates are collapsed.
        size: How many to draw.
        seed: The seed; the same seed and population always draw the same sample.

    Returns:
        The drawn ids, sorted.

    Raises:
        IngestError: ``size`` is not positive, the population is empty, or the population is
            smaller than ``size`` (a short draw is not a smaller sample, it is a different
            population, so it is refused rather than quietly reduced).
    """
    if size < 1:
        raise IngestError(f"a sample needs at least one trace, got {size}")
    candidates = sorted(set(population))
    if not candidates:
        raise IngestError(
            "nothing to sample: the population is empty. Widen the scope (drop --cell, or point "
            "at the project or files that hold the traces)"
        )
    if size > len(candidates):
        raise IngestError(
            f"asked for a sample of {size} but the population holds {len(candidates)} trace(s); "
            "a short draw would be a different population, so it is refused. Lower the sample "
            "size or widen the scope"
        )
    rng = random.Random(seed)  # noqa: S311  # nosec B311 - a seeded statistical draw, not crypto
    return sorted(rng.sample(candidates, size))


_STORE: RecordStore[SampleRecord] = RecordStore(
    SampleRecord,
    stem="sample_record",
    schema_version=SAMPLE_SCHEMA_VERSION,
    error=IngestError,
    noun="sample record",
    rerun="calibrated-agent-evaluation ingest --sample",
)


def dump_samples(records: Sequence[SampleRecord], path: str | Path, fmt: str = "csv") -> None:
    """Write ``records`` to ``sample_record.{csv,json}`` under ``path``.

    Args:
        records: The sample records to persist.
        path: The directory to write into.
        fmt: Which file a later :func:`load_samples` reads back (both are always written).

    Raises:
        IngestError: A bad ``fmt`` or any filesystem failure.
    """
    _STORE.dump(records, path, fmt)


def load_samples(path: str | Path, fmt: str = "csv") -> list[SampleRecord]:
    """Read sample records back from ``path``; a missing file is an empty store.

    Args:
        path: The directory :func:`dump_samples` wrote.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The stored records.

    Raises:
        IngestError: A schema-version mismatch, or any read, parse or validation failure.
    """
    return _STORE.load(path, fmt)
