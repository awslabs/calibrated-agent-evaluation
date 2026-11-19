# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The stored preference record and its CSV+JSON round-trip.

A :class:`PreferenceRecord` is one validated
:class:`~calibrated_agent_evaluation.subjective.PreferencePair` with where it came from: the
criterion whose :class:`~calibrated_agent_evaluation.subjective.AtomSet` declared the dimension, who
gave it, and whether it was imported from a file or recorded in the review surface's A/B page.
:func:`preference_record` builds one only after
:func:`~calibrated_agent_evaluation.subjective.record_preference` accepts the pair, so an objective,
undeclared or self-compared dimension never reaches the store.

``dump_preference_records`` writes ``preference_record.{csv,json}`` under a directory and
``load_preference_records`` reads them back; a stored ``schema_version`` other than the running
one, and every read or parse failure, raises :class:`SubjectiveStoreError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import ConfigDict

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._provenance import ProvenanceStamp, capture_provenance
from calibrated_agent_evaluation._recordstore import RecordStore
from calibrated_agent_evaluation.subjective.errors import SubjectiveStoreError
from calibrated_agent_evaluation.subjective.preference import record_preference
from calibrated_agent_evaluation.subjective.types import AtomSet, PairJudgeKind, PreferencePair

__all__ = [
    "SUBJECTIVE_SCHEMA_VERSION",
    "PreferenceRecord",
    "dump_preference_records",
    "load_preference_records",
    "preference_record",
]

SUBJECTIVE_SCHEMA_VERSION = "1"

PreferenceSource = Literal["file", "review"]


class PreferenceRecord(CalibratedAgentEvaluationModel):
    """One stored pairwise preference: trace A or trace B was better on one subjective dimension.

    ``source`` is ``"file"`` for ``calibrated-agent-evaluation prefer --pairs`` and ``"review"`` for
    the A/B page; ``source_ref`` locates it (``"<file>:<line>"``, or the review session).
    ``rater_id`` names the human who chose, when one did. Tracked, never gated: no CI decision reads
    these records.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = SubjectiveStoreError

    preference_id: str
    criterion: str
    dimension: str
    trace_id_a: str
    trace_id_b: str
    winner: Literal["a", "b"]
    judge_kind: PairJudgeKind
    rater_id: str | None = None
    source: PreferenceSource
    source_ref: str
    schema_version: str = SUBJECTIVE_SCHEMA_VERSION
    provenance: ProvenanceStamp

    def pair(self) -> PreferencePair:
        """The library pair this record stores."""
        return PreferencePair(
            trace_id_a=self.trace_id_a,
            trace_id_b=self.trace_id_b,
            dimension=self.dimension,
            winner=self.winner,
            judge_kind=self.judge_kind,
        )


def preference_record(
    pair: PreferencePair,
    atom_set: AtomSet,
    *,
    preference_id: str,
    source: PreferenceSource,
    source_ref: str,
    rater_id: str | None = None,
) -> PreferenceRecord:
    """Validate ``pair`` against ``atom_set`` and return it as a record to store.

    Raises:
        SubjectiveError: The pair names an objective or undeclared dimension, compares a trace
            with itself, or has a winner other than ``"a"`` or ``"b"``.
        SubjectiveStoreError: ``judge_kind`` is not ``human``, ``llm-judge`` or ``code``.
    """
    checked = record_preference(pair, atom_set)
    return PreferenceRecord(
        preference_id=preference_id,
        criterion=atom_set.criterion,
        dimension=checked.dimension,
        trace_id_a=checked.trace_id_a,
        trace_id_b=checked.trace_id_b,
        winner=checked.winner,
        judge_kind=checked.judge_kind,  # type: ignore[arg-type]  # validated by the model
        rater_id=rater_id,
        source=source,
        source_ref=source_ref,
        provenance=capture_provenance(),
    )


_STORE = RecordStore(
    PreferenceRecord,
    stem="preference_record",
    schema_version=SUBJECTIVE_SCHEMA_VERSION,
    error=SubjectiveStoreError,
    noun="preference record",
    rerun="calibrated-agent-evaluation prefer",
)


def dump_preference_records(
    records: Sequence[PreferenceRecord], path: str | Path, fmt: str = "csv"
) -> None:
    """Write ``records`` to ``preference_record.{csv,json}`` under ``path``.

    Raises:
        SubjectiveStoreError: A bad ``fmt`` or any filesystem failure.
    """
    _STORE.dump(records, path, fmt)


def load_preference_records(path: str | Path, fmt: str = "csv") -> list[PreferenceRecord]:
    """Read preference records back from ``path``; a missing file is an empty store.

    Raises:
        SubjectiveStoreError: A schema-version mismatch, or any read, parse or validation failure.
    """
    return _STORE.load(path, fmt)
