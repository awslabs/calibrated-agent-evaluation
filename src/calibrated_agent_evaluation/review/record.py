# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The records the labeling side persists: an inter-rater agreement result and a ratified codebook.

* :class:`AgreementRecord` is what ``calibrated-agent-evaluation label --agreement`` writes to
  ``<root>/records/agreement/``: the alpha gate verdict over the named raters' human labels within
  one codebook version, with every interval the gate read.
* :class:`CodebookRecord` is what ``calibrated-agent-evaluation cluster --ratify`` writes to
  ``<root>/records/codebook/``: the failure modes a named human ratified under one version, which
  the review surface then offers as its failure-mode list. :class:`CodebookFile` is the same
  vocabulary written by hand, the file ``calibrated-agent-evaluation label --serve --codebook``
  reads.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self, TypeVar

from pydantic import ConfigDict, Field, field_validator, model_validator

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._provenance import ProvenanceStamp, capture_provenance
from calibrated_agent_evaluation.review.errors import ReviewError
from calibrated_agent_evaluation.review.gate import (
    ALPHA_LOWER_BOUND_GATE,
    ALPHA_REVISE_FLOOR,
    AgreementAxis,
    GateVerdict,
)

if TYPE_CHECKING:
    from calibrated_agent_evaluation.review.agreement import InterRaterReport
    from calibrated_agent_evaluation.stats.agreement import AgreementInterval

__all__ = [
    "AGREEMENT_SCHEMA_VERSION",
    "CODEBOOK_SCHEMA_VERSION",
    "AgreementRecord",
    "AgreementStat",
    "CodebookFile",
    "CodebookMode",
    "CodebookRecord",
    "ProposedModeRecord",
    "agreement_record_of",
    "load_codebook_file",
    "mode_key",
    "pending_proposals",
    "with_content_id",
]

M = TypeVar("M", bound=CalibratedAgentEvaluationModel)

AGREEMENT_SCHEMA_VERSION = "1"
CODEBOOK_SCHEMA_VERSION = "1"
PROPOSED_MODE_SCHEMA_VERSION = "1"


class AgreementStat(CalibratedAgentEvaluationModel):
    """One agreement statistic with its two-sided percentile bootstrap interval over units."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ReviewError

    point: float
    lower: float
    upper: float
    confidence: float
    n_units: int
    n_resamples: int
    n_undefined: int
    seed: int

    @classmethod
    def of(cls, interval: AgreementInterval) -> AgreementStat:
        """The stored form of a bootstrap interval."""
        return cls(
            point=interval.point,
            lower=interval.lower,
            upper=interval.upper,
            confidence=interval.confidence,
            n_units=interval.n_units,
            n_resamples=interval.n_resamples,
            n_undefined=interval.n_undefined,
            seed=interval.seed,
        )


class AgreementRecord(CalibratedAgentEvaluationModel):
    """The gold-set agreement gate over one scope, one axis and one codebook version.

    ``gate`` is ``usable`` only when ``alpha.lower`` reaches ``alpha_gate`` (the 0.80 bound) on at
    least ``minimum_overlap`` pairable units; ``top_up_overlap`` / ``revise_codebook`` /
    ``codebook_broken`` otherwise (see
    :func:`calibrated_agent_evaluation.review.agreement.alpha_gate`). ``minimum_overlap`` is
    ``None`` on a record written before the gate had a minimum. ``codebook_version`` ``None`` is the
    unversioned bucket. ``trace_ids`` lists every trace with a unit at least one named rater coded,
    and ``n_pairable_units`` how many units two or more coded. ``n_superseded`` counts the older
    labels a rater's relabel replaced (newest label per rater and unit). ``cell`` / ``trace_id`` /
    ``frame`` record the scope flags the command ran with.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ReviewError

    agreement_id: str
    axis: AgreementAxis
    codebook_version: str | None
    raters: tuple[str, ...]
    trace_ids: tuple[str, ...]
    n_units: int
    n_pairable_units: int
    n_superseded: int
    alpha: AgreementStat
    cohen_kappa: AgreementStat | None
    per_class: dict[str, AgreementStat]
    gate: GateVerdict
    alpha_gate: float = ALPHA_LOWER_BOUND_GATE
    alpha_revise_floor: float = ALPHA_REVISE_FLOOR
    minimum_overlap: int | None = None
    cell: dict[str, str] | None = None
    trace_id: str | None = None
    frame: str | None = None
    provenance: ProvenanceStamp
    schema_version: str = AGREEMENT_SCHEMA_VERSION


def agreement_record_of(
    report: InterRaterReport,
    *,
    trace_ids: Collection[str],
    cell: Mapping[str, str] | None = None,
    trace_id: str | None = None,
    frame: str | None = None,
    provenance: ProvenanceStamp | None = None,
) -> AgreementRecord:
    """Build the :class:`AgreementRecord` for ``report``; the id is a hash of its content."""
    stamp = provenance or capture_provenance(
        seed=report.alpha.seed,
        n_resamples=report.alpha.n_resamples,
        thresholds={
            "alpha_gate": ALPHA_LOWER_BOUND_GATE,
            "alpha_revise": ALPHA_REVISE_FLOOR,
            "minimum_overlap": report.minimum_overlap,
        },
    )
    record = AgreementRecord(
        agreement_id="",
        axis=report.axis,
        codebook_version=report.codebook_version,
        raters=tuple(report.raters),
        trace_ids=tuple(sorted(set(trace_ids))),
        n_units=report.n_units,
        n_pairable_units=report.n_pairable_units,
        n_superseded=report.n_superseded,
        alpha=AgreementStat.of(report.alpha),
        cohen_kappa=None if report.cohen_kappa is None else AgreementStat.of(report.cohen_kappa),
        per_class={k: AgreementStat.of(v) for k, v in report.per_class.items()},
        gate=report.gate,
        minimum_overlap=report.minimum_overlap,
        cell=dict(cell) if cell else None,
        trace_id=trace_id,
        frame=frame,
        provenance=stamp,
    )
    return with_content_id(record, "agreement_id", "agreement")


def with_content_id(record: M, field: str, prefix: str) -> M:
    """``record`` with ``field`` set to ``<prefix>-<hash of every other field>``."""
    doc = record.model_dump(mode="json", exclude={field})
    digest = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    return record.model_copy(update={field: f"{prefix}-{digest[:16]}"})


class CodebookMode(CalibratedAgentEvaluationModel):
    """One failure mode: the name a label carries in ``failure_mode``, and what it means."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ReviewError

    name: str
    description: str = ""

    @field_validator("name")
    @classmethod
    def _named(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("a failure-mode name must be non-empty with no surrounding spaces")
        return value


def _unique_modes(modes: Sequence[CodebookMode]) -> None:
    names = [m.name for m in modes]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ReviewError(f"failure-mode names must be unique; repeated: {duplicates}")


class CodebookFile(CalibratedAgentEvaluationModel):
    """A hand-written failure-mode list: ``{"version": "1", "modes": [{"name": ..., ...}]}``."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ReviewError

    version: str = Field(min_length=1)
    modes: tuple[CodebookMode, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _unique_modes(self.modes)
        return self


class CodebookRecord(CalibratedAgentEvaluationModel):
    """A ratified failure-mode vocabulary under one codebook version.

    ``ratified_by`` names the human who ratified it and ``ratified_at`` when (UTC ISO-8601).
    ``parent_version`` is the version this one follows, ``None`` for the first. ``n_tagged`` counts
    the FAIL labels the ratification stamped with a mode, and ``source`` names the map file.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ReviewError

    codebook_id: str
    version: str = Field(min_length=1)
    modes: tuple[CodebookMode, ...] = Field(min_length=1)
    ratified_by: str = Field(min_length=1)
    ratified_at: str
    parent_version: str | None = None
    n_tagged: int = 0
    source: str | None = None
    merged: dict[str, str] = Field(default_factory=dict)
    provenance: ProvenanceStamp
    schema_version: str = CODEBOOK_SCHEMA_VERSION

    @model_validator(mode="after")
    def _check(self) -> Self:
        _unique_modes(self.modes)
        return self


def mode_key(name: str) -> str:
    """The name two failure modes are the same mode under: case and spacing ignored."""
    return " ".join(name.split()).casefold()


class ProposedModeRecord(CalibratedAgentEvaluationModel):
    """A failure mode a labeler proposed from the review form, waiting for the codebook owner.

    The FAIL label that proposed it already carries ``name`` as its ``failure_mode`` under
    ``codebook_version``. The proposal stays pending until
    ``calibrated-agent-evaluation cluster --ratify`` adopts it (it is then a ratified mode of the
    version) or merges it into another mode (the codebook record's ``merged`` maps it to that mode,
    and its labels are retagged).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ReviewError

    proposal_id: str
    codebook_version: str = Field(min_length=1)
    name: str
    note: str = Field(min_length=1)
    proposed_by: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    proposed_at: str
    provenance: ProvenanceStamp
    schema_version: str = PROPOSED_MODE_SCHEMA_VERSION

    @field_validator("name")
    @classmethod
    def _named(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("a failure-mode name must be non-empty with no surrounding spaces")
        return value

    @field_validator("note")
    @classmethod
    def _noted(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a proposed failure mode needs a note saying what it covers")
        return value


def pending_proposals(
    proposals: Sequence[ProposedModeRecord],
    codebooks: Sequence[CodebookRecord],
    version: str,
) -> tuple[ProposedModeRecord, ...]:
    """The proposals under ``version`` no codebook record has adopted or merged yet.

    One per mode: a later proposal of a name already proposed (case and spacing ignored) is the
    same proposal and is dropped. Oldest first.
    """
    settled = {
        mode_key(name)
        for record in codebooks
        if record.version == version
        for name in (*(m.name for m in record.modes), *record.merged)
    }
    seen: dict[str, ProposedModeRecord] = {}
    for proposal in sorted(proposals, key=lambda p: p.proposed_at):
        key = mode_key(proposal.name)
        if proposal.codebook_version == version and key not in settled:
            seen.setdefault(key, proposal)
    return tuple(seen.values())


def load_codebook_file(path: str | Path) -> CodebookFile:
    """Read a :class:`CodebookFile` from JSON.

    Raises:
        ReviewError: The file cannot be read, is not JSON, or is not a valid codebook.
    """
    src = Path(path)
    try:
        doc = json.loads(src.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReviewError(f"cannot read the codebook file {src}: {exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReviewError(f"the codebook file {src} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ReviewError(
            f'the codebook file {src} must be a JSON object: {{"version": "1", "modes": [...]}}'
        )
    return CodebookFile.model_validate(doc)
