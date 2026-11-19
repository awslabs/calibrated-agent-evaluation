# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CSV+JSON round-trip for the agreement and codebook records.

``dump_agreement_records`` / ``load_agreement_records`` write and read
``agreement_record.{csv,json}`` under a directory; ``dump_codebook_records`` /
``load_codebook_records`` do the same for ``codebook_record.{csv,json}``. A stored
``schema_version`` other than the running one, and every read or parse failure, raises
:class:`~calibrated_agent_evaluation.review.errors.ReviewError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from calibrated_agent_evaluation._recordstore import RecordStore
from calibrated_agent_evaluation.review.errors import ReviewError
from calibrated_agent_evaluation.review.record import (
    AGREEMENT_SCHEMA_VERSION,
    CODEBOOK_SCHEMA_VERSION,
    PROPOSED_MODE_SCHEMA_VERSION,
    AgreementRecord,
    CodebookRecord,
    ProposedModeRecord,
)

__all__ = [
    "dump_agreement_records",
    "dump_codebook_records",
    "dump_proposed_modes",
    "load_agreement_records",
    "load_codebook_records",
    "load_proposed_modes",
]

_AGREEMENT = RecordStore(
    AgreementRecord,
    stem="agreement_record",
    schema_version=AGREEMENT_SCHEMA_VERSION,
    error=ReviewError,
    noun="agreement record",
    rerun="calibrated-agent-evaluation label --agreement",
)
_CODEBOOK = RecordStore(
    CodebookRecord,
    stem="codebook_record",
    schema_version=CODEBOOK_SCHEMA_VERSION,
    error=ReviewError,
    noun="codebook record",
    rerun="calibrated-agent-evaluation cluster --ratify",
)


def dump_agreement_records(
    records: Sequence[AgreementRecord], path: str | Path, fmt: str = "csv"
) -> None:
    """Write ``records`` under ``path``; ``fmt`` picks the file a later load reads."""
    _AGREEMENT.dump(records, path, fmt)


def load_agreement_records(path: str | Path, fmt: str = "csv") -> list[AgreementRecord]:
    """Read agreement records back from ``path``; a missing file is an empty store."""
    return _AGREEMENT.load(path, fmt)


def dump_codebook_records(
    records: Sequence[CodebookRecord], path: str | Path, fmt: str = "csv"
) -> None:
    """Write ``records`` under ``path``; ``fmt`` picks the file a later load reads."""
    _CODEBOOK.dump(records, path, fmt)


def load_codebook_records(path: str | Path, fmt: str = "csv") -> list[CodebookRecord]:
    """Read codebook records back from ``path``; a missing file is an empty store."""
    return _CODEBOOK.load(path, fmt)


_PROPOSED = RecordStore(
    ProposedModeRecord,
    stem="proposed_mode",
    schema_version=PROPOSED_MODE_SCHEMA_VERSION,
    error=ReviewError,
    noun="proposed failure mode",
    rerun="calibrated-agent-evaluation label --serve",
)


def dump_proposed_modes(
    records: Sequence[ProposedModeRecord], path: str | Path, fmt: str = "csv"
) -> None:
    """Write proposed failure modes under ``path``; ``fmt`` picks the file a later load reads."""
    _PROPOSED.dump(records, path, fmt)


def load_proposed_modes(path: str | Path, fmt: str = "csv") -> list[ProposedModeRecord]:
    """Read proposed failure modes back from ``path``; a missing file is an empty store."""
    return _PROPOSED.load(path, fmt)
