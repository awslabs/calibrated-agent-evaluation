# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two-vocabulary versioned codebook.

A crew codebook versions **two** vocabularies -- failure modes AND agent/handoff responsibility
boundaries -- because "intake drops facts" and "policy-checker ignores exclusions" are different
bugs with different owners. Each entry must carry a definition + ≥1 positive + ≥1 boundary example;
an incomplete entry is refused, not silently stored. A codebook change is retroactive:
when a definition or a responsibility boundary changes (a new version), every Label written under a
superseded ``codebook_version`` is flagged for re-labeling -- old labels must not silently
drift from the new definition.
"""

from __future__ import annotations

from collections.abc import Sequence

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.attribution.errors import CodebookError
from calibrated_agent_evaluation.attribution.types import Codebook, CodebookEntry, Vocabulary
from calibrated_agent_evaluation.schema.types import Label


def register_entry(codebook: Codebook, vocabulary: Vocabulary, entry: CodebookEntry) -> Codebook:
    """Return a new codebook with ``entry`` appended to the named vocabulary.

    Raises :class:`CodebookError` if the entry has an empty definition, or fewer than one positive
    or one boundary example -- the examples are what make a category ratifiable, so an
    entry without them is refused rather than stored half-defined. Also refuses a name already
    present in the target vocabulary.
    """
    _validate_entry(entry)
    _refuse_duplicate_name(codebook, vocabulary, entry)
    if vocabulary == "failure_modes":
        return replace(codebook, failure_modes=(*codebook.failure_modes, entry))
    return replace(codebook, responsibility_boundaries=(*codebook.responsibility_boundaries, entry))


def _refuse_duplicate_name(
    codebook: Codebook, vocabulary: Vocabulary, entry: CodebookEntry
) -> None:
    """Refuse an entry whose name is already in the target vocabulary.

    `_validate_entry` checks an entry is individually well-formed but says nothing about the
    codebook it is joining, so two entries could share a name in one vocabulary. A label's
    `failure_mode` IS a name, so a duplicate makes the category it refers to ambiguous -- and the
    codebook is what a labeler and a judge are supposed to agree on.
    """
    existing = (
        codebook.failure_modes
        if vocabulary == "failure_modes"
        else codebook.responsibility_boundaries
    )
    if any(e.name == entry.name for e in existing):
        raise CodebookError(
            f"codebook vocabulary {vocabulary!r} already has an entry named {entry.name!r}; "
            "two entries with one name make the category a label refers to ambiguous. Rename the "
            "new entry, or bump_version() the codebook if you are redefining the existing one"
        )


def bump_version(
    codebook: Codebook,
    new_version: str,
    labels: Sequence[Label],
    *,
    change_note: str,
) -> tuple[Codebook, list[Label]]:
    """Return the re-versioned codebook and the labels to flag for re-labeling.

    Every Label whose ``codebook_version`` differs from ``new_version`` (i.e. was written under a
    superseded version, including ``None``) is returned in the flag list -- a definition change is
    retroactive, so labeled history must be re-checked against the new definition. Labels already at
    ``new_version`` are not flagged. ``change_note`` is appended to the changelog for audit.
    """
    if not new_version:
        raise CodebookError("A codebook version bump requires a non-empty new_version.")
    flagged = [label for label in labels if label.codebook_version != new_version]
    bumped = replace(
        codebook,
        version=new_version,
        changelog=(*codebook.changelog, change_note),
    )
    return bumped, flagged


def _validate_entry(entry: CodebookEntry) -> None:
    if not entry.definition.strip():
        raise CodebookError(f"codebook entry {entry.name!r} needs a non-empty definition.")
    if not entry.positive_examples:
        raise CodebookError(f"codebook entry {entry.name!r} needs at least one positive example.")
    if not entry.boundary_examples:
        raise CodebookError(f"codebook entry {entry.name!r} needs at least one boundary example.")
