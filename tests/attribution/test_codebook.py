# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-vocabulary versioned codebook.

Every test embeds its FR and asserts behavior (the populated codebook, the flagged-for-re-labeling
set, or the typed refusal), never mock calls.
"""

from __future__ import annotations

import pytest

from _helpers import make_label
from calibrated_agent_evaluation.attribution import (
    Codebook,
    CodebookEntry,
    CodebookError,
    bump_version,
    register_entry,
)

_DROPPED_FACT = CodebookEntry(
    name="dropped_material_fact",
    definition="An agent omits a fact present in its input that a downstream agent needs.",
    positive_examples=("intake emitted occupancy_status=null despite 'in Florida for a month'",),
    boundary_examples=("intake emitted a raw phrase but did not normalize it (see disputed R3)",),
)
_INTAKE_BOUNDARY = CodebookEntry(
    name="intake_owns_occupancy_normalization",
    definition="Intake must emit a structured occupancy_status, not a raw narrative phrase (R3).",
    positive_examples=("intake emits occupancy_status='vacant_>30d'",),
    boundary_examples=(
        "intake forwards the raw phrase and policy-checker is expected to parse it",
    ),
)


# --- register into both vocabularies → both populated ---


def test_register_populates_both_vocabularies() -> None:
    book = Codebook(version="v1")
    book = register_entry(book, "failure_modes", _DROPPED_FACT)
    book = register_entry(book, "responsibility_boundaries", _INTAKE_BOUNDARY)
    assert [e.name for e in book.failure_modes] == ["dropped_material_fact"]
    assert [e.name for e in book.responsibility_boundaries] == [
        "intake_owns_occupancy_normalization"
    ]


# --- version bump flags prior-version labels; current-version labels not flagged ---


def test_version_bump_flags_prior_version_labels() -> None:
    book = Codebook(version="v1")
    prior = make_label(trace_id="CLM-4471", codebook_version="v1")
    current = make_label(trace_id="CLM-4477", codebook_version="v2")
    unversioned = make_label(trace_id="CLM-4473", codebook_version=None)

    bumped, flagged = bump_version(
        book, "v2", [prior, current, unversioned], change_note="R3 boundary redefined"
    )
    assert bumped.version == "v2"
    flagged_traces = {label.trace_id for label in flagged}
    # v1 and unversioned labels are stale under v2; the v2 label is not flagged.
    assert flagged_traces == {"CLM-4471", "CLM-4473"}
    assert "R3 boundary redefined" in bumped.changelog


# --- (negative) incomplete entry refused ---


def test_entry_without_definition_refused() -> None:
    bad = CodebookEntry(
        name="x", definition="  ", positive_examples=("p",), boundary_examples=("b",)
    )
    with pytest.raises(CodebookError, match="non-empty definition"):
        register_entry(Codebook(version="v1"), "failure_modes", bad)


def test_entry_without_boundary_example_refused() -> None:
    bad = CodebookEntry(name="x", definition="d", positive_examples=("p",), boundary_examples=())
    with pytest.raises(CodebookError, match="boundary example"):
        register_entry(Codebook(version="v1"), "failure_modes", bad)


def test_entry_without_positive_example_refused() -> None:
    bad = CodebookEntry(name="x", definition="d", positive_examples=(), boundary_examples=("b",))
    with pytest.raises(CodebookError, match="positive example"):
        register_entry(Codebook(version="v1"), "failure_modes", bad)
