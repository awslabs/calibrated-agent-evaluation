# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atom decomposition and dimension classification."""

from __future__ import annotations

from calibrated_agent_evaluation.subjective.errors import SubjectiveError
from calibrated_agent_evaluation.subjective.types import Atom, AtomSet, SubjectiveDimension


def define_atoms(
    criterion: str,
    objective: list[tuple[str, str]],
    subjective: list[tuple[str, str]],
) -> AtomSet:
    """Define an AtomSet from objective atoms and subjective dimensions.

    Each tuple is (name, description). Raises SubjectiveError if names overlap.
    """
    if not objective and not subjective:
        raise SubjectiveError(
            "AtomSet must have at least 1 objective atom or 1 subjective dimension"
        )

    # Validate no empty/whitespace names
    for name, _ in [*objective, *subjective]:
        if not name or not name.strip():
            raise SubjectiveError("Dimension names must be non-empty strings")

    # Validate no duplicates within the same list
    obj_names_list = [name for name, _ in objective]
    subj_names_list = [name for name, _ in subjective]
    obj_names = set(obj_names_list)
    subj_names = set(subj_names_list)

    if len(obj_names) != len(obj_names_list):
        dupes = [n for n in obj_names_list if obj_names_list.count(n) > 1]
        raise SubjectiveError(f"Duplicate name {dupes[0]!r} in objective atoms")

    if len(subj_names) != len(subj_names_list):
        dupes = [n for n in subj_names_list if subj_names_list.count(n) > 1]
        raise SubjectiveError(f"Duplicate name {dupes[0]!r} in subjective dimensions")

    overlap = obj_names & subj_names
    if overlap:
        raise SubjectiveError(
            f"Dimension {sorted(overlap)[0]!r} appears in both objective and subjective "
            f"lanes; each must be in exactly one lane"
        )

    return AtomSet(
        criterion=criterion,
        objective_atoms=tuple(Atom(name=n, description=d) for n, d in objective),
        subjective_dimensions=tuple(
            SubjectiveDimension(name=n, description=d) for n, d in subjective
        ),
    )


def is_objective(dimension: str, atom_set: AtomSet) -> bool:
    """True if dimension is classified as objective in the AtomSet."""
    return any(a.name == dimension for a in atom_set.objective_atoms)


def is_subjective(dimension: str, atom_set: AtomSet) -> bool:
    """True if dimension is classified as subjective in the AtomSet."""
    return any(d.name == dimension for d in atom_set.subjective_dimensions)
