# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate/track boundary enforcement."""

from __future__ import annotations

from calibrated_agent_evaluation.subjective.classify import is_objective, is_subjective
from calibrated_agent_evaluation.subjective.errors import SubjectiveError, SubjectiveGateError
from calibrated_agent_evaluation.subjective.types import AtomSet


def validate_gateable(dimension: str, atom_set: AtomSet) -> None:
    """Confirm a dimension is objective (gatable). Raises if subjective or unknown.

    Passes silently if objective. Raises SubjectiveGateError if subjective.
    Raises SubjectiveError if unknown.
    """
    if is_subjective(dimension, atom_set):
        raise SubjectiveGateError(
            f"Dimension {dimension!r} is subjective "
            f"and is tracked via Elo ranking only, not gatable as a CI gate"
        )

    if not is_objective(dimension, atom_set):
        raise SubjectiveError(
            f"Dimension {dimension!r} not found in AtomSet "
            f"for criterion {atom_set.criterion!r}; declare it first"
        )
