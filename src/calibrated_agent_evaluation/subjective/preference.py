# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preference pair validation."""

from __future__ import annotations

from calibrated_agent_evaluation.subjective.classify import is_objective, is_subjective
from calibrated_agent_evaluation.subjective.errors import SubjectiveError
from calibrated_agent_evaluation.subjective.types import AtomSet, PreferencePair


def record_preference(pair: PreferencePair, atom_set: AtomSet) -> PreferencePair:
    """Validate a preference pair against the AtomSet and return it.

    Raises SubjectiveError if:
    - winner is not 'a' or 'b'
    - trace_id_a or trace_id_b is empty
    - dimension is objective
    - dimension is unknown
    - trace_id_a == trace_id_b
    """
    if pair.winner not in ("a", "b"):
        raise SubjectiveError(f"winner must be 'a' or 'b'; got {pair.winner!r}")

    if not pair.trace_id_a or not pair.trace_id_a.strip():
        raise SubjectiveError("trace_id_a must be a non-empty string")

    if not pair.trace_id_b or not pair.trace_id_b.strip():
        raise SubjectiveError("trace_id_b must be a non-empty string")

    if pair.trace_id_a == pair.trace_id_b:
        raise SubjectiveError(
            f"A trace cannot be preferred over itself (trace_id={pair.trace_id_a!r})"
        )

    if is_objective(pair.dimension, atom_set):
        raise SubjectiveError(
            f"Dimension {pair.dimension!r} is objective; "
            f"use binary PASS/FAIL labels, not preferences"
        )

    if not is_subjective(pair.dimension, atom_set):
        raise SubjectiveError(
            f"Dimension {pair.dimension!r} not found in AtomSet "
            f"for criterion {atom_set.criterion!r}; declare it first"
        )

    return pair
