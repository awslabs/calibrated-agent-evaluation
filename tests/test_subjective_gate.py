# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for gate/track boundary enforcement."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.schema.types import Label
from calibrated_agent_evaluation.subjective import (
    SubjectiveError,
    SubjectiveGateError,
    define_atoms,
    validate_gateable,
)


def _atom_set():
    return define_atoms(
        "quality",
        objective=[("answered_question", ""), ("factually_correct", ""), ("used_correct_tool", "")],
        subjective=[("tone", ""), ("helpfulness", "")],
    )


class TestValidateGateable:
    """Objective dimensions pass; subjective refused."""

    def test_objective_passes_silently(self) -> None:
        validate_gateable("answered_question", _atom_set())
        validate_gateable("factually_correct", _atom_set())
        validate_gateable("used_correct_tool", _atom_set())

    def test_subjective_raises_gate_error(self) -> None:
        with pytest.raises(SubjectiveGateError, match="subjective"):
            validate_gateable("tone", _atom_set())

    def test_error_message_says_elo_only(self) -> None:
        with pytest.raises(SubjectiveGateError, match="Elo ranking only, not gatable"):
            validate_gateable("helpfulness", _atom_set())

    def test_unknown_dimension_raises_subjective_error(self) -> None:
        with pytest.raises(SubjectiveError, match="not found"):
            validate_gateable("nonexistent", _atom_set())


class TestObjectiveAtomsCompatibleWithF001:
    """Objective atoms produce binary Labels (schema compatible)."""

    def test_objective_atoms_can_be_binary_labels(self) -> None:
        atom_set = _atom_set()
        for atom in atom_set.objective_atoms:
            validate_gateable(atom.name, atom_set)
            # Prove a binary Label can be constructed for this atom
            label = Label(
                trace_id="test-trace",
                span_id=None,
                kind="human",
                outcome="FAIL",
                critique=f"Failed on {atom.name}",
            )
            assert label.outcome in ("PASS", "FAIL")
