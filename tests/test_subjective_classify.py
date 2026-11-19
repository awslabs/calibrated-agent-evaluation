# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for atom decomposition and classification."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.subjective import (
    SubjectiveError,
    define_atoms,
    is_objective,
    is_subjective,
)


class TestDefineAtoms:
    """define_atoms produces a valid AtomSet."""

    def test_define_3_objective_2_subjective(self) -> None:
        atom_set = define_atoms(
            "quality",
            objective=[
                ("answered_question", "Did it answer?"),
                ("factually_correct", "Is it correct?"),
                ("used_correct_tool", "Right tool?"),
            ],
            subjective=[
                ("tone", "Was the tone appropriate?"),
                ("helpfulness", "Was it helpful?"),
            ],
        )
        assert atom_set.criterion == "quality"
        assert len(atom_set.objective_atoms) == 3
        assert len(atom_set.subjective_dimensions) == 2

    def test_overlapping_names_raises(self) -> None:
        with pytest.raises(SubjectiveError, match="appears in both"):
            define_atoms(
                "quality",
                objective=[("tone", "obj tone")],
                subjective=[("tone", "subj tone")],
            )


class TestClassification:
    """Every dimension classified as exactly objective or subjective."""

    def test_objective_atoms_classified_correctly(self) -> None:
        atom_set = define_atoms(
            "test",
            objective=[("a", ""), ("b", ""), ("c", "")],
            subjective=[("x", ""), ("y", "")],
        )
        assert is_objective("a", atom_set) is True
        assert is_objective("b", atom_set) is True
        assert is_objective("c", atom_set) is True
        assert is_subjective("a", atom_set) is False

    def test_subjective_dims_classified_correctly(self) -> None:
        atom_set = define_atoms(
            "test",
            objective=[("a", "")],
            subjective=[("x", ""), ("y", "")],
        )
        assert is_subjective("x", atom_set) is True
        assert is_subjective("y", atom_set) is True
        assert is_objective("x", atom_set) is False

    def test_unknown_dimension_not_objective(self) -> None:
        atom_set = define_atoms(
            "test",
            objective=[("a", "")],
            subjective=[("x", "")],
        )
        assert is_objective("unknown", atom_set) is False
        assert is_subjective("unknown", atom_set) is False
