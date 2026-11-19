# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration: define → record → compute → validate."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.subjective import (
    PreferencePair,
    SubjectiveGateError,
    compute_ratings,
    define_atoms,
    record_preference,
    validate_gateable,
)


class TestEndToEnd:
    """Full pipeline composition proof."""

    def test_full_pipeline(self) -> None:
        # 1. Define atoms
        atom_set = define_atoms(
            "orderbot_quality",
            objective=[
                ("answered_question", "Did the agent answer?"),
                ("used_correct_tool", "Did it use the right tool?"),
            ],
            subjective=[
                ("tone", "Was the tone appropriate?"),
                ("helpfulness", "How helpful was it?"),
            ],
        )

        # 2. Record preferences for subjective dimension
        pairs = []
        matchups = [
            ("ORD-1001", "CLM-4471", "a"),
            ("ORD-1001", "t3", "a"),
            ("CLM-4471", "t3", "a"),
            ("t3", "t4", "b"),
            ("ORD-1001", "t4", "a"),
        ]
        for a, b, w in matchups:
            pair = PreferencePair(
                trace_id_a=a,
                trace_id_b=b,
                dimension="tone",
                winner="a" if w == "a" else "b",
                judge_kind="human",
            )
            pairs.append(record_preference(pair, atom_set))

        assert len(pairs) == 5

        # 3. Compute Elo ratings
        ratings = compute_ratings("tone", pairs)
        assert ratings.treatment == "tracked"
        assert ratings.n_total_comparisons == 5
        assert len(ratings.ratings) == 4
        # ORD-1001 won 3 times → should be highest
        assert ratings.ratings[0].trace_id == "ORD-1001"

        # 4. Validate gate/track boundary
        validate_gateable("answered_question", atom_set)  # passes
        with pytest.raises(SubjectiveGateError):
            validate_gateable("tone", atom_set)  # refuses
