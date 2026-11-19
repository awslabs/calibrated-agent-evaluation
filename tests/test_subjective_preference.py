# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for preference pair validation."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.subjective import (
    PreferencePair,
    SubjectiveError,
    define_atoms,
    record_preference,
)


def _atom_set():
    return define_atoms(
        "quality",
        objective=[("answered_question", "")],
        subjective=[("tone", ""), ("helpfulness", "")],
    )


class TestRecordPreference:
    """Valid preferences accepted."""

    def test_valid_pair_accepted(self) -> None:
        pair = PreferencePair(
            trace_id_a="t1", trace_id_b="t2", dimension="tone", winner="a", judge_kind="human"
        )
        result = record_preference(pair, _atom_set())
        assert result == pair

    def test_all_fields_preserved(self) -> None:
        pair = PreferencePair(
            trace_id_a="trace-A",
            trace_id_b="trace-B",
            dimension="tone",
            winner="b",
            judge_kind="llm-judge",
        )
        result = record_preference(pair, _atom_set())
        assert result.trace_id_a == "trace-A"
        assert result.trace_id_b == "trace-B"
        assert result.dimension == "tone"
        assert result.winner == "b"
        assert result.judge_kind == "llm-judge"


class TestPreferenceRefusals:
    """Invalid preferences refused."""

    def test_objective_dimension_refused(self) -> None:
        pair = PreferencePair(
            trace_id_a="t1",
            trace_id_b="t2",
            dimension="answered_question",
            winner="a",
            judge_kind="human",
        )
        with pytest.raises(SubjectiveError, match="objective"):
            record_preference(pair, _atom_set())

    def test_unknown_dimension_refused(self) -> None:
        pair = PreferencePair(
            trace_id_a="t1",
            trace_id_b="t2",
            dimension="unknown_dim",
            winner="a",
            judge_kind="human",
        )
        with pytest.raises(SubjectiveError, match="not found"):
            record_preference(pair, _atom_set())

    def test_self_comparison_refused(self) -> None:
        pair = PreferencePair(
            trace_id_a="t1", trace_id_b="t1", dimension="tone", winner="a", judge_kind="human"
        )
        with pytest.raises(SubjectiveError, match="cannot be preferred over itself"):
            record_preference(pair, _atom_set())
