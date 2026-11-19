# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for calibrated_agent_evaluation.scorers.heuristic — concrete scorer classes."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from calibrated_agent_evaluation.scorers.errors import ScorerError
from calibrated_agent_evaluation.scorers.heuristic import (
    ExactMatchScorer,
    JSONDiffScorer,
    LevenshteinScorer,
    NumericDiffScorer,
    ValidJSONScorer,
)


class TestExactMatch:
    """ExactMatch match/mismatch behavior."""

    def test_match_returns_pass(self) -> None:
        s = ExactMatchScorer()
        label = s.score(output={"answer": "yes"}, expected={"answer": "yes"}, trace_id="t1")
        assert label.outcome == "PASS"
        assert "matches" in label.critique

    def test_mismatch_returns_fail_with_values(self) -> None:
        s = ExactMatchScorer()
        label = s.score(output={"answer": "refund"}, expected={"answer": "denied"}, trace_id="t2")
        assert label.outcome == "FAIL"
        assert "refund" in label.critique
        assert "denied" in label.critique

    def test_expected_none_raises_scorer_error(self) -> None:
        s = ExactMatchScorer()
        with pytest.raises(ScorerError, match="requires an expected value"):
            s.score(output={"x": 1}, expected=None, trace_id="t3")


class TestLevenshtein:
    """Levenshtein threshold behavior."""

    def test_above_threshold_returns_pass(self) -> None:
        s = LevenshteinScorer(threshold=0.7)
        label = s.score(output={"text": "hello"}, expected={"text": "helo"}, trace_id="lev1")
        assert label.outcome == "PASS"
        assert "threshold" in label.critique

    def test_below_threshold_returns_fail(self) -> None:
        s = LevenshteinScorer(threshold=0.95)
        label = s.score(output={"text": "abc"}, expected={"text": "xyz"}, trace_id="lev2")
        assert label.outcome == "FAIL"
        assert "threshold" in label.critique
        assert "0.95" in label.critique

    def test_expected_none_raises(self) -> None:
        s = LevenshteinScorer()
        with pytest.raises(ScorerError, match="requires an expected value"):
            s.score(output={"x": "a"}, expected=None, trace_id="lev3")


class TestNumericDiff:
    """NumericDiff behavior + error handling."""

    def test_close_values_pass(self) -> None:
        s = NumericDiffScorer(threshold=0.95)
        label = s.score(output={"val": 42}, expected={"val": 40}, trace_id="nd1")
        assert label.outcome == "PASS"

    def test_distant_values_fail(self) -> None:
        s = NumericDiffScorer(threshold=0.99)
        label = s.score(output={"val": 100}, expected={"val": 50}, trace_id="nd2")
        assert label.outcome == "FAIL"

    def test_non_numeric_raises_scorer_error(self) -> None:
        s = NumericDiffScorer()
        with pytest.raises(ScorerError, match="expected numeric values"):
            s.score(output={"val": "not a number"}, expected={"val": "also not"}, trace_id="nd3")

    def test_expected_none_raises(self) -> None:
        s = NumericDiffScorer()
        with pytest.raises(ScorerError, match="requires an expected value"):
            s.score(output={"val": 1}, expected=None, trace_id="nd4")


class TestJSONDiff:
    """JSONDiff structural comparison."""

    def test_identical_json_passes(self) -> None:
        s = JSONDiffScorer()
        label = s.score(output={"data": {"a": 1}}, expected={"data": {"a": 1}}, trace_id="jd1")
        assert label.outcome == "PASS"

    def test_different_json_fails(self) -> None:
        s = JSONDiffScorer()
        label = s.score(output={"data": {"a": 2}}, expected={"data": {"a": 1}}, trace_id="jd2")
        assert label.outcome == "FAIL"

    def test_expected_none_raises(self) -> None:
        s = JSONDiffScorer()
        with pytest.raises(ScorerError, match="requires an expected value"):
            s.score(output={"x": 1}, expected=None, trace_id="jd3")


class TestValidJSON:
    """ValidJSON does not require expected."""

    def test_valid_json_passes(self) -> None:
        s = ValidJSONScorer()
        label = s.score(output={"raw": '{"valid": true}'}, expected=None, trace_id="vj1")
        assert label.outcome == "PASS"
        assert "valid JSON" in label.critique

    def test_invalid_json_fails(self) -> None:
        s = ValidJSONScorer()
        label = s.score(output={"raw": "not json {{"}, expected=None, trace_id="vj2")
        assert label.outcome == "FAIL"
        assert "not valid JSON" in label.critique


class TestCommonProperties:
    """All scorers return kind='code' and non-empty critique."""

    @pytest.mark.parametrize(
        "scorer_cls",
        [ExactMatchScorer, LevenshteinScorer, NumericDiffScorer, JSONDiffScorer, ValidJSONScorer],
    )
    def test_all_scorers_return_kind_code(self, scorer_cls: type) -> None:
        if scorer_cls is ValidJSONScorer:
            s = scorer_cls()
            label = s.score(output={"x": '{"a":1}'}, expected=None, trace_id="kind")
        elif scorer_cls is NumericDiffScorer:
            s = scorer_cls()
            label = s.score(output={"x": 5}, expected={"x": 5}, trace_id="kind")
        else:
            s = scorer_cls()
            label = s.score(output={"x": "a"}, expected={"x": "a"}, trace_id="kind")
        assert label.kind == "code"
        assert label.critique  # non-empty


class TestLazyImport:
    """Missing autoevals raises ScorerError with installable string."""

    def test_missing_autoevals_raises_scorer_error(self) -> None:
        with patch.dict("sys.modules", {"autoevals": None}):
            with patch(
                "calibrated_agent_evaluation.scorers.heuristic._load_autoevals",
                side_effect=ScorerError(
                    "autoevals is not installed. Install via: pip install "
                    "calibrated-agent-evaluation[scorers]"
                ),
            ):
                s = ExactMatchScorer()
                with pytest.raises(ScorerError, match="calibrated-agent-evaluation\\[scorers\\]"):
                    s.score(output={"x": "a"}, expected={"x": "a"}, trace_id="lazy")


# --- an explicit key is a requirement, not a preference ---------------------


class TestExplicitKeyIsRequired:
    """A `key=` that is absent must refuse, never fall through to scoring a different field."""

    def test_missing_key_in_output_raises_naming_the_key_and_the_actual_keys(self) -> None:
        """The defect: `key="action"` against `{"answer": ...}` silently compared the answer.

        `_extract_value`'s guard was `if key and key in data`, so an absent key fell through to the
        single-key heuristic and the scorer returned a confident PASS/FAIL about a field the caller
        never asked about. A mistyped key is a caller bug and has to surface as one.
        """
        scorer = ExactMatchScorer(key="action")
        with pytest.raises(ScorerError) as excinfo:
            scorer.score(output={"answer": "x"}, expected={"action": "x"}, trace_id="t1")

        message = str(excinfo.value)
        assert "'action'" in message, "the message must name the key that was asked for"
        assert "['answer']" in message, "and the keys that are actually there"
        assert "output" in message, "and which side was missing it"

    def test_missing_key_in_expected_names_the_expected_side(self) -> None:
        scorer = ExactMatchScorer(key="answer")
        with pytest.raises(ScorerError, match="expected"):
            scorer.score(output={"answer": "x"}, expected={"other": "x"}, trace_id="t1")

    def test_a_present_key_still_scores_that_key(self) -> None:
        """Positive path: the refusal must not disturb the case the key was written for."""
        scorer = ExactMatchScorer(key="action")
        label = scorer.score(
            output={"action": "refund", "note": "n"},
            expected={"action": "refund", "note": "different"},
            trace_id="t1",
        )
        assert label.outcome == "PASS", "scored on `action` alone, ignoring the differing note"

    def test_no_key_keeps_the_single_key_and_json_dump_heuristics(self) -> None:
        """Unspecified `key` is unchanged: the heuristics are only bypassed when a key is given."""
        single = ExactMatchScorer()
        assert single.score(output={"a": 1}, expected={"a": 1}, trace_id="t").outcome == "PASS"

        multi = ExactMatchScorer()
        assert (
            multi.score(output={"a": 1, "b": 2}, expected={"b": 2, "a": 1}, trace_id="t").outcome
            == "PASS"
        ), "key order must not matter: both sides serialize sorted"
        assert (
            multi.score(output={"a": 1, "b": 2}, expected={"a": 1, "b": 3}, trace_id="t").outcome
            == "FAIL"
        )
