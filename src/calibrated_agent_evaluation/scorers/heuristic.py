# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete scorer classes wrapping autoevals heuristic functions.

Each scorer wraps one autoevals heuristic, maps its ``Score(score=float)`` result to a
``Label(outcome, critique)``, and declares ``lane=CI_GATE`` (all heuristics are deterministic
structured-target checks, valid as CI gates).

Written against ``autoevals==0.3.0``. The verified surface:
``ExactMatch().eval(output, expected) -> Score``, ``LevenshteinScorer().eval(...)``,
``NumericDiff().eval(...)``, ``JSONDiff().eval(...)``, ``ValidJSON().eval(...)``.
"""

from __future__ import annotations

from typing import Any

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.schema.types import Label
from calibrated_agent_evaluation.scorers.base import ScorerLane
from calibrated_agent_evaluation.scorers.errors import ScorerError


def _load_autoevals() -> Any:
    """Lazy-import autoevals; raise ScorerError if not installed."""
    try:
        import autoevals
    except ImportError as exc:
        raise ScorerError(
            f"autoevals is not installed. Install via: {install_hint('scorers')}"
        ) from exc
    return autoevals


def _extract_value(data: dict[str, Any], key: str | None = None, *, side: str = "output") -> Any:
    """Extract the scoreable value from an output/expected dict.

    A ``key`` is a **requirement**, not a preference: if it is given and absent, this raises rather
    than scoring something else. It used to fall through to the heuristics below, so a caller asking
    for ``key="action"`` against ``{"answer": ...}`` silently got the answer compared instead; a
    scorer reporting PASS/FAIL about a field nobody asked about, with no error anywhere. A typo in a
    key name is a caller bug and has to surface as one.

    With no ``key`` at all the heuristics stand: a single-key dict yields its value, and anything
    else is compared as its sorted JSON serialization.
    """
    if key is not None:
        if key not in data:
            raise ScorerError(
                f"scorer key {key!r} is not present in the {side} dict (its keys are "
                f"{sorted(data)}); refusing to score a different field instead; pass a key that "
                "exists, or omit key= to compare the whole dict"
            )
        return data[key]
    if len(data) == 1:
        return next(iter(data.values()))
    import json

    return json.dumps(data, sort_keys=True)


class ExactMatchScorer:
    """Binary exact-match check. PASS if output == expected, FAIL otherwise."""

    lane = ScorerLane.CI_GATE

    def __init__(self, *, key: str | None = None) -> None:
        self._key = key

    def score(
        self,
        output: dict[str, Any],
        expected: dict[str, Any] | None,
        *,
        trace_id: str,
    ) -> Label:
        """Score ``output`` against ``expected`` as PASS or FAIL.

        Args:
            output: The agent's output.
            expected: The expected output; required.
            trace_id: The trace the label is written for.

        Returns:
            A label carrying the verdict.

        Raises:
            ScorerError: ``expected`` is missing or a value can't be compared.
        """
        if expected is None:
            raise ScorerError("ExactMatchScorer requires an expected value")
        ae = _load_autoevals()
        out_val = _extract_value(output, self._key)
        exp_val = _extract_value(expected, self._key, side="expected")
        try:
            result = ae.ExactMatch().eval(output=str(out_val), expected=str(exp_val))
        except Exception as exc:
            raise ScorerError(f"ExactMatchScorer failed: {exc}") from exc
        outcome = "PASS" if result.score == 1.0 else "FAIL"
        if outcome == "PASS":
            critique = f"ExactMatch: output matches expected ('{exp_val}')"
        else:
            critique = f"ExactMatch: got '{out_val}', expected '{exp_val}'"
        return Label(
            trace_id=trace_id, span_id=None, kind="code", outcome=outcome, critique=critique
        )


class LevenshteinScorer:
    """Edit-distance similarity check. Configurable threshold."""

    lane = ScorerLane.CI_GATE

    def __init__(self, *, threshold: float = 0.9, key: str | None = None) -> None:
        self._threshold = threshold
        self._key = key

    def score(
        self,
        output: dict[str, Any],
        expected: dict[str, Any] | None,
        *,
        trace_id: str,
    ) -> Label:
        """Score ``output`` against ``expected`` as PASS or FAIL.

        Args:
            output: The agent's output.
            expected: The expected output; required.
            trace_id: The trace the label is written for.

        Returns:
            A label carrying the verdict.

        Raises:
            ScorerError: ``expected`` is missing or a value can't be compared.
        """
        if expected is None:
            raise ScorerError("LevenshteinScorer requires an expected value")
        ae = _load_autoevals()
        out_val = _extract_value(output, self._key)
        exp_val = _extract_value(expected, self._key, side="expected")
        try:
            result = ae.LevenshteinScorer().eval(output=str(out_val), expected=str(exp_val))
        except Exception as exc:
            raise ScorerError(f"LevenshteinScorer failed: {exc}") from exc
        score_val = result.score if result.score is not None else 0.0
        outcome = "PASS" if score_val >= self._threshold else "FAIL"
        critique = (
            f"Levenshtein: similarity {score_val:.3f} "
            f"{'≥' if outcome == 'PASS' else '<'} threshold {self._threshold}"
        )
        return Label(
            trace_id=trace_id, span_id=None, kind="code", outcome=outcome, critique=critique
        )


class NumericDiffScorer:
    """Numeric closeness check. Expects numeric values."""

    lane = ScorerLane.CI_GATE

    def __init__(self, *, threshold: float = 0.99, key: str | None = None) -> None:
        self._threshold = threshold
        self._key = key

    def score(
        self,
        output: dict[str, Any],
        expected: dict[str, Any] | None,
        *,
        trace_id: str,
    ) -> Label:
        """Score ``output`` against ``expected`` as PASS or FAIL.

        Args:
            output: The agent's output.
            expected: The expected output; required.
            trace_id: The trace the label is written for.

        Returns:
            A label carrying the verdict.

        Raises:
            ScorerError: ``expected`` is missing or a value can't be compared.
        """
        if expected is None:
            raise ScorerError("NumericDiffScorer requires an expected value")
        out_val = _extract_value(output, self._key)
        exp_val = _extract_value(expected, self._key, side="expected")
        try:
            out_num = float(out_val)
            exp_num = float(exp_val)
        except (TypeError, ValueError) as exc:
            raise ScorerError(
                f"NumericDiffScorer: expected numeric values, got output={out_val!r}, "
                f"expected={exp_val!r}"
            ) from exc
        ae = _load_autoevals()
        try:
            result = ae.NumericDiff().eval(output=out_num, expected=exp_num)
        except Exception as exc:
            raise ScorerError(f"NumericDiffScorer failed: {exc}") from exc
        score_val = result.score if result.score is not None else 0.0
        outcome = "PASS" if score_val >= self._threshold else "FAIL"
        critique = (
            f"NumericDiff: closeness {score_val:.4f} "
            f"{'≥' if outcome == 'PASS' else '<'} threshold {self._threshold} "
            f"(output={out_num}, expected={exp_num})"
        )
        return Label(
            trace_id=trace_id, span_id=None, kind="code", outcome=outcome, critique=critique
        )


class JSONDiffScorer:
    """Structural JSON comparison. Configurable threshold."""

    lane = ScorerLane.CI_GATE

    def __init__(self, *, threshold: float = 1.0, key: str | None = None) -> None:
        self._threshold = threshold
        self._key = key

    def score(
        self,
        output: dict[str, Any],
        expected: dict[str, Any] | None,
        *,
        trace_id: str,
    ) -> Label:
        """Score ``output`` against ``expected`` as PASS or FAIL.

        Args:
            output: The agent's output.
            expected: The expected output; required.
            trace_id: The trace the label is written for.

        Returns:
            A label carrying the verdict.

        Raises:
            ScorerError: ``expected`` is missing or a value can't be compared.
        """
        if expected is None:
            raise ScorerError("JSONDiffScorer requires an expected value")
        import json

        out_val = _extract_value(output, self._key)
        exp_val = _extract_value(expected, self._key, side="expected")
        out_str = json.dumps(out_val, sort_keys=True) if not isinstance(out_val, str) else out_val
        exp_str = json.dumps(exp_val, sort_keys=True) if not isinstance(exp_val, str) else exp_val
        ae = _load_autoevals()
        try:
            result = ae.JSONDiff().eval(output=out_str, expected=exp_str)
        except Exception as exc:
            raise ScorerError(f"JSONDiffScorer failed: {exc}") from exc
        score_val = result.score if result.score is not None else 0.0
        outcome = "PASS" if score_val >= self._threshold else "FAIL"
        critique = (
            f"JSONDiff: similarity {score_val:.3f} "
            f"{'≥' if outcome == 'PASS' else '<'} threshold {self._threshold}"
        )
        return Label(
            trace_id=trace_id, span_id=None, kind="code", outcome=outcome, critique=critique
        )


class ValidJSONScorer:
    """JSON validity check. Does not require expected value."""

    lane = ScorerLane.CI_GATE

    def __init__(self, *, key: str | None = None) -> None:
        self._key = key

    def score(
        self,
        output: dict[str, Any],
        expected: dict[str, Any] | None = None,  # noqa: ARG002
        *,
        trace_id: str,
    ) -> Label:
        """Score ``output`` as PASS when it is valid JSON, FAIL otherwise.

        Args:
            output: The agent's output.
            expected: Ignored.
            trace_id: The trace the label is written for.

        Returns:
            A label carrying the verdict.

        Raises:
            ScorerError: The value to check can't be read from ``output``.
        """
        ae = _load_autoevals()
        out_val = _extract_value(output, self._key)
        out_str = str(out_val) if not isinstance(out_val, str) else out_val
        try:
            result = ae.ValidJSON().eval(output=out_str, expected=None)
        except Exception as exc:
            raise ScorerError(f"ValidJSONScorer failed: {exc}") from exc
        outcome = "PASS" if result.score == 1 else "FAIL"
        if outcome == "PASS":
            critique = "ValidJSON: output is valid JSON"
        else:
            critique = f"ValidJSON: output is not valid JSON ('{out_str[:50]}...')"
        return Label(
            trace_id=trace_id, span_id=None, kind="code", outcome=outcome, critique=critique
        )
