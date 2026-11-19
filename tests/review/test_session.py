# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""ReviewSession behavior: defer, undo, timing, completion."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.review import ReviewError, ReviewSession, build_label
from calibrated_agent_evaluation.schema import Label


class _Clock:
    """Deterministic monotonic stand-in — returns the queued values in order (no wall-clock)."""

    def __init__(self, values: list[float]) -> None:
        self._values = values
        self._i = 0

    def __call__(self) -> float:
        value = self._values[min(self._i, len(self._values) - 1)]
        self._i += 1
        return value


def _pass(trace_id: str) -> Label:
    return build_label(trace_id=trace_id, outcome="PASS", critique="")


def test_defer_moves_current_to_end() -> None:
    session = ReviewSession.from_ids(["A", "B", "C"])
    new_current = session.defer()
    assert session.pending == ["B", "C", "A"]
    assert new_current == "B"


def test_defer_single_item_keeps_it_current() -> None:
    session = ReviewSession.from_ids(["A"])
    assert session.defer() == "A"
    assert session.pending == ["A"]


def test_undo_retracts_last_and_returns_to_that_trace() -> None:
    session = ReviewSession.from_ids(["A", "B"])
    session.record(_pass("A"))
    assert session.pending == ["B"]
    entry = session.undo()
    assert entry is not None
    assert entry.trace_id == "A"
    assert session.current() == "A"
    assert session.summary()["labeled_count"] == 0


def test_undo_on_empty_history_is_noop() -> None:
    session = ReviewSession.from_ids(["A"])
    assert session.undo() is None
    assert session.pending == ["A"]


def test_summary_reports_count_and_elapsed_on_trace() -> None:
    # shown@100 -> recorded@105 = 5.0s on A; shown@105 -> recorded@108 = 3.0s on B.
    clock = _Clock([100.0, 105.0, 105.0, 108.0])
    session = ReviewSession.from_ids(["A", "B"], clock=clock)
    session.mark_shown()
    session.record(_pass("A"))
    session.mark_shown()
    session.record(_pass("B"))
    summary = session.summary()
    assert summary["labeled_count"] == 2
    assert summary["elapsed_seconds"] == 8.0


def test_summary_zero_labeled_is_zero() -> None:
    session = ReviewSession.from_ids(["A"])
    assert session.summary() == {"labeled_count": 0, "labeled_seconds": 0, "elapsed_seconds": 0.0}


def test_is_complete_only_when_worklist_empty() -> None:
    session = ReviewSession.from_ids(["A"])
    assert session.is_complete is False
    session.record(_pass("A"))
    assert session.is_complete is True
    assert session.current() is None


def test_a_span_label_attaches_to_the_verdict_it_was_written_with() -> None:
    session = ReviewSession.from_ids(["A", "B"])
    session.record(_pass("A"))
    span = build_label(
        trace_id="A", span_id="A/x", outcome="FAIL", critique="bad", span_defect=True
    )
    entry = session.attach_span_label(span)
    assert entry.span_labels == (span,)
    assert session.undo() == entry  # undo hands back everything the submission wrote


def test_a_span_label_for_another_trace_is_refused() -> None:
    session = ReviewSession.from_ids(["A", "B"])
    with pytest.raises(ReviewError):
        session.attach_span_label(_pass("A"))  # nothing recorded yet
    session.record(_pass("A"))
    with pytest.raises(ReviewError):
        session.attach_span_label(_pass("B"))


# --- time is attributed to the trace on screen, and never dropped ---------


class _Manual:
    """A clock the test sets by hand."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_time_is_charged_to_the_trace_shown_not_the_queue_head() -> None:
    clock = _Manual()
    session = ReviewSession.from_ids(["A", "B", "C"], clock=clock)
    session.mark_shown("A")
    clock.t = 2.0
    session.mark_shown("B")  # the labeler stepped right to B: A's 2s are banked, not lost
    clock.t = 9.0
    entry = session.record(_pass("B"))
    assert entry.seconds_on_trace == 7.0
    assert session.summary()["elapsed_seconds"] == 9.0


def test_deferred_time_is_counted_in_the_session_total() -> None:
    clock = _Manual()
    session = ReviewSession.from_ids(["A", "B"], clock=clock)
    session.mark_shown("A")
    clock.t = 30.0
    session.defer("A")
    session.mark_shown("B")
    clock.t = 40.0
    session.record(_pass("B"))
    summary = session.summary()
    assert summary["labeled_seconds"] == 10.0
    assert summary["elapsed_seconds"] == 40.0  # the 30s spent on A before deferring it count


def test_undone_time_stays_in_the_session_total() -> None:
    clock = _Manual()
    session = ReviewSession.from_ids(["A"], clock=clock)
    session.mark_shown("A")
    clock.t = 5.0
    session.record(_pass("A"))
    session.undo()
    assert session.summary()["labeled_count"] == 0
    assert session.summary()["elapsed_seconds"] == 5.0


def test_re_rendering_the_same_trace_does_not_restart_its_clock() -> None:
    """A refused submit re-renders the trace; the time already spent on it must survive."""
    clock = _Manual()
    session = ReviewSession.from_ids(["A"], clock=clock)
    session.mark_shown("A")
    clock.t = 12.0
    session.mark_shown("A")
    clock.t = 15.0
    assert session.record(_pass("A")).seconds_on_trace == 15.0
