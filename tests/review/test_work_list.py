# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a review session covers: resume and scope.

``build_app`` used to seed the work-list from an unfiltered ``backend.query()`` and nothing else,
so a restarted session began at trace 1 and re-labeling duplicated labels, and there was no way to
review less than the whole store.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.review import ReviewError, build_label
from calibrated_agent_evaluation.review.session import WorkScope, build_work_list
from calibrated_agent_evaluation.schema import BackendError, Trace

from .conftest import GOLD_RATER, FakeBackend


def _label(trace_id: str, rater: str | None, span_id: str | None = None) -> Any:
    return build_label(
        trace_id=trace_id, span_id=span_id, outcome="PASS", critique="", rater_id=rater
    )


# --- resume ----------------------------------------------------------------------------------


def test_a_restarted_session_skips_what_this_rater_already_labeled(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    backend.put_score(_label("ORD-1001", GOLD_RATER))
    _client, app = make_client(backend)
    assert app.state.review_session.pending == ["ORD-1002", "CLM-4471"]
    assert app.state.already_labeled == ["ORD-1001"]


def test_another_raters_label_does_not_hide_a_trace(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    """Dual coding: a second rater must see traces the first rater already labeled."""
    backend.put_score(_label("ORD-1001", "someone-else"))
    backend.put_score(_label("ORD-1002", None))  # an unattributed pre-rater_id label
    _client, app = make_client(backend)
    assert app.state.review_session.pending == ["ORD-1001", "ORD-1002", "CLM-4471"]


def test_a_span_level_label_alone_does_not_count_as_labeling_the_trace(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    backend.put_score(_label("CLM-4471", GOLD_RATER, span_id="CLM-4471/intake"))
    _client, app = make_client(backend)
    assert "CLM-4471" in app.state.review_session.pending


def test_relabel_keeps_already_labeled_traces(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    backend.put_score(_label("ORD-1001", GOLD_RATER))
    _client, app = make_client(backend, scope=WorkScope(relabel=True))
    assert app.state.review_session.pending == ["ORD-1001", "ORD-1002", "CLM-4471"]


# --- scope -----------------------------------------------------------------------------------


def test_a_trace_scope_is_exactly_that_trace(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    _client, app = make_client(backend, scope=WorkScope(trace_id="CLM-4471"))
    assert app.state.review_session.pending == ["CLM-4471"]


def test_an_unknown_trace_is_refused_rather_than_opening_an_empty_session(
    backend: FakeBackend,
) -> None:
    with pytest.raises(BackendError):
        build_work_list(backend, WorkScope(trace_id="NOPE"), rater_id="r")


def test_frame_and_cell_narrow_the_work_list(corpus_traces: list[Trace]) -> None:
    # the corpus is all production-representative; make one trace a discovery trace
    traces = [replace(corpus_traces[0], frame="discovery"), *corpus_traces[1:]]
    backend = FakeBackend(traces)
    work = build_work_list(backend, WorkScope(frame="discovery"), rater_id="r")
    assert [t.trace_id for t in work.traces] == [traces[0].trace_id]

    cell = dict(corpus_traces[2].taxonomy_cell)
    work = build_work_list(backend, WorkScope(cell=cell), rater_id="r")
    assert corpus_traces[2].trace_id in [t.trace_id for t in work.traces]
    assert all(t.taxonomy_cell.items() >= cell.items() for t in work.traces)


def test_a_scope_matching_nothing_is_refused(backend: FakeBackend) -> None:
    with pytest.raises(ReviewError, match="matches the requested scope"):
        build_work_list(backend, WorkScope(cell={"feature": "no-such-feature"}), rater_id="r")


def test_limit_counts_traces_still_to_label(backend: FakeBackend) -> None:
    backend.put_score(_label("ORD-1001", "r"))
    work = build_work_list(backend, WorkScope(limit=1), rater_id="r")
    assert [t.trace_id for t in work.traces] == ["ORD-1002"]


@pytest.mark.parametrize("bad", [0, -3])
def test_a_non_positive_limit_is_refused(backend: FakeBackend, bad: int) -> None:
    with pytest.raises(ReviewError, match="limit"):
        build_work_list(backend, WorkScope(limit=bad), rater_id="r")
