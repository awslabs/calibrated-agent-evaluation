# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM pre-fill with human ratification."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.review import NoPrefill, RatificationError, Suggestion, build_label
from calibrated_agent_evaluation.schema import Trace


def test_accepted_suggestion_yields_a_human_label() -> None:
    suggestion = Suggestion(outcome="PASS", critique="in-window refund, correct")
    label = build_label(
        trace_id="ORD-1002",
        outcome=suggestion.outcome,
        critique=suggestion.critique,
        from_suggestion=True,
        ratified=True,  # the human accepted/overrode — this is the ratification act
    )
    assert label.kind == "human"
    assert label.outcome == "PASS"
    assert label.critique == "in-window refund, correct"


def test_unratified_suggestion_is_refused() -> None:
    with pytest.raises(RatificationError):
        build_label(
            trace_id="ORD-1002",
            outcome="PASS",
            critique="auto-suggested",
            from_suggestion=True,
            ratified=False,
        )


def test_noprefill_offers_no_suggestion(corpus_traces: list[Trace]) -> None:
    assert NoPrefill().suggest(corpus_traces[0]) is None
