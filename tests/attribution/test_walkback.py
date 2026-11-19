# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Counterfactual walk-back — validator + proposer.

Every test embeds its FR and asserts behavior (the validated ``Attribution`` or the typed refusal),
never mock calls. ``CLM-4471`` (intake seq=1 → payout seq=4, culprit≠manifestation) is the real
worked example; ``make_trace``/``make_span`` build the disputed and outcome-pass/span-fail shapes.
"""

from __future__ import annotations

import pytest

from _helpers import make_span, make_trace
from calibrated_agent_evaluation.attribution import (
    AttributionError,
    AttributionSuggestion,
    ratify_suggestion,
    validate_walkback,
)
from calibrated_agent_evaluation.review.errors import RatificationError
from calibrated_agent_evaluation.schema.types import Trace

# --- CLM-4471: culprit=intake precedes manifestation=payout → valid, single ---


def test_walkback_intake_before_payout_is_valid(corpus_clm: Trace, claimscrew) -> None:
    attribution = validate_walkback(
        corpus_clm,
        claimscrew,
        culprit_spans=("CLM-4471/intake",),
        manifestation_span="CLM-4471/payout",
        handoff_implicated="intake->policy-checker",
    )
    assert attribution.status == "single"
    assert attribution.culprit_spans == ("CLM-4471/intake",)
    assert attribution.manifestation_span == "CLM-4471/payout"
    assert attribution.handoff_implicated == "intake->policy-checker"


# --- two culprit spans → disputed, both retained ---


def test_two_culprits_marks_disputed_and_keeps_both(corpus_clm: Trace, claimscrew) -> None:
    attribution = validate_walkback(
        corpus_clm,
        claimscrew,
        culprit_spans=("CLM-4471/intake", "CLM-4471/policy-checker"),
        manifestation_span="CLM-4471/payout",
    )
    assert attribution.status == "disputed"
    assert set(attribution.culprit_spans) == {"CLM-4471/intake", "CLM-4471/policy-checker"}


# --- complementary single culprit → status "single", never collapsed to disputed ---


def test_single_culprit_is_not_disputed(corpus_clm: Trace, claimscrew) -> None:
    attribution = validate_walkback(
        corpus_clm,
        claimscrew,
        culprit_spans=("CLM-4471/intake",),
        manifestation_span="CLM-4471/payout",
    )
    assert attribution.status == "single"


# --- span_defect on a PASS-outcome trace → span-level FAIL, outcome unchanged ---


def _outcome_pass_crew_trace() -> Trace:
    """A 3-span crew whose terminal output is a correct denial (a PASS-outcome trace)."""
    supervisor = make_span(
        span_id="C/supervisor",
        trace_id="C",
        agent="supervisor",
        role="supervisor",
        parent=None,
        seq=0,
    )
    intake = make_span(
        span_id="C/intake",
        trace_id="C",
        agent="intake",
        role="specialist",
        parent="C/supervisor",
        seq=1,
        output={"amount_requested_usd": 12000},
    )
    policy = make_span(
        span_id="C/policy-checker",
        trace_id="C",
        agent="policy-checker",
        role="specialist",
        parent="C/supervisor",
        seq=2,
    )
    return make_trace(
        trace_id="C",
        output={"decision": "DENIED"},
        spans=[supervisor, intake, policy],
    )


def test_span_defect_on_pass_trace_does_not_change_outcome(claimscrew) -> None:
    trace = _outcome_pass_crew_trace()
    attribution = validate_walkback(
        trace,
        claimscrew,
        culprit_spans=("C/intake",),
        manifestation_span=None,
        span_defect=True,
    )
    # The span-level defect is recorded independently of the trace outcome (still a correct denial).
    assert attribution.span_defect is True
    assert trace.output == {"decision": "DENIED"}


# --- a proposed culprit is recorded only through a named human's ratification ---


def test_ratified_suggestion_is_returned_for_the_caller_to_record(corpus_clm: Trace) -> None:
    suggestion = AttributionSuggestion(
        culprit_span=f"{corpus_clm.trace_id}/intake", critique="intake dropped the vacancy signal"
    )
    # Ratifying with a named human returns the suggestion for the caller to record.
    ratified = ratify_suggestion(suggestion, ratified_by="dana@team")
    assert ratified is suggestion
    assert ratified.culprit_span == "CLM-4471/intake"


# --- (negative) absent span / culprit after manifestation → AttributionError ---


def test_culprit_span_absent_from_trace_refused(corpus_clm: Trace, claimscrew) -> None:
    with pytest.raises(AttributionError, match="not present in trace"):
        validate_walkback(
            corpus_clm,
            claimscrew,
            culprit_spans=("CLM-4471/nonexistent",),
            manifestation_span="CLM-4471/payout",
        )


def test_culprit_after_manifestation_refused(corpus_clm: Trace, claimscrew) -> None:
    # payout (seq=4) is AFTER intake (seq=1): a failure cannot originate after it surfaced.
    with pytest.raises(AttributionError, match="strictly after the manifestation"):
        validate_walkback(
            corpus_clm,
            claimscrew,
            culprit_spans=("CLM-4471/payout",),
            manifestation_span="CLM-4471/intake",
        )


def test_empty_culprit_list_refused(corpus_clm: Trace, claimscrew) -> None:
    with pytest.raises(AttributionError, match="at least one culprit"):
        validate_walkback(
            corpus_clm, claimscrew, culprit_spans=(), manifestation_span="CLM-4471/payout"
        )


# --- (negative) undeclared handoff seam → AttributionError ---


def test_undeclared_handoff_refused(corpus_clm: Trace, claimscrew) -> None:
    with pytest.raises(AttributionError, match="not a declared seam"):
        validate_walkback(
            corpus_clm,
            claimscrew,
            culprit_spans=("CLM-4471/intake",),
            manifestation_span="CLM-4471/payout",
            handoff_implicated="intake->payout",
        )


# --- (negative) unratified suggestion → RatificationError ---


def test_unratified_suggestion_refused() -> None:
    suggestion = AttributionSuggestion(culprit_span="CLM-4471/intake", critique="dropped fact")
    with pytest.raises(RatificationError, match="ratified by a human"):
        ratify_suggestion(suggestion, ratified_by=None)
