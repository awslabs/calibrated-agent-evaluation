# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure-relocation primitive.

Gate on the WORKFLOW outcome rate, not the touched agent's rate: a per-agent gain with a flat
outcome rate is relocation, not a fix (Appendix A). Every test embeds its FR + asserts the verdict.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.attribution import CrewOutcomes, relocation_check
from calibrated_agent_evaluation.attribution.errors import AttributionError

# --- failures drop but outcome flat → relocation; drop + improve → improved ---


def test_flat_outcome_with_agent_gain_is_relocation() -> None:
    before = CrewOutcomes(
        workflow_failures=20, workflow_total=100, per_agent_failures={"intake": 18, "policy": 15}
    )
    after = CrewOutcomes(
        workflow_failures=20, workflow_total=100, per_agent_failures={"intake": 3, "policy": 15}
    )
    result = relocation_check(before, after, touched_agent="intake")
    assert result.verdict == "relocation"
    assert result.relocated_from == "intake"
    assert result.workflow_outcome_rate_before == result.workflow_outcome_rate_after


def test_agent_gain_and_outcome_improves_is_improved() -> None:
    before = CrewOutcomes(
        workflow_failures=20, workflow_total=100, per_agent_failures={"intake": 18}
    )
    after = CrewOutcomes(workflow_failures=6, workflow_total=100, per_agent_failures={"intake": 3})
    result = relocation_check(before, after, touched_agent="intake")
    assert result.verdict == "improved"
    assert result.relocated_from is None


# --- (edge) agent gain but outcome REGRESSES → regressed (relocation never masks it)


def test_regression_not_masked_by_agent_gain() -> None:
    before = CrewOutcomes(
        workflow_failures=150, workflow_total=1000, per_agent_failures={"intake": 180}
    )
    after = CrewOutcomes(
        workflow_failures=250, workflow_total=1000, per_agent_failures={"intake": 30}
    )
    result = relocation_check(before, after, touched_agent="intake")
    assert result.verdict == "regressed"
    assert result.relocated_from is None
    assert result.workflow_rate_change is not None
    assert result.workflow_rate_change.lo > 0.0


# --- verdicts read intervals on rates, not raw differences or raw counts ---------


def test_a_bigger_before_slice_at_the_same_rates_is_flat_not_relocation() -> None:
    """1000 traces before, 500 after, same rates: the agent's count halves, its rate does not."""
    before = CrewOutcomes(
        workflow_failures=200, workflow_total=1000, per_agent_failures={"intake": 100}
    )
    after = CrewOutcomes(
        workflow_failures=100, workflow_total=500, per_agent_failures={"intake": 50}
    )
    result = relocation_check(before, after, touched_agent="intake")
    assert result.verdict == "flat"
    assert result.touched_agent_rate_change is not None
    assert result.touched_agent_rate_change.point == 0.0


def test_a_noise_sized_dip_in_the_workflow_rate_is_not_an_improvement() -> None:
    before = CrewOutcomes(
        workflow_failures=200, workflow_total=1000, per_agent_failures={"intake": 100}
    )
    after = CrewOutcomes(
        workflow_failures=199, workflow_total=1000, per_agent_failures={"intake": 100}
    )
    result = relocation_check(before, after, touched_agent="intake")
    assert result.workflow_outcome_rate_after < result.workflow_outcome_rate_before
    assert result.verdict == "flat"


def test_an_unshown_workflow_change_with_a_shown_agent_gain_is_relocation() -> None:
    """15 to 25 of 100 is not a shown regression, and nothing shows the crew improved either."""
    before = CrewOutcomes(
        workflow_failures=15, workflow_total=100, per_agent_failures={"intake": 18}
    )
    after = CrewOutcomes(workflow_failures=25, workflow_total=100, per_agent_failures={"intake": 3})
    result = relocation_check(before, after, touched_agent="intake")
    assert result.workflow_rate_change is not None
    assert result.workflow_rate_change.lo < 0.0 < result.workflow_rate_change.hi
    assert result.verdict == "relocation"


def test_a_failure_count_above_its_slice_is_refused() -> None:
    before = CrewOutcomes(workflow_failures=5, workflow_total=10, per_agent_failures={"intake": 12})
    after = CrewOutcomes(workflow_failures=5, workflow_total=10, per_agent_failures={"intake": 1})
    with pytest.raises(AttributionError, match="intake counts"):
        relocation_check(before, after, touched_agent="intake")


def test_no_movement_is_flat() -> None:
    before = CrewOutcomes(
        workflow_failures=20, workflow_total=100, per_agent_failures={"intake": 18}
    )
    after = CrewOutcomes(
        workflow_failures=20, workflow_total=100, per_agent_failures={"intake": 18}
    )
    result = relocation_check(before, after, touched_agent="intake")
    assert result.verdict == "flat"


# --- a rate over zero traces is not a rate ---------------------------------


def test_empty_after_slice_refuses_rather_than_reporting_a_verdict() -> None:
    """An empty slice used to be scored as a 0.0 rate, so an unmeasured run looked like a fix.

    `_rate` returned `failures / total if total else 0.0`, and 0.0 is a *rate* — so a query that
    matched nothing came back as "0% workflow failures" and the ladder read it as `improved` against
    any non-zero before-rate. A statistical-honesty hazard rather than a cosmetic one: the verdict
    was about an improvement nobody measured.
    """
    before = CrewOutcomes(
        workflow_failures=20, workflow_total=100, per_agent_failures={"intake": 18}
    )
    after = CrewOutcomes(workflow_failures=0, workflow_total=0, per_agent_failures={})

    with pytest.raises(AttributionError) as excinfo:
        relocation_check(before, after, touched_agent="intake")
    assert "after" in str(excinfo.value), "the message must name which side was empty"


def test_empty_before_slice_refuses_too() -> None:
    empty = CrewOutcomes(workflow_failures=0, workflow_total=0, per_agent_failures={})
    after = CrewOutcomes(workflow_failures=1, workflow_total=100, per_agent_failures={"intake": 1})

    with pytest.raises(AttributionError, match="before"):
        relocation_check(empty, after, touched_agent="intake")


def test_a_genuine_improvement_to_zero_failures_still_reports_improved() -> None:
    """The boundary the refusal must not cross: zero *failures* is real, zero traces is not.

    Without this, "refuse an empty slice" could be over-applied to any zero and would then reject
    the best possible outcome — a crew fixed to a 0% workflow failure rate.
    """
    before = CrewOutcomes(
        workflow_failures=20, workflow_total=100, per_agent_failures={"intake": 18}
    )
    after = CrewOutcomes(workflow_failures=0, workflow_total=100, per_agent_failures={"intake": 0})

    result = relocation_check(before, after, touched_agent="intake")
    assert result.verdict == "improved"
    assert result.workflow_outcome_rate_after == 0.0
