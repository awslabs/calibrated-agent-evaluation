# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generic-metric rule applied to the scorers behind a gate's verdicts."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.enforce import (
    GENERIC_GENERATION_METRICS,
    EnforcementContext,
    GenericMetricInCIError,
    lint_scorer_lanes,
)


def test_a_generic_scorer_in_ci_is_refused_with_its_decision_recorded() -> None:
    with pytest.raises(GenericMetricInCIError) as exc:
        lint_scorer_lanes(["exact", "ROUGE"], EnforcementContext(ci=True))
    decision = exc.value.decision
    assert decision is not None
    assert decision.tier == "hard_refusal"
    assert decision.inputs == {"scorers": "ROUGE", "ci": "true"}
    assert "'ROUGE'" in str(exc.value)


def test_outside_ci_each_generic_scorer_is_one_exploration_only_decision() -> None:
    decisions = lint_scorer_lanes(["bleu", "rouge", "rouge"], EnforcementContext(ci=False))
    assert [d.inputs["scorer"] for d in decisions] == ["bleu", "rouge"]
    assert {d.tier for d in decisions} == {"display_default"}


@pytest.mark.parametrize("ci", [True, False])
def test_structured_scorers_pass_with_no_decision(ci: bool) -> None:
    assert lint_scorer_lanes(["exact", "json-diff", "numeric"], EnforcementContext(ci=ci)) == []
    assert lint_scorer_lanes([], EnforcementContext(ci=ci)) == []


def test_every_listed_generic_metric_is_refused_in_ci() -> None:
    for name in sorted(GENERIC_GENERATION_METRICS):
        with pytest.raises(GenericMetricInCIError):
            lint_scorer_lanes([name], EnforcementContext(ci=True))
