# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder turns agreed trace-level human FAILs into scenarios, joined with the answer key."""

from __future__ import annotations

from typing import Any

import pytest

from calibrated_agent_evaluation.scenarios.build import (
    CONFLICTING_EXPECTED,
    RATERS_DISAGREE,
    build_slice,
    expected_text,
)
from calibrated_agent_evaluation.scenarios.errors import ScenarioError
from calibrated_agent_evaluation.scenarios.types import AnswerKey, PredefinedScenario
from calibrated_agent_evaluation.schema.types import Label, Outcome
from cli_store_helpers import trace

POLICY = "trusted_broken_policy_tool"
LOOKUP = "skipped_order_lookup"
DENIED = {"action": "REFUND_DENIED"}


def _label(
    trace_id: str,
    outcome: Outcome,
    *,
    mode: str | None = POLICY,
    rater: str = "ana",
    expected: dict[str, Any] | None = None,
    span_id: str | None = None,
) -> Label:
    return Label(
        trace_id=trace_id,
        span_id=span_id,
        kind="human",
        outcome=outcome,
        critique="The refund broke the return policy.",
        failure_mode=mode if outcome == "FAIL" else None,
        expected=expected,
        rater_id=rater,
    )


def _key(**data: Any) -> AnswerKey:
    return AnswerKey.model_validate(data)


def _scenario(result: Any, scenario_id: str) -> PredefinedScenario:
    assert result.slice is not None
    found = {s.scenario_id: s for s in result.slice.scenarios}
    scenario = found[scenario_id]
    assert isinstance(scenario, PredefinedScenario)
    return scenario


def test_only_trace_level_fails_become_scenarios() -> None:
    traces = [trace("T-PASS-SPAN-FAIL"), trace("T-FAIL-NO-EXPECTED"), trace("T-PASS")]
    labels = {
        # A trace-level PASS whose FAIL is on one span is not a failure of the trace.
        "T-PASS-SPAN-FAIL": [
            _label("T-PASS-SPAN-FAIL", "PASS"),
            _label("T-PASS-SPAN-FAIL", "FAIL", span_id="T-PASS-SPAN-FAIL/intake"),
        ],
        "T-FAIL-NO-EXPECTED": [_label("T-FAIL-NO-EXPECTED", "FAIL")],
        "T-PASS": [_label("T-PASS", "PASS")],
    }

    result = build_slice(traces, labels)

    assert result.slice is not None
    assert [s.scenario_id for s in result.slice.scenarios] == ["T-FAIL-NO-EXPECTED"]
    scenario = _scenario(result, "T-FAIL-NO-EXPECTED")
    assert scenario.turns[0].input == {"message": "refund T-FAIL-NO-EXPECTED"}
    assert scenario.turns[0].expected_response is None
    assert scenario.calibrated_agent_evaluation.expected is None
    assert result.left_out == {}


def test_scenario_carries_the_labels_expected_output_and_the_trace_context() -> None:
    labels = {"ORD-1": [_label("ORD-1", "FAIL", expected=DENIED)]}

    scenario = _scenario(build_slice([trace("ORD-1")], labels), "ORD-1")

    assert scenario.turns[0].expected_response == expected_text(DENIED)
    assert scenario.calibrated_agent_evaluation.expected == DENIED
    assert scenario.calibrated_agent_evaluation.source_trace_id == "ORD-1"
    assert scenario.calibrated_agent_evaluation.frame == "production-representative"
    assert scenario.calibrated_agent_evaluation.taxonomy_cell == {"feature": "refund"}
    assert scenario.calibrated_agent_evaluation.failure_modes == [POLICY]


def test_answer_key_joins_by_failure_mode_with_trace_overrides() -> None:
    labels = {
        "TWO-MODES": [
            _label("TWO-MODES", "FAIL", mode=POLICY, rater="ana"),
            _label("TWO-MODES", "FAIL", mode=LOOKUP, rater="bo"),
        ],
        "OVERRIDDEN": [_label("OVERRIDDEN", "FAIL")],
    }
    key = _key(
        by_failure_mode={
            POLICY: {
                "assertions": ["Agent checks the return window", "Agent stays polite"],
                "expected_trajectory": ["lookup_order", "check_refund_policy"],
            },
            LOOKUP: {"assertions": ["Agent looks up the order first", "Agent stays polite"]},
        },
        by_trace={
            "OVERRIDDEN": {
                "assertions": ["Agent escalates to a human"],
                "expected_trajectory": ["escalate"],
            }
        },
    )

    result = build_slice([trace("TWO-MODES"), trace("OVERRIDDEN")], labels, key)

    both = _scenario(result, "TWO-MODES")
    assert both.assertions == [
        "Agent checks the return window",
        "Agent stays polite",
        "Agent looks up the order first",
    ]
    assert both.expected_trajectory == ["lookup_order", "check_refund_policy"]
    overridden = _scenario(result, "OVERRIDDEN")
    assert overridden.assertions == ["Agent escalates to a human"]
    assert overridden.expected_trajectory == ["escalate"]


def test_trace_override_replaces_only_the_fields_it_sets() -> None:
    labels = {"ORD-1": [_label("ORD-1", "FAIL")]}
    key = _key(
        by_failure_mode={
            POLICY: {"assertions": ["mode rule"], "expected_trajectory": ["lookup_order"]},
            "never_seen_mode": {"assertions": ["unused"]},
        },
        by_trace={"ORD-1": {"assertions": ["trace rule"]}},
    )

    result = build_slice([trace("ORD-1")], labels, key)

    scenario = _scenario(result, "ORD-1")
    assert scenario.assertions == ["trace rule"]
    assert scenario.expected_trajectory == ["lookup_order"]
    # A key entry no scenario uses is reported, not refused.
    assert result.unused_modes == ["never_seen_mode"]


def test_conflicting_traces_are_left_out_and_named() -> None:
    labels = {
        "DISPUTED-EXPECTED": [
            _label("DISPUTED-EXPECTED", "FAIL", rater="ana", expected=DENIED),
            _label("DISPUTED-EXPECTED", "FAIL", rater="bo", expected={"action": "REFUND_ISSUED"}),
        ],
        "DISPUTED-OUTCOME": [
            _label("DISPUTED-OUTCOME", "FAIL", rater="ana"),
            _label("DISPUTED-OUTCOME", "PASS", rater="bo"),
        ],
        "AGREED": [_label("AGREED", "FAIL", rater="ana"), _label("AGREED", "FAIL", rater="bo")],
    }
    traces = [trace("DISPUTED-EXPECTED"), trace("DISPUTED-OUTCOME"), trace("AGREED")]

    result = build_slice(traces, labels)

    assert result.left_out == {
        CONFLICTING_EXPECTED: ["DISPUTED-EXPECTED"],
        RATERS_DISAGREE: ["DISPUTED-OUTCOME"],
    }
    assert result.slice is not None
    assert [s.scenario_id for s in result.slice.scenarios] == ["AGREED"]


def test_a_raters_relabel_to_pass_removes_the_trace() -> None:
    labels = {"ORD-1": [_label("ORD-1", "FAIL"), _label("ORD-1", "PASS")]}

    result = build_slice([trace("ORD-1")], labels)

    assert result.slice is None
    assert result.left_out == {}


def test_ambiguous_or_unknown_answer_key_entries_are_refused() -> None:
    labels = {
        "TWO-MODES": [
            _label("TWO-MODES", "FAIL", mode=POLICY, rater="ana"),
            _label("TWO-MODES", "FAIL", mode=LOOKUP, rater="bo"),
        ]
    }
    clashing = _key(
        by_failure_mode={
            POLICY: {"expected_trajectory": ["lookup_order", "check_refund_policy"]},
            LOOKUP: {"expected_trajectory": ["lookup_order"]},
        }
    )

    with pytest.raises(ScenarioError, match="TWO-MODES") as refused:
        build_slice([trace("TWO-MODES")], labels, clashing)
    assert POLICY in str(refused.value)
    assert LOOKUP in str(refused.value)

    # Setting the trace's tool order under by_trace resolves it.
    resolved = clashing.model_copy(
        update={"by_trace": _key(by_trace={"TWO-MODES": {"expected_trajectory": ["x"]}}).by_trace}
    )
    resolved_scenario = _scenario(build_slice([trace("TWO-MODES")], labels, resolved), "TWO-MODES")
    assert resolved_scenario.expected_trajectory == ["x"]

    with pytest.raises(ScenarioError, match="NOT-IN-SCOPE"):
        build_slice([trace("TWO-MODES")], labels, _key(by_trace={"NOT-IN-SCOPE": {}}))


def test_expected_comes_from_each_raters_newest_label_that_records_one() -> None:
    # The rule `calibrated-agent-evaluation score` uses: a later FAIL without an expected does not
    # erase an earlier one.
    labels = {
        "ORD-1": [
            _label("ORD-1", "FAIL", rater="ana", expected=DENIED),
            _label("ORD-1", "FAIL", rater="ana"),
        ]
    }

    scenario = _scenario(build_slice([trace("ORD-1")], labels), "ORD-1")

    assert scenario.calibrated_agent_evaluation.expected == DENIED
    assert scenario.turns[0].expected_response == expected_text(DENIED)
