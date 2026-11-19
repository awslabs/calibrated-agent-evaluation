# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The scenario model accepts well-formed test cases and slices and refuses every malformed one."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pytest

from calibrated_agent_evaluation.scenarios.errors import ScenarioError
from calibrated_agent_evaluation.scenarios.types import (
    ActorProfile,
    CalibratedAgentEvaluationScenarioMeta,
    DatasetSlice,
    PredefinedScenario,
    SimulatedScenario,
    Turn,
)


def _predefined(scenario_id: str = "ORD-1001", **extra: Any) -> PredefinedScenario:
    return PredefinedScenario(
        scenario_id=scenario_id,
        turns=[Turn(input={"message": "I want a refund for order A-7781."})],
        **extra,
    )


def _simulated(scenario_id: str = "angry-refund", **extra: Any) -> SimulatedScenario:
    return SimulatedScenario(
        scenario_id=scenario_id,
        actor_profile=ActorProfile(context="Order A-7781 arrived late", goal="Get a refund"),
        input="Where is my refund?",
        **extra,
    )


def test_valid_scenarios_and_slices_construct() -> None:
    multi_turn = PredefinedScenario(
        scenario_id="two-turns",
        turns=[
            Turn(input="What is 15 + 27?", expected_response="42"),
            Turn(input={"role": "user", "content": "And the weather?"}),
        ],
        expected_trajectory=["calculator", "weather"],
        assertions=["Agent uses the calculator for arithmetic"],
        metadata={"owner": "orderbot-team"},
        calibrated_agent_evaluation=CalibratedAgentEvaluationScenarioMeta(
            source_trace_id="ORD-9", frame="discovery"
        ),
    )
    simulated = _simulated(
        assertions=["Agent refunds to the original card"],
        max_turns=1,
        scenario_description="An impatient customer",
    )

    predefined_slice = DatasetSlice(form="predefined", scenarios=[_predefined(), multi_turn])
    simulated_slice = DatasetSlice(form="simulated", scenarios=[simulated])

    assert [s.scenario_id for s in predefined_slice.scenarios] == ["ORD-1001", "two-turns"]
    assert predefined_slice.published is None
    assert isinstance(simulated_slice.scenarios[0], SimulatedScenario)
    # A slice read back from plain data picks each scenario's form from its "form" field.
    reread = DatasetSlice.model_validate(simulated_slice.model_dump(mode="json"))
    assert reread == simulated_slice


_INVALID: dict[str, tuple[Callable[[], object], str]] = {
    "empty scenario id": (lambda: _predefined(scenario_id=""), "scenario_id"),
    "no turns": (
        lambda: PredefinedScenario(scenario_id="a", turns=[]),
        "turns",
    ),
    "empty string input": (lambda: Turn(input="  "), "input"),
    "empty object input": (lambda: Turn(input={}), "input"),
    "empty actor context": (lambda: ActorProfile(context="", goal="g"), "context"),
    "empty actor goal": (lambda: ActorProfile(context="c", goal=""), "goal"),
    "zero max turns": (lambda: _simulated(max_turns=0), "max_turns"),
    "simulated with a tool order": (
        lambda: SimulatedScenario.model_validate(
            {**_simulated().model_dump(), "expected_trajectory": ["lookup_order"]}
        ),
        "expected_trajectory",
    ),
    "simulated with scripted turns": (
        lambda: SimulatedScenario.model_validate(
            {**_simulated().model_dump(), "turns": [{"input": "hi", "expected_response": "ok"}]}
        ),
        "turns",
    ),
    "empty assertion": (lambda: _predefined(assertions=[""]), "assertions"),
    "reserved metadata key": (
        lambda: _predefined(metadata={"calibrated_agent_evaluation": {}}),
        "calibrated_agent_evaluation",
    ),
    "slice mixing forms": (
        lambda: DatasetSlice(form="predefined", scenarios=[_predefined(), _simulated()]),
        "another form",
    ),
    "duplicate scenario ids": (
        lambda: DatasetSlice(form="predefined", scenarios=[_predefined(), _predefined()]),
        "ORD-1001",
    ),
    "empty slice": (lambda: DatasetSlice(form="predefined", scenarios=[]), "scenarios"),
    "unknown slice schema version": (
        lambda: DatasetSlice.model_validate(
            {"schema_version": "2", "form": "predefined", "scenarios": [_predefined().model_dump()]}
        ),
        "schema_version",
    ),
}


@pytest.mark.parametrize("case", sorted(_INVALID))
def test_invalid_scenario_shapes_are_refused_naming_the_field(case: str) -> None:
    build, named = _INVALID[case]

    with pytest.raises(ScenarioError, match=named):
        build()


def test_scenario_package_loads_no_aws_sdk() -> None:
    code = (
        "import sys\n"
        "import calibrated_agent_evaluation.scenarios\n"
        "from calibrated_agent_evaluation.scenarios.build import build_slice, starter_key\n"
        "from calibrated_agent_evaluation.schema.types import Label, Trace\n"
        "t = Trace(trace_id='a', frame='discovery', ownership='probed', input={'m': 'hi'}, "
        "output={}, taxonomy_cell={})\n"
        "lb = Label(trace_id='a', span_id=None, kind='human', outcome='FAIL', critique='c', "
        "failure_mode='m')\n"
        "assert build_slice([t], {'a': [lb]}).slice is not None\n"
        "assert starter_key([t], {'a': [lb]}).by_failure_mode\n"
        "loaded = sorted(m for m in ('boto3', 'botocore', 'bedrock_agentcore', 'strands_evals')"
        " if m in sys.modules)\n"
        "print(loaded)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"
