# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test scenarios built from caught failures, grouped into dataset slices.

A scenario is a test case for a future run: an input, the expected answer and, optionally, the
expected tool order and plain-language assertions. Scenarios are built from human FAIL labels plus
a hand-written answer key and grouped into a :class:`DatasetSlice`. Simulated scenarios (a persona
and a goal instead of scripted turns) use the same model, with assertions as their only ground
truth. This package is vendor-free; the AgentCore codec and publish/pull live in
:mod:`calibrated_agent_evaluation.agentcore`.
"""

from __future__ import annotations

from calibrated_agent_evaluation.scenarios.build import (
    BuildResult,
    build_slice,
    expected_text,
    starter_key,
)
from calibrated_agent_evaluation.scenarios.errors import ScenarioError
from calibrated_agent_evaluation.scenarios.slicefile import read_answer_key, read_slice, write_slice
from calibrated_agent_evaluation.scenarios.types import (
    ActorProfile,
    AnswerEntry,
    AnswerKey,
    CalibratedAgentEvaluationScenarioMeta,
    DatasetSlice,
    PredefinedScenario,
    PublishedVersion,
    Scenario,
    SimulatedScenario,
    Turn,
)

__all__ = [
    "ActorProfile",
    "AnswerEntry",
    "AnswerKey",
    "BuildResult",
    "CalibratedAgentEvaluationScenarioMeta",
    "DatasetSlice",
    "PredefinedScenario",
    "PublishedVersion",
    "Scenario",
    "ScenarioError",
    "SimulatedScenario",
    "Turn",
    "build_slice",
    "expected_text",
    "read_answer_key",
    "read_slice",
    "starter_key",
    "write_slice",
]
