# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The scenario model: test cases built from caught failures, and the slices that group them.

A :class:`PredefinedScenario` scripts the user's turns and may carry an expected answer per turn,
an expected tool order and assertions. A :class:`SimulatedScenario` gives a persona and a goal
instead, so the conversation is not known in advance and assertions are its only ground truth.
Fields only Calibrated Agent Evaluation uses (the source trace, frame, taxonomy cell, failure modes
and the structured expected output) live on :class:`CalibratedAgentEvaluationScenarioMeta`, which a
codec stores under a reserved ``metadata`` key so a round trip loses nothing.
"""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.scenarios.errors import ScenarioError
from calibrated_agent_evaluation.schema.types import Frame

__all__ = [
    "CALIBRATED_AGENT_EVALUATION_METADATA_KEY",
    "SLICE_SCHEMA_VERSION",
    "ActorProfile",
    "AnswerEntry",
    "AnswerKey",
    "CalibratedAgentEvaluationScenarioMeta",
    "DatasetSlice",
    "Form",
    "PredefinedScenario",
    "PublishedVersion",
    "Scenario",
    "SimulatedScenario",
    "Turn",
]

SLICE_SCHEMA_VERSION = "1"
CALIBRATED_AGENT_EVALUATION_METADATA_KEY = "calibrated_agent_evaluation"

Form = Literal["predefined", "simulated"]

_CONFIG = ConfigDict(frozen=True, extra="forbid")


def _non_empty_input(value: str | dict[str, Any]) -> str | dict[str, Any]:
    if not value or (isinstance(value, str) and not value.strip()):
        raise ValueError("must not be empty")
    return value


def _non_empty_strings(values: list[str]) -> list[str]:
    if any(not v.strip() for v in values):
        raise ValueError("entries must not be empty")
    return values


class _Model(CalibratedAgentEvaluationModel):
    model_config = _CONFIG
    invalid_error: ClassVar[type[Exception]] = ScenarioError


class Turn(_Model):
    """One scripted user turn and, when known, the answer the agent should give to it."""

    input: str | dict[str, Any]
    expected_response: str | None = None

    @field_validator("input")
    @classmethod
    def _check_input(cls, value: str | dict[str, Any]) -> str | dict[str, Any]:
        return _non_empty_input(value)


class ActorProfile(_Model):
    """Who a simulated user is and what they want."""

    context: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    traits: dict[str, Any] = Field(default_factory=dict)


class CalibratedAgentEvaluationScenarioMeta(_Model):
    """The scenario fields only the library uses; ``None`` and empty mean "not recorded"."""

    source_trace_id: str | None = None
    frame: Frame | None = None
    taxonomy_cell: dict[str, str] = Field(default_factory=dict)
    failure_modes: list[str] = Field(default_factory=list)
    expected: dict[str, Any] | None = None

    def is_empty(self) -> bool:
        """Whether every field is unset, so an encoder can leave the block out."""
        return self == CalibratedAgentEvaluationScenarioMeta()


class _ScenarioBase(_Model):
    scenario_id: str = Field(min_length=1)
    assertions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    calibrated_agent_evaluation: CalibratedAgentEvaluationScenarioMeta = Field(
        default_factory=CalibratedAgentEvaluationScenarioMeta
    )

    @field_validator("assertions")
    @classmethod
    def _check_assertions(cls, values: list[str]) -> list[str]:
        return _non_empty_strings(values)

    @field_validator("metadata")
    @classmethod
    def _reserved_key(cls, value: dict[str, Any]) -> dict[str, Any]:
        if CALIBRATED_AGENT_EVALUATION_METADATA_KEY in value:
            raise ValueError(
                f"the {CALIBRATED_AGENT_EVALUATION_METADATA_KEY!r} key is reserved; set the "
                "scenario's "
                "calibrated-agent-evaluation fields"
            )
        return value


class PredefinedScenario(_ScenarioBase):
    """A scripted conversation: fixed user turns, replayed as written."""

    form: Literal["predefined"] = "predefined"
    turns: list[Turn] = Field(min_length=1)
    expected_trajectory: list[str] = Field(default_factory=list)

    @field_validator("expected_trajectory")
    @classmethod
    def _check_trajectory(cls, values: list[str]) -> list[str]:
        return _non_empty_strings(values)


class SimulatedScenario(_ScenarioBase):
    """A conversation an LLM-played user drives toward a goal, so only assertions can judge it."""

    form: Literal["simulated"] = "simulated"
    actor_profile: ActorProfile
    input: str | dict[str, Any]
    max_turns: int = Field(default=10, ge=1)
    scenario_description: str = ""

    @field_validator("input")
    @classmethod
    def _check_input(cls, value: str | dict[str, Any]) -> str | dict[str, Any]:
        return _non_empty_input(value)


Scenario = Annotated[PredefinedScenario | SimulatedScenario, Field(discriminator="form")]


class PublishedVersion(_Model):
    """Where a slice was published: the AgentCore dataset and the immutable version it became."""

    service: Literal["agentcore"] = "agentcore"
    dataset_arn: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    version: str = Field(pattern=r"^[0-9]+$")
    region: str = Field(min_length=1)
    content_digest: str = Field(min_length=1)
    published_at: str = Field(min_length=1)


class DatasetSlice(_Model):
    """Scenarios of one form with unique ids, and the version they were published as, if any."""

    schema_version: str = SLICE_SCHEMA_VERSION
    form: Form
    scenarios: list[Scenario] = Field(min_length=1)
    published: PublishedVersion | None = None

    # Raised directly: an after-validator runs outside CalibratedAgentEvaluationModel's
    # ValidationError wrapper.
    @model_validator(mode="after")
    def _one_form_unique_ids(self) -> Self:
        if self.schema_version != SLICE_SCHEMA_VERSION:
            raise ScenarioError(
                f"slice schema_version {self.schema_version!r} is not the supported "
                f"{SLICE_SCHEMA_VERSION!r}"
            )
        other = sorted({s.scenario_id for s in self.scenarios if s.form != self.form})
        if other:
            raise ScenarioError(f"a {self.form} slice holds scenarios of another form: {other}")
        counts = Counter(s.scenario_id for s in self.scenarios)
        dupes = sorted(sid for sid, n in counts.items() if n > 1)
        if dupes:
            raise ScenarioError(f"scenario ids must be unique; repeated: {dupes}")
        return self


class AnswerEntry(_Model):
    """The expected tool order and assertions an answer key gives one failure mode or trace."""

    assertions: list[str] = Field(default_factory=list)
    expected_trajectory: list[str] = Field(default_factory=list)

    @field_validator("assertions", "expected_trajectory")
    @classmethod
    def _check_entries(cls, values: list[str]) -> list[str]:
        return _non_empty_strings(values)


class AnswerKey(_Model):
    """The hand-written answer key: entries per failure mode, with optional per-trace overrides."""

    by_failure_mode: dict[str, AnswerEntry] = Field(default_factory=dict)
    by_trace: dict[str, AnswerEntry] = Field(default_factory=dict)
