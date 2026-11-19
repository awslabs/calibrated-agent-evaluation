# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flagship-proof data types -- the frozen artifacts the harness assembles and persists.

Pure, frozen data holders (the "typed functional core" house style): the orchestration work
lives in :mod:`calibrated_agent_evaluation.flagship.orchestrator` / ``stages`` / ``gates`` /
``oracle``, never here. No vendor SDK type appears in this module -- the methodology-plane boundary.
A stage's headline rate is the calibration :class:`~calibrated_agent_evaluation.calibrate.Interval`
(reused, never re-modelled) so a refused correction carries ``point=None`` + ``refusal_reason``
rather than a fabricated float; the harness computes no rate of its own.

  * :class:`OracleCheck` -- one stage's output vs its committed corpus oracle.
  * :class:`~calibrated_agent_evaluation.enforce.admission.GateDecisionRef` -- the three-gate
    judge-trust outcome, re-exported here because the report carries it.
  * :class:`StageOutcome` -- one ``(agent, world, stage)`` result.
  * :class:`WorldParity` -- cross-world equality for one agent.
  * :class:`FlagshipConfig` -- the input to :func:`calibrated_agent_evaluation.flagship.prove`.
  * :class:`FlagshipReport` -- the single reconstructable proof artifact.

The report is persisted through the self-contained :mod:`calibrated_agent_evaluation.flagship.store`
codec (a proof spans two agents x two worlds x eight stages, not a single ``trace_id``, so the
trace-store ``dump()`` cannot carry it -- the ``run/store.py`` precedent).
"""

from __future__ import annotations

import re
from typing import ClassVar, Literal, Self

from pydantic import ConfigDict, model_validator

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._provenance import ProvenanceStamp
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.enforce.admission import GateDecisionRef as GateDecisionRef
from calibrated_agent_evaluation.flagship.errors import FlagshipError

# Stamped on every persisted FlagshipReport; load refuses on mismatch (no coercion). Homed here
# (not __init__) so the store + types share one constant without an import cycle.
# "2": provenance + OracleCheck.pinned; "3": Interval kind fields; "4": the provenance stamp
# names calibrated_agent_evaluation_version (records from before the rename are refused)
FLAGSHIP_SCHEMA_VERSION = "4"

# An agent id is an open string, validated against the AgentPlans the caller registers with
# ``prove`` (an id with no plan is refused there), not a closed enum: a third agent must not need a
# library edit. The id is used as a report key and a path segment, so it must be a plain slug.
Agent = str
_AGENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
World = Literal["local", "langfuse"]
ModelPlane = Literal["cassette", "live"]
# The eight canonical pipeline stages, in traversal order (error analysis precedes scoring).
StageName = Literal["probe", "define", "generate", "label", "cluster", "judge", "run", "improve"]
CANONICAL_STAGES: tuple[StageName, ...] = (
    "probe",
    "define",
    "generate",
    "label",
    "cluster",
    "judge",
    "run",
    "improve",
)
# "unpinned": nothing failed, but at least one stage had no committed oracle to check against, so
# the run is not a proof. Only a plan that opts out of oracle pinning can produce it.
Verdict = Literal["proven", "unpinned", "failed"]


class OracleCheck(CalibratedAgentEvaluationModel):
    """One stage's output compared against its committed corpus oracle.

    ``oracle_path`` is ``None`` only for an oracle-less stage proven structurally (documented in the
    ``note``); every stage that claims a committed oracle names it. ``matched=False`` carries the
    ``divergence`` -- what differed -- so a failed proof is diagnosable.

    ``pinned=False`` means the stage named an oracle that does not exist and its plan opted out of
    oracle pinning: nothing was compared, so ``matched`` is False with no ``divergence``. The row
    stays visible as unpinned rather than failing the run or reading as a match.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = FlagshipError

    stage: StageName
    oracle_path: str | None
    matched: bool
    divergence: str | None = None
    pinned: bool = True


class StageOutcome(CalibratedAgentEvaluationModel):
    """One ``(agent, world, stage)`` result.

    ``ok`` is True iff the stage ran and its oracle matched. ``headline`` is the calibration
    ``Interval`` where the stage emits a headline rate (judge/run/improve), ``None`` otherwise --
    never a bare float. ``error_type`` names the typed error class when a stage's shipped feature
    raised, so the failure is diagnosable from the record alone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = FlagshipError

    agent: Agent
    world: World
    stage: StageName
    ok: bool
    oracle: OracleCheck
    headline: Interval | None = None
    artifact_ids: tuple[str, ...] = ()
    error_type: str | None = None
    note: str = ""


class WorldParity(CalibratedAgentEvaluationModel):
    """Cross-world equality for one agent.

    ``equal=False`` names the divergent field/artifact -- a divergence is a blocker that
    flips the proof to ``failed``, not a footnote.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = FlagshipError

    agent: Agent
    compared: tuple[World, World]
    equal: bool
    divergence: str | None = None


class FlagshipConfig(CalibratedAgentEvaluationModel):
    """The input to :func:`calibrated_agent_evaluation.flagship.prove`.

    ``agents`` selects which agents to prove (each needs a registered ``AgentPlan``), ``worlds``
    selects the trace-store axis (local, and langfuse when configured), ``model_plane`` selects
    the model axis (cassette default/CI vs live sign-off), and ``fixtures_root`` is the seeded
    corpus root whose ``VERSION`` is checked against the running ``SCHEMA_VERSION``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = FlagshipError

    agents: tuple[Agent, ...]
    worlds: tuple[World, ...]
    model_plane: ModelPlane
    fixtures_root: str


class FlagshipReport(CalibratedAgentEvaluationModel):
    """The single reconstructable proof artifact.

    ``verdict == "proven"`` iff every ``StageOutcome.ok`` is True for every configured
    ``(agent, world)`` AND every ``WorldParity.equal`` is True AND every ``GateDecisionRef`` used to
    score was fully admitted; otherwise ``failed`` with ``reasons`` naming the offending
    stage/parity/gate. Round-trips through the self-contained
    :mod:`calibrated_agent_evaluation.flagship.store` codec; ``load`` refuses on a
    ``schema_version`` mismatch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = FlagshipError

    flagship_id: str
    verdict: Verdict
    agents: tuple[Agent, ...]
    worlds: tuple[World, ...]
    model_plane: ModelPlane
    fixture_version: str
    stages: tuple[StageOutcome, ...]
    gates: tuple[GateDecisionRef, ...]
    parity: tuple[WorldParity, ...]
    reasons: tuple[str, ...]
    inputs: dict[str, str]
    schema_version: str = FLAGSHIP_SCHEMA_VERSION
    # The environment the proof ran in; the stages' own records carry their seeds.
    provenance: ProvenanceStamp

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        self._post_init()
        return self

    def _post_init(self) -> None:
        # A proof with no recorded inputs is not reproducible (the RunRecord/ImproveRecord
        # precedent -- refuse at construction rather than persist an un-reconstructable artifact).
        if not self.inputs:
            raise FlagshipError(
                f"FlagshipReport(flagship_id={self.flagship_id!r}) requires non-empty inputs "
                "(a proof with no recorded inputs is not reproducible --)"
            )


def validate_agent_id(agent: str) -> str:
    """Return ``agent`` if it is a usable agent id, else raise :class:`FlagshipError`."""
    if not isinstance(agent, str) or not _AGENT_ID.match(agent):
        raise FlagshipError(
            f"agent id {agent!r} is not a plain slug (letters, digits, '.', '_', '-'); it keys the "
            "report and names oracle paths, so a blank or spaced id is refused"
        )
    return agent
