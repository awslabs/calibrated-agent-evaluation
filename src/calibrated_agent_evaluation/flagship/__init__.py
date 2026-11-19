# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The flagship proof harness -- the end-to-end proof of the whole pipeline.

Runs both reference agents (OrderBot single-agent, ClaimsCrew crew) end-to-end through all eight
pipeline stages (``probe -> define -> generate -> label -> cluster -> judge -> run -> improve``)
with calibrated, bias-gated judges, in both worlds (local + self-hosted Langfuse; cassette-replay +
live Bedrock), and emits one reconstructable :class:`FlagshipReport`. It is a **pure orchestration**
feature: it composes the shipped ``calibrated_agent_evaluation.*`` surfaces and introduces no
estimator, gate, judge, scorer, or statistic. The corpus-specific binding (which
trace/cassette/judge/crew per agent) is injected by the caller as an :class:`AgentPlan`, so this
package imports only ``calibrated_agent_evaluation.*`` (no ``scripts``/``examples``) and no vendor
SDK type. :class:`ReferenceCorpus` builds the two reference plans from a source checkout's
``fixtures/`` tree, and :func:`run_reference_proof` runs them and writes the record.

Public surface:
  * :func:`prove` -- run the whole proof for a :class:`FlagshipConfig`, return a ``FlagshipReport``.
  * :func:`run_agent_pipeline` -- run the eight stages for one ``(agent, world)`` cell.
  * :func:`run_reference_proof` -- prove the reference agents and write the flagship record.
  * :class:`ReferenceCorpus` -- the reference agents' plans over a checkout's ``fixtures/``.
  * :func:`compare_oracle` -- compare a stage's output against its committed corpus oracle.
  * :class:`AgentPlan` -- the injected per-agent corpus binding the orchestrator drives.
  * the frozen report types + the self-contained CSV+JSON codec.
"""

from __future__ import annotations

from calibrated_agent_evaluation.flagship.errors import FlagshipError, FlagshipStoreError
from calibrated_agent_evaluation.flagship.oracle import compare_oracle, load_oracle
from calibrated_agent_evaluation.flagship.orchestrator import (
    AgentPlan,
    StageContext,
    StagePlan,
    StageResult,
    assert_worlds_equal,
    prove,
    run_agent_pipeline,
)
from calibrated_agent_evaluation.flagship.reference import ReferenceCorpus
from calibrated_agent_evaluation.flagship.report import (
    CANONICAL_STAGES,
    FLAGSHIP_SCHEMA_VERSION,
    Agent,
    FlagshipConfig,
    FlagshipReport,
    GateDecisionRef,
    ModelPlane,
    OracleCheck,
    StageName,
    StageOutcome,
    Verdict,
    World,
    WorldParity,
)
from calibrated_agent_evaluation.flagship.runner import (
    REFERENCE_AGENTS,
    format_report,
    run_reference_proof,
)
from calibrated_agent_evaluation.flagship.stages import (
    attribution_stage,
    cluster_stage,
    improve_stage,
    judge_stage,
    probe_stage,
    run_gate_stage,
    structural_stage,
)
from calibrated_agent_evaluation.flagship.store import dump_flagship_reports, load_flagship_reports

__all__ = [
    "CANONICAL_STAGES",
    "FLAGSHIP_SCHEMA_VERSION",
    "REFERENCE_AGENTS",
    "Agent",
    "AgentPlan",
    "FlagshipConfig",
    "FlagshipError",
    "FlagshipReport",
    "FlagshipStoreError",
    "GateDecisionRef",
    "ModelPlane",
    "OracleCheck",
    "ReferenceCorpus",
    "StageContext",
    "StageName",
    "StageOutcome",
    "StagePlan",
    "StageResult",
    "Verdict",
    "World",
    "WorldParity",
    "assert_worlds_equal",
    "attribution_stage",
    "cluster_stage",
    "compare_oracle",
    "dump_flagship_reports",
    "format_report",
    "improve_stage",
    "judge_stage",
    "load_flagship_reports",
    "load_oracle",
    "probe_stage",
    "prove",
    "run_agent_pipeline",
    "run_gate_stage",
    "run_reference_proof",
    "structural_stage",
]
