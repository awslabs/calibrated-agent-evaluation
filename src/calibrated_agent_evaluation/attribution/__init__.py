# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crew attribution layer.

Makes a multi-agent crew evaluable per-agent, not just at the outcome:
- **Crew Define artifact** -- expected agent set / order / rules per scenario, so a skip is a diff.
- **Counterfactual walk-back** -- validate a recorded culprit/manifestation against the span tree
  (plus an opt-in LLM proposer the human ratifies).
- **Two-vocabulary codebook** -- failure modes + responsibility boundaries, versioned.
- **Per-agent corrected rates** -- each agent corrected by its own judge's TPR/TNR (reuses the
  calibration engine).
- **Relocation primitive** -- gate crew improvement on the workflow outcome rate, not one agent's.

Pure methodology plane: reads normalized ``Trace``/``Span``/``Label`` and reuses the
calibration engine; no trace-store/executor vendor type appears here. It does NOT run LLM
judges nor cluster critiques into modes.

Public API re-exports from submodules.
"""

from __future__ import annotations

from calibrated_agent_evaluation.attribution.agreement import attribution_agreement
from calibrated_agent_evaluation.attribution.codebook import bump_version, register_entry
from calibrated_agent_evaluation.attribution.definition import (
    build_crew_definition,
    require_expected_process,
)
from calibrated_agent_evaluation.attribution.errors import (
    AgentRecordMismatchError,
    AttributionError,
    CodebookError,
    CrewDefinitionError,
)
from calibrated_agent_evaluation.attribution.labels import SpanAgents, attributed_agent, span_agents
from calibrated_agent_evaluation.attribution.rates import per_agent_rates
from calibrated_agent_evaluation.attribution.relocation import relocation_check
from calibrated_agent_evaluation.attribution.types import (
    AgentRate,
    AgreementReport,
    Attribution,
    AttributionStatus,
    AttributionSuggestion,
    Codebook,
    CodebookEntry,
    CrewDefinition,
    CrewOutcomes,
    Denominator,
    ExpectedProcess,
    MandatedRule,
    RelocationResult,
    RelocationVerdict,
    Vocabulary,
)
from calibrated_agent_evaluation.attribution.walkback import ratify_suggestion, validate_walkback

__all__ = [
    "AgentRate",
    "AgentRecordMismatchError",
    "AgreementReport",
    "Attribution",
    "AttributionError",
    "AttributionStatus",
    "AttributionSuggestion",
    "Codebook",
    "CodebookEntry",
    "CodebookError",
    "CrewDefinition",
    "CrewDefinitionError",
    "CrewOutcomes",
    "Denominator",
    "ExpectedProcess",
    "MandatedRule",
    "RelocationResult",
    "RelocationVerdict",
    "SpanAgents",
    "Vocabulary",
    "attributed_agent",
    "attribution_agreement",
    "build_crew_definition",
    "bump_version",
    "per_agent_rates",
    "ratify_suggestion",
    "register_entry",
    "relocation_check",
    "require_expected_process",
    "span_agents",
    "validate_walkback",
]
