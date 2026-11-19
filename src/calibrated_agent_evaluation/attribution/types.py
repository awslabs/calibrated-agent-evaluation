# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data model for the crew-attribution layer.

Pure, frozen data holders; no methods, no I/O (typed, functional core). The
operations that build and validate them are free functions in the sibling modules
(``definition``/``walkback``/``codebook``/``rates``/``relocation``/``agreement``), mirroring the
``TaxonomyGrid`` + ``discovery_frame(grid)`` house style rather than dataclass methods. No
trace-store/executor vendor type appears here; only the normalized ``Trace``/``Span``/``Label``
and the ``Interval`` this layer reuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import ConfigDict

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.attribution.errors import AttributionError
from calibrated_agent_evaluation.calibrate.record import Interval

AttributionStatus = Literal["single", "disputed"]
Denominator = Literal["share-of-failures", "rate-over-traffic"]
Vocabulary = Literal["failure_modes", "responsibility_boundaries"]
RelocationVerdict = Literal["improved", "relocation", "regressed", "flat"]


@dataclass(frozen=True)
class MandatedRule:
    """A named responsibility boundary: antecedent ⇒ required consequent agent.

    ``deterministic=False`` marks a rule the process judge cannot decide by agent set-difference
    alone; it must be routed to a calibrated sub-judge.
    """

    rule_id: str
    antecedent: str
    consequent_agent: str
    deterministic: bool = True


@dataclass(frozen=True)
class ExpectedProcess:
    """Per-scenario expected process: the diff target for the process judge.

    ``expected_agents`` is the valid *ordering* (a sequence, not just a set); a skipped/mis-routed
    agent is only detectable as *expected minus actual*, so it must exist before any trace.
    """

    scenario: str
    expected_agents: tuple[str, ...]
    terminal_decision: str
    payout_allowed: bool
    escalation_required: bool


@dataclass(frozen=True)
class CrewDefinition:
    """The crew Define artifact.

    Built + validated by
    :func:`calibrated_agent_evaluation.attribution.definition.build_crew_definition` (a skipped
    agent is a diff against ``scenarios[...].expected_agents``; a violated boundary is a ``rules``
    clause). ``handoffs`` are declared seams like ``"intake->policy-checker"``.
    """

    agents: tuple[str, ...]
    handoffs: tuple[str, ...]
    scenarios: dict[str, ExpectedProcess]
    rules: tuple[MandatedRule, ...]


@dataclass(frozen=True)
class Attribution:
    """A validated counterfactual walk-back over the span tree.

    ``culprit_spans`` has ≥1 entry; more than one ⇒ ``status="disputed"`` (an undefined
    responsibility boundary, recorded; never collapsed to a single culprit). ``span_defect`` is
    the outcome-pass / span-fail flag: a defect recorded independently of the trace
    outcome, supplying per-span gold.
    """

    trace_id: str
    culprit_spans: tuple[str, ...]
    manifestation_span: str | None
    handoff_implicated: str | None
    status: AttributionStatus
    span_defect: bool = False


@dataclass(frozen=True)
class AttributionSuggestion:
    """An LLM proposal for a crew FAIL: NEVER a Label until a human ratifies it.

    ``critique`` precedes the (human-ratified) verdict, mirroring the review-UI ``Suggestion``.
    Whatever produced it,
    :func:`~calibrated_agent_evaluation.attribution.walkback.ratify_suggestion` refuses it until a
    named human ratifies it. No proposer ships: the ``WalkBackProposer`` protocol and its
    ``NoProposer`` default were removed, since nothing implemented or consumed them and no planned
    feature owns one.
    """

    culprit_span: str
    critique: str


@dataclass(frozen=True)
class CodebookEntry:
    """One versioned vocabulary entry.

    It needs a definition, at least one positive example and at least one boundary example, and is
    registered through :func:`calibrated_agent_evaluation.attribution.codebook.register_entry`.
    """

    name: str
    definition: str
    positive_examples: tuple[str, ...]
    boundary_examples: tuple[str, ...]


@dataclass(frozen=True)
class Codebook:
    """A crew's two versioned vocabularies: failure modes and agent or handoff boundaries.

    A version bump flags labels written under the prior version for re-labeling.
    """

    version: str
    failure_modes: tuple[CodebookEntry, ...] = ()
    responsibility_boundaries: tuple[CodebookEntry, ...] = ()
    changelog: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentRate:
    """One agent's Rogan-Gladen-corrected rate.

    ``interval`` is the :class:`~calibrated_agent_evaluation.calibrate.record.Interval` (corrected
    point and its interval, or a refusal reason) produced by *that agent's own* judge; rows do not
    share a fate. ``denominator`` states whether this is a share-of-failures or a rate-over-traffic
    number, so a discovery-pile rate is never quoted as the customer SLA.
    """

    agent: str
    interval: Interval
    denominator: Denominator


@dataclass(frozen=True)
class CrewOutcomes:
    """A before/after snapshot for the relocation check.

    ``workflow_failures`` / ``workflow_total`` give the crew's workflow outcome rate (the gate
    metric); ``per_agent_failures`` counts each agent's span-level failures over the same slice.
    """

    workflow_failures: int
    workflow_total: int
    per_agent_failures: dict[str, int]


class RelocationResult(CalibratedAgentEvaluationModel):
    """The before/after crew-improvement verdict.

    ``verdict="relocation"`` when the touched agent's failure rate fell (its interval below 0) but
    the workflow outcome rate was not shown to improve; a per-agent gain with a flat outcome rate is
    failure *relocation*, not a fix. ``workflow_rate_change`` and ``touched_agent_rate_change`` are
    the after-minus-before intervals the verdict read; ``None`` on results made before they were
    recorded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = AttributionError

    workflow_outcome_rate_before: float
    workflow_outcome_rate_after: float
    per_agent_failures_before: dict[str, int]
    per_agent_failures_after: dict[str, int]
    verdict: RelocationVerdict
    relocated_from: str | None
    workflow_rate_change: Interval | None = None
    touched_agent_rate_change: Interval | None = None


@dataclass(frozen=True)
class AgreementReport:
    """Per-axis inter-coder agreement over a dual-coded subset.

    A high ``outcome_kappa`` with a low ``culprit_kappa`` is the most useful multiagent signal; a
    fuzzy agent/handoff responsibility boundary (a real product bug), not labeler noise.

    Each κ carries its 95% interval. An axis on which both coders used one and the same value
    throughout has κ = 0/0: its field is ``None`` and its ``*_undefined_reason`` says so, rather
    than the conventional 1.0 that would read as proven agreement. ``n_dual_coded`` counts the
    traces both coders labeled (the outcome axis); ``n_culprit_coded`` the ones both called FAIL
    (the culprit axis).
    """

    outcome_kappa: Interval | None
    culprit_kappa: Interval | None
    n_dual_coded: int
    outcome_undefined_reason: str | None = None
    culprit_undefined_reason: str | None = None
    n_culprit_coded: int = 0
