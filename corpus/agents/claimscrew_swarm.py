# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClaimsTriage build #3b — a genuine Strands ``Swarm``, emitting Strands-native OTel.

``corpus-companion.md`` deliverable #3b. Where ClaimsCrew (#3) wires a *fixed* route with a
``Graph``, a ``Swarm`` is the idiomatic Strands construct for a **dynamic** crew: agents decide
among themselves who acts next, and coordination is the framework's own registered
``handoff_to_agent`` tool (``Swarm._create_handoff_tool``, verified against
``strands-agents==1.45.0`` ``swarm.py``). This is the corpus' proof for the trajectory
normalizer's ``strands_adapter`` **Swarm** path:

  * the captured trace carries an ``invoke_swarm`` root and a real ``execute_tool handoff_to_agent``
    span (the handoff is a *tool call*, not a span primitive) — the exact shape the normalizer must
    land as a ``ToolCall`` (with the ``agent_name``/``message`` args), **never** a phantom agent
    ``Span``;
  * the spans are produced by the **real Strands tracer**, so the span names are genuine, not
    invented — closing the docs-derived surface for the Swarm pattern the way #3 did for Graph.

Build shape (a genuine Strands application — the agent conforms to Strands, not to Calibrated Agent
Evaluation):
  * two real ``strands.Agent`` nodes (``triage`` -> ``adjuster``) wired into a
    ``strands.multiagent.Swarm`` with ``triage`` as the entry point; ``triage`` classifies the
    claim and hands off to ``adjuster`` via the framework's ``handoff_to_agent`` tool;
  * each agent's domain tool is a real ``@strands.tool`` backed by deterministic example data;
  * the live model calls are recorded/replayed by the custom ``CassetteModel`` **Strands provider**
    (composition behind the public ``Model`` interface — no monkey-patching), so the run is
    deterministic and offline on replay.

Determinism comes from cassette replay + ``temperature=0`` + deterministic tool backends — all
things a real reproducible example does — not from constraining what the framework may emit.
"""

from __future__ import annotations

import logging
from typing import Any

from corpus.generate.cassette import Cassette
from corpus.generate.otel_capture import SpanCapture

logger = logging.getLogger("calibrated_agent_evaluation.corpus.claimscrew_swarm")

_MODEL_ID = "global.anthropic.claude-sonnet-4-6"

# A small, self-contained worked example (a genuine claim, not derived from an oracle: the Swarm's
# normalized topology is observed FROM the real capture, never assumed before it — see
# corpus-companion "assume nothing"). Tool results are deterministic so the frozen trace replays.
_TASK = (
    "Adjudicate water-damage claim CLM-7720 for $6,400. Narrative: a supply line under the "
    "upstairs sink failed and soaked the ceiling below; homeowner was present and reported it "
    "the same day."
)


def run(
    cassette: Cassette,
    capture: SpanCapture,
    *,
    region: str = "us-west-2",
) -> dict[str, Any]:
    """Run the claims-triage Swarm; Strands emits OTel into ``capture``.

    Returns the swarm's final result summary. The spans are captured via the global provider
    ``capture`` installs; the caller serializes them to OTLP-JSON.
    """
    # Local imports: strands is a corpus-only dep, never importable from
    # src/calibrated_agent_evaluation.
    from strands.models.bedrock import BedrockModel

    from corpus.agents.strands_cassette_model import CassetteModel

    # Install the capture provider as the OTel global so the framework's native tracer emits here
    # (the way a real app wires OTel once at startup; Strands' tracer reads the global provider).
    capture.register_global()

    inner = BedrockModel(model_id=_MODEL_ID, region_name=region, temperature=0.0)
    model = CassetteModel(inner, cassette)

    swarm = build_claims_swarm(model)
    result = swarm(_TASK)
    logger.info("claims swarm finished: status=%s", getattr(result, "status", "?"))
    return {"task": _TASK, "status": str(getattr(result, "status", "?"))}


def build_claims_swarm(model: Any) -> Any:
    """Build the claims-triage Strands ``Swarm`` on ``model`` (shared topology seam).

    Extracted so both this corpus ``run()`` and the ``StrandsExecutor`` build the *identical*
    swarm (same agents, prompts, tools), a prerequisite for the executor's cassette replay to hit
    the fingerprints this swarm recorded. ``model`` is the (cassette-wrapped) Strands ``Model``.
    """
    import strands
    from strands import Agent
    from strands.multiagent import Swarm

    @strands.tool(name="classify_claim", description="Classify a claim's peril and severity.")
    def classify_claim(claim_id: str, narrative: str) -> dict[str, Any]:
        # Deterministic example backend: a real triage tool would parse the narrative; here it
        # returns this example's fixed classification so the captured trace is reproducible.
        return {"claim_id": claim_id, "peril": "water_damage", "severity": "moderate"}

    @strands.tool(name="assess_and_decide", description="Assess a classified claim and decide.")
    def assess_and_decide(claim_id: str, peril: str, severity: str) -> dict[str, Any]:
        return {
            "claim_id": claim_id,
            "decision": "COVERED",
            "amount_usd": 6400,
            "basis": f"{peril} ({severity}); dwelling occupied at loss, no exclusion applies",
        }

    triage = Agent(
        name="triage",
        model=model,
        system_prompt=(
            "You are the claims triage agent in a small claims swarm. Use the classify_claim tool "
            "to classify the claim's peril and severity, then hand off to the 'adjuster' agent to "
            "make the coverage decision. Always hand off to adjuster after classifying — do not "
            "decide coverage yourself."
        ),
        tools=[classify_claim],
    )
    adjuster = Agent(
        name="adjuster",
        model=model,
        system_prompt=(
            "You are the claims adjuster in a small claims swarm. Use the assess_and_decide tool "
            "to assess the classified claim and produce the final coverage decision and amount. "
            "Summarize the decision for the customer."
        ),
        tools=[assess_and_decide],
    )

    return Swarm(nodes=[triage, adjuster], entry_point=triage)
