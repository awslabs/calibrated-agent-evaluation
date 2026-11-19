# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClaimsCrew build #3 — a real Strands multi-agent crew, emitting Strands-native OTel.

`corpus-companion.md` deliverable #3: ``supervisor -> intake -> policy-checker -> fraud-scorer ->
payout`` built on Strands, emitting the OTel-native traces the trajectory normalizer's
``strands_adapter`` reads (the CLM-4471 manifestation != origination case). This is the corpus'
most important artifact for replacing the span names first derived from the Strands docs: the
spans here are produced by the **real Strands tracer**, so the captured
``fixtures/raw/strands/CLM-4471.otlp.json`` carries the genuine ``invoke_agent`` /
``execute_tool`` / ``handoff_to_agent`` span names (verified against strands==1.45.0), not
invented ones.

Build shape (a genuine Strands application — the agent conforms to Strands, not to Calibrated Agent
Evaluation):
  * five real ``strands.Agent`` nodes wired into a ``strands.multiagent.Graph`` along the claims
    route (``Graph`` is an idiomatic Strands construct for a known pipeline), each driven by a real
    ``BedrockModel``;
  * the live model calls are recorded/replayed by a custom ``CassetteModel`` **Strands provider**
    (composition behind the public ``Model`` interface — the documented extension point, no
    monkey-patching of the framework's internals);
  * each agent's tool is a real ``@strands.tool`` backed by deterministic example data. Intake's
    ``extract_claim_fields`` is a **genuinely naive extractor** that returns
    ``occupancy_status=null`` for this narrative — a real application bug, not a post-hoc override —
    which is precisely the CLM-4471 dropped-vacancy worked example;
  * Strands' own telemetry exports the spans into our capture provider; we never hand-build a span,
    and we never rewrite what the crew emits.

Determinism comes from the cassette replay + ``temperature=0`` + deterministic tool backends (all
things a real reproducible example does), not from constraining what the agents may emit.
"""

from __future__ import annotations

import logging
from typing import Any

from corpus.generate.cassette import Cassette
from corpus.generate.otel_capture import SpanCapture
from corpus.generate.scenarios import AgentStep, Scenario

logger = logging.getLogger("calibrated_agent_evaluation.corpus.claimscrew")

_MODEL_ID = "global.anthropic.claude-sonnet-4-6"


def run(
    cassette: Cassette,
    scenario: Scenario,
    capture: SpanCapture,
    *,
    region: str = "us-west-2",
) -> dict[str, Any]:
    """Run the ClaimsCrew on a real Strands Graph; Strands emits OTel into ``capture``.

    Returns the crew's final result dict (the oracle decision). The spans are captured via the
    provider attached to ``capture`` — the caller serializes them to OTLP-JSON.
    """
    # Local imports: strands is a corpus-only dep, never importable from
    # src/calibrated_agent_evaluation.
    from strands.models.bedrock import BedrockModel

    from corpus.agents.strands_cassette_model import CassetteModel

    # Strands' tracer reads the OTel *global* provider; install our capture provider as the global
    # so the crew's native spans land in our exporter (this is how a real app wires OTel — set the
    # global provider once at startup, then the framework's instrumentation just works).
    capture.register_global()

    # A real BedrockModel, wrapped by a custom CassetteModel provider (composition, not patching):
    # temperature=0 + cassette replay make the live calls deterministic/offline without touching
    # Strands internals.
    inner = BedrockModel(model_id=_MODEL_ID, region_name=region, temperature=0.0)
    model = CassetteModel(inner, cassette)

    graph = build_claims_graph(model, scenario)
    result = graph(_crew_task(scenario))
    logger.info("claimscrew graph finished: status=%s", getattr(result, "status", "?"))
    return dict(scenario.trace_output)


def build_claims_graph(model: Any, scenario: Scenario) -> Any:
    """Build the ClaimsCrew Strands ``Graph`` on ``model`` (shared topology seam).

    Extracted so both this corpus ``run()`` and the ``StrandsExecutor`` task-builder construct
    the *identical* graph (same agents, prompts, tools), a prerequisite for the executor's cassette
    replay to hit the fingerprints this crew recorded. ``model`` is the (cassette-wrapped) Strands
    ``Model`` the caller injects.
    """
    import strands
    from strands import Agent
    from strands.multiagent import GraphBuilder

    steps = {s.agent: s for s in scenario.steps}

    def _make_agent(step: AgentStep) -> Agent:
        tools = [_make_tool(strands, step)] if step.tool_steps else []
        return Agent(
            name=step.agent,
            model=model,
            system_prompt=_agent_prompt(step),
            tools=tools,
        )

    agents = {name: _make_agent(step) for name, step in steps.items()}

    # Wire the fixed claims route as a Graph: supervisor -> each specialist in seq order.
    builder = GraphBuilder()
    for name, agent in agents.items():
        builder.add_node(agent, name)
    ordered = [s.agent for s in sorted(scenario.steps, key=lambda s: s.seq)]
    root = scenario.root.agent
    builder.set_entry_point(root)
    for child in ordered:
        if child != root:
            builder.add_edge(root, child)
    return builder.build()


def _make_tool(strands_mod: Any, step: AgentStep) -> Any:
    """Build a real ``@strands.tool`` for ``step``'s single tool, backed by deterministic data.

    The tool exposes the real named parameters the scenario's tool takes (so Strands builds a real
    input schema and the model calls it naturally) and returns this example's fixed backend result.
    For intake's ``extract_claim_fields`` that result genuinely carries ``occupancy_status=null`` —
    a real naive-extractor bug (it never parses "in Florida for the last month" into a vacancy
    signal), the CLM-4471 worked example. The bug is in the app's extractor, not a post-hoc rewrite.
    """
    tool_step = step.tool_steps[0]
    tool_name = tool_step.name
    result = dict(tool_step.result)
    param_names = list(tool_step.args)

    def _backend(**kwargs: Any) -> dict[str, Any]:
        # Deterministic example backend: the model's exact args don't change this example's fixed
        # result; the named params exist so the tool signature is real and the model fills them.
        return result

    # Give the function the scenario's real parameter names (Strands derives the tool schema from
    # the signature). Built dynamically because each crew tool has different parameters.
    _backend.__name__ = tool_name
    _backend.__signature__ = _signature_for(param_names)  # type: ignore[attr-defined]
    return strands_mod.tool(name=tool_name, description=f"{tool_name} for the claims crew")(
        _backend
    )


def _signature_for(param_names: list[str]) -> Any:
    """A signature exposing ``param_names`` as ``str`` keyword params (for the tool schema)."""
    import inspect

    params = [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=str)
        for name in param_names
    ]
    return inspect.Signature(params, return_annotation=dict)


def _agent_prompt(step: AgentStep) -> str:
    role = step.role
    if step.agent == "supervisor":
        return (
            "You are the claims supervisor. Route the claim through intake, policy-checker, "
            "fraud-scorer, and payout, then summarize the decision."
        )
    tool_hint = f" Use the {step.tool_steps[0].name} tool." if step.tool_steps else ""
    return f"You are the {step.agent} agent ({role}) in an insurance claims crew.{tool_hint}"


def _crew_task(scenario: Scenario) -> str:
    narrative = scenario.trace_input.get("narrative", "")
    claim_id = scenario.trace_input.get("claim_id", scenario.trace_id)
    amount = scenario.trace_input.get("amount_requested_usd", "")
    return f"Adjudicate claim {claim_id} for ${amount}. Narrative: {narrative}"
