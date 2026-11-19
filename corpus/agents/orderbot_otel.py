# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OrderBot build #1 — a hand-coded agent loop, instrumented with the OpenTelemetry SDK.

This is the "own agent loop" customer (``corpus-companion.md`` §1, deliverable #1): no agent
framework, just a plain Python tool-use loop over Bedrock Converse, instrumented by hand with the
OTel SDK. It is the target-agnosticism twin of build #2 (Anthropic-SDK + MCP) — same observable
behavior (``lookup_order -> check_refund_policy -> issue_refund``), different build, both emitting
OTel. One gold ``Label`` must apply to both.

The same logical run is emitted in **two content placements** — span *events* and span *attributes*
(the two placements the OTel convention allows), so the normalizer's OTel adapter is exercised on
both. A third capture (``build2``)
is produced by the Anthropic build for the equivalence pair.

Run via ``corpus/generate/record.py`` (which owns the cassette/Bedrock wiring); the functions here
take an already-built :class:`BedrockConverseClient` so they replay offline in CI.
"""

from __future__ import annotations

from corpus.agents.common import ORDERBOT_SYSTEM_PROMPT, ExecutionRecord, run_orderbot
from corpus.agents.otel_emit import ContentLocation, emit_trace
from corpus.generate.bedrock_client import BedrockConverseClient
from corpus.generate.otel_capture import SpanCapture
from corpus.generate.scenarios import Scenario


def run(
    client: BedrockConverseClient,
    scenario: Scenario,
    capture: SpanCapture,
    *,
    content_location: ContentLocation = ContentLocation.EVENTS,
) -> ExecutionRecord:
    """Execute the OrderBot scenario and emit its OTel span tree into ``capture``.

    Returns the build-agnostic :class:`ExecutionRecord` so callers (e.g. the equivalence check) can
    compare build #1 against build #2 on tool sequence + final output.
    """
    record = run_orderbot(client, scenario, ORDERBOT_SYSTEM_PROMPT)
    tracer = capture.tracer("calibrated_agent_evaluation.orderbot.otel")
    emit_trace(
        tracer,
        [record],
        trace_input=scenario.trace_input,
        content_location=content_location,
    )
    return record


def run_record_only(client: BedrockConverseClient, scenario: Scenario) -> ExecutionRecord:
    """Run the OrderBot loop and return its record WITHOUT emitting spans.

    Used for the degraded variant (a thinly instrumented agent): the same genuine run, re-emitted
    by the caller with thin instrumentation so the normalizer must degrade it to outcome-only
    ``probed``.
    """
    return run_orderbot(client, scenario, ORDERBOT_SYSTEM_PROMPT)
