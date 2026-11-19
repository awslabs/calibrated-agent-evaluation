# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route a Strands crew through Calibrated Agent Evaluation, replaying its recorded model calls.

Runs the corpus ClaimsCrew (a Strands ``Graph`` built with ``GraphBuilder``) through
``StrandsExecutor`` against the committed cassette in replay mode, so no model is called and no
AWS credentials are needed. The executor returns normalized traces, and the run prints what one
holds. Run from the repository root with the ``strands`` extra.

    uv run --extra strands python docs/site/examples/integrate/route_strands_replay.py
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from corpus.agents.claimscrew_strands import build_claims_graph
from corpus.generate.scenarios import load_scenario

from calibrated_agent_evaluation.execute import Mode, StrandsExecutor, StrandsTask

logging.getLogger("strands").setLevel(logging.ERROR)

# --8<-- [start:replay]
scenario = load_scenario("CLM-4471")
claim = scenario.trace_input
task = StrandsTask(
    build=lambda model: build_claims_graph(model, scenario),
    input=(
        f"Adjudicate claim {claim['claim_id']} for ${claim['amount_requested_usd']}. "
        f"Narrative: {claim['narrative']}"
    ),
    mode="legacy",
    trace_id="CLM-4471",
    taxonomy_cell=dict(scenario.taxonomy_cell),
)

with (
    StrandsExecutor(
        cassette_path="fixtures/raw/cassettes/claimscrew_strands.cassette.json",
        cassette_mode=Mode.REPLAY,
    ) as executor,
    contextlib.redirect_stdout(io.StringIO()),
):
    # Strands agents stream their replies to stdout by default; keep the page output to the trace.
    (trace,) = executor.run(task)

print(f"trace {trace.trace_id}: ownership={trace.ownership}, {len(trace.spans)} spans")
print(f"taxonomy cell: {trace.taxonomy_cell}")
# The specialists run in parallel, so their order varies between runs; list them by name.
for span in sorted(trace.spans, key=lambda s: (s.parent is not None, s.agent)):
    tools = ", ".join(call.name for call in span.tool_calls) or "no tools"
    parent = span.parent.split("/", 1)[-1] if span.parent else "root"
    print(f"  {span.agent:<15} parent={parent:<12} {tools}")
# --8<-- [end:replay]
