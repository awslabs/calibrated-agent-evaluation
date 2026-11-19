# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route an Inspect task through Calibrated Agent Evaluation, replaying its recorded model calls.

Runs the corpus OrderBot Inspect task through ``InspectExecutor`` with the ``cassette/orderbot``
model, which replays the committed cassette, so no model is called and no AWS credentials are
needed. Run from the repository root with the ``inspect`` extra.

    uv run --extra inspect python docs/site/examples/integrate/route_inspect_replay.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Inspect prints a live progress display and the log path; the page shows only the trace.
os.environ.setdefault("INSPECT_DISPLAY", "none")
sys.path.insert(0, str(Path.cwd()))

from corpus.agents.orderbot_inspect import orderbot_fail

from calibrated_agent_evaluation.execute import InspectExecutor, Mode

# --8<-- [start:replay]
with InspectExecutor(
    model="cassette/orderbot",
    cassette_path="fixtures/raw/cassettes/inspect_ORD-1001.cassette.json",
    cassette_mode=Mode.REPLAY,
) as executor:
    (trace,) = executor.run(orderbot_fail)

print(f"trace {trace.trace_id}: ownership={trace.ownership}, {len(trace.spans)} span(s)")
print(f"input:  {trace.input}")
for span in trace.spans:
    for call in span.tool_calls:
        print(f"  {span.agent} called {call.name}({call.args})")
print(f"output: {trace.output}")
# --8<-- [end:replay]
