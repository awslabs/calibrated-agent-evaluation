# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture an instrumented agent from Python with ``capture_and_ingest``; read back its tool call.

Runs ``orderbot.py`` (next to this file) in process and stores the trace in ``./store`` under the
working directory, so run it from an empty directory with the ``strands`` extra, which carries the
OTel SDK the capture uses:

    uv run --extra strands python docs/site/examples/integrate/capture_from_python.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# --8<-- [start:capture]
from orderbot import run_once

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.ingest import capture_and_ingest

backend = LocalBackend(root="./store")
report = capture_and_ingest(run_once, service_name="orderbot", backend=backend)
trace = backend.get_trace(report.results[0].trace_id)
call = trace.spans[0].tool_calls[0]
print(call.name, call.args, call.result)
# --8<-- [end:capture]
