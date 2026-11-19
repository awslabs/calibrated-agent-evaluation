# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture an instrumented agent with ``calibrated-agent-evaluation capture`` and show each outcome
the page describes.

Runs the installed CLI on ``orderbot.py`` (next to this file), one process per command as you would
from a shell: a dry run, a stored capture, a second capture under the same id (refused), and a
callable that emits no spans (refused). The temporary OTLP file a capture writes is shown as
``<tmp>``, and the random OTel trace id of the dry run as ``<otel-trace-id>``. Run from the
repository root with the ``strands`` extra, which carries the OTel SDK the capture uses.

    uv run --extra strands python docs/site/examples/integrate/capture_orderbot.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CALIBRATED_AGENT_EVALUATION = str(Path(sys.executable).parent / "calibrated-agent-evaluation")
HERE = Path(__file__).resolve().parent
_SCRUBS = (
    (re.compile(r"\S*calibrated-agent-evaluation-capture-[^/\s]+/"), "<tmp>/"),
    (re.compile(r"\b[0-9a-f]{32}\b"), "<otel-trace-id>"),
)


def run(args: list[str], store: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(HERE)}
    done = subprocess.run(  # noqa: S603 - fixed argv
        [CALIBRATED_AGENT_EVALUATION, "--root", str(store), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    output = (done.stdout + done.stderr).rstrip()
    for pattern, replacement in _SCRUBS:
        output = pattern.sub(replacement, output)
    print(f"$ calibrated-agent-evaluation {' '.join(args)}")
    print(output)
    print(f"(exit {done.returncode})\n")


agent = ["--agent", "orderbot:run_once", "--service", "orderbot"]
with tempfile.TemporaryDirectory() as tmp:
    store = Path(tmp) / "store"
    run(["capture", *agent, "--dry-run"], store)
    run(["capture", *agent, "--trace-id", "ORD-A1001"], store)
    run(["capture", *agent, "--trace-id", "ORD-A1001"], store)
    run(["capture", "--agent", "os:getcwd"], store)
