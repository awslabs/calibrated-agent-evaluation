# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe a black-box agent endpoint with the ``calibrated-agent-evaluation probe`` CLI, against a
server on localhost.

Starts a small HTTP agent on 127.0.0.1 that answers the generic ``{"message": ...}`` payload with
a server-sent event stream (one tool event and one answer), then runs the real CLI against it:
first without ``--authorized``, which sends nothing, then with it. Nothing leaves the machine.
Run from the repository root.

    uv run python docs/site/examples/integrate/probe_local_endpoint.py
"""

from __future__ import annotations

import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli.dispatch import main


class RefundAgent(BaseHTTPRequestHandler):
    """An agent you do not own: you see its HTTP replies, never its code or its traces."""

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "message" not in body:
            self.send_error(400)
            return
        events = [
            {"type": "tool_use", "tool_name": "lookup_order"},
            {"type": "response", "response": "I can look up orders and start refunds."},
        ]
        payload = "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:
        pass


# A fixed port keeps the captured output identical from run to run.
server = ThreadingHTTPServer(("127.0.0.1", 47821), RefundAgent)
threading.Thread(target=server.serve_forever, daemon=True).start()
endpoint = "http://127.0.0.1:47821/chat"
prompts = ["--prompt", "What can you help me with?", "--prompt", "Refund order 1001."]

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "store"
    common = ["--root", str(root), "probe", "--endpoint", endpoint]

    print(
        f"$ calibrated-agent-evaluation probe --endpoint {endpoint} --prompt ... --prompt ...",
        flush=True,
    )
    print(f"exit {main([*common, *prompts])}\n", flush=True)

    print(
        f"$ calibrated-agent-evaluation probe --endpoint {endpoint} --authorized --prompt ... "
        "--prompt ...",
        flush=True,
    )
    print(f"exit {main([*common, '--authorized', *prompts])}\n", flush=True)

    with LocalBackend(root) as backend:
        for trace in sorted(backend.query(), key=lambda t: t.trace_id):
            print(f"{trace.trace_id}: ownership={trace.ownership}, spans={len(trace.spans)}")
            print(f"  input:  {trace.input}")
            print(f"  output: {trace.output}")

server.shutdown()
