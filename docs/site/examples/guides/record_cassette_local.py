# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record a judge cassette with ``calibrated-agent-evaluation calibrate --cassette-mode record``,
against a local model.

Record mode calls the judge's live model. So that this runs offline, the "live model" here is a
small OpenAI-compatible server on 127.0.0.1 that answers every judge call with the same verdict
through the forced ``record_verdict`` tool call, the way a model without native structured output
answers. The judge file points at it with an ``openai-api/local/...`` model string, as a judge on
your own hardware would. Run from an empty directory holding a store seeded by
``calibrated-agent-evaluation quickstart --certify``
(``CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT=demo``), with the ``inspect`` extra and the ``openai``
package:

    python docs/site/examples/guides/record_cassette_local.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CALIBRATED_AGENT_EVALUATION = str(Path(sys.executable).parent / "calibrated-agent-evaluation")


class StubModel(BaseHTTPRequestHandler):
    """A chat-completions endpoint that fails every trace, with a one-line critique."""

    def do_POST(self) -> None:
        json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        verdict = {"critique": "The stub model fails every trace.", "verdict": "fail"}
        call = {
            "id": "call_0",
            "type": "function",
            "function": {"name": "record_verdict", "arguments": json.dumps(verdict)},
        }
        reply = {
            "id": "stub",
            "object": "chat.completion",
            "created": 0,
            "model": "stub-judge",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        payload = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        """Keep the output to the CLI's; the base class logs every request to stderr."""


server = ThreadingHTTPServer(("127.0.0.1", 0), StubModel)
threading.Thread(target=server.serve_forever, daemon=True).start()

judge = json.loads(Path("demo/quickstart/judge-v1.json").read_text(encoding="utf-8"))
judge["model"] = "openai-api/local/stub-judge"
Path("judge-local.json").write_text(json.dumps(judge, indent=2) + "\n", encoding="utf-8")

env = {
    **os.environ,
    "LOCAL_BASE_URL": f"http://127.0.0.1:{server.server_address[1]}/v1",
    "LOCAL_API_KEY": "not-a-secret",
}
command = [
    "calibrate",
    "--judge",
    "judge-local.json",
    "--cell",
    "label_round=1",
    "--cassette",
    "judge-local.cassette.json",
    "--cassette-mode",
    "record",
]
print("$ calibrated-agent-evaluation " + " ".join(command))
done = subprocess.run(  # noqa: S603 - fixed argv
    [CALIBRATED_AGENT_EVALUATION, *command],
    env=env,
    capture_output=True,
    text=True,
    check=False,
    timeout=300,
)
print((done.stdout + done.stderr).rstrip())
print(f"exit {done.returncode}")
entries = json.loads(Path("judge-local.cassette.json").read_text(encoding="utf-8"))
print(f"\njudge-local.cassette.json holds {len(entries.get('entries', entries))} recorded replies")
server.shutdown()
