# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live test: ``calibrated-agent-evaluation probe --battery`` over a real published
benchmark.

The target is a loopback-only endpoint started here (nothing is reachable off the machine), but
everything else is real: StrongREJECT's own dataset, its own scorer, and a live Bedrock grader. What
only this can show is that the bridge presents the endpoint to a published benchmark the benchmark
accepts, and that its verdicts come back as trial outcomes with a bound over them.

Run: ``AWS_PROFILE=<your-profile> uv run pytest -m integration -k probe_battery_live``
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_OK
from calibrated_agent_evaluation.cli.records import load_records_of

pytestmark = pytest.mark.integration

_TRIALS = 2


class _RefusingAgent(BaseHTTPRequestHandler):
    """An endpoint that refuses everything: the benchmark should find no break."""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        payload = json.dumps(
            {"response": "I can't help with that. I won't assist with anything harmful."}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        """Keep the test output clean; the base class logs every request to stderr."""


@pytest.fixture
def endpoint() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RefusingAgent)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}/chat"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_live_the_battery_runs_a_published_benchmark_against_the_endpoint(
    endpoint: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("inspect_evals")
    if not (os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID")):
        pytest.skip(
            "no AWS credentials configured (set AWS_PROFILE to a profile with Bedrock access)"
        )

    from _bedrock import inspect_bedrock_model

    root = tmp_path / "root"
    code = main(
        [
            "--root",
            str(root),
            "probe",
            "--endpoint",
            endpoint,
            "--authorized",
            "--battery",
            "strong_reject",
            "--trials",
            str(_TRIALS),
            "--grader-model",
            inspect_bedrock_model(),
            "--explain",
        ]
    )
    out = capsys.readouterr()
    assert code == EXIT_OK, out.err
    assert f"of {_TRIALS} trial(s) broke" in out.out
    assert "(95% upper bound)" in out.out
    assert "no calibrated point rate" in out.out or "do not own" in out.out
    # --explain carries the interval and the method behind the headline bound.
    assert "one-sided 95% upper bound" in out.out

    [record] = load_records_of(root, "probe")
    [finding] = record.findings
    assert finding.payload["benchmark"] == "strong_reject"
    assert finding.payload["k"] == str(_TRIALS)
    # The endpoint refuses everything, so the benchmark's scorer should see no break; what matters
    # for the bound is that the trials ran and were scored.
    assert finding.payload["breaks"] in {"0", "1", "2"}
    assert 0.0 < float(finding.payload["break_rate_ub95"]) <= 1.0
