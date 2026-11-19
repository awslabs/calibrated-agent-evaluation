# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live test: `calibrated-agent-evaluation run --publish langfuse` against a real Langfuse
stack.

Runs the real CLI over a local store (the trace store stays local: the dashboard sink reads the
``LANGFUSE_*`` keys whatever ``--backend`` is), then reads the scores back through Langfuse's
public REST API and checks they equal the run record's drift, so the claim "published" means
"readable in Langfuse", not "the SDK accepted the call". Skips when the stack is unreachable:

    just langfuse-up
    just test-integration tests/integration/test_cli_run_publish_live.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from _helpers import langfuse_reachable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_OK
from calibrated_agent_evaluation.cli.records import load_records_of
from cli_store_helpers import owned_trace, seed_traces

pytestmark = pytest.mark.integration

_READ_BACK_SECONDS = 90.0


def _client():
    langfuse = pytest.importorskip("langfuse")
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    client = langfuse.Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-dev"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-calibrated-agent-evaluation-dev"),
        host=host,
        timeout=5,
    )
    reachable = langfuse_reachable(client)
    if not reachable:
        pytest.skip(f"no reachable Langfuse at {host} (run `just langfuse-up`)")
    return client


def test_published_drift_scores_are_readable_in_langfuse(tmp_path, monkeypatch, capsys):
    client = _client()
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    monkeypatch.setenv("LANGFUSE_HOST", os.environ.get("LANGFUSE_HOST", "http://localhost:3000"))
    monkeypatch.setenv(
        "LANGFUSE_PUBLIC_KEY",
        os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-dev"),
    )
    monkeypatch.setenv(
        "LANGFUSE_SECRET_KEY",
        os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-calibrated-agent-evaluation-dev"),
    )
    baseline, now = tmp_path / "baseline", tmp_path / "now"
    seed_traces(baseline, [owned_trace(f"B{i}", chars=80 + i % 20) for i in range(60)])
    seed_traces(now, [owned_trace(f"T{i}", chars=300 + i % 20) for i in range(60)], n_fail=2)

    argv = ["--root", str(now), "run", "--n-resamples", "199", "--publish", "langfuse"]
    argv += ["--drift-baseline", str(baseline), "--drift-feature", "input-chars"]
    assert main(argv) == EXIT_OK, capsys.readouterr().err
    (record,) = load_records_of(now, "run")
    assert record.input_drift is not None
    expected = {f"drift.input-chars.{m.name}": m.value for m in record.input_drift.metrics}

    trace_id = client.create_trace_id(seed=f"calibrated-agent-evaluation-run-{record.run_id}")
    deadline = time.monotonic() + _READ_BACK_SECONDS
    seen: dict[str, float] = {}
    while time.monotonic() < deadline:
        page = client.api.score_v_2.get(trace_id=trace_id, limit=50)
        seen = {s.name: s.value for s in page.data}
        if seen.keys() >= expected.keys():
            break
        time.sleep(2)
    assert seen == pytest.approx(expected)
