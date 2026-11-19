# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live test: `calibrated-agent-evaluation ingest --from-langfuse` reads a foreign trace.

Writes one trace with the plain Langfuse SDK (an agent observation with a generation and a tool
under it, the way an application instrumented with Langfuse directly records a run), waits for the
server to make it readable (ingestion is asynchronous), then runs the real CLI against the project
into a temporary local store and checks that trace came through with its agent, content and tool
call. Each run uses a fresh seed, so it never re-puts a trace (a re-put duplicates observations).

Skipped unless the stack is up: ``just langfuse-up`` then ``just test-integration``.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest

from _helpers import langfuse_reachable
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli.dispatch import main

pytestmark = pytest.mark.integration


@pytest.fixture
def plain_client():  # type: ignore[no-untyped-def]
    langfuse = pytest.importorskip("langfuse")
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    client = langfuse.Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-dev"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-calibrated-agent-evaluation-dev"),
        host=host,
    )
    reachable = langfuse_reachable(client)
    if not reachable:
        pytest.skip(f"no reachable Langfuse at {host} (run `just langfuse-up`)")
    return client


def _write_with_plain_sdk(client, seed: str) -> str:  # type: ignore[no-untyped-def]
    trace_id = client.create_trace_id(seed=seed)
    with client.start_as_current_observation(
        trace_context={"trace_id": trace_id},
        name="refund-agent",
        as_type="agent",
        input={"message": "refund order A-7781"},
    ) as agent:
        with client.start_as_current_observation(
            name="claude-sonnet",
            as_type="generation",
            model="claude-sonnet-4-6",
            input=[{"role": "user", "content": "refund order A-7781"}],
        ) as generation:
            generation.update(output={"role": "assistant", "content": "calling lookup_order"})
        with client.start_as_current_observation(
            name="lookup_order", as_type="tool", input={"order_id": "A-7781"}
        ) as tool:
            tool.update(output={"status": "delivered", "amount_usd": 89.99})
        agent.update(output={"answer": "Refunded $89.99"})
    client.flush()
    return trace_id


def _wait_until_readable(client, trace_id: str, observations: int) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            if len(client.api.trace.get(trace_id).observations) >= observations:
                return
        except Exception:
            pass
        time.sleep(1)
    pytest.fail(f"Langfuse never made trace {trace_id} readable")


def test_calibrated_agent_evaluation_ingest_from_langfuse_reads_a_trace_written_with_the_plain_sdk(
    plain_client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    trace_id = _write_with_plain_sdk(plain_client, f"cli-wiring-17-{uuid.uuid4()}")
    _wait_until_readable(plain_client, trace_id, observations=3)
    project_id = plain_client.api.projects.get().data[0].id

    monkeypatch.chdir(tmp_path)
    for name, default in (
        ("LANGFUSE_HOST", "http://localhost:3000"),
        ("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-dev"),
        ("LANGFUSE_SECRET_KEY", "sk-lf-calibrated-agent-evaluation-dev"),
    ):
        monkeypatch.setenv(name, os.environ.get(name, default))
    code = main(["ingest", "--from-langfuse", project_id])
    out = capsys.readouterr().out

    line = next((ln for ln in out.splitlines() if f"langfuse:{trace_id}" in ln), None)
    assert line is not None, out
    assert "[ok]" in line, line
    got = LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).get_trace(trace_id)
    assert got.ownership == "owned"
    [agent] = got.spans
    assert agent.agent == "refund-agent"
    assert agent.input == {"message": "refund order A-7781"}
    assert agent.output == {"answer": "Refunded $89.99"}
    assert [(c.name, c.args, c.result) for c in agent.tool_calls] == [
        ("lookup_order", {"order_id": "A-7781"}, {"status": "delivered", "amount_usd": 89.99})
    ]
    # Other traces in a shared dev project may be skipped for reasons of their own; that is the
    # per-trace contract. What must not happen is the pull aborting before reporting this one.
    assert code in (0, 1)
