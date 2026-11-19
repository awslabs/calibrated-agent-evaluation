# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture in a process whose OpenTelemetry global provider is already fixed.

The global provider can be set once per process, and a tracer obtained before that binds to the
first real provider for good, so each case runs in a fresh interpreter.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("opentelemetry.sdk")

_PRELUDE = """
import json, tempfile
from opentelemetry import trace as trace_api
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.ingest import IngestError, capture_and_ingest

tracer = trace_api.get_tracer("module.level")  # bound at import, before any capture

def agent():
    with tracer.start_as_current_span(
        "invoke_agent orderbot",
        attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "orderbot"},
    ):
        pass

backend = LocalBackend(root=tempfile.mkdtemp())
"""


def _run_isolated(body: str) -> dict[str, object]:
    done = subprocess.run(
        [sys.executable, "-c", _PRELUDE + body],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_repeated_captures_in_one_process_each_get_their_spans() -> None:
    """A capture that swapped the global provider saw nothing on its second run and refused the
    agent as uninstrumented, because the module-level tracer stayed bound to the first provider."""
    out = _run_isolated(
        """
counts = []
for _ in range(3):
    report = capture_and_ingest(agent, service_name="orderbot", backend=backend)
    counts.append(report.exit_code)
print(json.dumps({"counts": counts, "stored": len(backend.query())}))
"""
    )
    assert out == {"counts": [0, 0, 0], "stored": 3}


def test_capture_joins_an_sdk_provider_the_application_installed() -> None:
    """The application's own provider stays global and still receives its spans."""
    out = _run_isolated(
        """
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

own = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(own))
trace_api.set_tracer_provider(provider)
codes = [
    capture_and_ingest(agent, service_name="orderbot", backend=backend).exit_code
    for _ in range(2)
]
print(json.dumps({
    "codes": codes,
    "app_spans": len(own.get_finished_spans()),
    "same_provider": trace_api.get_tracer_provider() is provider,
}))
"""
    )
    assert out == {"codes": [0, 0], "app_spans": 2, "same_provider": True}


def test_a_non_sdk_global_provider_is_refused_by_name() -> None:
    out = _run_isolated(
        """
trace_api.set_tracer_provider(trace_api.NoOpTracerProvider())
try:
    capture_and_ingest(agent, service_name="orderbot", backend=backend)
    print(json.dumps({"raised": None, "stored": len(backend.query())}))
except IngestError as exc:
    print(json.dumps({"raised": str(exc), "stored": len(backend.query())}))
"""
    )
    assert out["stored"] == 0
    assert "NoOpTracerProvider" in str(out["raised"])
    assert "cannot attach" in str(out["raised"])


def test_concurrent_captures_each_receive_only_their_own_spans() -> None:
    """Two captures open at once on two threads used to receive each other's spans."""
    out = _run_isolated(
        """
import threading
from calibrated_agent_evaluation.execute.strands_capture import capture_spans

def run(name, barrier, results):
    with capture_spans(name) as cap:
        cap.register_global()
        barrier.wait()
        with tracer.start_as_current_span(f"invoke_agent {name}"):
            barrier.wait()
    results[name] = sorted(s.name for s in cap.finished_spans)

barrier = threading.Barrier(2)
results = {}
threads = [threading.Thread(target=run, args=(n, barrier, results)) for n in ("a", "b")]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(json.dumps(results))
"""
    )
    assert out == {"a": ["invoke_agent a"], "b": ["invoke_agent b"]}


def test_host_telemetry_outside_the_capture_is_not_captured() -> None:
    """A span the host process started outside the capture's block stays out of the trace."""
    out = _run_isolated(
        """
import threading
from calibrated_agent_evaluation.execute.strands_capture import capture_spans

started = threading.Event()
release = threading.Event()

def host_request():
    with tracer.start_as_current_span("GET /health"):
        started.set()
        release.wait()

host = threading.Thread(target=host_request)
with capture_spans("orderbot") as cap:
    cap.register_global()
    host.start()
    started.wait()
    with tracer.start_as_current_span("invoke_agent orderbot"):
        with tracer.start_as_current_span("execute_tool lookup_order"):
            pass
    release.set()
    host.join()
print(json.dumps({"names": sorted(s.name for s in cap.finished_spans)}))
"""
    )
    assert out == {"names": ["execute_tool lookup_order", "invoke_agent orderbot"]}


def test_a_child_span_on_a_thread_without_the_context_still_reaches_its_capture() -> None:
    """Ownership follows the parent, so a worker that did not copy the context is still caught."""
    out = _run_isolated(
        """
import threading
from calibrated_agent_evaluation.execute.strands_capture import capture_spans

with capture_spans("orderbot") as cap:
    cap.register_global()
    with tracer.start_as_current_span("invoke_agent orderbot") as root:
        parent = trace_api.set_span_in_context(root)
        def worker():
            with tracer.start_as_current_span("execute_tool lookup_order", context=parent):
                pass
        t = threading.Thread(target=worker)
        t.start()
        t.join()
print(json.dumps({"names": sorted(s.name for s in cap.finished_spans)}))
"""
    )
    assert out == {"names": ["execute_tool lookup_order", "invoke_agent orderbot"]}


def test_strands_spans_do_not_reach_the_applications_global_exporter() -> None:
    """With a Langfuse-style global provider installed, Strands spans went to its exporter too."""
    pytest.importorskip("strands")
    out = _run_isolated(
        """
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

app_exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(app_exporter))
trace_api.set_tracer_provider(provider)

import strands.telemetry.tracer as strands_tracer
from calibrated_agent_evaluation.execute.strands_capture import capture_spans

strands = strands_tracer.get_tracer()  # built before the capture, as an Agent would hold it
with capture_spans("orderbot") as cap:
    cap.register_global()
    with strands.tracer.start_as_current_span("invoke_agent orderbot"):
        pass
print(json.dumps({
    "captured": [s.name for s in cap.finished_spans],
    "app_saw": [s.name for s in app_exporter.get_finished_spans()],
    "restored": strands.tracer_provider is provider,
}))
"""
    )
    assert out == {"captured": ["invoke_agent orderbot"], "app_saw": [], "restored": True}
