# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live test: ``calibrated-agent-evaluation ingest --from-langfuse --sample`` against a real
Langfuse project.

Writes a small population of Calibrated Agent Evaluation traces that share a taxonomy cell unique to
this run, then draws a seeded sample of them through the real CLI into a temporary local store. What
the live stack has to prove over the hermetic tests: the cell restriction works against the real
pull (a shared dev project holds other people's traces), the sampled traces come back tagged
production-representative, and the same seed draws the same traces twice.

Skipped unless the stack is up: ``just langfuse-up`` then ``just test-integration``.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest

from _helpers import langfuse_reachable, make_span, make_trace
from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli.dispatch import main
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.schema.types import Trace

pytestmark = pytest.mark.integration

_POPULATION = 6
_DRAW = 3


@pytest.fixture
def live_backend() -> LangfuseBackend:
    pytest.importorskip("langfuse")
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    backend = LangfuseBackend.from_credentials(
        os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-dev"),
        os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-calibrated-agent-evaluation-dev"),
        host,
    )
    reachable = langfuse_reachable(backend._client)
    if not reachable:
        pytest.skip(f"no reachable Langfuse at {host} (run `just langfuse-up`)")
    return backend


def _seed_population(backend: LangfuseBackend, run_tag: str) -> list[Trace]:
    traces = [
        make_trace(
            trace_id=f"review-2026-10-26-sample-{run_tag}-{i}",
            frame="discovery",
            taxonomy_cell={"feature": "refund", "run": run_tag},
            spans=[
                make_span(
                    trace_id=f"review-2026-10-26-sample-{run_tag}-{i}",
                    span_id=f"review-2026-10-26-sample-{run_tag}-{i}/root",
                )
            ],
        )
        for i in range(_POPULATION)
    ]
    for trace in traces:
        backend.put_trace(trace)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        readable = 0
        for trace in traces:
            try:
                backend.get_trace(trace.trace_id)
            except Exception:
                continue
            readable += 1
        if readable == len(traces):
            return traces
        time.sleep(2)
    pytest.fail("Langfuse never made the seeded population readable")


def _ingest(project_id: str, root: Path, run_tag: str, seed: int) -> int:
    return main(
        [
            "--root",
            str(root),
            "ingest",
            "--from-langfuse",
            project_id,
            "--sample",
            str(_DRAW),
            "--seed",
            str(seed),
            "--cell",
            f"run={run_tag}",
        ]
    )


def test_live_a_sampled_pull_draws_one_cell_reproducibly_and_tags_it(
    live_backend: LangfuseBackend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_tag = uuid.uuid4().hex[:8]
    _seed_population(live_backend, run_tag)
    project_id, _name = live_backend.project()
    for name, default in (
        ("LANGFUSE_HOST", "http://localhost:3000"),
        ("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-dev"),
        ("LANGFUSE_SECRET_KEY", "sk-lf-calibrated-agent-evaluation-dev"),
    ):
        monkeypatch.setenv(name, os.environ.get(name, default))

    first_root = tmp_path / "first"
    assert _ingest(project_id, first_root, run_tag, seed=5) == 0
    out = capsys.readouterr().out
    assert f"sampled {_DRAW} of {_POPULATION} trace(s)" in out, out
    assert "production-representative" in out

    stored = LocalBackend(root=str(first_root)).query()
    assert len(stored) == _DRAW
    assert {trace.frame for trace in stored} == {"production-representative"}
    assert {trace.taxonomy_cell["run"] for trace in stored} == {run_tag}

    [record] = load_records_of(first_root, "sample")
    assert record.population_size == _POPULATION
    assert record.sample_size == _DRAW
    assert record.provenance.seed == 5
    assert record.cell == {"run": run_tag}
    assert sorted(record.trace_ids) == sorted(record.drawn)

    second_root = tmp_path / "second"
    assert _ingest(project_id, second_root, run_tag, seed=5) == 0
    capsys.readouterr()
    [again] = load_records_of(second_root, "sample")
    assert again.drawn == record.drawn

    third_root = tmp_path / "third"
    assert _ingest(project_id, third_root, run_tag, seed=99) == 0
    capsys.readouterr()
    [other] = load_records_of(third_root, "sample")
    assert other.population_size == _POPULATION
