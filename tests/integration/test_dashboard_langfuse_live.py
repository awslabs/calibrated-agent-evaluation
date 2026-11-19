# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live integration test: the dashboard sink against a REAL Langfuse instance.

The binding check for the "publish numeric scores to Langfuse dashboards" claim and the both-worlds
both-worlds parity of the dashboard series -- it exercises the real ``langfuse`` v3 SDK against the
real OSS stack (``docker/langfuse/docker-compose.yml``), not the unit-test stub. Marked
``@pytest.mark.integration`` and **skipped by default** (the default ``just test`` run is
hermetic/offline); run it with:

    just langfuse-up
    just test-integration
    just langfuse-down

It skips cleanly (not fails) when the ``[langfuse]`` extra is absent or the stack is unreachable, so
it never breaks a developer who hasn't started Docker.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

from _helpers import TEST_INTERVAL_KIND, langfuse_reachable
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.backends.dashboard import LangfuseDashboardSink, LocalDashboardSink
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.run.report import (
    DriftMetric,
    InputDriftReport,
    ModeKey,
    RunReport,
    TrendPoint,
    numeric_series,
)

pytestmark = pytest.mark.integration


def _report(run_id: str) -> RunReport:
    return RunReport(
        run_id=run_id,
        mode="cadence",
        trend=(
            TrendPoint(
                mode=ModeKey(failure_type="over_refund"),
                n_total=50,
                corrected_rate=Interval(point=0.06, lo=0.02, hi=0.11, **TEST_INTERVAL_KIND),
                calib_id="c1",
                raw_rate=Interval(point=0.16, lo=0.08, hi=0.29, **TEST_INTERVAL_KIND),
            ),
        ),
        input_drift=InputDriftReport(
            metrics=(
                DriftMetric(
                    name="psi",
                    value=0.31,
                    threshold=0.25,
                    exceeded=True,
                    p_value=0.001,
                    method="test",
                ),
                DriftMetric(
                    name="jsd",
                    value=0.10,
                    threshold=0.20,
                    exceeded=False,
                    p_value=0.001,
                    method="test",
                ),
                DriftMetric(
                    name="wasserstein",
                    value=0.50,
                    threshold=1.0,
                    exceeded=False,
                    p_value=0.001,
                    method="test",
                ),
            ),
            dimension="taxonomy_cell",
        ),
    )


def _live_sink(local: LocalDashboardSink | None = None) -> LangfuseDashboardSink:
    pytest.importorskip("langfuse")
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-dev")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-calibrated-agent-evaluation-dev")
    sink = LangfuseDashboardSink.from_credentials(public_key, secret_key, host, local=local)
    # Skip (don't fail) if the stack isn't actually up; keeps the suite friendly off-Docker.
    reachable = langfuse_reachable(sink._client)
    if not reachable:
        pytest.skip(f"no reachable Langfuse at {host} (run `just langfuse-up`)")
    return sink


def _scores_on(sink: LangfuseDashboardSink, run_id: str, expected: int) -> dict[str, object]:
    client = sink._client  # type: ignore[attr-defined]
    trace_id = client.create_trace_id(seed=f"calibrated-agent-evaluation-run-{run_id}")
    deadline = time.monotonic() + 60
    while True:
        rows = client.api.score_v_2.get(trace_id=trace_id, limit=100).data
        if len(rows) >= expected or time.monotonic() > deadline:
            return {row.name: row.value for row in rows}
        time.sleep(2)


def test_live_numeric_series_publishes_to_real_langfuse(tmp_path) -> None:
    """The numeric series publishes to real Langfuse; local leg parity holds."""
    local = LocalDashboardSink(tmp_path)
    sink = _live_sink(local)

    report = _report(f"review-2026-10-26-dash-{uuid.uuid4().hex[:8]}")
    # Raises on an unreachable backend or a refused score, so reaching here means the server
    # acknowledged every score.
    sink.publish(report)

    # The local leg is authoritative and must round-trip to the exact vendor-neutral series
    # (both-worlds
    # parity: the same numbers went to both worlds).
    assert tuple(local.load()) == numeric_series(report)

    series = numeric_series(report)
    assert _scores_on(sink, report.run_id, len(series)) == {p.name: p.value for p in series}


def test_live_a_score_the_server_rejects_fails_the_publish() -> None:
    """A score body over the server's 1 MB limit is refused; the publish raises, not succeeds."""
    from calibrated_agent_evaluation.schema.errors import BackendError

    sink = _live_sink()
    run_id = f"review-2026-10-26-dashbig-{uuid.uuid4().hex[:8]}"
    report = _report(run_id)
    (point,) = report.trend
    oversized = replace(point, mode=ModeKey(failure_type="x" * 2_000_000))
    with pytest.raises(BackendError, match="while writing score"):
        sink.publish(replace(report, trend=(oversized,)))
