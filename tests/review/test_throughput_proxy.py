# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Throughput proxy: a scripted 50-label run reports elapsed in the summary.

The real timed-throughput proof (under 30 minutes for 50 traces) is a human-run session; this
only asserts the session's per-trace timing + summary machinery works at 50-label scale.
"""

from __future__ import annotations

from calibrated_agent_evaluation.review import ReviewSession, build_label


class _TickClock:
    """Monotonic stand-in that advances 1.0 per call (mark_shown then record → 1.0s per trace)."""

    def __init__(self) -> None:
        self._t = 0.0

    def __call__(self) -> float:
        self._t += 1.0
        return self._t


def test_scripted_50_label_run_reports_elapsed() -> None:
    ids = [f"T-{i:03d}" for i in range(50)]
    session = ReviewSession.from_ids(ids, clock=_TickClock())
    for trace_id in ids:
        session.mark_shown()
        session.record(build_label(trace_id=trace_id, outcome="PASS", critique=""))
    summary = session.summary()
    assert summary["labeled_count"] == 50
    assert summary["elapsed_seconds"] == 50.0  # 1.0s per trace, deterministic clock
    assert session.is_complete
