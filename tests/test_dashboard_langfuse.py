# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Langfuse dashboard sink against a verified-surface stub.

Behavioural: publish maps each numeric point to a numeric create_score whose value equals the
report's; an unreachable/rejecting backend raises BackendError BEFORE any score is written (the
health-probe-before-success discipline); the stub mirrors the real langfuse==3.15.0 surface.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from _helpers import TEST_INTERVAL_KIND
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.backends.dashboard import (
    LangfuseDashboardSink,
    LangfuseScoreClient,
)
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.run.report import (
    DriftMetric,
    InputDriftReport,
    ModeKey,
    RunReport,
    TrendPoint,
    numeric_series,
)
from calibrated_agent_evaluation.schema.errors import BackendError


class _StubSpan:
    def update_trace(self, *, name: str | None = None, metadata: Any | None = None) -> None:
        return None

    def end(self) -> None:
        return None


class _StubScoreApi:
    """Mirrors ``client.api.score.create(request=CreateScoreRequest)`` (langfuse==3.15.0)."""

    def __init__(self, owner: _StubClient) -> None:
        self._owner = owner

    def create(self, *, request: Any) -> Any:
        if self._owner.reject_scores:
            raise RuntimeError("status_code: 413, body: request entity too large")
        self._owner.scores.append((request.name, request.value, request.data_type.value))
        self._owner.events.append(("score", request.trace_id))
        return SimpleNamespace(id=None if self._owner.no_score_id else f"s-{request.name}")


class _StubClient:
    """Mirrors the verified langfuse==3.15.0 surface the sink uses (api.score/auth_check/...).

    Records the numeric scores written and the order of operations, so the test asserts on the
    round-tripped VALUES (not call counts as the assertion) and on the probe-before-write ordering.
    """

    def __init__(
        self,
        *,
        reachable: bool = True,
        auth_raises: bool = False,
        flush_hangs: bool = False,
        reject_scores: bool = False,
        no_score_id: bool = False,
    ) -> None:
        self._reachable = reachable
        self._auth_raises = auth_raises
        self._flush_hangs = flush_hangs
        self.reject_scores = reject_scores
        self.no_score_id = no_score_id
        self.scores: list[tuple[str, float | str, str | None]] = []
        self.auth_checked = False
        self.flushed = False
        # Each event in order: ("trace", id) when a trace is opened, ("flush", ""), ("score", id).
        self.events: list[tuple[str, str]] = []
        self.api = SimpleNamespace(score=_StubScoreApi(self))

    def create_trace_id(self, *, seed: str) -> str:
        return f"lf-{seed}"

    def start_span(self, *, trace_context: Any, name: str, metadata: Any | None = None) -> Any:
        self.events.append(("trace", trace_context["trace_id"]))
        return _StubSpan()

    def auth_check(self) -> bool:
        self.auth_checked = True
        if self._auth_raises:
            raise ConnectionError("connection refused")
        return self._reachable

    def flush(self) -> None:
        if self._flush_hangs:
            import time

            time.sleep(30)  # simulate the SDK's unbounded queue.join() wedging
        self.flushed = True
        self.events.append(("flush", ""))


# The stub structurally satisfies the sink's client protocol (keeps the double honest at type time).
_STUB_MATCHES: LangfuseScoreClient = _StubClient()


def _report() -> RunReport:
    return RunReport(
        run_id="run-1",
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
            ),
            dimension="taxonomy_cell",
        ),
    )


def test_publishes_numeric_scores_equal_to_the_series() -> None:
    """Each numeric point is a numeric score whose value equals the series."""
    stub = _StubClient(reachable=True)
    report = _report()
    LangfuseDashboardSink(stub).publish(report)

    written = {name: (value, data_type) for name, value, data_type in stub.scores}
    for point in numeric_series(report):
        assert point.name in written
        assert written[point.name] == (point.value, "NUMERIC")


def test_unreachable_backend_raises_before_writing() -> None:
    """An unreachable backend raises BackendError and writes NO score."""
    stub = _StubClient(reachable=False)
    with pytest.raises(BackendError):
        LangfuseDashboardSink(stub).publish(_report())
    assert stub.auth_checked is True
    assert stub.scores == []  # probe fails BEFORE any write


def test_auth_check_exception_becomes_backend_error() -> None:
    """A raising auth_check surfaces as a typed BackendError, not a raw error."""
    stub = _StubClient(auth_raises=True)
    with pytest.raises(BackendError):
        LangfuseDashboardSink(stub).publish(_report())
    assert stub.scores == []


def test_wedged_flush_raises_backend_error_not_hang() -> None:
    """A wedged SDK flush is bounded and raises BackendError, never hangs indefinitely.

    Mirrors LangfuseBackend's bounded flush: the v3 flush() does an unbounded queue.join(); the sink
    caps it in a daemon thread and raises on timeout rather than blocking the caller forever.
    """
    stub = _StubClient(reachable=True, flush_hangs=True)
    sink = LangfuseDashboardSink(stub, flush_timeout_seconds=0.2)
    with pytest.raises(BackendError, match="did not confirm"):
        sink.publish(_report())


def test_local_leg_written_first_when_configured(tmp_path) -> None:
    """When a local sink is attached, it is written and remains authoritative."""
    from calibrated_agent_evaluation.backends.dashboard import LocalDashboardSink

    local = LocalDashboardSink(tmp_path)
    stub = _StubClient(reachable=True)
    report = _report()
    LangfuseDashboardSink(stub, local=local).publish(report)

    assert tuple(local.load()) == numeric_series(report)  # local round-trip stands
    assert len(stub.scores) == len(numeric_series(report))  # langfuse leg also written


def test_refused_correction_is_published_as_a_categorical_refusal() -> None:
    """A refused rate reaches the dashboard as "refused", not a gap."""

    refused = TrendPoint(
        mode=ModeKey(failure_type="dropped_fact", agent="intake"),
        n_total=20,
        corrected_rate=Interval(
            point=None, lo=0.0, hi=1.0, refusal_reason="youden_j=0.400<0.5", **TEST_INTERVAL_KIND
        ),
        calib_id="c2",
        raw_rate=Interval(point=0.5, lo=0.3, hi=0.7, **TEST_INTERVAL_KIND),
    )
    base = _report()
    report = replace(base, trend=(*base.trend, refused))
    stub = _StubClient(reachable=True)
    LangfuseDashboardSink(stub).publish(report)
    written = {name: (value, data_type) for name, value, data_type in stub.scores}
    assert written["rate.intake/dropped_fact"] == ("refused", "CATEGORICAL")
    assert written["rate.over_refund"] == (0.06, "NUMERIC")


def test_the_run_trace_is_opened_before_any_score_is_attached_to_it() -> None:
    """Langfuse never shows a score whose trace it has not seen, so the trace comes first."""
    stub = _StubClient(reachable=True)
    LangfuseDashboardSink(stub).publish(_report())
    kinds = [kind for kind, _ in stub.events]
    assert kinds[:2] == ["trace", "flush"]
    assert kinds.count("trace") == 1
    trace_id = stub.events[0][1]
    assert all(tid == trace_id for kind, tid in stub.events if kind == "score")
    assert "trace" not in kinds[1:]
    assert len(stub.scores) == len(numeric_series(_report()))


def test_a_score_the_server_rejects_raises_instead_of_vanishing() -> None:
    """The queued write dropped a rejected score silently; the REST create's error surfaces."""
    stub = _StubClient(reject_scores=True)
    with pytest.raises(BackendError, match="while writing score"):
        LangfuseDashboardSink(stub).publish(_report())
    assert stub.scores == []


def test_a_score_write_with_no_id_back_is_not_reported_as_published() -> None:
    stub = _StubClient(no_score_id=True)
    with pytest.raises(BackendError, match="not confirmed"):
        LangfuseDashboardSink(stub).publish(_report())
