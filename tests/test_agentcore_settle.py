# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The read-after-write settling policy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from calibrated_agent_evaluation.agentcore.errors import AgentCoreSettlingLag
from calibrated_agent_evaluation.agentcore.settle import SettlePolicy
from calibrated_agent_evaluation.agentcore.sources import TimeWindow

_END = datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC)


def _window(end: datetime = _END) -> TimeWindow:
    return TimeWindow(start=(end - timedelta(hours=1)).isoformat(), end=end.isoformat())


class _Clock:
    """A fake wall clock whose ``sleep`` advances time, so a poll loop makes progress."""

    def __init__(self, start: datetime) -> None:
        self.t = start
        self.slept: list[float] = []

    def now(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += timedelta(seconds=seconds)


def test_wait_returns_once_the_window_has_settled() -> None:
    # Start 100s after the window end (inside the 300s lag); each polled sleep advances the clock
    # until it clears the lag, within the 600s bound.
    clock = _Clock(_END + timedelta(seconds=100))
    policy = SettlePolicy(lag_seconds=300, max_wait_seconds=600, now=clock.now, sleep=clock.sleep)
    policy.wait_for_window(_window())
    assert clock.slept, "expected the wait to poll at least once before settling"
    assert policy.is_settled(_window(), now=clock.now())


def test_wait_refuses_a_window_that_cannot_settle_within_the_bound() -> None:
    # The window end is now, so it needs a full 300s lag, but the wait is bounded to 10s.
    clock = _Clock(_END)
    policy = SettlePolicy(lag_seconds=300, max_wait_seconds=10, now=clock.now, sleep=clock.sleep)
    with pytest.raises(AgentCoreSettlingLag):
        policy.wait_for_window(_window())


def test_refuse_fires_when_the_window_ends_inside_the_lag() -> None:
    policy = SettlePolicy(lag_seconds=300)
    inside = _END + timedelta(seconds=120)
    with pytest.raises(AgentCoreSettlingLag):
        policy.refuse_if_unsettled(_window(), now=inside)


def test_a_window_safely_in_the_past_is_settled() -> None:
    policy = SettlePolicy(lag_seconds=300)
    well_past = _END + timedelta(seconds=3600)
    assert policy.is_settled(_window(), now=well_past) is True
    policy.refuse_if_unsettled(_window(), now=well_past)  # does not raise
