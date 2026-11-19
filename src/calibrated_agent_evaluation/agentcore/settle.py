# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-after-write settling policy for AgentCore reads.

AgentCore telemetry is written asynchronously: a trace that finished moments ago is not yet fully
readable, so a window whose end is too recent returns a partial trace. This mirrors the pattern in
``calibrated_agent_evaluation.cli.consistency``: wait a bounded interval for the window to settle,
and refuse when it cannot settle in time rather than return a mid-write trace.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from calibrated_agent_evaluation.agentcore.errors import AgentCoreSettlingLag
from calibrated_agent_evaluation.agentcore.sources import TimeWindow

SETTLE_ENV_VAR = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_SETTLE_SECONDS"
# On a live account a trace became fully readable within about five minutes; that is the default
# lag.
DEFAULT_LAG_SECONDS = 300.0
_POLL_SECONDS = 5.0


def _parse_end(window: TimeWindow) -> datetime:
    return datetime.fromisoformat(window.end)


class SettlePolicy:
    """Decides whether a window has settled, waiting a bounded interval when it has not."""

    def __init__(
        self,
        lag_seconds: float = DEFAULT_LAG_SECONDS,
        *,
        max_wait_seconds: float | None = None,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.lag_seconds = lag_seconds
        # Without an explicit bound, wait at most one full lag — long enough for a window that just
        # entered the lag to clear it, bounded so a far-future window cannot block forever.
        self.max_wait_seconds = lag_seconds if max_wait_seconds is None else max_wait_seconds
        self._now = now or (lambda: datetime.now(tz=UTC))
        self._sleep = sleep or time.sleep

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> SettlePolicy:
        """A policy whose lag is the one the environment sets, when set.

        The variable is ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_SETTLE_SECONDS``.
        """
        env = environ if environ is not None else dict(os.environ)
        raw = env.get(SETTLE_ENV_VAR, "").strip()
        if not raw:
            return cls()
        try:
            return cls(float(raw))
        except ValueError as exc:
            raise AgentCoreSettlingLag(
                f"{SETTLE_ENV_VAR}={raw!r} is not a number of seconds"
            ) from exc

    def is_settled(self, window: TimeWindow, *, now: datetime | None = None) -> bool:
        """Whether the window ended at least ``lag_seconds`` before ``now``."""
        current = now or self._now()
        return (current - _parse_end(window)).total_seconds() >= self.lag_seconds

    def refuse_if_unsettled(self, window: TimeWindow, *, now: datetime | None = None) -> None:
        """Raise :class:`AgentCoreSettlingLag` when the window ends inside the lag."""
        if not self.is_settled(window, now=now):
            raise AgentCoreSettlingLag(
                f"the window ends at {window.end}, inside the {self.lag_seconds:g}s settling lag; "
                f"reading now would return a partial trace. Retry after the lag, or raise "
                f"{SETTLE_ENV_VAR} if the account settles faster"
            )

    def wait_for_window(self, window: TimeWindow) -> None:
        """Wait up to ``max_wait_seconds`` for the window to settle, else refuse.

        Returns as soon as the window has settled. Raises :class:`AgentCoreSettlingLag` if it
        cannot settle within the bounded wait, so a window too recent to ever settle in time is a
        loud refusal, not a silent partial read.
        """
        deadline = self._now() + timedelta(seconds=self.max_wait_seconds)
        while True:
            if self.is_settled(window):
                return
            if self._now() >= deadline:
                self.refuse_if_unsettled(window)
                return
            self._sleep(_POLL_SECONDS)
