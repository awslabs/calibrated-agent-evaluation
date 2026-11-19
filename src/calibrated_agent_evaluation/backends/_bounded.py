# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a blocking SDK call with a deadline, on a daemon thread.

Some SDK calls can block without bound, which is as dishonest as a silent drop: the caller can
neither succeed nor fail. Two adapters run one here: the Langfuse backend's ``flush()``, which
blocks on an unbounded ``queue.join()`` when the ingestion queue wedges (verified
langfuse==3.15.0), and the AgentCore Runtime session stop, a ``StopRuntimeSession`` call that can
hang on a wedged connection. The caller waits at most its timeout and turns a timeout or an error
into its own typed outcome; the worker's exception, an interrupt included, is handed back for the
caller to raise or report in its own thread.

A thread that outlives the deadline is abandoned, since Python has no safe thread-kill. It is a
daemon, so it never keeps the process alive, and holds only what the call holds (the Langfuse
client's queue lock, or one connection of the AWS client's pool). The call may still finish
later, so a timeout says only that the caller stopped waiting: the Langfuse write is not
confirmed, and the session stop is recorded as failed because it can't be shown to have worked.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["BoundedOutcome", "call_bounded"]


@dataclass(frozen=True)
class BoundedOutcome:
    """How a bounded call ended: ``finished`` within the deadline, and the ``error`` it raised."""

    finished: bool
    error: BaseException | None = None


def call_bounded(fn: Callable[[], object], *, timeout_seconds: float, name: str) -> BoundedOutcome:
    """Run ``fn`` on a daemon thread named ``name`` and wait at most ``timeout_seconds``.

    Args:
        fn: The blocking call.
        timeout_seconds: How long to wait before giving up on it.
        name: The worker thread's name, which shows up in thread dumps.

    Returns:
        Whether the call finished in time, and the exception it raised if it did and failed.
    """
    error: list[BaseException] = []

    def _run() -> None:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - handed back to the calling thread
            error.append(exc)

    worker = threading.Thread(target=_run, name=name, daemon=True)
    worker.start()
    worker.join(timeout=timeout_seconds)
    if worker.is_alive():
        return BoundedOutcome(finished=False)
    return BoundedOutcome(finished=True, error=error[0] if error else None)
