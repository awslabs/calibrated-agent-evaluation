# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""An advisory, cross-process exclusive lock on a sidecar file.

``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows. Advisory: it serializes cooperating
writers (two ``calibrated-agent-evaluation`` processes recording one cassette), not arbitrary
programs. Not reentrant.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = ["exclusive_lock"]


@contextmanager
def exclusive_lock(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``lock_path`` (created if absent) for the ``with`` block.

    Raises:
        OSError: The lock file cannot be created or locked.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as fh:
        _lock(fh.fileno())
        try:
            yield
        finally:
            _unlock(fh.fileno())


if os.name == "nt":  # pragma: no cover - exercised on Windows only
    import msvcrt

    def _lock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
