# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A directory lock holds every other process off the directory until it is released, and a
directory that cannot hold its lock file is refused with the operating system's error."""

from __future__ import annotations

import multiprocessing
import os
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event
from pathlib import Path

import pytest

from calibrated_agent_evaluation._recordstore import directory_lock

# Upper bounds only, for a spawned interpreter to start and to finish: no assertion depends on how
# long anything takes.
_START_TIMEOUT_SECONDS = 60.0


def _lock_is_free(path: Path) -> bool:
    """Whether another process could take the lock on ``path`` right now, asked without waiting."""
    with path.open("a+b") as fh:
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows only
                import msvcrt

                os.lseek(fh.fileno(), 0, os.SEEK_SET)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            return False
    return True


def _probe_then_lock(
    directory: str, free_while_held: Synchronized[int], probed: Event, released: Event
) -> None:
    free_while_held.value = int(_lock_is_free(Path(directory) / ".lock"))
    probed.set()
    if released.wait(_START_TIMEOUT_SECONDS):
        with directory_lock(Path(directory)):
            pass


def test_a_directory_lock_holds_off_a_second_process_until_released(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    free_while_held = ctx.Value("i", -1)
    probed, released = ctx.Event(), ctx.Event()
    second = ctx.Process(
        target=_probe_then_lock, args=(str(tmp_path), free_while_held, probed, released)
    )

    with directory_lock(tmp_path):
        second.start()
        assert probed.wait(_START_TIMEOUT_SECONDS), "the second process never probed the lock"

    # Released: the second process now takes the lock itself, through directory_lock.
    released.set()
    second.join(_START_TIMEOUT_SECONDS)
    assert free_while_held.value == 0
    assert second.exitcode == 0


@pytest.mark.parametrize(
    ("directory_is", "error"),
    [("missing", FileNotFoundError), ("a-file", NotADirectoryError)],
)
def test_a_directory_that_cannot_hold_a_lock_file_raises_os_error(
    tmp_path: Path, directory_is: str, error: type[OSError]
) -> None:
    directory = tmp_path / "logs"
    if directory_is == "a-file":
        directory.write_text("not a directory", encoding="utf-8")
    entered = False

    with pytest.raises(error), directory_lock(directory):
        entered = True

    assert not entered
    assert not (directory / ".lock").exists()
