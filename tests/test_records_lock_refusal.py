# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A records directory that cannot hold its lock is refused with a CLI error naming the directory,
before the block that needed the lock runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrated_agent_evaluation.cli.errors import CliError
from calibrated_agent_evaluation.cli.records import records_dir, records_lock

_KIND = "calibration"


@pytest.mark.parametrize(
    "blocked_by",
    ["a-file-in-place-of-the-directory", "a-directory-in-place-of-the-lock-file"],
)
def test_a_records_directory_that_cannot_hold_a_lock_is_refused_naming_it(
    tmp_path: Path, blocked_by: str
) -> None:
    directory = records_dir(tmp_path, _KIND)
    directory.parent.mkdir(parents=True)
    if blocked_by == "a-file-in-place-of-the-directory":
        directory.write_text("not a directory", encoding="utf-8")
    else:
        (directory / ".lock").mkdir(parents=True)
    entered = False

    with pytest.raises(CliError) as refused, records_lock(tmp_path, _KIND):
        entered = True

    assert not entered
    message = str(refused.value)
    assert str(directory) in message
    assert f"cannot lock the {_KIND} records" in message
    assert isinstance(refused.value.__cause__, OSError)
