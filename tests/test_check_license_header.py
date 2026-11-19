# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The license-header checker covers Python, JavaScript and CSS, and skips vendored files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "check_license_header.py"


def _check(*paths: Path, fix: bool = False) -> subprocess.CompletedProcess[str]:
    args = [sys.executable, str(_SCRIPT), *(["--fix"] if fix else []), *map(str, paths)]
    return subprocess.run(args, capture_output=True, text=True, check=False)


@pytest.mark.parametrize("name", ["a.py", "a.js", "a.css"])
def test_a_file_without_the_header_fails_and_fix_adds_it(tmp_path, name) -> None:
    path = tmp_path / name
    path.write_text("body\n", encoding="utf-8")
    assert _check(path).returncode == 1
    assert _check(path, fix=True).returncode == 0
    assert "SPDX-License-Identifier: Apache-2.0" in path.read_text(encoding="utf-8")
    assert _check(path).returncode == 0


def test_vendored_and_minified_files_are_skipped(tmp_path) -> None:
    (tmp_path / "vendor").mkdir()
    vendored = tmp_path / "vendor" / "lib.js"
    minified = tmp_path / "lib.min.js"
    for path in (vendored, minified):
        path.write_text("upstream\n", encoding="utf-8")
    assert _check(vendored, minified).returncode == 0


def test_every_tracked_first_party_file_carries_the_header() -> None:
    tracked = subprocess.run(
        ["git", "-C", str(_REPO), "ls-files", "*.py", "*.js", "*.css"],
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        pytest.skip("not a git checkout")
    paths = [_REPO / p for p in tracked.stdout.split()]
    result = _check(*paths)
    assert result.returncode == 0, result.stderr
