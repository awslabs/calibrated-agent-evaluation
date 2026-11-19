# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the hotkeys.js controller tests (tests/js/) under Node's built-in test runner.

The review surface's throughput and integrity guards (key-repeat, in-flight lock, dwell,
reconcile-on-non-recorded, Ctrl/Cmd+Enter) live in ``static/hotkeys.js``. They are tested in
JavaScript with no npm dependency (``node:test`` + ``node:vm``); this wrapper makes ``just check``
run them. Without Node on PATH the test is skipped with that reason in the report, not passed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_JS_TESTS = Path(__file__).resolve().parents[1] / "js"


def test_hotkeys_controller_js_suite_passes() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH; the hotkeys.js suite in tests/js/ did not run")
    # Node >= 22 resolves a bare directory argument as a module, so pass the files explicitly.
    test_files = sorted(str(p) for p in _JS_TESTS.glob("*.test.js"))
    assert test_files, f"no *.test.js files under {_JS_TESTS}"
    result = subprocess.run(
        [node, "--test", *test_files],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
