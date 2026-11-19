# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The flagship harness introduces no copyleft node.

adds no new dependency and no new extra -- it exercises the already-pinned
[langfuse]/[inspect]/[strands]/[bedrock] extras through their shipped adapters. This asserts the
flagship package declares no runtime dependency of its own and that the extras it drives carry no
copyleft node (running the repository's scanner). The full-tree scan lives in test_license_scan.py.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "license_scan.py"


def test_flagship_adds_no_runtime_dependency() -> None:
    """The default runtime tree is pydantic alone; flagship added no runtime dep."""
    pyproject = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["project"]["dependencies"]
    assert [d.split(">")[0].split("=")[0].split("<")[0] for d in deps] == ["pydantic"]


@pytest.mark.parametrize("extra", ["inspect", "strands", "bedrock", "langfuse"])
def test_flagship_extras_carry_no_copyleft(extra: str) -> None:
    """Each extra the flagship exercises resolves to a copyleft-free tree."""
    run = subprocess.run(
        [sys.executable, str(_SCRIPT), "--extra", extra],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert run.returncode == 0, f"[{extra}] resolved a copyleft node: {run.stderr}"
