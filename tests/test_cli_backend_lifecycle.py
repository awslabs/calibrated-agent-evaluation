# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CLI stages close the local store they open.

``ingest``, ``dump``, ``load``, ``capture``, ``probe`` and ``quickstart`` opened a local store and
never closed it, leaking one sqlite connection per command (a ``ResourceWarning`` once collected).
They now close it when the command ends. Each command runs in a fresh interpreter, so no other
test's unclosed connection can be counted against it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_FIXTURE = _REPO / "fixtures" / "raw" / "otel" / "ORD-1001.events.otlp.json"

_RUN = textwrap.dedent(
    """
    import gc
    import sys
    import warnings

    warnings.simplefilter("always", ResourceWarning)
    from calibrated_agent_evaluation.cli import main

    code = main(sys.argv[1:])
    gc.collect()
    print("exit", code)
    """
)


def _run(tmp_path: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _RUN, *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        check=False,
        timeout=300,
    )


@pytest.mark.parametrize(
    "command",
    [
        ("ingest", str(_FIXTURE)),
        ("dump", "{dump}"),
        ("load", "{dump}"),
    ],
    ids=lambda c: c[0],
)
def test_a_stage_leaves_no_unclosed_store(tmp_path: Path, command: tuple[str, ...]) -> None:
    root = tmp_path / "store"
    dump_dir = tmp_path / "dump"
    seeded = _run(tmp_path, "--root", str(root), "ingest", str(_FIXTURE))
    assert seeded.stdout.strip().endswith("exit 0"), seeded.stderr[-2000:]
    if command[0] == "load":
        assert _run(tmp_path, "--root", str(root), "dump", str(dump_dir)).returncode == 0
        root = tmp_path / "fresh"
    argv = [part.format(dump=dump_dir) for part in command]
    result = _run(tmp_path, "--root", str(root), *argv)
    assert result.stdout.strip().endswith("exit 0"), result.stderr[-2000:]
    assert "ResourceWarning" not in result.stderr, result.stderr[-2000:]


def test_quickstart_leaves_no_unclosed_store(tmp_path: Path) -> None:
    result = _run(tmp_path, "--root", str(tmp_path / "qs"), "quickstart")
    assert result.stdout.strip().endswith("exit 0"), result.stderr[-2000:]
    assert "ResourceWarning" not in result.stderr, result.stderr[-2000:]
