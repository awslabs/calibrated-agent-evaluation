# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dumps, dashboard series and the verdict cache are replaced atomically.

Each file used to be truncated and rewritten in place, so a crash mid-write left a partial file:
for the verdict cache that is paid verdicts lost. A write now goes to a sibling temporary file
that is renamed over the old one, which shows on disk as a new inode and no temporary left behind.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends.dashboard import LocalDashboardSink
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.dumpload import MANIFEST_NAME, dump
from calibrated_agent_evaluation.ingest.core import run_ingest
from calibrated_agent_evaluation.judges.cache import VerdictCache
from calibrated_agent_evaluation.run.report import RunReport
from calibrated_agent_evaluation.schema.types import Label

_OTEL = Path(__file__).resolve().parent.parent / "fixtures" / "raw" / "otel"
_MESSAGES = _OTEL / "messages" / "swebench_tool_call.otlp.json"


def _store_with(path: Path, *, label: bool) -> LocalBackend:
    backend = LocalBackend(":memory:")
    (result,) = run_ingest([path], backend=backend).results
    assert result.trace_id is not None
    if label:
        backend.put_score(
            Label(
                trace_id=result.trace_id, span_id=None, kind="human", outcome="FAIL", critique="x"
            )
        )
    return backend


def _inodes(directory: Path) -> dict[str, int]:
    return {p.name: p.stat().st_ino for p in directory.iterdir()}


def _assert_replaced(before: dict[str, int], directory: Path) -> None:
    after = _inodes(directory)
    assert sorted(after) == sorted(before), "a temporary file was left behind"
    assert all(after[name] != before[name] for name in before), (before, after)


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_a_dump_replaces_its_files_atomically(tmp_path: Path, fmt: str) -> None:
    source = _store_with(_MESSAGES, label=True)
    dump(source, tmp_path, fmt)
    # Without its manifest the directory is an unfinished dump, whose files a new dump replaces
    # (a sealed one is removed first).
    (tmp_path / MANIFEST_NAME).unlink()
    before = _inodes(tmp_path)
    dump(source, tmp_path, fmt)
    after = _inodes(tmp_path)
    assert sorted(after) == sorted([*before, MANIFEST_NAME]), "a temporary file was left behind"
    assert all(after[name] != before[name] for name in before), (before, after)


def test_the_dashboard_series_is_replaced_atomically(tmp_path: Path) -> None:
    sink = LocalDashboardSink(tmp_path)
    report = RunReport(run_id="r1", mode="cadence", trend=())
    sink.publish(report)
    before = _inodes(tmp_path)
    sink.publish(report)
    _assert_replaced(before, tmp_path)


def test_the_verdict_cache_is_replaced_atomically(tmp_path: Path) -> None:
    cache = VerdictCache()
    cache.dump(tmp_path)
    before = _inodes(tmp_path)
    cache.dump(tmp_path)
    _assert_replaced(before, tmp_path)
    assert len(VerdictCache.load(tmp_path)) == 0
