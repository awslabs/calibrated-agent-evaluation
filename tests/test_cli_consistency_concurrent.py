# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two calibrated-agent-evaluation processes on one root keep each other's pending writes.

The write manifest was one JSON file read and rewritten whole by each command, so two commands on
one root overwrote each other's entries and a later read stopped waiting for a write that had
happened. Each write is now its own entry file.
``calibrated-agent-evaluation ingest --from-langfuse`` also waits for the store's listing, not only
for each earlier write one at a time.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends import SourceTrace
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli.consistency import (
    ConsistentBackend,
    WriteManifest,
    manifest_path,
)
from calibrated_agent_evaluation.cli.errors import StaleReadError
from calibrated_agent_evaluation.ingest import ingest_langfuse
from calibrated_agent_evaluation.schema.types import Label
from cli_store_helpers import trace
from test_cli_consistency import LAG, Clock, LaggingStore

_N = 40

# One calibrated-agent-evaluation process: open the manifest, wait for the go file, then record _N
# traces and _N labels.
_WRITER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from types import SimpleNamespace
    from calibrated_agent_evaluation.cli.consistency import WriteManifest
    from calibrated_agent_evaluation.schema.types import Label

    path, name, n, go = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    manifest = WriteManifest(path)
    while not go.exists():
        time.sleep(0.01)
    for i in range(n):
        manifest.record_trace(SimpleNamespace(trace_id=f"{name}-{i}", spans=[]))
        label = Label(
            trace_id=f"{name}-{i}", span_id=None, kind="llm-judge", outcome="FAIL",
            critique="v", failure_mode="m",
        )
        manifest.record_label(label, retracted=False)
    """
)


@pytest.fixture
def manifest_dir(tmp_path) -> Path:
    return tmp_path / "root" / "pending-writes" / "store"


def test_two_concurrent_writers_keep_every_entry(manifest_dir, tmp_path):
    go = tmp_path / "go"
    procs = [
        subprocess.Popen([sys.executable, "-c", _WRITER, str(manifest_dir), name, str(_N), str(go)])
        for name in ("A", "B")
    ]
    time.sleep(1.0)  # both have read the (empty) manifest before either writes
    go.touch()
    assert [p.wait(timeout=60) for p in procs] == [0, 0]

    merged = WriteManifest(manifest_dir)
    expected = {f"{name}-{i}" for name in ("A", "B") for i in range(_N)}
    assert set(merged.traces) == expected
    assert {doc["trace_id"] for doc in merged.labels.values()} == expected
    assert len(merged.labels) == 2 * _N


def test_confirming_drops_only_the_entries_read_not_a_later_write(manifest_dir):
    """A command confirms what it read; a re-put another command made meanwhile still waits."""
    WriteManifest(manifest_dir).record_trace(trace("T1"))
    reader = WriteManifest(manifest_dir)
    assert reader.traces == {"T1": 0}
    WriteManifest(manifest_dir).record_trace(trace("T1"))  # another command re-puts T1
    reader.confirm({"T1"}, set(), set())
    assert WriteManifest(manifest_dir).traces == {"T1": 0}


def test_the_newer_of_a_put_and_a_retraction_is_the_one_awaited(manifest_dir):
    label = Label(
        trace_id="T1",
        span_id=None,
        kind="llm-judge",
        outcome="FAIL",
        critique="v",
        failure_mode="m",
    )
    WriteManifest(manifest_dir).record_label(label, retracted=False)
    WriteManifest(manifest_dir).record_label(label, retracted=True)
    seen = WriteManifest(manifest_dir)
    assert seen.labels == {}
    assert len(seen.retractions) == 1
    seen.confirm(set(), set(), set(seen.retractions))
    assert WriteManifest(manifest_dir).empty()
    assert list(manifest_dir.glob("*.json")) == []


def test_an_entry_another_command_confirmed_meanwhile_is_not_an_error(manifest_dir):
    WriteManifest(manifest_dir).record_trace(trace("T1"))
    first, second = WriteManifest(manifest_dir), WriteManifest(manifest_dir)
    first.confirm({"T1"}, set(), set())
    second.confirm({"T1"}, set(), set())
    assert WriteManifest(manifest_dir).empty()


def test_an_entry_of_another_version_is_refused(manifest_dir):
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "0001.json").write_text(
        json.dumps({"version": 1, "kind": "trace", "trace_id": "T1", "n_spans": 1})
    )
    with pytest.raises(
        StaleReadError, match="version 1; this calibrated-agent-evaluation reads version 2"
    ):
        WriteManifest(manifest_dir)


def test_the_single_file_manifest_of_an_earlier_version_is_refused(tmp_path):
    path = manifest_path(tmp_path, "langfuse|h|pk")
    legacy = path.with_name(path.name + ".json")
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"traces": {"T1": 0}, "labels": {}, "retractions": {}}))
    with pytest.raises(StaleReadError) as exc:
        WriteManifest(path)
    assert str(legacy) in str(exc.value)
    assert "single-file" in str(exc.value)


# --- ingest --from-langfuse waits for the listing -------------------------------------------


class ListingLagStore(LaggingStore):
    """A store whose listing (``pull``) shows a trace LAG seconds after ``get_trace`` does."""

    def __init__(self, root: Path, clock: Clock) -> None:
        super().__init__(root, clock)
        self._readable_at: dict[str, float] = {}

    def get_trace(self, trace_id):  # type: ignore[no-untyped-def]
        got = super().get_trace(trace_id)
        self._readable_at.setdefault(trace_id, self._clock())
        return got

    def pull(self):  # type: ignore[no-untyped-def]
        self._apply()
        for t in self._real.query():
            since = self._readable_at.get(t.trace_id)
            if since is not None and self._clock() >= since + LAG:
                yield SourceTrace(source_id=f"lf-{t.trace_id}", trace=t)


def _command(store, manifest_dir, clock, wait=30.0):  # type: ignore[no-untyped-def]
    return ConsistentBackend(
        store, WriteManifest(manifest_dir), wait_seconds=wait, clock=clock, sleep=clock.sleep
    )


def test_ingest_from_langfuse_waits_for_the_listing(tmp_path, manifest_dir):
    clock = Clock()
    store = ListingLagStore(tmp_path / "store", clock)
    writer = _command(store, manifest_dir, clock)
    for i in range(3):
        writer.put_trace(trace(f"T{i}"))

    source = _command(store, manifest_dir, clock)
    with LocalBackend(root=tmp_path / "dest") as dest:
        report = ingest_langfuse(source, backend=dest)
        assert report.ingested == 3
        assert sorted(t.trace_id for t in dest.query()) == ["T0", "T1", "T2"]
    assert clock.now >= 2 * LAG  # the writes' lag, then the listing's


def test_a_listing_that_never_catches_up_is_refused_before_anything_is_ingested(
    tmp_path, manifest_dir
):
    clock = Clock()

    class NeverListed(ListingLagStore):
        def pull(self):  # type: ignore[no-untyped-def]
            yield from ()

    store = NeverListed(tmp_path / "store", clock)
    _command(store, manifest_dir, clock).put_trace(trace("T1"))
    source = _command(store, manifest_dir, clock, wait=12.0)
    with LocalBackend(root=tmp_path / "dest") as dest:
        with pytest.raises(StaleReadError, match=r"listing still lacks 1 trace\(s\).*'T1'"):
            ingest_langfuse(source, backend=dest)
        assert dest.query() == []
