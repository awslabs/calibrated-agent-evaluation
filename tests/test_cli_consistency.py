# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A command reads every write the commands before it made.

The store under test is a local backend behind a lag: a write becomes readable only ``LAG``
seconds of a fake clock after it was made, the way Langfuse's asynchronous ingestion behaves
(verified live; see tests/integration/test_cli_read_after_write_live.py). Each
``ConsistentBackend`` is one command.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from calibrated_agent_evaluation.backends import TraceScan
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.consistency import (
    READ_WAIT_ENV,
    ConsistentBackend,
    WriteManifest,
)
from calibrated_agent_evaluation.cli.dispatch import EXIT_OK
from calibrated_agent_evaluation.cli.errors import StaleReadError
from calibrated_agent_evaluation.schema.errors import BackendError
from calibrated_agent_evaluation.schema.types import Label, Trace
from cli_store_helpers import admit_default_judge, default_judge_rater_id, trace

LAG = 5.0


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class LaggingStore:
    """A store whose writes become readable LAG seconds after they are made."""

    def __init__(self, root: Path, clock: Clock, lag: float = LAG) -> None:
        self._real = LocalBackend(root=root)
        self._clock = clock
        self._lag = lag
        self._queue: list[tuple[float, str, Any]] = []

    def _apply(self) -> None:
        due = [item for item in self._queue if item[0] <= self._clock()]
        self._queue = [item for item in self._queue if item[0] > self._clock()]
        for _, op, obj in due:
            getattr(self._real, op)(obj)

    def _later(self, op: str, obj: Any) -> None:
        self._queue.append((self._clock() + self._lag, op, obj))

    def put_trace(self, t: Trace) -> None:
        self._later("put_trace", t)

    def put_score(self, label: Label) -> None:
        self._later("put_score", label)

    def retract_score(self, label: Label) -> None:
        self._later("retract_score", label)

    def get_trace(self, trace_id: str) -> Trace:
        self._apply()
        return self._real.get_trace(trace_id)

    def get_scores(self, trace_id: str, span_id: str | None = None) -> list[Label]:
        self._apply()
        return self._real.get_scores(trace_id, span_id)

    def query(self, frame=None, cell=None, ownership=None) -> list[Trace]:  # type: ignore[no-untyped-def]
        self._apply()
        return self._real.query(frame=frame, cell=cell, ownership=ownership)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path, clock) -> LaggingStore:
    return LaggingStore(tmp_path / "store", clock)


@pytest.fixture
def manifest_file(tmp_path) -> Path:
    return tmp_path / "root" / "pending-writes" / "store"


def command(store: LaggingStore, manifest_file: Path, clock: Clock, wait: float = 30.0):
    """One calibrated-agent-evaluation command's view of the store."""
    return ConsistentBackend(
        store,  # type: ignore[arg-type]
        WriteManifest(manifest_file),
        wait_seconds=wait,
        clock=clock,
        sleep=clock.sleep,
    )


def _verdict(trace_id: str = "T1", outcome: str = "FAIL") -> Label:
    return Label(
        trace_id=trace_id,
        span_id=None,
        kind="llm-judge",
        outcome=outcome,  # type: ignore[arg-type]
        critique="v",
        failure_mode="over_refund",
        rater_id=default_judge_rater_id(),
    )


def test_the_lag_is_real_without_the_manifest(store, clock):
    """The defect: a read straight after a write misses it, and nothing says so."""
    store.put_trace(trace("T1"))
    assert store.query() == []


def test_the_next_command_waits_for_the_last_ones_writes(store, manifest_file, clock):
    judge = command(store, manifest_file, clock)
    judge.put_trace(trace("T1"))
    judge.put_score(_verdict())

    run = command(store, manifest_file, clock)
    assert [lb.outcome for lb in run.get_scores("T1")] == ["FAIL"]
    assert sum(clock.sleeps) >= LAG
    # Confirmed writes are dropped: the command after this one does not wait again.
    assert WriteManifest(manifest_file).empty()
    clock.sleeps.clear()
    command(store, manifest_file, clock).get_scores("T1")
    assert clock.sleeps == []


def test_a_whole_store_read_waits_for_the_listing(store, manifest_file, clock):
    writer = command(store, manifest_file, clock)
    for i in range(3):
        writer.put_trace(trace(f"T{i}"))
    scan = command(store, manifest_file, clock).scan()
    assert isinstance(scan, TraceScan)
    assert sorted(t.trace_id for t in scan.traces) == ["T0", "T1", "T2"]


def test_a_write_that_never_shows_is_a_refusal_naming_it(store, manifest_file, clock):
    writer = command(store, manifest_file, clock)
    writer.put_trace(trace("T1"))
    writer.put_score(_verdict())
    store._queue.clear()  # the writes were lost in transit
    reader = command(store, manifest_file, clock, wait=3.0)
    with pytest.raises(StaleReadError) as exc:
        reader.get_scores("T1")
    message = str(exc.value)
    assert "after waiting 3s it still does not show 1 trace(s) ('T1')" in message
    assert "1 label(s) ('llm-judge FAIL over_refund on T1')" in message
    assert str(manifest_file) in message
    assert READ_WAIT_ENV in message
    # Still pending: the next command refuses too, until the writes show or the file is removed.
    assert not WriteManifest(manifest_file).empty()


def test_a_command_does_not_wait_on_its_own_writes(store, manifest_file, clock):
    """The review surface reads right after each label it writes; it must not stall."""
    review = command(store, manifest_file, clock)
    review.put_score(_verdict())
    review.get_scores("T1")
    assert clock.sleeps == []


def test_a_retraction_is_awaited_until_the_label_is_gone(store, manifest_file, clock):
    store._real.put_trace(trace("T1"))
    store._real.put_score(_verdict())
    command(store, manifest_file, clock).retract_score(_verdict())
    assert command(store, manifest_file, clock).get_scores("T1") == []
    assert sum(clock.sleeps) >= LAG


def test_a_trace_is_awaited_until_all_its_spans_show(tmp_path, manifest_file, clock):
    from _helpers import load_corpus_trace

    crew = load_corpus_trace("CLM-4471")

    class PartialFirst(LaggingStore):
        def get_trace(self, trace_id: str) -> Trace:
            got = super().get_trace(trace_id)
            if self._clock() < 2 * LAG:  # the root lands before its observations
                return Trace(**{**got.__dict__, "spans": got.spans[:1]})
            return got

    store = PartialFirst(tmp_path / "s", clock)
    command(store, manifest_file, clock).put_trace(crew)
    assert len(command(store, manifest_file, clock).get_trace("CLM-4471").spans) == 5
    assert clock.now >= 2 * LAG


def test_the_wait_is_configurable_and_checked(monkeypatch, store, manifest_file, clock):
    monkeypatch.setenv(READ_WAIT_ENV, "soon")
    with pytest.raises(StaleReadError, match="number of seconds"):
        ConsistentBackend(store, WriteManifest(manifest_file))  # type: ignore[arg-type]
    monkeypatch.setenv(READ_WAIT_ENV, "-1")
    with pytest.raises(StaleReadError, match="must not be negative"):
        ConsistentBackend(store, WriteManifest(manifest_file))  # type: ignore[arg-type]


def test_a_corrupt_manifest_entry_is_refused(manifest_file):
    manifest_file.mkdir(parents=True)
    (manifest_file / "0001.json").write_text("{not json")
    with pytest.raises(StaleReadError, match="cannot read the write manifest entry"):
        WriteManifest(manifest_file)


def test_a_failed_write_is_not_recorded(tmp_path, manifest_file, clock):
    class Refusing(LaggingStore):
        def put_score(self, label: Label) -> None:
            raise BackendError("unreachable")

    refusing = Refusing(tmp_path / "s", clock)
    with pytest.raises(BackendError):
        command(refusing, manifest_file, clock).put_score(_verdict())
    assert WriteManifest(manifest_file).empty()


def test_the_langfuse_backend_is_wrapped_and_the_local_one_is_not(monkeypatch, tmp_path):
    import calibrated_agent_evaluation.cli.backends as cli_backends

    real = cli_backends.backend_from_flag
    fake = SimpleNamespace(name="langfuse-stub")
    monkeypatch.setattr(
        cli_backends,
        "backend_from_flag",
        lambda flag, **kw: fake if flag == "langfuse" else real(flag, **kw),
    )
    monkeypatch.setenv("LANGFUSE_HOST", "http://lf.example")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-a")
    args = SimpleNamespace(backend="langfuse", root=str(tmp_path))
    wrapped = cli_backends.backend_from_args(args)  # type: ignore[arg-type]
    assert isinstance(wrapped, ConsistentBackend)
    assert wrapped.inner is fake
    assert not isinstance(
        cli_backends.backend_from_args(SimpleNamespace(backend="local", root=str(tmp_path))),  # type: ignore[arg-type]
        ConsistentBackend,
    )
    # Another project gets another manifest.
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-b")
    other = cli_backends.backend_from_args(args)  # type: ignore[arg-type]
    assert isinstance(other, ConsistentBackend)
    assert other.manifest.path != wrapped.manifest.path


# --- the ticket's case through the CLI: calibrated-agent-evaluation judge, then
# calibrated-agent-evaluation run, back to back -----------------


def test_run_right_after_judge_gates_every_new_verdict(tmp_path, monkeypatch, clock, capsys):
    import calibrated_agent_evaluation.cli.backends as cli_backends
    import calibrated_agent_evaluation.cli.consistency as consistency

    root = tmp_path / "root"
    store = LaggingStore(tmp_path / "store", clock)
    for i in range(10):  # already readable: written long before
        store._real.put_trace(trace(f"T{i:03d}"))
        store._real.put_score(_verdict(f"T{i:03d}", "PASS"))
    monkeypatch.setattr(
        cli_backends, "backend_from_flag", lambda flag, **kw: store if flag == "langfuse" else None
    )
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(consistency, "time", SimpleNamespace(monotonic=clock, sleep=clock.sleep))

    # What `calibrated-agent-evaluation judge` does to the store: ten more traces scored FAIL,
    # written just now.
    judge = cli_backends.backend_from_args(SimpleNamespace(backend="langfuse", root=str(root)))  # type: ignore[arg-type]
    for i in range(10, 20):
        judge.put_trace(trace(f"T{i:03d}"))
        judge.put_score(_verdict(f"T{i:03d}", "FAIL"))

    admit_default_judge(root)
    argv = ["--backend", "langfuse", "--root", str(root), "run", "--n-resamples", "199"]
    assert main(argv) == EXIT_OK
    out = capsys.readouterr().out
    assert "CI gate over 20 judge verdicts on 20 stored trace(s)" in out
    assert "FAILED" in out
