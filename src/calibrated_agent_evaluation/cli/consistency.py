# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read what the earlier commands wrote: a write manifest for an eventually consistent store.

Langfuse ingests asynchronously. A trace or score written and flushed by one
``calibrated-agent-evaluation`` command can take seconds to become readable, so
``calibrated-agent-evaluation run`` straight after ``calibrated-agent-evaluation judge`` read the
store before the judge's verdicts were queryable, gated on the older set, and exited normally.
Nothing in the output said the read was incomplete.

:class:`ConsistentBackend` wraps the Langfuse backend a command opens:

  * every write it forwards (``put_trace``, ``put_score``, ``retract_score``) is recorded in a
    manifest under the local root, ``<root>/pending-writes/<store>/``, keyed by the store the
    credentials address, once the write has returned. Each write is its own entry file, so two
    commands on one root never lose each other's entries;
  * before its first read it waits until every write the manifest held when the command started
    is readable (a trace with all its spans, a label present, a retracted label gone), polling up
    to a bounded wait (``CALIBRATED_AGENT_EVALUATION_READ_WAIT_SECONDS``, default 30), and deletes
    the entries it confirmed. A write still missing after the wait is a :class:`StaleReadError`
    naming it: the command refuses rather than computes over a partial store.
  * a whole-store read (``scan``, and the ``pull`` that
    ``calibrated-agent-evaluation ingest --from-langfuse`` reads) also waits until every trace the
    earlier commands wrote is in the listing, which lags a single-trace read.

Only writes made before the command started are awaited, so a command never waits on its own
writes (the review surface reads right after each label it writes); the next command does. The
local backend is synchronous and is not wrapped.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from time import time_ns
from typing import Any, TypeVar

from calibrated_agent_evaluation.backends import (
    SourceTrace,
    TraceScan,
    TraceStoreBackend,
    scan_traces,
)
from calibrated_agent_evaluation.cli.errors import StaleReadError
from calibrated_agent_evaluation.schema.errors import BackendError
from calibrated_agent_evaluation.schema.serde import label_identity, to_dict
from calibrated_agent_evaluation.schema.types import Frame, Label, Ownership, Trace

__all__ = [
    "MANIFEST_VERSION",
    "READ_WAIT_ENV",
    "ConsistentBackend",
    "WriteManifest",
    "manifest_path",
]

READ_WAIT_ENV = "CALIBRATED_AGENT_EVALUATION_READ_WAIT_SECONDS"
_DEFAULT_WAIT_SECONDS = 30.0
_POLL_SECONDS = 1.0
# Gap entries named in a refusal; the count is always given in full.
_SHOW = 5
_R = TypeVar("_R")


def manifest_path(root: Path, store_key: str) -> Path:
    """``<root>/pending-writes/<store>/``, the entry directory for the store ``store_key`` names."""
    digest = hashlib.sha256(store_key.encode("utf-8")).hexdigest()[:16]
    return root / "pending-writes" / digest


# Version 1 was the single-file manifest that the per-entry layout replaced; it is refused.
MANIFEST_VERSION = 2
# Tells apart one process's entries written in the same nanosecond. Lock-guarded: the review
# surface writes labels from request threads.
_SEQ = itertools.count()
_SEQ_LOCK = threading.Lock()


class WriteManifest:
    """The writes to one store that no later read has confirmed yet.

    Each write is its own entry file under ``path``, created atomically under a name no other
    process picks, so two ``calibrated-agent-evaluation`` commands on one root never overwrite each
    other's entries. Entry names sort in write order; when one key has several entries the
    newest decides it (a retraction after a put, a re-put with more spans).

    ``traces`` maps a trace id to its span count; ``labels`` and ``retractions`` map a label
    identity (``schema.serde.label_identity``) to the label document, kept so a refusal can say
    which verdict is missing. They hold the entries present when the manifest was read: recording
    a write adds an entry file and leaves them alone, so a command never waits on its own writes.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._refuse_single_file()
        self.traces: dict[str, int] = {}
        self.labels: dict[str, dict[str, Any]] = {}
        self.retractions: dict[str, dict[str, Any]] = {}
        # Every entry file read for a key, superseded ones included, so confirming the key deletes
        # them all. An entry written after this read is never in here, so it is never deleted.
        self._files: dict[tuple[str, str], list[Path]] = {}
        for entry, doc in self._read_entries():
            self._fold(entry, doc)

    def _refuse_single_file(self) -> None:
        legacy = self.path.with_name(self.path.name + ".json")
        for candidate in (legacy, self.path):
            if candidate.is_file():
                raise StaleReadError(
                    f"{candidate} is a single-file write manifest from an earlier release, which "
                    "two concurrent commands could overwrite; it is no longer read. Once the "
                    "writes it lists are readable in the store, or are known lost, delete it"
                )

    def _read_entries(self) -> list[tuple[Path, dict[str, Any]]]:
        if not self.path.exists():
            return []
        try:
            names = sorted(self.path.glob("*.json"))
        except OSError as exc:
            raise StaleReadError(f"cannot list the write manifest {self.path}: {exc}") from exc
        entries: list[tuple[Path, dict[str, Any]]] = []
        for entry in names:
            try:
                doc = json.loads(entry.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue  # another command confirmed it after the listing: nothing to wait for
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise StaleReadError(
                    f"cannot read the write manifest entry {entry}: {exc}; it records a write an "
                    "earlier command made to the store, so a read cannot be checked against it"
                ) from exc
            entries.append((entry, _check_entry(entry, doc)))
        return entries

    def _fold(self, entry: Path, doc: dict[str, Any]) -> None:
        kind = doc["kind"]
        if kind == "trace":
            key = ("trace", doc["trace_id"])
            self.traces[doc["trace_id"]] = doc["n_spans"]
        else:
            key = ("label", doc["key"])
            bucket = self.retractions if kind == "retraction" else self.labels
            other = self.labels if kind == "retraction" else self.retractions
            bucket[doc["key"]] = doc["label"]
            other.pop(doc["key"], None)  # the newer write is the one a read must see
        self._files.setdefault(key, []).append(entry)

    def empty(self) -> bool:
        """Whether every recorded write has been confirmed by a read."""
        return not (self.traces or self.labels or self.retractions)

    def record_trace(self, trace: Trace) -> None:
        """Record a trace write that later reads must see."""
        doc = {"kind": "trace", "trace_id": trace.trace_id, "n_spans": len(trace.spans)}
        self._write(doc, trace.trace_id)

    def record_label(self, label: Label, *, retracted: bool) -> None:
        """Record a label write (or retraction) that later reads must see."""
        kind = "retraction" if retracted else "label"
        doc = {"kind": kind, "key": label_identity(label), "label": to_dict(label)}
        self._write(doc, label.trace_id)

    def confirm(self, traces: set[str], labels: set[str], retractions: set[str]) -> None:
        """Delete the entries read for the confirmed writes, so later reads do not wait on them.

        Only entry files this manifest read are deleted. One written since, for the same trace or
        label, stays pending for the next command.
        """
        keys = [("trace", t) for t in traces] + [("label", k) for k in labels | retractions]
        for tid in traces:
            self.traces.pop(tid, None)
        for key in labels:
            self.labels.pop(key, None)
        for key in retractions:
            self.retractions.pop(key, None)
        for key in keys:
            for entry in self._files.pop(key, []):
                try:
                    entry.unlink(missing_ok=True)
                except OSError as exc:
                    raise StaleReadError(
                        f"cannot delete the confirmed write manifest entry {entry}: "
                        f"{exc.strerror or exc}; the next command would wait on it again"
                    ) from exc

    def _write(self, doc: dict[str, Any], trace_id: str) -> None:
        with _SEQ_LOCK:
            seq = next(_SEQ)
        name = f"{time_ns():020d}-{os.getpid()}-{seq:06d}-{uuid.uuid4().hex[:8]}.json"
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            tmp = self.path / f".{name}.tmp"
            body = {"version": MANIFEST_VERSION, **doc}
            tmp.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path / name)
        except OSError as exc:
            raise StaleReadError(
                f"cannot record the write to {trace_id!r} in the write manifest {self.path}: "
                f"{exc.strerror or exc}; the next command could not check that it reads this write"
            ) from exc


_ENTRY_FIELDS: dict[str, dict[str, type]] = {
    "trace": {"trace_id": str, "n_spans": int},
    "label": {"key": str, "label": dict},
    "retraction": {"key": str, "label": dict},
}


def _check_entry(entry: Path, doc: Any) -> dict[str, Any]:
    """``doc`` if it is a current-version entry of a known kind with its fields, else a refusal."""
    if not isinstance(doc, dict):
        raise StaleReadError(f"the write manifest entry {entry} is not a JSON object")
    version = doc.get("version")
    if version != MANIFEST_VERSION:
        raise StaleReadError(
            f"the write manifest entry {entry} is version {version}; this "
            "calibrated-agent-evaluation reads version "
            f"{MANIFEST_VERSION}. Once the write it records is readable in the store, delete it"
        )
    fields = _ENTRY_FIELDS.get(str(doc.get("kind")))
    if fields is None or any(not isinstance(doc.get(f), t) for f, t in fields.items()):
        raise StaleReadError(
            f"the write manifest entry {entry} is not a trace, label or retraction entry"
        )
    return doc


def read_wait_seconds() -> float:
    """The bounded wait: ``CALIBRATED_AGENT_EVALUATION_READ_WAIT_SECONDS``, else 30 seconds."""
    raw = os.environ.get(READ_WAIT_ENV)
    if raw is None or not raw.strip():
        return _DEFAULT_WAIT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise StaleReadError(f"{READ_WAIT_ENV} must be a number of seconds, got {raw!r}") from exc
    if value < 0:
        raise StaleReadError(f"{READ_WAIT_ENV} must not be negative, got {value}")
    return value


class ConsistentBackend:
    """A trace store whose reads include every write the earlier commands made (see module doc).

    ``clock`` and ``sleep`` are injection seams for tests.
    """

    def __init__(
        self,
        inner: TraceStoreBackend,
        manifest: WriteManifest,
        *,
        wait_seconds: float | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._inner = inner
        self._manifest = manifest
        self._wait = read_wait_seconds() if wait_seconds is None else wait_seconds
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        # What the command must see: the writes recorded before it started. Its own writes are
        # recorded for the next command and never awaited here.
        self._await_traces = dict(manifest.traces)
        self._await_labels = dict(manifest.labels)
        self._await_retractions = dict(manifest.retractions)
        self._settled = manifest.empty()

    @property
    def inner(self) -> TraceStoreBackend:
        """The wrapped trace store."""
        return self._inner

    @property
    def manifest(self) -> WriteManifest:
        """The writes not yet confirmed by a read."""
        return self._manifest

    # --- writes: forward, then record -------------------------------------------------------

    def put_trace(self, trace: Trace) -> None:
        """Write ``trace`` to the wrapped store and record it in the manifest."""
        self._inner.put_trace(trace)
        self._manifest.record_trace(trace)

    def put_score(self, label: Label) -> None:
        """Write ``label`` to the wrapped store and record it in the manifest."""
        self._inner.put_score(label)
        self._manifest.record_label(label, retracted=False)

    def retract_score(self, label: Label) -> None:
        """Retract ``label`` in the wrapped store and record the retraction in the manifest."""
        self._inner.retract_score(label)
        self._manifest.record_label(label, retracted=True)

    # --- reads: settle first ----------------------------------------------------------------

    def get_trace(self, trace_id: str) -> Trace:
        """Read a trace once every recorded write is visible."""
        self._settle()
        return self._inner.get_trace(trace_id)

    def get_scores(self, trace_id: str, span_id: str | None = None) -> list[Label]:
        """Read a trace's labels once every recorded write is visible."""
        self._settle()
        return self._inner.get_scores(trace_id, span_id)

    def query(
        self,
        frame: Frame | None = None,
        cell: dict[str, str] | None = None,
        ownership: Ownership | None = None,
    ) -> list[Trace]:
        """Query the wrapped store once every recorded write is visible."""
        return self.scan(frame=frame, cell=cell, ownership=ownership).traces

    def scan(
        self,
        frame: Frame | None = None,
        cell: dict[str, str] | None = None,
        ownership: Ownership | None = None,
    ) -> TraceScan:
        """Scan the wrapped store once every recorded write is visible."""
        self._settle()
        if frame is not None or cell is not None or ownership is not None:
            # A filter may rightly exclude a written trace; there is nothing to expect.
            return scan_traces(self._inner, frame=frame, cell=cell, ownership=ownership)
        return self._until_listed(
            lambda: scan_traces(self._inner),
            lambda scan: {t.trace_id for t in scan.traces} | {s.trace_id for s in scan.skipped},
        )

    def pull(self) -> Iterator[SourceTrace]:
        """The wrapped store's ``pull``, once it lists every trace the earlier commands wrote.

        ``calibrated-agent-evaluation ingest --from-langfuse`` reads this. Each earlier write is
        first awaited on its own (``_settle``), then the listing, which lags a single-trace read.
        The pull is read whole before anything is returned, so a listing that never catches up is
        refused before a single trace is ingested.
        """
        self._settle()
        pulled = self._until_listed(
            lambda: list(self._inner.pull()),  # type: ignore[attr-defined]
            lambda items: (
                {i.source_id for i in items}
                | {i.trace.trace_id for i in items if i.trace is not None}
            ),
        )
        return iter(pulled)

    def _until_listed(self, read: Callable[[], _R], listed: Callable[[_R], set[str]]) -> _R:
        """``read()`` again until its result lists every awaited trace, up to the bounded wait."""
        deadline = self._clock() + self._wait
        while True:
            result = read()
            missing = sorted(set(self._await_traces) - listed(result))
            if not missing:
                return result
            if self._clock() >= deadline:
                raise StaleReadError(
                    f"the store's listing still lacks {len(missing)} trace(s) an earlier command "
                    f"wrote ({_names(missing)}) after waiting {self._wait:g}s; the whole-store "
                    f"read would be partial. {self._advice()}"
                )
            self._sleep(_POLL_SECONDS)

    def __getattr__(self, name: str) -> Any:
        # pull / project / close and anything else the wrapped backend offers.
        return getattr(self._inner, name)

    def __enter__(self) -> ConsistentBackend:
        return self

    def __exit__(self, *_exc: object) -> None:
        close = getattr(self._inner, "__exit__", None)
        if callable(close):
            close(None, None, None)

    # --- the wait ---------------------------------------------------------------------------

    def _settle(self) -> None:
        if self._settled:
            return
        deadline = self._clock() + self._wait
        while True:
            gaps = self._gaps()
            seen_traces = set(self._await_traces) - set(gaps["trace"])
            seen_labels = set(self._await_labels) - set(gaps["label"])
            seen_retractions = set(self._await_retractions) - set(gaps["retraction"])
            self._manifest.confirm(seen_traces, seen_labels, seen_retractions)
            if not any(gaps.values()):
                self._settled = True
                return
            if self._clock() >= deadline:
                raise StaleReadError(self._gap_message(gaps))
            self._sleep(_POLL_SECONDS)

    def _gaps(self) -> dict[str, list[str]]:
        """The awaited writes not yet readable, by kind (trace ids / label identities)."""
        gaps: dict[str, list[str]] = {"trace": [], "label": [], "retraction": []}
        for tid, n_spans in self._await_traces.items():
            try:
                if len(self._inner.get_trace(tid).spans) < n_spans:
                    gaps["trace"].append(tid)
            except BackendError:
                gaps["trace"].append(tid)
        visible: dict[str, set[str]] = {}
        for key, doc in self._await_labels.items():
            if key not in self._identities(doc["trace_id"], visible):
                gaps["label"].append(key)
        for key, doc in self._await_retractions.items():
            if key in self._identities(doc["trace_id"], visible):
                gaps["retraction"].append(key)
        return gaps

    def _identities(self, trace_id: str, cache: dict[str, set[str]]) -> set[str]:
        if trace_id not in cache:
            cache[trace_id] = {label_identity(lb) for lb in self._inner.get_scores(trace_id)}
        return cache[trace_id]

    def _gap_message(self, gaps: dict[str, list[str]]) -> str:
        parts = []
        if gaps["trace"]:
            parts.append(f"{len(gaps['trace'])} trace(s) ({_names(gaps['trace'])})")
        for kind, docs in (("label", self._await_labels), ("retraction", self._await_retractions)):
            if gaps[kind]:
                described = [_describe(docs[k]) for k in gaps[kind]]
                what = "label(s)" if kind == "label" else "retraction(s) still visible"
                parts.append(f"{len(described)} {what} ({_names(described)})")
        return (
            f"earlier commands wrote to the store and after waiting {self._wait:g}s it still does "
            f"not show {'; '.join(parts)}; reading now would compute over a partial store. "
            f"{self._advice()}"
        )

    def _advice(self) -> str:
        return (
            f"Retry, raise {READ_WAIT_ENV}, or, if those writes are known lost, delete the "
            f"directory {self._manifest.path} to stop waiting for them"
        )


def _describe(doc: dict[str, Any]) -> str:
    span = f"/{doc['span_id']}" if doc.get("span_id") else ""
    mode = f" {doc['failure_mode']}" if doc.get("failure_mode") else ""
    return f"{doc['kind']} {doc['outcome']}{mode} on {doc['trace_id']}{span}"


def _names(items: list[str]) -> str:
    shown = ", ".join(repr(i) for i in items[:_SHOW])
    return shown + (f" and {len(items) - _SHOW} more" if len(items) > _SHOW else "")
