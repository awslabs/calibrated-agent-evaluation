# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The local SQLite/CSV backend.

The no-config default backend: pure-filesystem, no network on any path. SQLite gives
indexed retrieval for the locked ``query(frame, cell, ownership)`` surface; nested objects
ride as JSON ``TEXT`` columns and the full artifact is stored as a JSON blob so ``get`` /
``query`` reconstruct it field-for-field. ``put_*`` runs :func:`schema.validate`
first, so an invalid artifact (e.g. empty ``trace_id``) raises ``SchemaError`` and never
reaches a row.

This module is a backend adapter (I/O), so it may use stdlib infrastructure freely; but it
still speaks only normalized types at its surface (no vendor type; ``sqlite3`` is stdlib).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.schema.errors import BackendError, TraceNotFoundError
from calibrated_agent_evaluation.schema.serde import (
    dedup_labels,
    label_from_dict,
    label_identity,
    to_dict,
    trace_from_dict,
)
from calibrated_agent_evaluation.schema.types import Frame, Label, Ownership, Trace
from calibrated_agent_evaluation.schema.validate import validate

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id   TEXT NOT NULL PRIMARY KEY,
    frame      TEXT NOT NULL,
    ownership  TEXT NOT NULL,
    cell_feature TEXT,
    doc        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_frame_ownership ON traces (frame, ownership);

CREATE TABLE IF NOT EXISTS spans (
    span_id   TEXT NOT NULL,
    trace_id  TEXT NOT NULL,
    seq       INTEGER NOT NULL,
    doc       TEXT NOT NULL,
    PRIMARY KEY (trace_id, span_id)
);
CREATE INDEX IF NOT EXISTS idx_spans_trace_seq ON spans (trace_id, seq);

CREATE TABLE IF NOT EXISTS labels (
    rowid_pk  INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id  TEXT NOT NULL,
    span_id   TEXT,
    doc       TEXT NOT NULL,
    rater_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_labels_trace_span ON labels (trace_id, span_id);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring a store written by an older release up to the current column set, in place.

    ``CREATE TABLE IF NOT EXISTS`` leaves an existing table alone, so a store created before
    ``labels.rater_id`` existed would otherwise keep the old shape and fail its first insert. The
    added column is NULL for every old row, which is the truth: those labels carry no rater in
    their ``doc`` either, and ``label_from_dict`` reads them back as ``rater_id=None``.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(labels)")}
    if "rater_id" not in columns:
        conn.execute("ALTER TABLE labels ADD COLUMN rater_id TEXT")
    _rekey_spans(conn)
    conn.commit()


def _rekey_spans(conn: sqlite3.Connection) -> None:
    """Key ``spans`` on ``(trace_id, span_id)`` in a store that keyed it on ``span_id`` alone.

    Under the old key, a span id two traces share made the second ``INSERT OR REPLACE`` delete the
    first trace's row. The table is rebuilt from the trace docs, which are the source of truth and
    still hold every span.
    """
    pk = [row[1] for row in conn.execute("PRAGMA table_info(spans)") if row[5]]
    if pk != ["span_id"]:
        return
    with conn:
        conn.execute("DROP TABLE spans")
        conn.executescript(_SCHEMA)
        for (doc,) in conn.execute("SELECT doc FROM traces").fetchall():
            trace = _trace_from_doc(doc)
            conn.executemany(
                "INSERT INTO spans (span_id, trace_id, seq, doc) VALUES (?, ?, ?, ?)",
                [(s.span_id, s.trace_id, s.seq, json.dumps(to_dict(s))) for s in trace.spans],
            )


def _trace_from_doc(doc: str) -> Trace:
    """A stored trace doc decoded, a corrupt one raised as the store's typed error."""
    try:
        return trace_from_dict(json.loads(doc))
    except json.JSONDecodeError as exc:
        raise BackendError(
            f"the local store holds a trace doc that is not valid JSON: {exc}"
        ) from exc


class LocalBackend:
    """A ``TraceStoreBackend`` backed by stdlib ``sqlite3`` and the local filesystem.

    Spans are stored in their own table (for the indexed tree reconstruction) but are also
    embedded in the trace ``doc`` blob, which is the source of truth on read. Use it as a context
    manager, or call :meth:`close`, to release the database connection.

    Args:
        root: The directory holding the store, created if missing. ``":memory:"`` keeps the
            store in memory, for tests.
    """

    def __init__(self, root: str | Path = ".calibrated-agent-evaluation") -> None:
        if root == ":memory:":
            self._db_path = ":memory:"
        else:
            root_path = Path(root)
            root_path.mkdir(parents=True, exist_ok=True)
            self._db_path = str(root_path / "store.sqlite3")
        # One connection, used from the thread that opened it (sqlite3's default check).
        self._conn = sqlite3.connect(self._db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        _migrate(self._conn)

    # --- TraceStoreBackend surface --------------------------------------------------------

    def put_trace(self, trace: Trace) -> None:
        """Store ``trace`` and its spans, replacing any trace with the same ``trace_id``.

        Args:
            trace: The normalized trace to store.

        Raises:
            SchemaError: The trace fails validation; nothing is written.
        """
        validate(trace)  # refuse invalid artifacts before any write
        doc = json.dumps(to_dict(trace))
        cell_feature = trace.taxonomy_cell.get("feature")
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO traces "
                "(trace_id, frame, ownership, cell_feature, doc) VALUES (?, ?, ?, ?, ?)",
                (trace.trace_id, trace.frame, trace.ownership, cell_feature, doc),
            )
            self._conn.execute("DELETE FROM spans WHERE trace_id = ?", (trace.trace_id,))
            self._conn.executemany(
                "INSERT OR REPLACE INTO spans (span_id, trace_id, seq, doc) VALUES (?, ?, ?, ?)",
                [
                    (span.span_id, span.trace_id, span.seq, json.dumps(to_dict(span)))
                    for span in trace.spans
                ],
            )

    def get_trace(self, trace_id: str) -> Trace:
        """Return the stored trace for ``trace_id``, field for field as it was written.

        Args:
            trace_id: The id of the trace to read.

        Returns:
            The stored trace.

        Raises:
            TraceNotFoundError: No trace with that id is stored.
        """
        row = self._conn.execute(
            "SELECT doc FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if row is None:
            raise TraceNotFoundError(f"no trace with trace_id {trace_id!r} in local store")
        return _trace_from_doc(row[0])

    def query(
        self,
        frame: Frame | None = None,
        cell: dict[str, str] | None = None,
        ownership: Ownership | None = None,
    ) -> list[Trace]:
        """Return the stored traces that match every given filter, ordered by ``trace_id``.

        Args:
            frame: Keep only traces in this sampling frame.
            cell: Keep only traces whose taxonomy cell has all of these key/value pairs.
            ownership: Keep only owned or only probed traces.

        Returns:
            The matching traces; an empty list when none match.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if frame is not None:
            clauses.append("frame = ?")
            params.append(frame)
        if ownership is not None:
            clauses.append("ownership = ?")
            params.append(ownership)
        if cell is not None and "feature" in cell:
            clauses.append("cell_feature = ?")
            params.append(cell["feature"])
        sql = "SELECT doc FROM traces"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY trace_id"
        traces = [_trace_from_doc(r[0]) for r in self._conn.execute(sql, params)]
        # Remaining cell keys (beyond the indexed `feature`) filter in-memory.
        if cell:
            extra = {k: v for k, v in cell.items() if k != "feature"}
            if extra:
                traces = [
                    t for t in traces if all(t.taxonomy_cell.get(k) == v for k, v in extra.items())
                ]
        return traces

    def put_score(self, label: Label) -> None:
        """Append a label. Identity-duplicate rows are collapsed by :func:`dedup_labels` on read.

        Deliberately an append rather than an upsert. `labels.rowid_pk` is an autoincrement primary
        key and nothing in the existing schema carries the identity, so an upsert needs either a
        migration of every store on disk or an identity column that the Langfuse backend could not
        mirror anyway (its `create_score` API has no update). Deduping on read gives both backends
        one behaviour and keeps every already-written store readable.

        ``rater_id`` is projected into its own column, like ``trace_id``/``span_id``, but stays out
        of the ``(trace_id, span_id)`` index: every read filters on the target, none on the rater,
        and the index is a lookup path, not a uniqueness key. Two raters' labels on one target are
        kept apart by the full-content identity :func:`dedup_labels` applies, which includes
        ``rater_id``, so widening the index would change no result.
        """
        validate(label)  # refuse invalid labels before any write
        with self._conn:
            self._conn.execute(
                "INSERT INTO labels (trace_id, span_id, doc, rater_id) VALUES (?, ?, ?, ?)",
                (label.trace_id, label.span_id, json.dumps(to_dict(label)), label.rater_id),
            )

    def get_scores(self, trace_id: str, span_id: str | None = None) -> list[Label]:
        """Return the labels stored for a trace, oldest first, with duplicates collapsed.

        Args:
            trace_id: The trace whose labels to read.
            span_id: Narrow the result to labels on this span.

        Returns:
            The labels; an empty list when the trace has none.
        """
        if span_id is None:
            rows = self._conn.execute(
                "SELECT doc FROM labels WHERE trace_id = ? ORDER BY rowid_pk", (trace_id,)
            )
        else:
            rows = self._conn.execute(
                "SELECT doc FROM labels WHERE trace_id = ? AND span_id = ? ORDER BY rowid_pk",
                (trace_id, span_id),
            )
        # `ORDER BY rowid_pk` makes the last byte-identical row the last written, which is what
        # `dedup_labels` keeps. `put_score` appends (see its docstring), so re-putting one label
        # leaves several identical rows, and an undeduped read doubled every rate over them.
        return dedup_labels(label_from_dict(json.loads(r[0])) for r in rows)

    def retract_score(self, label: Label) -> None:
        """Delete every stored row carrying exactly ``label`` (see ``TraceStoreBackend``).

        A real delete, because SQLite's is synchronous and exact. Every byte-identical row goes:
        an idempotent re-put leaves several rows that read back as one label, and leaving any of
        them would make the retraction a no-op on read. Retracting a label that is not stored
        deletes nothing, matching the Langfuse backend, which cannot check synchronously.
        """
        identity = label_identity(label)
        rows = self._conn.execute(
            "SELECT rowid_pk, doc FROM labels WHERE trace_id = ? AND span_id IS ?",
            (label.trace_id, label.span_id),
        ).fetchall()
        doomed = [
            (pk,)
            for pk, doc in rows
            if label_identity(label_from_dict(json.loads(doc))) == identity
        ]
        with self._conn:
            self._conn.executemany("DELETE FROM labels WHERE rowid_pk = ?", doomed)

    # --- round-trip support (used by dump/load and acceptance wipe→load) ------------------

    def all_traces(self) -> list[Trace]:
        """Return every stored trace (dump support)."""
        rows = self._conn.execute("SELECT doc FROM traces ORDER BY trace_id")
        return [_trace_from_doc(r[0]) for r in rows]

    def all_scores(self) -> list[Label]:
        """Return every stored label (dump support)."""
        rows = self._conn.execute("SELECT doc FROM labels ORDER BY rowid_pk")
        # `ORDER BY rowid_pk` makes the last byte-identical row the last written, which is what
        # `dedup_labels` keeps. `put_score` appends (see its docstring), so re-putting one label
        # leaves several identical rows, and an undeduped read doubled every rate over them.
        return dedup_labels(label_from_dict(json.loads(r[0])) for r in rows)

    def wipe(self) -> None:
        """Delete every artifact (the dump→wipe→load no-lock-in proof)."""
        with self._conn:
            self._conn.execute("DELETE FROM traces")
            self._conn.execute("DELETE FROM spans")
            self._conn.execute("DELETE FROM labels")

    def close(self) -> None:
        """Close the database connection. The backend can't be used afterwards."""
        self._conn.close()

    def __enter__(self) -> LocalBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
