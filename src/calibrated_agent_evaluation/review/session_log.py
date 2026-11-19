# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append-only JSONL record of a labeling session.

``ReviewSession`` times every label, but the timings lived only in memory: a crashed session lost
all of them and a finished one left no per-item record, and the at-scale gold-set sessions are the
only place the labeling-throughput measurement can come from. One line is appended per event, at the
moment it happens, so a crash keeps everything up to the last submission.

Process telemetry stays out of ``Label``, which is the published methodology-plane record.

Each line is a JSON object with ``event`` (``"label"`` or ``"undo"``), ``trace_id``,
``rater_id`` and ``recorded_at`` (UTC, ISO-8601). A ``label`` line adds ``outcome``,
``seconds_on_trace`` and ``span_labels`` (the span ids of defect labels written with it). An
``undo`` line names the trace whose verdict was retracted; a reader keeps a trace's last ``label``
only when no later ``undo`` for it follows.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.review.errors import ReviewError
from calibrated_agent_evaluation.review.session import LabeledEntry


class SessionLog:
    """Appends session events to a JSONL file. Opens the file per write; holds no handle."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        try:  # fail at startup, not at the first verdict, if the log cannot be written
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8"):
                pass
        except OSError as exc:
            raise ReviewError(f"cannot write the session log {self.path}: {exc}") from exc

    def _append(self, record: dict[str, Any]) -> None:
        record = {**record, "recorded_at": datetime.now(tz=UTC).isoformat()}
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError as exc:
            raise ReviewError(f"could not append to the session log {self.path}: {exc}") from exc

    def label(self, entry: LabeledEntry, *, rater_id: str) -> None:
        """Record a verdict that was just stored, with the time spent on its trace."""
        self._append(
            {
                "event": "label",
                "trace_id": entry.trace_id,
                "rater_id": rater_id,
                "outcome": entry.label.outcome,
                "seconds_on_trace": entry.seconds_on_trace,
                "span_labels": [lbl.span_id for lbl in entry.span_labels],
            }
        )

    def undo(self, entry: LabeledEntry, *, rater_id: str) -> None:
        """Record that the verdict for ``entry.trace_id`` was retracted."""
        self._append({"event": "undo", "trace_id": entry.trace_id, "rater_id": rater_id})
