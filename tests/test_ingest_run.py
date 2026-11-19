# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""run_ingest tests: ingesting a dump stores a normalized Trace equal to the oracle; the recovery
summary; a batch skips a bad file, keeps going and exits non-zero; dry-run writes nothing; and the
run is backend-agnostic across both worlds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.ingest import run_ingest
from calibrated_agent_evaluation.schema.errors import BackendError, TraceNotFoundError
from calibrated_agent_evaluation.schema.serde import to_dict
from calibrated_agent_evaluation.schema.types import Frame, Label, Ownership, Trace

_REPO = Path(__file__).resolve().parent.parent
_RAW = _REPO / "fixtures" / "raw"
_NORMALIZED = _REPO / "fixtures" / "normalized"


def _oracle_tool_names(trace_id: str) -> list[str]:
    data = json.loads((_NORMALIZED / f"{trace_id}.json").read_text(encoding="utf-8"))
    return sorted(tc["name"] for s in data["spans"] for tc in s["tool_calls"])


def _tool_names(trace: Trace) -> list[str]:
    return sorted(tc.name for s in trace.spans for tc in s.tool_calls)


class _FakeBackend:
    """A minimal in-memory ``TraceStoreBackend`` — the 'other world' for the parity check."""

    def __init__(self) -> None:
        self.stored: dict[str, Trace] = {}

    def put_trace(self, trace: Trace) -> None:
        self.stored[trace.trace_id] = trace

    def get_trace(self, trace_id: str) -> Trace:
        if trace_id not in self.stored:
            raise TraceNotFoundError(trace_id)
        return self.stored[trace_id]

    def query(
        self,
        frame: Frame | None = None,
        cell: dict[str, str] | None = None,
        ownership: Ownership | None = None,
    ) -> list[Trace]:
        return list(self.stored.values())

    def put_score(self, label: Label) -> None:  # pragma: no cover - unused by ingest
        raise NotImplementedError

    def get_scores(self, trace_id: str, span_id: str | None = None) -> list[Label]:
        return []

    def retract_score(self, label: Label) -> None:  # pragma: no cover - unused here
        raise NotImplementedError


def test_ingest_strands_dump_reproduces_oracle_in_store(tmp_path: Path) -> None:
    """Ingesting a Strands OTLP dump persists a Trace whose tool calls equal the
    frozen oracle, and reports an accurate per-source summary."""
    backend = LocalBackend(root=str(tmp_path / "store"))
    report = run_ingest([_RAW / "strands" / "CLM-4471.otlp.json"], backend=backend)

    assert report.exit_code == 0
    [res] = report.results
    assert res.status == "ingested"
    assert res.adapter == "strands"
    assert res.trace_id == "CLM-4471"
    assert res.ownership == "owned"
    got = backend.get_trace("CLM-4471")
    assert _tool_names(got) == _oracle_tool_names("CLM-4471")
    assert res.tool_call_count == len(_oracle_tool_names("CLM-4471"))
    assert res.span_count == len(got.spans)


def test_ingest_generic_otel_dump_reproduces_oracle(tmp_path: Path) -> None:
    """A generic gen_ai OTLP dump reproduces its oracle via the OTel adapter."""
    backend = LocalBackend(root=str(tmp_path / "store"))
    report = run_ingest([_RAW / "otel" / "ORD-1002.events.otlp.json"], backend=backend)

    assert report.exit_code == 0
    [res] = report.results
    assert res.status == "ingested"
    assert res.adapter == "gen_ai/latest"
    assert _tool_names(backend.get_trace("ORD-1002")) == _oracle_tool_names("ORD-1002")


def test_batch_skips_bad_file_keeps_going_and_exits_nonzero(tmp_path: Path) -> None:
    """Negative/batch: a corrupt file is skipped with a reason, the good file still ingests,
    and the run exits non-zero — a dropped source is never a silent exit-zero."""
    bad = tmp_path / "broken.otlp.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    backend = LocalBackend(root=str(tmp_path / "store"))

    report = run_ingest([_RAW / "strands" / "CLM-4471.otlp.json", bad], backend=backend)

    assert report.exit_code != 0
    assert report.ingested == 1
    assert report.skipped == 1
    skipped = next(r for r in report.results if r.status == "skipped")
    assert skipped.reason is not None
    backend.get_trace("CLM-4471")  # the good one still persisted (no BackendError)


def test_dry_run_reports_summary_but_writes_nothing(tmp_path: Path) -> None:
    """--dry-run yields a per-source summary and exit 0, but persists no Trace."""
    backend = LocalBackend(root=str(tmp_path / "store"))
    report = run_ingest([_RAW / "strands" / "CLM-4471.otlp.json"], backend=backend, dry_run=True)

    assert report.exit_code == 0
    assert report.results[0].tool_call_count > 0  # summary computed
    with pytest.raises(BackendError):
        backend.get_trace("CLM-4471")  # nothing written


def test_run_ingest_persists_identically_in_both_worlds(tmp_path: Path) -> None:
    """The same normalized Trace lands in a local backend and an independent backend —
    run_ingest depends only on the TraceStoreBackend protocol, not a world's internals."""
    src: list[Any] = [_RAW / "strands" / "CLM-4471.otlp.json"]
    local = LocalBackend(root=str(tmp_path / "store"))
    other = _FakeBackend()

    run_ingest(src, backend=local)
    run_ingest(src, backend=other)

    assert to_dict(local.get_trace("CLM-4471")) == to_dict(other.get_trace("CLM-4471"))
