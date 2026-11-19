# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Langfuse-pull tests: every trace is pulled from a source into the destination, and an
unreachable source raises before anything is persisted. Most tests use a source
stand-in that returns already-normalized traces; the pagination tests at the end read through a
real ``LangfuseBackend`` over a stub client that pages like the v3 endpoint."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from calibrated_agent_evaluation.backends.langfuse import (
    _QUERY_PAGE_SIZE,
    LangfuseBackend,
    LangfuseLike,
)
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.ingest import ingest_langfuse
from calibrated_agent_evaluation.normalize import OTelSpans, StrandsSession, normalize
from calibrated_agent_evaluation.schema.errors import BackendError
from calibrated_agent_evaluation.schema.types import Frame, Label, Ownership, Trace
from test_backends_langfuse import StubLangfuseClient

_RAW = Path(__file__).resolve().parent.parent / "fixtures" / "raw"


class _SourceBackend:
    """A Langfuse-source stand-in: ``query`` returns pre-normalized traces (as the backend does)."""

    def __init__(self, traces: list[Trace]) -> None:
        self._traces = list(traces)

    def query(
        self,
        frame: Frame | None = None,
        cell: dict[str, str] | None = None,
        ownership: Ownership | None = None,
    ) -> list[Trace]:
        return list(self._traces)

    def get_trace(self, trace_id: str) -> Trace:  # pragma: no cover - unused by ingest_langfuse
        for trace in self._traces:
            if trace.trace_id == trace_id:
                return trace
        raise BackendError(trace_id)

    def put_trace(self, trace: Trace) -> None:  # pragma: no cover
        self._traces.append(trace)

    def put_score(self, label: Label) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_scores(self, trace_id: str, span_id: str | None = None) -> list[Label]:
        return []

    def retract_score(self, label: Label) -> None:  # pragma: no cover - unused here
        raise NotImplementedError


class _UnreachableSource(_SourceBackend):
    def query(
        self,
        frame: Frame | None = None,
        cell: dict[str, str] | None = None,
        ownership: Ownership | None = None,
    ) -> list[Trace]:
        raise BackendError("langfuse unreachable")


def _fixture_traces() -> list[Trace]:
    strands = StrandsSession(path=_RAW / "strands" / "CLM-4471.otlp.json", trace_id="CLM-4471")
    otel = OTelSpans(path=_RAW / "otel" / "ORD-1002.events.otlp.json", trace_id="ORD-1002")
    return [normalize(strands), normalize(otel)]


def test_a_langfuse_ingest_continues_to_name_each_trace_as_a_langfuse_source(
    tmp_path: Path,
) -> None:
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = ingest_langfuse(_SourceBackend(_fixture_traces()), backend=dest)
    assert report.results
    assert all(r.source.startswith("langfuse:") for r in report.results)


def test_pulls_all_source_traces_into_destination(tmp_path: Path) -> None:
    """Every trace from the source lands in the destination backend."""
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = ingest_langfuse(_SourceBackend(_fixture_traces()), backend=dest)

    assert report.exit_code == 0
    assert report.ingested == 2
    assert {r.trace_id for r in report.results} == {"CLM-4471", "ORD-1002"}
    assert dest.get_trace("CLM-4471").trace_id == "CLM-4471"
    assert dest.get_trace("ORD-1002").trace_id == "ORD-1002"


def test_unreachable_source_raises_before_persisting_anything(tmp_path: Path) -> None:
    """Negative: an unreachable source raises BackendError; the destination stays empty."""
    dest = LocalBackend(root=str(tmp_path / "store"))
    with pytest.raises(BackendError):
        ingest_langfuse(_UnreachableSource([]), backend=dest)
    assert dest.query() == []


def test_dry_run_pulls_but_writes_nothing(tmp_path: Path) -> None:
    """For the pull path: a dry run reports the traces but persists none."""
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = ingest_langfuse(_SourceBackend(_fixture_traces()), backend=dest, dry_run=True)
    assert report.exit_code == 0
    assert report.ingested == 2
    assert dest.query() == []


def test_persists_every_source_trace_without_truncation(tmp_path: Path) -> None:
    """Edge: ingest_langfuse persists EVERY trace the source returns (here 5) — no dropped
    tail. The multi-page walk is covered by the pagination tests below."""
    src_file = _RAW / "strands" / "CLM-4471.otlp.json"
    many = [normalize(StrandsSession(path=src_file, trace_id=f"MANY-{i}")) for i in range(5)]
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = ingest_langfuse(_SourceBackend(many), backend=dest)
    assert report.ingested == 5
    assert {t.trace_id for t in dest.query()} == {f"MANY-{i}" for i in range(5)}


# --- the pull walks a paged source to exhaustion -----------


def _paged_source(client: object) -> LangfuseBackend:
    return LangfuseBackend(cast("LangfuseLike", client))


def _seed(source: LangfuseBackend, n: int) -> list[Trace]:
    src_file = _RAW / "strands" / "CLM-4471.otlp.json"
    seeded = [normalize(StrandsSession(path=src_file, trace_id=f"PAGE-{i}")) for i in range(n)]
    for trace in seeded:
        source.put_trace(trace)
    return seeded


def test_pull_reads_every_page_of_the_source_listing(tmp_path: Path) -> None:
    """A source listing more than two pages lands whole, every trace equal to what was put.

    The source is a real ``LangfuseBackend`` over the stub client whose ``trace.list`` pages the
    way the v3 endpoint does (``page``/``limit``, ``meta.total_pages``).
    """
    source = _paged_source(StubLangfuseClient())
    seeded = _seed(source, _QUERY_PAGE_SIZE * 2 + 3)
    dest = LocalBackend(root=str(tmp_path / "store"))

    report = ingest_langfuse(source, backend=dest)

    assert report.exit_code == 0
    assert report.ingested == len(seeded)
    assert sorted(str(r.trace_id) for r in report.results) == sorted(t.trace_id for t in seeded)
    for trace in seeded:
        assert dest.get_trace(trace.trace_id) == trace


def test_pull_stops_on_an_empty_page_without_dropping_or_repeating(tmp_path: Path) -> None:
    """Edge: a listing with no page count, holding an exact multiple of the page size.

    Without ``meta.total_pages`` the walk only knows it is done when a page comes back short, so it
    must read the empty page after the last full one: every trace once, none twice, none dropped.
    """
    client = StubLangfuseClient()
    paged_list = client.api.trace.list
    pages_read: list[int] = []

    def list_without_meta(*, tags: list[str] | None = None, page: int = 1, limit: int = 50) -> Any:
        pages_read.append(page)
        return SimpleNamespace(data=paged_list(tags=tags, page=page, limit=limit).data)

    client.api.trace.list = list_without_meta  # type: ignore[method-assign]
    source = _paged_source(client)
    seeded = _seed(source, _QUERY_PAGE_SIZE * 2)
    dest = LocalBackend(root=str(tmp_path / "store"))

    report = ingest_langfuse(source, backend=dest)

    assert pages_read == [1, 2, 3]
    assert report.ingested == len(seeded)
    ids = [r.trace_id for r in report.results]
    assert len(ids) == len(set(ids)) == len(seeded)
