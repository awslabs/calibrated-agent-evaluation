# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingest refuses or reports sources that would otherwise lose or fuse data without a word.

Each case is a real source shape run through the real ingest path: several traces in one file, a
tool span the adapter cannot link, a file whose top level is not an object, an Inspect log whose
tool events went missing, a store that fails part-way through a batch, and a Langfuse pull that
holds two copies of one run.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.backends.base import SourceTrace
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.ingest import IngestAbortedError, ingest_langfuse, run_ingest
from calibrated_agent_evaluation.normalize import OTelSpans, normalize
from calibrated_agent_evaluation.normalize.otlp_json import attributes
from calibrated_agent_evaluation.schema.errors import BackendError
from calibrated_agent_evaluation.schema.types import Trace

_RAW = Path(__file__).resolve().parent.parent / "fixtures" / "raw"
_OTEL = _RAW / "otel"


def _load(name: str) -> dict[str, Any]:
    return json.loads((_OTEL / name).read_text(encoding="utf-8"))


def _spans(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return doc["resourceSpans"][0]["scopeSpans"][0]["spans"]


def test_a_file_holding_two_traces_is_refused_not_fused(tmp_path: Path) -> None:
    first = _load("ORD-1001.events.otlp.json")
    second = _load("ORD-1002.events.otlp.json")
    _spans(first).extend(_spans(second))
    path = tmp_path / "mixed.otlp.json"
    path.write_text(json.dumps(first), encoding="utf-8")

    with LocalBackend(tmp_path / "store") as backend:
        report = run_ingest([path], backend=backend)
        stored = backend.query()

    (result,) = report.results
    assert result.status == "refused"
    assert "2 different traces" in (result.reason or "")
    assert report.exit_code == 1
    assert stored == []


def test_a_tool_span_the_adapter_cannot_link_is_refused_unless_partial_tools_are_allowed(
    tmp_path: Path,
) -> None:
    doc = _load("ORD-1001.events.otlp.json")
    spans = _spans(doc)
    tool = next(s for s in spans if attributes(s).get("gen_ai.operation.name") == "execute_tool")
    orphan = copy.deepcopy(tool)
    orphan["spanId"] = "AAAAAAAAAAA="
    orphan["parentSpanId"] = "BBBBBBBBBBB="
    spans.append(orphan)
    path = tmp_path / "ORD-1001.otlp.json"
    path.write_text(json.dumps(doc), encoding="utf-8")

    with LocalBackend(tmp_path / "store") as backend:
        refused = run_ingest([path], backend=backend, dry_run=True)
        allowed = run_ingest([path], backend=backend, dry_run=True, allow_partial_tools=True)

    (result,) = refused.results
    assert result.status == "refused"
    assert result.unrecovered_tool_spans == 1
    (result,) = allowed.results
    assert result.status == "ingested"
    assert result.tool_call_count == 3
    assert result.unrecovered_tool_spans == 1


def test_a_sub_agent_run_as_a_tool_is_not_counted_as_a_lost_tool(tmp_path: Path) -> None:
    path = _RAW / "strands" / "ORD-3310-as-tool.otlp.json"
    with LocalBackend(tmp_path / "store") as backend:
        report = run_ingest([path], backend=backend, dry_run=True)

    (result,) = report.results
    assert result.status == "ingested"
    assert result.unrecovered_tool_spans == 0


def test_a_file_whose_top_level_is_not_an_object_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "wrapped.otlp.json"
    path.write_text(json.dumps([_load("ORD-1001.events.otlp.json")]), encoding="utf-8")

    with LocalBackend(tmp_path / "store") as backend:
        report = run_ingest([path], backend=backend)
        stored = backend.query()

    (result,) = report.results
    assert result.status == "skipped"
    assert "not an OTLP-JSON object" in (result.reason or "")
    assert stored == []


def test_an_inspect_log_whose_tool_events_were_lost_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("inspect_ai")
    from calibrated_agent_evaluation.normalize import inspect_adapter

    monkeypatch.setattr(inspect_adapter, "_tool_calls_and_answer", lambda _events: ([], None))
    with LocalBackend(tmp_path / "store") as backend:
        report = run_ingest([_RAW / "inspect" / "ORD-1001.eval"], backend=backend)
        stored = backend.query()

    (result,) = report.results
    assert result.status == "refused"
    assert "recovered 0 tool calls" in (result.reason or "")
    assert stored == []


class _FailsOnSecondWrite(LocalBackend):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.writes = 0

    def put_trace(self, trace: Trace) -> None:
        self.writes += 1
        if self.writes == 2:
            raise BackendError("disk full")
        super().put_trace(trace)


def test_a_store_failure_mid_batch_names_what_was_already_written(tmp_path: Path) -> None:
    paths = [_OTEL / "ORD-1001.events.otlp.json", _OTEL / "ORD-1002.events.otlp.json"]
    with _FailsOnSecondWrite(tmp_path / "store") as backend:
        with pytest.raises(IngestAbortedError) as caught:
            run_ingest(paths, backend=backend)
        stored = [t.trace_id for t in backend.query()]

    assert caught.value.written == ("ORD-1001",)
    assert stored == ["ORD-1001"]
    assert "ORD-1002" in str(caught.value)
    assert isinstance(caught.value.__cause__, BackendError)


def test_a_store_failure_on_the_first_write_is_the_store_error(tmp_path: Path) -> None:
    class _Down(LocalBackend):
        def put_trace(self, trace: Trace) -> None:
            raise BackendError("unreachable")

    with _Down(tmp_path / "store") as backend, pytest.raises(BackendError) as caught:
        run_ingest([_OTEL / "ORD-1001.events.otlp.json"], backend=backend)
    assert not isinstance(caught.value, IngestAbortedError)


class _Pull:
    """A Langfuse-shaped source: a foreign export listed before the Calibrated Agent
    Evaluation-written copy."""

    def __init__(self, items: list[SourceTrace]) -> None:
        self._items = items

    def pull(self) -> Iterator[SourceTrace]:
        return iter(self._items)

    def put_trace(self, trace: Trace) -> None:  # pragma: no cover - a source is only read
        raise NotImplementedError

    def get_trace(self, trace_id: str) -> Trace:  # pragma: no cover
        raise NotImplementedError

    def query(self, **_filters: Any) -> list[Trace]:  # pragma: no cover
        return []


def test_a_langfuse_pull_keeps_the_faithful_copy_of_a_repeated_run(tmp_path: Path) -> None:
    raw = _load("ORD-1001.events.otlp.json")
    faithful = normalize(OTelSpans(path=_OTEL / "ORD-1001.events.otlp.json"))
    pull = _Pull(
        [
            SourceTrace(source_id=faithful.trace_id, otlp=raw),
            SourceTrace(source_id=faithful.trace_id, trace=faithful),
        ]
    )
    with LocalBackend(tmp_path / "store") as backend:
        report = ingest_langfuse(pull, backend=backend)  # type: ignore[arg-type]
        stored = backend.get_trace(faithful.trace_id)

    statuses = sorted((r.adapter == "langfuse", r.status) for r in report.results)
    assert statuses == [(False, "refused"), (True, "ingested")]
    assert stored == faithful
