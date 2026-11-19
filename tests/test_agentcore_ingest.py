# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""ingest_agentcore: trace persistence, advisory-score dedup, and re-key notes."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from _agentcore_capture import captured_run
from calibrated_agent_evaluation.agentcore.ingest import ingest_agentcore
from calibrated_agent_evaluation.agentcore.settle import SettlePolicy
from calibrated_agent_evaluation.agentcore.sources import (
    AgentCoreSourceConfig,
    AgentCoreTraceSource,
    RawTrace,
    TimeWindow,
)
from calibrated_agent_evaluation.backends.base import SourceTrace
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.schema.types import Span, Trace


def _trace(trace_id: str) -> Trace:
    return Trace(
        trace_id=trace_id,
        frame="production-representative",
        ownership="owned",
        input={"q": "x"},
        output={"a": "y"},
        taxonomy_cell={},
        spans=[
            Span(
                span_id="sp-1",
                trace_id=trace_id,
                agent="agent",
                role="specialist",
                parent=None,
                seq=0,
                status="completed",
                input={"q": "x"},
                output={"a": "y"},
            )
        ],
    )


def _score(trace_id: str = "t-1", value: float = 1.0, label: str = "Correct") -> dict[str, Any]:
    return {
        "evaluatorArn": "arn:aws:bedrock-agentcore:us-west-2:111122223333:evaluator/correctness",
        "evaluatorName": "correctness",
        "explanation": "matches expected",
        "context": {"spanContext": {"traceId": trace_id}},
        "value": value,
        "label": label,
    }


class _FakeSource:
    """Stands in for AgentCoreTraceSource: yields prebuilt normalized traces + result records."""

    def __init__(
        self,
        traces: list[Trace],
        scores: list[dict] | None = None,
        rekeyed: list[tuple[str, str]] | None = None,
    ) -> None:
        self._traces = traces
        self._scores = scores or []
        self.rekeyed = rekeyed or []

    def pull(self) -> Iterator[SourceTrace]:
        for trace in self._traces:
            yield SourceTrace(source_id=trace.trace_id, trace=trace)

    def read_scores(self) -> Iterator[dict]:
        yield from self._scores


def test_traces_are_persisted_to_the_backend(tmp_path: Any) -> None:
    backend = LocalBackend(root=str(tmp_path / "store"))
    report = ingest_agentcore(_FakeSource([_trace("t-1")]), backend=backend)
    assert report.ingested == 1
    assert backend.get_trace("t-1").trace_id == "t-1"


def test_an_agentcore_ingest_names_each_trace_as_an_agentcore_source(tmp_path: Any) -> None:
    backend = LocalBackend(root=str(tmp_path / "store"))
    report = ingest_agentcore(_FakeSource([_trace("t-1"), _trace("t-2")]), backend=backend)
    assert sorted(r.source for r in report.results) == ["agentcore:t-1", "agentcore:t-2"]


def test_an_agentcore_trace_that_cannot_be_read_is_reported_as_agentcore(tmp_path: Any) -> None:
    class _Unreadable(_FakeSource):
        def pull(self) -> Iterator[SourceTrace]:
            yield SourceTrace(source_id="t-9", error="the run's content records are missing")

    backend = LocalBackend(root=str(tmp_path / "store"))
    (result,) = ingest_agentcore(_Unreadable([]), backend=backend).results
    assert result.status == "skipped"
    assert (result.source, result.adapter) == ("agentcore:t-9", "agentcore")


def test_dry_run_writes_nothing(tmp_path: Any) -> None:
    backend = LocalBackend(root=str(tmp_path / "store"))
    report = ingest_agentcore(
        _FakeSource([_trace("t-1")], scores=[_score()]),
        backend=backend,
        dry_run=True,
        with_scores=True,
    )
    assert report.dry_run
    assert backend.query() == []
    assert backend.get_scores("t-1") == []


def test_with_scores_stores_advisory_labels(tmp_path: Any) -> None:
    backend = LocalBackend(root=str(tmp_path / "store"))
    ingest_agentcore(
        _FakeSource([_trace("t-1")], scores=[_score()]), backend=backend, with_scores=True
    )
    labels = backend.get_scores("t-1")
    assert len(labels) == 1
    assert labels[0].kind == "llm-judge"


def test_without_scores_no_label_is_stored(tmp_path: Any) -> None:
    backend = LocalBackend(root=str(tmp_path / "store"))
    ingest_agentcore(
        _FakeSource([_trace("t-1")], scores=[_score()]), backend=backend, with_scores=False
    )
    assert backend.get_scores("t-1") == []


def test_re_reading_scores_adds_no_duplicate(tmp_path: Any) -> None:
    backend = LocalBackend(root=str(tmp_path / "store"))
    args = {"backend": backend, "with_scores": True}
    ingest_agentcore(_FakeSource([_trace("t-1")], scores=[_score()]), **args)
    ingest_agentcore(_FakeSource([_trace("t-1")], scores=[_score()]), **args)
    assert len(backend.get_scores("t-1")) == 1


def test_a_score_for_a_trace_not_ingested_this_pull_is_dropped(tmp_path: Any) -> None:
    # A result addressing a trace that did not ingest in this pull (another runtime's evaluation
    # that shared the window, or a refused trace) is dropped rather than written against an id that
    # is not in the store.
    backend = LocalBackend(root=str(tmp_path / "store"))
    source = _FakeSource([_trace("t-1")], scores=[_score("other-trace")])
    ingest_agentcore(source, backend=backend, with_scores=True)
    assert backend.get_scores("other-trace") == []
    assert backend.get_scores("t-1") == []


def test_a_score_is_attached_to_the_rekeyed_id_not_the_original(tmp_path: Any) -> None:
    # The trace ingested under its runtime session id "rt-9"; a result addressed to the original
    # mixed-session id "t-1" must attach to "rt-9" (where the trace actually lives), not "t-1".
    backend = LocalBackend(root=str(tmp_path / "store"))
    source = _FakeSource([_trace("rt-9")], scores=[_score("t-1")], rekeyed=[("t-1", "rt-9")])
    ingest_agentcore(source, backend=backend, with_scores=True)
    assert len(backend.get_scores("rt-9")) == 1
    assert backend.get_scores("t-1") == []


def test_a_result_with_no_verdict_is_skipped_not_stored(tmp_path: Any) -> None:
    # A result for an ingested trace that carries neither a value nor a label has no verdict to
    # record; it is collected as a failure and skipped, never stored as an empty label, and one
    # such record does not abort the rest of the pass.
    backend = LocalBackend(root=str(tmp_path / "store"))
    no_verdict = {
        "evaluatorArn": "arn:aws:bedrock-agentcore:us-west-2:111122223333:evaluator/correctness",
        "context": {"spanContext": {"traceId": "t-1"}},
        "value": None,
        "label": "",
    }
    source = _FakeSource([_trace("t-1")], scores=[no_verdict, _score("t-1")])
    ingest_agentcore(source, backend=backend, with_scores=True)
    # The empty record was skipped; the valid one was still stored (no abort after the bad record).
    assert len(backend.get_scores("t-1")) == 1


def test_a_result_from_an_evaluator_with_no_pass_direction_is_skipped_not_stored(
    tmp_path: Any,
) -> None:
    # A Refusal score has no defined pass, so it is counted unmappable and skipped; the pass goes on
    # to store the valid result after it.
    backend = LocalBackend(root=str(tmp_path / "store"))
    refusal = {
        "evaluatorArn": "arn:aws:bedrock-agentcore:::evaluator/Builtin.Refusal",
        "evaluatorName": "Builtin.Refusal",
        "context": {"spanContext": {"traceId": "t-1"}},
        "value": 1.0,
        "label": "Yes",
    }
    source = _FakeSource([_trace("t-1")], scores=[refusal, _score("t-1")])
    ingest_agentcore(source, backend=backend, with_scores=True)
    (stored,) = backend.get_scores("t-1")
    assert "Builtin.Refusal" not in stored.critique


def test_a_rekeyed_trace_is_noted_on_its_result(tmp_path: Any) -> None:
    backend = LocalBackend(root=str(tmp_path / "store"))
    source = _FakeSource([_trace("rt-9")], rekeyed=[("t-1", "rt-9")])
    report = ingest_agentcore(source, backend=backend)
    noted = [r for r in report.results if r.trace_id == "rt-9"]
    assert len(noted) == 1
    assert noted[0].reason is not None
    assert "t-1" in noted[0].reason


# --- the session an ingested trace belongs to ------------------------------------------------

_CAPTURE_ID = captured_run()["trace_id"]
_PAST = TimeWindow(start="2026-10-01T00:00:00+00:00", end="2026-10-01T01:00:00+00:00")


_KEEP = object()


def _capture(*, session: object = _KEEP) -> RawTrace:
    """The committed AgentCore capture as a reader decodes it: session ids from the spans."""
    run = captured_run()
    spans, logs = run["span_records"], run["log_records"]
    if session is not _KEEP:
        for span in spans:
            span["attributes"].pop("session.id", None)
            if session is not None:
                span["attributes"]["session.id"] = str(session)
    # As readers._group_traces builds them: the distinct non-empty values, sorted, first is the
    # runtime's own session.
    sessions = sorted({s["attributes"].get("session.id") for s in spans} - {None, ""})
    return RawTrace(
        trace_id=_CAPTURE_ID,
        span_records=spans,
        log_records=logs,
        session_ids=tuple(sessions),
        runtime_session_id=sessions[0] if sessions else None,
    )


def _with_sessions(raw: RawTrace, sessions: tuple[str, ...], runtime: str | None) -> RawTrace:
    return dataclasses.replace(raw, session_ids=tuple(sorted(sessions)), runtime_session_id=runtime)


class _FileReader:
    """A reader that serves decoded traces, like the Omni and Logs Insights readers yield."""

    def __init__(self, traces: list[RawTrace]) -> None:
        self._traces = traces

    def read_traces(self) -> Iterator[RawTrace]:
        yield from self._traces

    def read_scores(self) -> Iterator[dict]:
        return iter(())


def _pull(traces: list[RawTrace], tmp_path: Path, **config: Any) -> tuple[Any, LocalBackend]:
    source = AgentCoreTraceSource(
        _FileReader(traces),
        AgentCoreSourceConfig(runtime="nl_sql_agent", window=_PAST, **config),
        settle=SettlePolicy(lag_seconds=0),
    )
    backend = LocalBackend(root=str(tmp_path / "store"))
    return ingest_agentcore(source, backend=backend), backend


def test_an_ingested_trace_keeps_the_session_its_spans_carry(tmp_path: Path) -> None:
    raw = _capture()
    [expected] = raw.session_ids

    report, backend = _pull([raw], tmp_path)

    assert report.ingested == 1
    assert backend.get_trace(_CAPTURE_ID).session_id == expected


def test_a_trace_with_no_session_on_its_spans_is_stored_with_none(tmp_path: Path) -> None:
    _, backend = _pull([_capture(session=None)], tmp_path)

    assert backend.get_trace(_CAPTURE_ID).session_id is None


def test_a_blank_span_session_is_stored_with_none_and_the_pull_continues(tmp_path: Path) -> None:
    blank = _with_sessions(_capture(), ("  ",), "  ")
    mixed_with_blank = _with_sessions(
        dataclasses.replace(_capture(), trace_id="t-real"), ("s-1", "  "), "  "
    )

    report, backend = _pull([blank, mixed_with_blank], tmp_path)

    assert report.ingested == 2
    assert backend.get_trace(_CAPTURE_ID).session_id is None
    # The blank value is no session, so the other trace is not mixed: it keeps its real one.
    assert backend.get_trace("t-real").session_id == "s-1"


def test_a_rekeyed_trace_carries_the_runtime_session_it_was_rekeyed_to(tmp_path: Path) -> None:
    raw = _with_sessions(_capture(), ("s-1", "s-2"), "s-1")

    report, backend = _pull([raw], tmp_path, rekey_sessions=True)

    assert report.ingested == 1
    assert backend.get_trace("s-1").session_id == "s-1"


def test_a_rekey_with_no_runtime_session_sets_no_session(tmp_path: Path) -> None:
    raw = _with_sessions(_capture(), ("s-1", "s-2"), None)

    _, backend = _pull([raw], tmp_path, rekey_sessions=True)

    # Re-keyed to its own trace id, which is not a session, so no session is stamped.
    assert backend.get_trace(_CAPTURE_ID).session_id is None


def test_a_mixed_session_trace_is_still_refused_without_rekeying(tmp_path: Path) -> None:
    raw = _with_sessions(_capture(), ("s-1", "s-2"), "s-1")

    report, backend = _pull([raw], tmp_path)

    assert report.ingested == 0
    [result] = report.results
    assert result.status == "skipped"
    assert "2 session ids" in (result.reason or "")
    assert backend.query() == []


def test_a_sampled_ingest_given_a_frame_is_refused_before_the_source_is_read(
    tmp_path: Any,
) -> None:
    from calibrated_agent_evaluation.ingest import ingest_langfuse
    from calibrated_agent_evaluation.ingest.errors import IngestError

    class _Unread:
        def pull(self) -> Any:
            raise AssertionError("the source was read")

    backend = LocalBackend(root=str(tmp_path / "store"))
    with pytest.raises(IngestError, match="takes no frame"):
        ingest_langfuse(
            _Unread(),  # type: ignore[arg-type]
            backend=backend,
            sample_size=5,
            sample_seed=0,
            frame="production-representative",
        )
    assert backend.query() == []
