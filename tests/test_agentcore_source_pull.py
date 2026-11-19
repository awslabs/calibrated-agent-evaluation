# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AgentCore trace source's pull orchestration (over a fake reader, no AWS)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from calibrated_agent_evaluation.agentcore.errors import AgentCoreSettlingLag
from calibrated_agent_evaluation.agentcore.settle import SettlePolicy
from calibrated_agent_evaluation.agentcore.sources import (
    AgentCoreSourceConfig,
    AgentCoreTraceSource,
    RawTrace,
    TimeWindow,
)
from calibrated_agent_evaluation.backends.base import TraceStoreBackend
from calibrated_agent_evaluation.schema.errors import BackendError

_PAST = TimeWindow(start="2026-10-01T00:00:00+00:00", end="2026-10-01T01:00:00+00:00")
_FUTURE = TimeWindow(start="2099-01-01T00:00:00+00:00", end="2099-01-01T01:00:00+00:00")
# A settle policy that never blocks for a past window (lag already cleared).
_SETTLED = SettlePolicy(lag_seconds=0)


def _agent_span(span_id: str, trace_id: str = "t-1") -> dict[str, Any]:
    return {
        "spanId": span_id,
        "traceId": trace_id,
        "attributes": {"gen_ai.operation.name": "invoke_agent"},
    }


def _chat_span(span_id: str, parent: str, trace_id: str = "t-1") -> dict[str, Any]:
    return {
        "spanId": span_id,
        "traceId": trace_id,
        "parentSpanId": parent,
        "attributes": {"gen_ai.operation.name": "chat"},
    }


def _log(span_id: str, event: str, body: dict[str, Any]) -> dict[str, Any]:
    return {"spanId": span_id, "eventName": event, "body": body}


def _raw(
    trace_id: str = "t-1", *, session_ids: tuple[str, ...] = ("s-1",), runtime: str | None = None
) -> RawTrace:
    # One invoke_agent with a chat child carrying the content the converter re-homes onto it.
    return RawTrace(
        trace_id=trace_id,
        span_records=[
            _agent_span("sp-1", trace_id),
            _chat_span("sp-c", "sp-1", trace_id),
        ],
        log_records=[
            _log("sp-c", "gen_ai.user.message", {"content": [{"text": "hello"}]}),
            _log(
                "sp-c",
                "gen_ai.choice",
                {"message": {"content": [{"text": "hi"}]}},
            ),
        ],
        session_ids=session_ids,
        runtime_session_id=runtime,
    )


class _FakeReader:
    def __init__(self, traces: list[RawTrace], scores: list[dict] | None = None) -> None:
        self._traces = traces
        self._scores = scores or []

    def read_traces(self) -> Iterator[RawTrace]:
        yield from self._traces

    def read_scores(self) -> Iterator[dict]:
        yield from self._scores


class _UnreachableReader:
    def read_traces(self) -> Iterator[RawTrace]:
        raise BackendError("AgentCore account unreachable")
        yield  # pragma: no cover - makes this a generator

    def read_scores(self) -> Iterator[dict]:
        return iter(())


def _cfg(**over: object) -> AgentCoreSourceConfig:
    base: dict[str, Any] = {"runtime": "nl_sql_agent", "window": _PAST}
    base.update(over)
    return AgentCoreSourceConfig(**base)


def _source(
    reader: Any, cfg: AgentCoreSourceConfig | None = None, settle: SettlePolicy = _SETTLED
) -> AgentCoreTraceSource:
    return AgentCoreTraceSource(reader, cfg or _cfg(), settle=settle)


def test_pull_yields_one_otlp_source_trace_per_trace() -> None:
    src = _source(_FakeReader([_raw("t-1"), _raw("t-2")]))
    pulled = list(src.pull())
    assert [p.source_id for p in pulled] == ["t-1", "t-2"]
    assert all(p.otlp is not None and p.error is None for p in pulled)


def test_an_unreachable_account_raises_before_any_trace_is_yielded() -> None:
    src = _source(_UnreachableReader())
    with pytest.raises(BackendError):
        list(src.pull())


def test_the_source_is_not_a_trace_store() -> None:
    src = _source(_FakeReader([]))
    assert not isinstance(src, TraceStoreBackend)
    assert not hasattr(src, "put_trace")


def test_a_mixed_session_trace_is_refused_naming_both_ids() -> None:
    src = _source(_FakeReader([_raw("t-1", session_ids=("s-1", "s-2"))]))
    (pulled,) = list(src.pull())
    assert pulled.otlp is None
    assert pulled.error is not None
    assert "s-1" in pulled.error
    assert "s-2" in pulled.error


def test_rekey_collapses_a_mixed_session_trace_and_records_it() -> None:
    src = _source(
        _FakeReader([_raw("t-1", session_ids=("s-1", "s-2"), runtime="rt-9")]),
        _cfg(rekey_sessions=True),
    )
    (pulled,) = list(src.pull())
    assert pulled.source_id == "rt-9"
    assert pulled.otlp is not None
    assert src.rekeyed == [("t-1", "rt-9")]


def test_two_mixed_session_traces_rekeying_to_one_id_refuse_rather_than_overwrite() -> None:
    # Both collapse to runtime session "rt-9"; storing both under one id would silently drop one,
    # so the second is refused (with its own error) rather than overwriting the first.
    src = _source(
        _FakeReader(
            [
                _raw("t-1", session_ids=("s-1", "s-2"), runtime="rt-9"),
                _raw("t-2", session_ids=("s-3", "s-4"), runtime="rt-9"),
            ]
        ),
        _cfg(rekey_sessions=True),
    )
    first, second = list(src.pull())
    assert first.source_id == "rt-9"
    assert first.otlp is not None
    assert second.otlp is None
    assert second.error is not None
    assert "overwrite" in second.error
    assert src.rekeyed == [("t-1", "rt-9")]


def test_a_trace_missing_its_content_is_refused_not_emptied() -> None:
    no_content = RawTrace(
        trace_id="t-1",
        span_records=[_agent_span("sp-1")],  # a content-bearing span...
        log_records=[],  # ...with no content records to re-home
        session_ids=("s-1",),
    )
    (pulled,) = list(_source(_FakeReader([no_content])).pull())
    assert pulled.otlp is None
    assert pulled.error is not None


def test_a_window_too_recent_to_settle_is_refused() -> None:
    src = _source(
        _FakeReader([_raw()]),
        _cfg(window=_FUTURE),
        SettlePolicy(lag_seconds=300, max_wait_seconds=0),
    )
    with pytest.raises(AgentCoreSettlingLag):
        list(src.pull())
