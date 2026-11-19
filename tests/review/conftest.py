# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures + an in-memory fake backend for the review-surface tests.

``FakeBackend`` implements the ``TraceStoreBackend`` and ``LabelBackend`` protocols against
plain dicts — the review surface is written to those protocols, so the tests drive the
real code path through a stub that mirrors the real surface (no vendor dependency). Fixtures load
the shared corpus (`fixtures/normalized/*.json`) and its gold labels (`fixtures/labels/gold.json`)
so tests assert against the same oracle the acceptance checklist cites.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.schema import BackendError, Label, Trace
from calibrated_agent_evaluation.schema.serde import label_from_dict, trace_from_dict

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_CORPUS_IDS = ("ORD-1001", "ORD-1002", "CLM-4471")
GOLD_RATER = "rater-1"  # the rater every fixtures/labels/gold.json row is attributed to


class FakeBackend:
    """In-memory backend for both protocols; configurable to raise for error paths."""

    def __init__(
        self,
        traces: list[Trace] | None = None,
        *,
        query_error: Exception | None = None,
        submit_error: Exception | None = None,
    ) -> None:
        traces = traces or []
        self._traces: dict[str, Trace] = {t.trace_id: t for t in traces}
        self._order: list[str] = [t.trace_id for t in traces]
        self.labels: list[Label] = []
        self.query_error = query_error
        self.submit_error = submit_error

    # --- TraceStoreBackend ---
    def put_trace(self, trace: Trace) -> None:
        if trace.trace_id not in self._traces:
            self._order.append(trace.trace_id)
        self._traces[trace.trace_id] = trace

    def get_trace(self, trace_id: str) -> Trace:
        if trace_id not in self._traces:
            raise BackendError(f"no trace {trace_id!r}")
        return self._traces[trace_id]

    def query(
        self,
        frame: str | None = None,
        cell: dict[str, str] | None = None,
        ownership: str | None = None,
    ) -> list[Trace]:
        if self.query_error is not None:
            raise self.query_error
        out = [self._traces[tid] for tid in self._order]
        if frame is not None:
            out = [t for t in out if t.frame == frame]
        if ownership is not None:
            out = [t for t in out if t.ownership == ownership]
        if cell:
            out = [t for t in out if all(t.taxonomy_cell.get(k) == v for k, v in cell.items())]
        return out

    def put_score(self, label: Label) -> None:
        if self.submit_error is not None:
            raise self.submit_error
        self.labels.append(label)

    def get_scores(self, trace_id: str, span_id: str | None = None) -> list[Label]:
        return [
            lab
            for lab in self.labels
            if lab.trace_id == trace_id and (span_id is None or lab.span_id == span_id)
        ]

    def retract_score(self, label: Label) -> None:
        self.labels = [lab for lab in self.labels if lab != label]

    # --- LabelBackend ---
    def next_item(self) -> Trace | None:
        labeled = {lab.trace_id for lab in self.labels if lab.span_id is None}
        for tid in self._order:
            if tid not in labeled:
                return self._traces[tid]
        return None

    def submit(self, label: Label) -> None:
        self.put_score(label)


def _load_trace(trace_id: str) -> Trace:
    data = json.loads((_FIXTURES / "normalized" / f"{trace_id}.json").read_text(encoding="utf-8"))
    return trace_from_dict(data)


@pytest.fixture
def corpus_traces() -> list[Trace]:
    """The shared-corpus normalized traces in a deterministic order (ORD-1001/1002, CLM-4471)."""
    return [_load_trace(tid) for tid in _CORPUS_IDS]


@pytest.fixture
def gold_labels() -> list[Label]:
    """The gold `Label`s from `fixtures/labels/gold.json` — the acceptance oracle."""
    rows: list[dict[str, Any]] = json.loads(
        (_FIXTURES / "labels" / "gold.json").read_text(encoding="utf-8")
    )
    return [label_from_dict(r) for r in rows]


@pytest.fixture
def backend(corpus_traces: list[Trace]) -> FakeBackend:
    """A fake backend seeded with the corpus, satisfying both protocols."""
    return FakeBackend(list(corpus_traces))


@pytest.fixture
def make_backend(corpus_traces: list[Trace]) -> Callable[..., FakeBackend]:
    """Factory for a fake backend: override the corpus and/or inject query/submit errors."""

    def _make(traces: list[Trace] | None = None, **kwargs: Any) -> FakeBackend:
        seed = list(corpus_traces) if traces is None else list(traces)
        return FakeBackend(seed, **kwargs)

    return _make


@pytest.fixture
def make_client() -> Callable[..., Any]:
    """Factory: build the review app over a backend and return a Starlette TestClient.

    Imports FastHTML lazily so the pure-core tests (labeling/session/renderers) don't require the
    ``[ui]`` extra just to collect this module.
    """
    from starlette.testclient import TestClient

    from calibrated_agent_evaluation.review.app import build_app

    class _PageClient(TestClient):
        def request(self, method: str, url: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            if method.upper() not in {"GET", "HEAD", "OPTIONS"}:
                headers = dict(kwargs.pop("headers", None) or {})
                headers.setdefault("HX-Request", "true")
                kwargs["headers"] = headers
            return super().request(method, url, **kwargs)

    def _make(backend: FakeBackend, **kwargs: Any) -> tuple[Any, Any]:
        # The gold fixture's rater, so a UI-produced label can match its gold row field-for-field.
        kwargs.setdefault("rater_id", GOLD_RATER)
        app = build_app(backend, **kwargs)
        # The surface answers only to local host names, and its writes must come from its own page,
        # which sends HX-Request on every write; the test client stands in for that page. Reads
        # stay plain so a direct visit still gets the whole page.
        return _PageClient(app, base_url="http://127.0.0.1"), app

    return _make
