# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangfuseBackend adapter.

The adapter is exercised against an in-memory stub that **mirrors the real Langfuse v3 SDK
surface** (verified against ``langfuse==3.15.0``): the stub implements only
methods that exist on the real client (``create_trace_id``, ``start_span``,
``start_observation``, ``flush``, and the ``api.trace`` / ``api.score`` / ``api.score_v_2``
REST resources). Because the stub mirrors the real surface, these tests exercise the real
mapping code, not a fiction — and the binding to the live SDK is the opt-in integration
test against the reusable Docker stack.

Tests assert observable behavior — round-trip equality vs. a LocalBackend, recovered tags /
seq / metadata fields, raised BackendError — never mock call counts (rubric rule 2).
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, cast

import pytest

from _helpers import make_label, make_span, make_trace
from calibrated_agent_evaluation.backends.langfuse import (
    EE_GATED_METHODS,
    LangfuseBackend,
    LangfuseLike,
)
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.schema.errors import BackendError, SchemaError, TraceNotFoundError
from calibrated_agent_evaluation.schema.types import Label, Trace


def _backend(client: object) -> LangfuseBackend:
    """Inject a stub client. The stub mirrors the real Langfuse surface (verified);
    the cast tells the type checker the test double conforms — behavior is the real assertion."""
    return LangfuseBackend(cast("LangfuseLike", client))


# --- a stub mirroring the real langfuse v3 client surface --------------------------------


class _StubObservation:
    """Mirror of a LangfuseSpan handle: an id + update_trace/end, recording to the trace."""

    def __init__(self, store: _StoredTrace, obs_id: str, metadata: dict[str, Any]) -> None:
        self._store = store
        self.id = obs_id
        self.metadata = metadata

    def update_trace(
        self,
        *,
        tags: list[str] | None = None,
        metadata: Any | None = None,
        input: Any | None = None,
        output: Any | None = None,
        session_id: str | None = None,
    ) -> Any:
        # Kept as the trace's own session, as the real service stores it (sessionId).
        if session_id is not None:
            self._store.session_id = session_id
        # The real server merges a trace's tags and metadata keys across updates rather than
        # replacing them (verified live against server 3.202.1), so re-puts accumulate tags.
        if tags is not None:
            self._store.tags = list(dict.fromkeys([*self._store.tags, *tags]))
        if metadata is not None:
            self._store.metadata = {**self._store.metadata, **metadata}
        return self

    def end(self) -> Any:
        return self


class _StoredTrace:
    def __init__(self, trace_id: str) -> None:
        self.id = trace_id
        self.tags: list[str] = []
        self.metadata: dict[str, Any] = {}
        self.observations: list[_StubObservation] = []
        self.session_id: str | None = None


class _StoredScore:
    def __init__(self, value: str, comment: str, metadata: dict[str, Any]) -> None:
        self.value = value
        self.comment = comment
        self.metadata = metadata


class _TraceApi:
    def __init__(self, traces: dict[str, _StoredTrace]) -> None:
        self._traces = traces

    def get(self, trace_id: str) -> Any:
        # The real v3 client raises NotFoundError (an ApiError with status_code 404) for an
        # unknown id; it never returns None.
        if trace_id not in self._traces:
            from langfuse.api import NotFoundError

            raise NotFoundError(body={"message": f"Trace {trace_id} not found"})
        return self._traces[trace_id]

    def list(self, *, tags: list[str] | None = None, page: int = 1, limit: int = 50) -> Any:
        # Mirror the real v3 endpoint: tag filter + page/limit pagination, returning a
        # `.data` slice plus a `.meta.total_pages` bound (verified shape, langfuse==3.15.0).
        want = set(tags or [])
        matched = [t for t in self._traces.values() if want.issubset(set(t.tags))]
        total_pages = max(1, (len(matched) + limit - 1) // limit)
        start = (page - 1) * limit
        page_items = matched[start : start + limit]
        meta = type("Meta", (), {"page": page, "limit": limit, "total_pages": total_pages})()
        return type("Listing", (), {"data": page_items, "meta": meta})()


class _ScoreApi:
    """Mirror of ``api.score``/``api.score_v_2``: a synchronous create plus a paginated read."""

    def __init__(self, scores: dict[str, list[_StoredScore]]) -> None:
        self._scores = scores
        self.created = 0

    def create(self, *, request: Any, request_options: Any = None) -> Any:
        # The real REST create answers synchronously with the new score's id and raises on a
        # non-2xx (verified langfuse==3.15.0 against server 3.202.1).
        doc = request.dict()
        self._scores.setdefault(str(doc["traceId"]), []).append(
            _StoredScore(
                str(doc["value"]), str(doc.get("comment") or ""), doc.get("metadata") or {}
            )
        )
        self.created += 1
        return type("CreateScoreResponse", (), {"id": f"score-{self.created}"})()

    def get(self, *, trace_id: str, page: int = 1, limit: int = 50, **_: Any) -> Any:
        # Mirror the real v3 score_v_2.get: page/limit pagination with a `.meta.total_pages`
        # bound (verified shape, langfuse==3.15.0), so the adapter must walk every page.
        all_scores = list(self._scores.get(trace_id, []))
        total_pages = max(1, (len(all_scores) + limit - 1) // limit)
        start = (page - 1) * limit
        page_items = all_scores[start : start + limit]
        meta = type("Meta", (), {"page": page, "limit": limit, "total_pages": total_pages})()
        return type("Resp", (), {"data": page_items, "meta": meta})()


class _Api:
    def __init__(
        self, traces: dict[str, _StoredTrace], scores: dict[str, list[_StoredScore]]
    ) -> None:
        self.trace = _TraceApi(traces)
        self.score = _ScoreApi(scores)
        self.score_v_2 = self.score


class StubLangfuseClient:
    """In-memory stand-in for ``langfuse.Langfuse`` — only real-surface methods are present."""

    def __init__(self) -> None:
        self._traces: dict[str, _StoredTrace] = {}
        self._scores: dict[str, list[_StoredScore]] = {}
        self.api = _Api(self._traces, self._scores)
        self.calls: list[str] = []  # informational only; never the primary assertion

    def create_trace_id(self, *, seed: str | None = None) -> str:
        # Deterministic like the real SDK: same seed → same 32-hex id.
        return hashlib.sha256((seed or "").encode()).hexdigest()[:32]

    def start_span(
        self,
        *,
        trace_context: Any,
        name: str,
        input: Any | None = None,
        output: Any | None = None,
        metadata: Any | None = None,
    ) -> _StubObservation:
        self.calls.append("start_span")
        tid = trace_context["trace_id"]
        store = self._traces.setdefault(tid, _StoredTrace(tid))
        return _StubObservation(store, f"{tid}/root", metadata or {})

    def start_observation(
        self,
        *,
        trace_context: Any,
        name: str,
        as_type: str = "span",
        input: Any | None = None,
        output: Any | None = None,
        metadata: Any | None = None,
    ) -> _StubObservation:
        self.calls.append("start_observation")
        tid = trace_context["trace_id"]
        store = self._traces.setdefault(tid, _StoredTrace(tid))
        obs = _StubObservation(store, f"{tid}/obs{len(store.observations)}", metadata or {})
        store.observations.append(obs)
        return obs

    def flush(self) -> None:
        self.calls.append("flush")

    def shutdown(self) -> None:
        self.calls.append("shutdown")

    def auth_check(self) -> bool:
        self.calls.append("auth_check")
        return True


class _UnreachableClient(StubLangfuseClient):
    """Models a REAL unreachable Langfuse v3 instance, where a write once reported success.

    Crucially, the v3 SDK writes via a fire-and-forget async OTel exporter, so ``start_span`` /
    the queued ``create_score`` do NOT raise when the backend is down — they log and drop in the
    background. Only the synchronous ``auth_check`` surfaces the failure. So this stub leaves the
    write methods non-raising and makes ``auth_check`` raise, exactly like the real SDK; the
    adapter must still produce a typed ``BackendError`` (via its synchronous reachability probe),
    never a silent no-op. (The earlier stub raised inside ``start_span`` — synchronous behavior
    the real SDK does not have — which masked the bug the both-worlds gate caught.)
    """

    def auth_check(self) -> bool:
        raise ConnectionError("langfuse unreachable")


@pytest.fixture
def lf() -> LangfuseBackend:
    return _backend(StubLangfuseClient())


# --- both-worlds parity: Langfuse round-trip == Local round-trip ----------


def test_trace_and_label_roundtrip_matches_local_backend(
    lf: LangfuseBackend, corpus_clm: Trace, corpus_labels: list[Label], tmp_path
) -> None:
    crew_label = next(
        lbl for lbl in corpus_labels if lbl.trace_id == "CLM-4471" and lbl.span_id is None
    )
    # Langfuse world
    lf.put_trace(corpus_clm)
    lf.put_score(crew_label)
    lf_trace = lf.get_trace("CLM-4471")
    [lf_label] = lf.get_scores("CLM-4471")
    # Local world
    local = LocalBackend(root=tmp_path / "store")
    local.put_trace(corpus_clm)
    local.put_score(crew_label)
    local_trace = local.get_trace("CLM-4471")
    [local_label] = local.get_scores("CLM-4471")
    # parity: both worlds reconstruct the SAME normalized artifact as the original.
    assert lf_trace == corpus_clm == local_trace
    assert lf_label == crew_label == local_label


def test_re_put_trace_is_idempotent_on_read(
    lf: LangfuseBackend, corpus_clm: Trace, tmp_path
) -> None:
    # The v3 OTel write path is append-only: re-putting the same trace_id mints fresh observation
    # ids for our deterministic span ids, so the server accumulates duplicate observations. And
    # api.trace.delete is async across two stores, so a delete-then-rewrite can't fix it (verified
    # live). get_trace therefore dedupes reconstructed spans by span_id — so a re-put stays
    # idempotent on read, matching LocalBackend's DELETE+INSERT upsert (both-worlds parity).
    lf.put_trace(corpus_clm)
    lf.put_trace(corpus_clm)  # duplicate write — the append-only hazard
    lf.put_trace(corpus_clm)  # and again
    got = lf.get_trace("CLM-4471")
    assert len(got.spans) == len(corpus_clm.spans)  # deduped: not 3x the spans

    local = LocalBackend(root=tmp_path / "store")
    local.put_trace(corpus_clm)
    local.put_trace(corpus_clm)
    assert got == corpus_clm == local.get_trace("CLM-4471")  # parity holds after re-puts


def _two_span_trace(**over: object) -> Trace:
    root = make_span(span_id="T-1/root", seq=0)
    leaf = make_span(span_id="T-1/leaf", agent="helper", parent="T-1/root", seq=1)
    return make_trace(spans=[root, leaf], **over)


def test_re_put_of_a_changed_trace_reads_back_only_the_new_content(lf: LangfuseBackend) -> None:
    first = _two_span_trace(frame="discovery")
    second = make_trace(frame="production-representative", spans=[first.spans[0]])
    lf.put_trace(first)
    lf.put_trace(second)

    got = lf.get_trace("T-1")
    assert got == second
    assert [s.span_id for s in got.spans] == ["T-1/root"]


def test_a_reframed_trace_leaves_the_old_frames_query(lf: LangfuseBackend) -> None:
    lf.put_trace(_two_span_trace(frame="discovery"))
    lf.put_trace(_two_span_trace(frame="production-representative"))

    assert lf.query(frame="discovery") == []
    assert [t.trace_id for t in lf.query(frame="production-representative")] == ["T-1"]
    assert lf.scan(frame="discovery").traces == []


def test_an_owned_trace_re_put_as_probed_stays_readable(lf: LangfuseBackend) -> None:
    lf.put_trace(_two_span_trace())
    probed = make_trace(ownership="probed", spans=[])
    lf.put_trace(probed)

    assert lf.get_trace("T-1") == probed
    assert lf.query(cell={"feature": "refund"}) == [probed]


def test_a_cell_value_holding_a_colon_matches_only_its_own_cell(lf: LangfuseBackend) -> None:
    lf.put_trace(make_trace(trace_id="A", spans=[], ownership="probed", taxonomy_cell={"a:b": "c"}))
    lf.put_trace(make_trace(trace_id="B", spans=[], ownership="probed", taxonomy_cell={"a": "b:c"}))

    assert [t.trace_id for t in lf.query(cell={"a": "b:c"})] == ["B"]
    assert [t.trace_id for t in lf.query(cell={"a:b": "c"})] == ["A"]


def test_a_trace_missing_spans_of_its_latest_write_is_refused(lf: LangfuseBackend) -> None:
    lf.put_trace(_two_span_trace())
    client = cast("StubLangfuseClient", lf._client)  # type: ignore[attr-defined]
    stored = next(iter(client._traces.values()))
    stored.observations.pop()  # one span still in Langfuse's ingestion queue

    with pytest.raises(BackendError, match="holds 1 of the 2 span"):
        lf.get_trace("T-1")


def test_a_trace_written_before_generations_keeps_every_enveloped_span(
    lf: LangfuseBackend,
) -> None:
    trace = _two_span_trace()
    lf.put_trace(trace)
    client = cast("StubLangfuseClient", lf._client)  # type: ignore[attr-defined]
    stored = next(iter(client._traces.values()))
    stored.metadata = {
        "calibrated_agent_evaluation_trace": stored.metadata["calibrated_agent_evaluation_trace"]
    }
    for obs in stored.observations:
        obs.metadata = {
            k: v for k, v in obs.metadata.items() if k != "calibrated_agent_evaluation_generation"
        }

    assert lf.get_trace("T-1") == trace


# --- public methods only; no schema change; no silent drop ---------------


def test_adapter_uses_only_public_client_methods() -> None:
    # "via the public API": the adapter must reach only documented public client
    # members — never a private/underscored attribute. A spying client records every attribute
    # the adapter requests on it; we assert none is underscore-prefixed. (The stub's own
    # internal state is accessed via object.__getattribute__ inside its methods, so it does not
    # appear here — only attribute access made *through* the client object surfaces.)
    requested: list[str] = []
    _PUBLIC_SURFACE = {
        "auth_check",
        "create_trace_id",
        "start_span",
        "start_observation",
        "flush",
        "api",
        "calls",
    }

    class _SpyClient(StubLangfuseClient):
        def __getattribute__(self, item: str) -> Any:
            if not item.startswith("__"):
                requested.append(item)
            return object.__getattribute__(self, item)

    backend = _backend(_SpyClient())
    backend.put_trace(
        make_trace(trace_id="P-1", spans=[make_span(trace_id="P-1", span_id="P-1/r")])
    )
    got = backend.get_trace("P-1")
    assert got.trace_id == "P-1"
    # Every attribute the adapter asked the client for is a documented public member.
    adapter_requested = {name for name in requested if not name.startswith("_")}
    assert adapter_requested
    assert adapter_requested <= _PUBLIC_SURFACE, (
        f"adapter touched non-public client attrs: {adapter_requested - _PUBLIC_SURFACE}"
    )
    # And it never reached for a private/underscored attribute on the client surface.
    assert not [n for n in requested if n.startswith("_") and n not in {"_traces", "_scores"}]


def test_normalized_field_with_no_langfuse_home_is_not_silently_dropped(
    lf: LangfuseBackend,
) -> None:
    # A trace whose metadata envelope is stripped (simulating a lost field) must surface, not
    # silently reconstruct a half-trace. The adapter raises rather than dropping data.
    trace = make_trace(trace_id="P-9", spans=[make_span(trace_id="P-9", span_id="P-9/r")])
    lf.put_trace(trace)
    # Corrupt the stored envelope to simulate a field with no Langfuse home being lost.
    stored = lf._client.api.trace.get(lf._client.create_trace_id(seed="P-9"))  # type: ignore[attr-defined]
    stored.metadata = {}
    with pytest.raises(BackendError, match="calibrated_agent_evaluation_trace metadata envelope"):
        lf.get_trace("P-9")


# --- tags / seq / Score.metadata conventions ------------------------------


def test_queryable_dims_land_as_tags_and_seq_persisted(lf: LangfuseBackend) -> None:
    trace = make_trace(
        trace_id="P-1",
        frame="production-representative",
        taxonomy_cell={"feature": "claims", "scenario": "S2"},
        spans=[
            make_span(trace_id="P-1", span_id="P-1/a", seq=0, parent=None),
            make_span(trace_id="P-1", span_id="P-1/b", seq=1, parent="P-1/a"),
        ],
    )
    lf.put_trace(trace)
    stored = lf._client.api.trace.get(lf._client.create_trace_id(seed="P-1"))  # type: ignore[attr-defined]
    assert "frame:production-representative" in stored.tags
    assert "ownership:owned" in stored.tags
    assert "cell.feature:claims" in stored.tags
    # seq persisted explicitly on each observation's metadata.
    assert sorted(obs.metadata["seq"] for obs in stored.observations) == [0, 1]


def test_bespoke_label_fields_recovered_from_score_metadata(
    lf: LangfuseBackend, corpus_clm: Trace, corpus_labels: list[Label]
) -> None:
    crew_label = next(
        lbl for lbl in corpus_labels if lbl.trace_id == "CLM-4471" and lbl.span_id is None
    )
    lf.put_trace(corpus_clm)
    lf.put_score(crew_label)
    [recovered] = lf.get_scores("CLM-4471")
    # culprit_span has no native Langfuse home — it must survive via Score.metadata.
    assert recovered.culprit_span == "CLM-4471/intake"
    assert recovered.handoff_implicated == "intake->policy-checker"


def test_query_pages_through_all_results_no_silent_truncation(lf: LangfuseBackend) -> None:
    # Regression: query() must walk every page of the paginated v3 list endpoint. With more
    # traces than one page (_QUERY_PAGE_SIZE) holds, reading only page 1 would silently drop the
    # rest — exactly the silent truncation the project forbids. Seed 2.5 pages' worth.
    from calibrated_agent_evaluation.backends.langfuse import _QUERY_PAGE_SIZE

    n = _QUERY_PAGE_SIZE * 2 + 7
    for i in range(n):
        lf.put_trace(
            make_trace(
                trace_id=f"D-{i}",
                frame="discovery",
                spans=[make_span(trace_id=f"D-{i}", span_id=f"D-{i}/r")],
            )
        )
    matched = lf.query(frame="discovery")
    assert len(matched) == n
    assert {t.trace_id for t in matched} == {f"D-{i}" for i in range(n)}


def test_corrupt_metadata_json_surfaces_as_backend_error(lf: LangfuseBackend) -> None:
    # A stored metadata string that is not valid JSON must raise a typed BackendError, never an
    # uncaught json.JSONDecodeError escaping the CalibratedAgentEvaluationDataError family.
    trace = make_trace(trace_id="P-7", spans=[make_span(trace_id="P-7", span_id="P-7/r")])
    lf.put_trace(trace)
    stored = lf._client.api.trace.get(lf._client.create_trace_id(seed="P-7"))  # type: ignore[attr-defined]
    stored.metadata = "{not valid json"
    with pytest.raises(BackendError, match="not valid JSON"):
        lf.get_trace("P-7")


def test_partial_trace_envelope_surfaces_as_backend_error(lf: LangfuseBackend) -> None:
    # A valid-JSON but structurally-incomplete envelope (missing a required key) makes
    # trace_from_dict raise KeyError; that must be wrapped as a BackendError, not escape raw.
    trace = make_trace(trace_id="P-8", spans=[make_span(trace_id="P-8", span_id="P-8/r")])
    lf.put_trace(trace)
    stored = lf._client.api.trace.get(lf._client.create_trace_id(seed="P-8"))  # type: ignore[attr-defined]
    stored.metadata = {
        "calibrated_agent_evaluation_trace": {"trace_id": "P-8"}
    }  # drop everything else
    with pytest.raises(BackendError):
        lf.get_trace("P-8")


def test_partial_label_envelope_surfaces_as_backend_error(lf: LangfuseBackend) -> None:
    # Same guarantee on the score read path: a partial label envelope wraps, never a raw KeyError.
    trace = make_trace(trace_id="P-9", spans=[make_span(trace_id="P-9", span_id="P-9/r")])
    lf.put_trace(trace)
    label = make_label(trace_id="P-9", kind="human", outcome="FAIL")
    lf.put_score(label)
    [stored] = lf._client.api.score_v_2.get(  # type: ignore[attr-defined]
        trace_id=lf._client.create_trace_id(seed="P-9")  # type: ignore[attr-defined]
    ).data
    stored.metadata = {
        "calibrated_agent_evaluation_label": {"trace_id": "P-9"}
    }  # drop required fields
    with pytest.raises(BackendError):
        lf.get_scores("P-9")


def test_get_scores_pages_through_all_results_no_silent_truncation(
    lf: LangfuseBackend,
) -> None:
    # More labels than one page (limit 50): get_scores must return ALL of them, not just page 1.
    trace = make_trace(trace_id="P-10", spans=[make_span(trace_id="P-10", span_id="P-10/r")])
    lf.put_trace(trace)
    for i in range(120):
        lf.put_score(make_label(trace_id="P-10", kind="human", outcome="FAIL", critique=f"c{i}"))
    labels = lf.get_scores("P-10")
    assert len(labels) == 120  # every page walked, no silent tail drop


def test_from_credentials_does_not_permanently_mutate_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Constructing a backend must not leave a global os.environ side effect behind: the
    # TELEMETRY_ENABLED set is scoped to client construction and restored after.
    monkeypatch.delenv("TELEMETRY_ENABLED", raising=False)

    def factory(**_: Any) -> StubLangfuseClient:
        return StubLangfuseClient()

    LangfuseBackend.from_credentials("pk", "sk", "http://localhost:3000", client_factory=factory)
    assert "TELEMETRY_ENABLED" not in __import__("os").environ


def test_from_credentials_restores_prior_environment_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TELEMETRY_ENABLED", "preexisting")

    def factory(**_: Any) -> StubLangfuseClient:
        return StubLangfuseClient()

    LangfuseBackend.from_credentials("pk", "sk", "http://localhost:3000", client_factory=factory)
    # The caller's prior value is restored, not clobbered.
    assert __import__("os").environ["TELEMETRY_ENABLED"] == "preexisting"


def test_langfuse_backend_is_a_context_manager() -> None:
    client = StubLangfuseClient()
    with _backend(client) as backend:
        backend.put_trace(
            make_trace(trace_id="P-1", spans=[make_span(trace_id="P-1", span_id="P-1/r")])
        )
    # __exit__ FLUSHES (durability boundary) but does NOT shutdown(): the SDK client is a
    # process-global singleton (keyed by public_key) whose shared daemon threads are cleaned up by
    # its own atexit hook. Calling shutdown() per exit would permanently kill the shared consumer
    # threads and wedge every later backend for the same key (see the dedicated regression tests).
    assert "flush" in client.calls
    assert "shutdown" not in client.calls


def test_query_by_tag_only_returns_matching_subset(lf: LangfuseBackend) -> None:
    disc = make_trace(
        trace_id="D-1", frame="discovery", spans=[make_span(trace_id="D-1", span_id="D-1/r")]
    )
    prod = make_trace(
        trace_id="P-1",
        frame="production-representative",
        spans=[make_span(trace_id="P-1", span_id="P-1/r")],
    )
    lf.put_trace(disc)
    lf.put_trace(prod)
    matched = lf.query(frame="discovery")
    assert [t.trace_id for t in matched] == ["D-1"]


# --- OSS build, telemetry off, no EE method invoked -----------------------


def test_adapter_invokes_no_ee_gated_method() -> None:
    # The adapter must never call an Enterprise-Edition-gated method. Drive a full put/get and
    # assert none of the recorded calls is an EE method name (negative guard: OSS build only).
    client = StubLangfuseClient()
    backend = _backend(client)
    backend.put_trace(
        make_trace(trace_id="P-1", spans=[make_span(trace_id="P-1", span_id="P-1/r")])
    )
    backend.get_trace("P-1")
    assert not (set(client.calls) & EE_GATED_METHODS)


def test_ee_denylist_tracks_the_real_client_surface() -> None:
    # Critic warning honored: prove the EE_GATED_METHODS denylist reflects reality, not just the
    # stub — none of the denied names exists on the real langfuse.Langfuse client surface, so
    # "the adapter never calls one" is a meaningful guarantee. Skips cleanly if the opt-in
    # [langfuse] extra is not installed.
    langfuse = pytest.importorskip("langfuse")
    real_surface = set(dir(langfuse.Langfuse))
    assert not (EE_GATED_METHODS & real_surface), (
        "an EE_GATED_METHODS name appears on the real client — update the denylist to match"
    )


def test_from_credentials_disables_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEMETRY_ENABLED", raising=False)
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> StubLangfuseClient:
        captured["env_at_build"] = __import__("os").environ.get("TELEMETRY_ENABLED")
        return StubLangfuseClient()

    backend = LangfuseBackend.from_credentials(
        "pk", "sk", "http://localhost:3000", client_factory=factory
    )
    assert backend.telemetry_disabled is True
    # TELEMETRY_ENABLED=false was set BEFORE the client was constructed.
    assert captured["env_at_build"] == "false"


def test_from_credentials_passes_a_request_timeout_to_the_client() -> None:
    # A bounded timeout reaches the real client so an unreachable backend (incl. the auth_check
    # probe) fails fast instead of hanging. Default is explicit (5s); callers can tune it.
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> StubLangfuseClient:
        captured.update(kwargs)
        return StubLangfuseClient()

    LangfuseBackend.from_credentials(
        "pk", "sk", "http://localhost:3000", timeout_seconds=12, client_factory=factory
    )
    assert captured["timeout"] == 12

    LangfuseBackend.from_credentials("pk", "sk", "http://localhost:3000", client_factory=factory)
    assert captured["timeout"] == 5  # explicit default


# --- unreachable backend → BackendError, no LocalBackend fallback ---------


def test_put_trace_failure_raises_backend_error_and_builds_no_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Sentinel: if the adapter ever silently fell back to a LocalBackend, constructing one
    # would be observable. Fail the test if LocalBackend is instantiated during the failure.
    import calibrated_agent_evaluation.backends.local as local_mod

    created: list[int] = []
    original_init = local_mod.LocalBackend.__init__

    def spy_init(self, *args: Any, **kwargs: Any) -> None:
        created.append(1)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(local_mod.LocalBackend, "__init__", spy_init)

    backend = _backend(_UnreachableClient())
    with pytest.raises(BackendError, match="put_trace"):
        backend.put_trace(
            make_trace(trace_id="P-1", spans=[make_span(trace_id="P-1", span_id="P-1/r")])
        )
    assert created == []  # no silent LocalBackend fallback


def test_put_score_failure_surfaces_as_backend_error() -> None:
    backend = _backend(_UnreachableClient())
    with pytest.raises(BackendError, match="put_score"):
        backend.put_score(make_label(trace_id="P-1", outcome="PASS", critique="ok"))


def test_unreachable_write_is_not_a_silent_noop_despite_async_exporter() -> None:
    # Regression for the dead-server finding from acceptance: the v3 SDK's OTel exporter is
    # fire-and-forget, so start_span does not raise when the backend is down. Prove
    # the adapter still refuses LOUDLY (synchronous reachability probe) instead of silently
    # accepting a write that will never land. The unreachable client's write methods never
    # raise; only auth_check does — exactly the real SDK's behavior.
    client = _UnreachableClient()
    backend = _backend(client)
    with pytest.raises(BackendError):
        backend.put_trace(
            make_trace(trace_id="Z-1", spans=[make_span(trace_id="Z-1", span_id="Z-1/r")])
        )
    # The write was refused at the probe — no span/score write was even attempted.
    assert "start_span" not in client.calls
    assert client.api.score.created == 0


class _RefusedKeysClient(StubLangfuseClient):
    """Langfuse answers, but refuses the keys: the SDK's ``auth_check`` returns ``False``."""

    def auth_check(self) -> bool:
        self.calls.append("auth_check")
        return False


@pytest.mark.parametrize(
    "client",
    [_RefusedKeysClient(), _UnreachableClient()],
    ids=["keys-refused", "host-unreachable"],
)
def test_check_reachable_raises_a_backend_error_when_the_host_or_keys_are_refused(
    client: StubLangfuseClient,
) -> None:
    backend = _backend(client)

    with pytest.raises(BackendError, match="reachability check"):
        backend.check_reachable()

    assert not {"start_span", "start_observation", "create_score", "flush"} & set(client.calls)


def test_check_reachable_passes_when_auth_is_accepted_and_writes_nothing() -> None:
    client = StubLangfuseClient()

    _backend(client).check_reachable()

    assert client.calls == ["auth_check"]
    assert client.api.trace.list(page=1, limit=50).data == []


class _WedgedFlushClient(StubLangfuseClient):
    """Models the real langfuse==3.15.0 hang: flush() blocks on an unbounded queue.join() when the
    ingestion queue wedges. Its flush() blocks until released (or forever), like the SDK's."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def flush(self) -> None:
        self.calls.append("flush")
        self.release.wait()  # blocks like the SDK's queue.join() until (if ever) released


def test_put_trace_bounded_flush_raises_backend_error_instead_of_hanging() -> None:
    # Regression for the end-to-end hang: the SDK's flush() does an unbounded queue.join() that
    # blocks forever when the ingestion queue wedges. The adapter must cap it and raise a typed
    # BackendError, never block the caller indefinitely (an infinite hang is as dishonest as a
    # silent drop). A tiny flush_timeout keeps the test fast; the wedged client never releases.
    client = _WedgedFlushClient()
    backend = LangfuseBackend(cast("LangfuseLike", client), flush_timeout_seconds=0.2)
    t0 = time.monotonic()
    with pytest.raises(BackendError, match="flush did not complete"):
        backend.put_trace(
            make_trace(trace_id="W-1", spans=[make_span(trace_id="W-1", span_id="W-1/r")])
        )
    elapsed = time.monotonic() - t0
    # The span write itself was attempted before the flush wedged…
    assert "start_span" in client.calls
    # …and the bounded wait returned promptly rather than hanging.
    assert elapsed < 5.0
    client.release.set()  # let the abandoned daemon flush thread finish cleanly


class _RejectingScoreApi(_ScoreApi):
    """Mirror of a server that rejects the score create (oversized body, a validation 4xx)."""

    def create(self, *, request: Any, request_options: Any = None) -> Any:
        from langfuse.api.core.api_error import ApiError

        raise ApiError(status_code=413, body="Body exceeded 1mb limit")


class _IdlessScoreApi(_ScoreApi):
    """Mirror of a create that answers 2xx without the score id the real one always carries."""

    def create(self, *, request: Any, request_options: Any = None) -> Any:
        return type("CreateScoreResponse", (), {"id": ""})()


def _client_with_score_api(api_cls: type[_ScoreApi]) -> StubLangfuseClient:
    client = StubLangfuseClient()
    client.api.score = api_cls(client._scores)
    return client


def test_a_rejected_score_write_raises_instead_of_reporting_success() -> None:
    # The queued create_score path swallows a rejection in its background consumer and flush()
    # returns normally, so a label from the review UI would be gone while the UI said saved.
    # The adapter writes through the synchronous REST create, so a rejection reaches the caller.
    backend = _backend(_client_with_score_api(_RejectingScoreApi))
    label = make_label(trace_id="R-1", outcome="FAIL", critique="rejected by the server")
    with pytest.raises(BackendError, match="put_score"):
        backend.put_score(label)
    with pytest.raises(BackendError, match="retract_score"):
        backend.retract_score(label)


def test_a_score_create_with_no_id_is_not_treated_as_confirmed() -> None:
    backend = _backend(_client_with_score_api(_IdlessScoreApi))
    with pytest.raises(BackendError, match="no score id"):
        backend.put_score(make_label(trace_id="R-2", outcome="PASS", critique="unconfirmed"))


def test_context_exit_flushes_but_does_not_shutdown_the_shared_singleton() -> None:
    # The langfuse v3 client is a thin wrapper over a PROCESS-GLOBAL singleton resource manager
    # (keyed by public_key) that owns shared daemon threads + queues. Calling shutdown() on context
    # exit permanently kills those SHARED consumer threads and leaves the dead manager cached, so a
    # LATER backend for the same key reuses a manager whose next flush() hangs forever. So __exit__
    # must FLUSH (durability) but NOT shutdown() — the SDK's own atexit hook owns teardown.
    client = StubLangfuseClient()
    with LangfuseBackend(cast("LangfuseLike", client)):
        pass
    assert "flush" in client.calls  # flushed on exit (durability boundary)
    assert "shutdown" not in client.calls  # did NOT poison the shared singleton


def test_context_exit_wedged_flush_neither_hangs_nor_raises() -> None:
    # __exit__ may run while an exception propagates, so it must never raise; and a wedged flush
    # must not hang the `with` close. The bounded flush suppresses its BackendError here (each
    # put_* already flushed+raised at its own call site — this is a best-effort final drain).
    client = _WedgedFlushClient()
    t0 = time.monotonic()
    with LangfuseBackend(cast("LangfuseLike", client), flush_timeout_seconds=0.2):
        pass  # exit triggers a bounded flush that wedges, but is suppressed
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0  # returned promptly, did not hang
    client.release.set()  # release the abandoned daemon


def test_multiple_backends_same_key_reused_do_not_wedge() -> None:
    # The end-to-end regression: opening a FRESH backend per operation (as the demo does per stage)
    # must not wedge later operations. With flush-only __exit__ (no shutdown), a shared stub client
    # stays usable across many enter/exit cycles — each flush returns, none block.
    client = StubLangfuseClient()  # one shared client, like the SDK's public_key singleton
    for i in range(6):
        with LangfuseBackend(cast("LangfuseLike", client)) as backend:
            backend.put_score(make_label(trace_id=f"M-{i}", outcome="PASS", critique="ok"))
    # every write landed; the shared client was never shut down mid-sequence
    assert client.calls.count("shutdown") == 0
    assert client.api.score.created == 6


# --- the read path validates, like the write path does -----------------------


def test_observations_without_the_span_envelope_surface_as_backend_error(
    lf: LangfuseBackend,
) -> None:
    """An `owned` envelope whose observations lack `calibrated_agent_evaluation_span` must refuse,
    not degrade.

    `_reconstruct_trace` skipped any observation with no span envelope and returned whatever was
    left — and it was the one place a Trace entered the methodology plane **unvalidated**, because
    only the write path called `validate`. So an `owned` trace came back with `spans == []`, a shape
    `validate` forbids outright, and every consumer read a zero-step trace as a
    complete one.
    """
    trace = make_trace(trace_id="P-20", spans=[make_span(trace_id="P-20", span_id="P-20/r")])
    lf.put_trace(trace)
    stored = lf._client.api.trace.get(lf._client.create_trace_id(seed="P-20"))  # type: ignore[attr-defined]
    for obs in stored.observations:
        obs.metadata = {"something_else": {"not": "our envelope"}}

    with pytest.raises(BackendError) as excinfo:
        lf.get_trace("P-20")
    message = str(excinfo.value)
    assert "P-20" in message, "the message must name the trace that could not be reconstructed"
    assert "calibrated_agent_evaluation_span" in message, (
        "and say what was missing from the observations"
    )


def test_the_skipped_observation_count_is_reported(lf: LangfuseBackend) -> None:
    """The count is what distinguishes "nothing was written" from "we could not read it"."""
    spans = [
        make_span(trace_id="P-21", span_id="P-21/r"),
        make_span(trace_id="P-21", span_id="P-21/a", parent="P-21/r", seq=1),
    ]
    lf.put_trace(make_trace(trace_id="P-21", spans=spans))
    stored = lf._client.api.trace.get(lf._client.create_trace_id(seed="P-21"))  # type: ignore[attr-defined]
    assert len(stored.observations) == 2, "precondition: both spans were written"
    for obs in stored.observations:
        obs.metadata = {}

    with pytest.raises(BackendError, match="2 observation"):
        lf.get_trace("P-21")


def test_a_correctly_enveloped_trace_still_round_trips(lf: LangfuseBackend) -> None:
    """Regression guard: adding validation must not refuse a trace that was written properly."""
    spans = [
        make_span(trace_id="P-22", span_id="P-22/r"),
        make_span(trace_id="P-22", span_id="P-22/a", parent="P-22/r", seq=1),
    ]
    original = make_trace(trace_id="P-22", spans=spans)
    lf.put_trace(original)

    got = lf.get_trace("P-22")
    assert got.trace_id == "P-22"
    assert got.ownership == original.ownership
    assert [s.span_id for s in got.spans] == ["P-22/r", "P-22/a"]


# --- retraction on an append-only score store ---------------------------


def test_retract_score_hides_exactly_that_label_on_read(lf: LangfuseBackend) -> None:
    keep = make_label(trace_id="T-1", outcome="FAIL", critique="kept")
    gone = make_label(trace_id="T-1", outcome="PASS", critique="retracted")
    lf.put_score(keep)
    lf.put_score(gone)

    lf.retract_score(gone)

    assert lf.get_scores("T-1") == [keep]


def test_a_label_re_put_after_its_retraction_reads_back_on_langfuse(lf: LangfuseBackend) -> None:
    """Last write wins per label: the tombstone hides earlier puts, never a later one."""
    label = make_label(trace_id="T-1", outcome="PASS", critique="same words")
    lf.put_score(label)
    lf.retract_score(label)
    assert lf.get_scores("T-1") == []
    lf.put_score(label)
    assert lf.get_scores("T-1") == [label]


def test_a_retraction_hides_a_put_that_lands_after_it_but_was_written_before(
    lf: LangfuseBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Score ingestion is asynchronous (seconds, verified live): an undo pressed right after a
    verdict retracts a label the API cannot return yet. Its stamp predates the tombstone's, so it
    stays hidden when it does land."""
    import calibrated_agent_evaluation.backends.langfuse as lf_mod

    label = make_label(trace_id="T-1", outcome="PASS", critique="undone quickly")
    lf.retract_score(label)  # nothing readable yet: no error
    monkeypatch.setattr(lf_mod, "_written_at", lambda: "2000-01-01T00:00:00.000000+00:00")
    lf.put_score(label)  # the put, stamped before the tombstone, becomes readable late
    assert lf.get_scores("T-1") == []


def test_retraction_parity_with_the_local_backend(lf: LangfuseBackend, tmp_path) -> None:
    """The same put/retract sequence reads back identically in both worlds."""
    local = LocalBackend(root=tmp_path / "store")
    a = make_label(trace_id="T-1", outcome="FAIL", critique="a")
    b = make_label(trace_id="T-1", outcome="PASS", critique="b")
    for backend in (lf, local):
        backend.put_score(a)
        backend.put_score(b)
        backend.retract_score(a)
    assert lf.get_scores("T-1") == local.get_scores("T-1") == [b]


def test_get_trace_of_an_unknown_id_is_a_typed_not_found(lf: LangfuseBackend) -> None:
    with pytest.raises(TraceNotFoundError, match="NOPE"):
        lf.get_trace("NOPE")


class _FailingReadClient(StubLangfuseClient):
    """A server that answers every trace read with a 500."""

    def __init__(self) -> None:
        super().__init__()

        def _fail(trace_id: str) -> Any:
            from langfuse.api.core.api_error import ApiError

            raise ApiError(status_code=500, body="internal error")

        self.api.trace.get = _fail  # type: ignore[method-assign]


def test_a_failed_trace_read_is_not_reported_as_not_found() -> None:
    lf = _backend(_FailingReadClient())
    with pytest.raises(BackendError) as excinfo:
        lf.get_trace("ANY")
    assert not isinstance(excinfo.value, TraceNotFoundError)


# --- the session a trace belongs to ----------------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   "])
def test_an_empty_session_id_is_refused_before_langfuse_holds_anything(blank: str) -> None:
    client = StubLangfuseClient()
    lf = _backend(client)
    with pytest.raises(SchemaError, match="session_id is blank"):
        lf.put_trace(_two_span_trace(session_id=blank))
    assert client.api.trace.list().data == []


@pytest.mark.parametrize(
    "session",
    ["calibrated-agent-evaluation-0123456789abcdef0123456789abcdef", "s" * 199],
    ids=["runtime", "199"],
)
def test_a_trace_session_is_also_set_as_the_langfuse_session(session: str) -> None:
    client = StubLangfuseClient()
    lf = _backend(client)
    trace = _two_span_trace(session_id=session)

    lf.put_trace(trace)

    [stored] = client.api.trace.list().data
    assert stored.session_id == session
    assert lf.get_trace("T-1") == trace
    assert lf.query() == [trace]


def test_a_trace_with_no_session_sets_no_langfuse_session() -> None:
    client = StubLangfuseClient()
    lf = _backend(client)

    lf.put_trace(_two_span_trace())

    [stored] = client.api.trace.list().data
    assert stored.session_id is None
    assert lf.get_trace("T-1").session_id is None


@pytest.mark.parametrize(
    ("session", "reason"),
    [("s" * 200, "is 200 characters"), ("séance-1", "is not US-ASCII")],
    ids=["200", "non-ascii"],
)
def test_a_session_langfuse_cannot_hold_is_kept_in_the_envelope_and_warned(
    session: str, reason: str, caplog: pytest.LogCaptureFixture
) -> None:
    client = StubLangfuseClient()
    lf = _backend(client)

    with caplog.at_level("WARNING", logger="calibrated_agent_evaluation.backends.langfuse"):
        lf.put_trace(_two_span_trace(session_id=session))

    [stored] = client.api.trace.list().data
    assert stored.session_id is None
    assert lf.get_trace("T-1").session_id == session
    [warning] = [r for r in caplog.records if "session_id" in r.getMessage()]
    assert "trace T-1" in warning.getMessage()
    assert reason in warning.getMessage()
