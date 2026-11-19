# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live integration test: LangfuseBackend against a REAL unmodified instance.

This is the binding check for the "unmodified deployment" claim and the both-worlds
parity — it exercises the real ``langfuse`` v3 SDK against the real OSS stack
(``docker/langfuse/docker-compose.yml``), not the unit-test stub. It is marked
``@pytest.mark.integration`` and is **skipped by default** (the default ``just test`` run is
hermetic and offline); run it with:

    just langfuse-up
    just test-integration
    just langfuse-down

It skips cleanly (not fails) when the ``[langfuse]`` extra is absent or the stack is
unreachable, so it never breaks a developer who hasn't started Docker.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, cast

import pytest

from _helpers import langfuse_reachable
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.schema.errors import BackendError
from calibrated_agent_evaluation.schema.serde import label_from_dict, trace_from_dict
from calibrated_agent_evaluation.schema.types import Label, Span, Trace

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures"


def _load_trace(trace_id: str) -> Trace:
    return trace_from_dict(json.loads((_FIXTURES / "normalized" / f"{trace_id}.json").read_text()))


def _remap_trace(trace: Trace, new_id: str) -> Trace:
    """Rewrite a corpus trace onto ``new_id`` (trace_id + every span's id/parent), preserving
    shape and content — so the live test is idempotent against a persistent instance."""
    old = trace.trace_id

    def remap_span(s: Span) -> Span:
        return replace(
            s,
            span_id=s.span_id.replace(old, new_id, 1),
            trace_id=new_id,
            parent=s.parent.replace(old, new_id, 1) if s.parent else None,
        )

    return replace(trace, trace_id=new_id, spans=[remap_span(s) for s in trace.spans])


def _load_labels() -> list[Label]:
    raw = json.loads((_FIXTURES / "labels" / "gold.json").read_text())
    return [label_from_dict(x) for x in raw]


@pytest.fixture
def live_backend() -> LangfuseBackend:
    pytest.importorskip("langfuse")
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-dev")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-calibrated-agent-evaluation-dev")
    backend = LangfuseBackend.from_credentials(public_key, secret_key, host)
    # Skip (don't fail) if the stack isn't actually up — keeps the suite friendly off-Docker.
    # auth_check() RAISES (httpx.ConnectError) when nothing is listening, so probe inside a
    # try and treat any failure-or-False as "not reachable → skip", never an ERROR.
    reachable = langfuse_reachable(backend._client)
    if not reachable:
        pytest.skip(f"no reachable Langfuse at {host} (run `just langfuse-up`)")
    return backend


def test_live_trace_and_label_roundtrip_against_real_langfuse(
    live_backend: LangfuseBackend, tmp_path
) -> None:
    # Remap the corpus CLM-4471 onto a UNIQUE per-run trace_id so the test is idempotent
    # against a persistent Langfuse instance (it cannot assume a clean DB; re-putting the same
    # id would append duplicate observations). The shape/content is the corpus oracle; only the
    # ids are rewritten.
    run_id = f"CLM-4471-it-{uuid.uuid4().hex[:8]}"
    base = _load_trace("CLM-4471")
    trace = _remap_trace(base, run_id)
    base_label = next(
        lbl for lbl in _load_labels() if lbl.trace_id == "CLM-4471" and lbl.span_id is None
    )
    crew_label = replace(
        base_label,
        trace_id=run_id,
        culprit_span=run_id + "/intake",
        manifestation_span=run_id + "/payout",
    )

    live_backend.put_trace(trace)
    live_backend.put_score(crew_label)

    # Langfuse ingestion is async (OTel export + worker); poll get_trace until it lands.
    got: Trace | None = None
    for _ in range(30):
        try:
            got = live_backend.get_trace(run_id)
            break
        except Exception:
            time.sleep(1)
    assert got is not None, "trace did not appear in Langfuse within the ingestion window"

    # parity: the real-Langfuse round-trip reconstructs the SAME normalized artifact as
    # a LocalBackend round-trip of the same input — field-for-field.
    local = LocalBackend(root=tmp_path / "store")
    local.put_trace(trace)
    assert got == trace == local.get_trace(run_id)

    # Scores ingest asynchronously too — poll until the gold label lands, then assert the
    # bespoke crew-attribution field survived via Score.metadata.
    scores: list[Label] = []
    for _ in range(30):
        scores = live_backend.get_scores(run_id)
        if scores:
            break
        time.sleep(1)
    assert any(s.culprit_span == run_id + "/intake" for s in scores), (
        "gold label with culprit_span did not appear / round-trip in Langfuse"
    )


def test_live_re_put_trace_is_idempotent_on_read(live_backend: LangfuseBackend) -> None:
    # The v3 OTel write path is append-only and api.trace.delete is async across Postgres+
    # ClickHouse (a delete-then-rewrite races and cannot be made reliable — verified). get_trace
    # dedupes reconstructed spans by span_id, so re-putting the same trace stays idempotent on
    # read. This is the live binding of the parity guarantee after a re-put.
    run_id = f"CLM-4471-idem-{uuid.uuid4().hex[:8]}"
    trace = _remap_trace(_load_trace("CLM-4471"), run_id)

    live_backend.put_trace(trace)
    got: Trace | None = None
    for _ in range(30):
        try:
            got = live_backend.get_trace(run_id)
            break
        except Exception:
            time.sleep(1)
    assert got is not None, "trace did not appear within the ingestion window"

    live_backend.put_trace(trace)  # duplicate write against the persistent instance
    live_backend.put_trace(trace)  # and again
    time.sleep(8)  # let the duplicate observations land

    got = live_backend.get_trace(run_id)
    # Deduped on read: one span per span_id despite three writes.
    assert len(got.spans) == len(trace.spans)
    assert got == trace


def _poll_trace(backend: LangfuseBackend, trace_id: str, want: Trace) -> Trace | None:
    got: Trace | None = None
    for _ in range(40):
        try:
            got = backend.get_trace(trace_id)
        except Exception:
            got = None
        if got == want:
            return got
        time.sleep(1)
    return got


def test_live_re_put_of_a_changed_trace_drops_the_old_spans_and_frame(
    live_backend: LangfuseBackend,
) -> None:
    run_id = f"review-2026-10-26-changed-{uuid.uuid4().hex[:8]}"
    first = replace(_remap_trace(_load_trace("CLM-4471"), run_id), frame="discovery")
    leaf = first.spans[-1]
    assert all(s.parent != leaf.span_id for s in first.spans)
    second = replace(first, frame="production-representative", spans=first.spans[:-1])

    live_backend.put_trace(first)
    assert _poll_trace(live_backend, run_id, first) == first
    live_backend.put_trace(second)
    assert _poll_trace(live_backend, run_id, second) == second

    cell = dict(first.taxonomy_cell)
    assert run_id not in {t.trace_id for t in live_backend.query(frame="discovery", cell=cell)}
    reframed = live_backend.query(frame="production-representative", cell=cell)
    assert run_id in {t.trace_id for t in reframed}


def test_live_owned_trace_re_put_as_probed_stays_readable(live_backend: LangfuseBackend) -> None:
    run_id = f"review-2026-10-26-probed-{uuid.uuid4().hex[:8]}"
    owned = _remap_trace(_load_trace("ORD-1001"), run_id)
    probed = replace(owned, ownership="probed", spans=[])

    live_backend.put_trace(owned)
    assert _poll_trace(live_backend, run_id, owned) == owned
    live_backend.put_trace(probed)
    assert _poll_trace(live_backend, run_id, probed) == probed
    cell = dict(owned.taxonomy_cell)
    assert run_id in {t.trace_id for t in live_backend.query(cell=cell, ownership="probed")}


def _poll_scores(backend: LangfuseBackend, trace_id: str, want: int) -> list[Label]:
    scores: list[Label] = []
    for _ in range(30):
        scores = backend.get_scores(trace_id)
        if len(scores) == want:
            return scores
        time.sleep(1)
    return scores


def test_live_rater_ids_and_retraction_tombstones_against_real_langfuse(
    live_backend: LangfuseBackend,
) -> None:
    """On the real stack: two raters stay two labels, a retraction
    tombstone hides exactly the retracted one, and a re-put after it reads back again."""
    run_id = f"ORD-1002-rt-{uuid.uuid4().hex[:8]}"
    live_backend.put_trace(_remap_trace(_load_trace("ORD-1002"), run_id))
    alice = Label(
        trace_id=run_id, span_id=None, kind="human", outcome="PASS", critique="ok", rater_id="alice"
    )
    bob = replace(alice, rater_id="bob")
    live_backend.put_score(alice)
    live_backend.put_score(bob)
    got = _poll_scores(live_backend, run_id, 2)
    assert sorted(str(lbl.rater_id) for lbl in got) == ["alice", "bob"]

    live_backend.retract_score(bob)
    got = _poll_scores(live_backend, run_id, 1)
    assert got == [alice]

    live_backend.put_score(bob)
    got = _poll_scores(live_backend, run_id, 2)
    assert sorted(str(lbl.rater_id) for lbl in got) == ["alice", "bob"]

    # An undo pressed straight after a verdict: carol's label is not readable yet when it is
    # retracted, and must stay hidden once ingestion catches up.
    carol = replace(alice, rater_id="carol")
    live_backend.put_score(carol)
    live_backend.retract_score(carol)
    time.sleep(10)
    assert sorted(str(lbl.rater_id) for lbl in live_backend.get_scores(run_id)) == ["alice", "bob"]


def test_live_doctor_confirms_the_langfuse_keys(
    live_backend: LangfuseBackend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from calibrated_agent_evaluation.cli import main

    monkeypatch.setenv("LANGFUSE_HOST", os.environ.get("LANGFUSE_HOST", "http://localhost:3000"))
    monkeypatch.setenv(
        "LANGFUSE_PUBLIC_KEY",
        os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-dev"),
    )
    monkeypatch.setenv(
        "LANGFUSE_SECRET_KEY",
        os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-calibrated-agent-evaluation-dev"),
    )
    assert main(["--backend", "langfuse", "--root", str(tmp_path), "doctor"]) == 0
    assert "Langfuse backend: reachable, keys accepted" in capsys.readouterr().out


def test_live_a_score_the_server_rejects_raises_instead_of_reporting_success(
    live_backend: LangfuseBackend,
) -> None:
    """A label whose body the real server refuses must fail loudly, not vanish.

    The queued ``create_score`` path drops a rejected score in its background consumer and
    ``flush()`` returns normally, so the review UI would show a saved verdict that no longer
    exists. A critique over the server's 1 MB request limit is the cheapest real rejection.
    """
    run_id = f"review-2026-10-26-bigscore-{uuid.uuid4().hex[:8]}"
    live_backend.put_trace(_remap_trace(_load_trace("ORD-1002"), run_id))
    oversized = Label(
        trace_id=run_id,
        span_id=None,
        kind="human",
        outcome="FAIL",
        critique="x" * 2_000_000,
        rater_id="alice",
    )
    with pytest.raises(BackendError, match="put_score"):
        live_backend.put_score(oversized)
    time.sleep(5)
    assert live_backend.get_scores(run_id) == []


def test_a_trace_session_reads_back_and_groups_in_langfuse(live_backend: LangfuseBackend) -> None:
    run = uuid.uuid4().hex
    session = f"calibrated-agent-evaluation-{run}"
    turns = [
        replace(_remap_trace(_load_trace("ORD-1002"), f"ORD-1002-s{i}-{run}"), session_id=session)
        for i in range(2)
    ]
    for trace in turns:
        live_backend.put_trace(trace)

    for trace in turns:
        assert _poll_trace(live_backend, trace.trace_id, trace) == trace

    # Langfuse groups the two turns as one session (its native sessionId), not only the envelope.
    from langfuse.api import NotFoundError

    api = cast("Any", live_backend._client.api)
    want = {live_backend._client.create_trace_id(seed=t.trace_id) for t in turns}
    grouped: set[str] = set()
    last: NotFoundError | None = None
    for _ in range(30):
        try:
            grouped = {t.id for t in api.sessions.get(session).traces}
        except NotFoundError as exc:
            # The session appears once Langfuse has processed the traces; keep waiting.
            last = exc
        if grouped >= want:
            break
        time.sleep(1)
    assert grouped >= want, f"session {session} lists {grouped}, want {want}; last error: {last}"
