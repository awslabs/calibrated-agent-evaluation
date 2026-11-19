# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LocalBackend (stdlib sqlite3 + CSV/JSON, offline default).

Covers put→get equality and query, the no-config local default, filesystem-only access with no
network, and refusing an invalid artifact with nothing persisted.
Tests assert observable behavior — reconstructed values and raised errors — never sqlite
internals or call counts (rubric rule 2). They exercise the real shared corpus traces so
the round-trip is proven on the genuine OrderBot/ClaimsCrew shapes (the acceptance oracle).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _helpers import make_label, make_span, make_trace
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.backends.factory import get_backend
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.schema.errors import BackendError, SchemaError
from calibrated_agent_evaluation.schema.types import Label, Trace


@pytest.fixture
def backend(tmp_path) -> LocalBackend:
    return LocalBackend(root=tmp_path / "store")


# --- put_trace → get_trace equality + query subset ------------------------


def test_owned_two_span_trace_roundtrips_field_for_field(
    backend: LocalBackend, corpus_clm: Trace
) -> None:
    # CLM-4471 is the 5-span ClaimsCrew tree (supervisor + 4 specialists).
    backend.put_trace(corpus_clm)
    got = backend.get_trace("CLM-4471")
    assert got == corpus_clm  # frozen dataclasses compare field-for-field, spans included
    assert [s.seq for s in got.spans] == [0, 1, 2, 3, 4]
    assert got.spans[1].tool_calls[0].name == "extract_claim_fields"


def test_per_span_token_usage_survives_the_store(backend: LocalBackend, corpus_clm: Trace) -> None:
    """`usage` is a nested optional, so it needs its own reverse mapping in serde — and the
    field-for-field test above cannot catch a missing one, because the frozen normalized fixture
    carries no usage and `None == None` either way."""
    from calibrated_agent_evaluation._model import replace
    from calibrated_agent_evaluation.schema.types import TokenUsage

    spans = list(corpus_clm.spans)
    spans[0] = replace(spans[0], usage=TokenUsage(input_tokens=89, output_tokens=17))
    stored = replace(corpus_clm, spans=spans)

    backend.put_trace(stored)
    got = backend.get_trace("CLM-4471")

    assert got == stored
    assert got.spans[0].usage == TokenUsage(input_tokens=89, output_tokens=17)
    # An absent record stays absent rather than becoming an empty one: "the source reported
    # nothing" and "the source reported no numbers" are different facts.
    assert got.spans[1].usage is None


def test_the_occurrence_of_a_re_executed_node_survives_the_store(
    backend: LocalBackend, corpus_clm: Trace
) -> None:
    """`occurrence` answers "which attempt" for a cyclic topology, so losing it in storage would
    make two attempts of one agent indistinguishable again — the thing the unique span id fixed."""
    from calibrated_agent_evaluation._model import replace

    spans = list(corpus_clm.spans)
    spans[1] = replace(spans[1], span_id=f"{spans[1].span_id}#2", occurrence=2)
    stored = replace(corpus_clm, spans=spans)

    backend.put_trace(stored)
    got = backend.get_trace("CLM-4471")

    assert got == stored
    assert got.spans[1].occurrence == 2
    assert got.spans[0].occurrence == 1


def test_label_with_crew_attribution_roundtrips(
    backend: LocalBackend, corpus_clm: Trace, corpus_labels: list[Label]
) -> None:
    # Critic warning honored: assert the attribution clause survives the round-trip,
    # not just outcome/critique.
    backend.put_trace(corpus_clm)
    crew_label = next(
        lbl for lbl in corpus_labels if lbl.trace_id == "CLM-4471" and lbl.span_id is None
    )
    backend.put_score(crew_label)
    [got] = backend.get_scores("CLM-4471")
    assert got == crew_label
    assert got.culprit_span == "CLM-4471/intake"
    assert got.manifestation_span == "CLM-4471/payout"
    assert got.handoff_implicated == "intake->policy-checker"
    assert got.codebook_version == "v1"
    assert got.expected == {
        "decision": "DENIED",
        "amount_paid_usd": 0,
        "basis": "exclusion W-7 (dwelling vacant > 30 days)",
    }


def test_query_returns_only_the_matching_subset(backend: LocalBackend) -> None:
    discovery = make_trace(
        trace_id="D-1",
        frame="discovery",
        spans=[make_span(trace_id="D-1", span_id="D-1/root")],
        taxonomy_cell={"feature": "refund"},
    )
    prod = make_trace(
        trace_id="P-1",
        frame="production-representative",
        spans=[make_span(trace_id="P-1", span_id="P-1/root")],
        taxonomy_cell={"feature": "claims"},
    )
    backend.put_trace(discovery)
    backend.put_trace(prod)
    assert [t.trace_id for t in backend.query(frame="discovery")] == ["D-1"]
    assert [t.trace_id for t in backend.query(cell={"feature": "claims"})] == ["P-1"]
    assert [t.trace_id for t in backend.query(ownership="owned")] == ["D-1", "P-1"]


def test_get_scores_can_narrow_to_one_span(
    backend: LocalBackend, corpus_clm: Trace, corpus_labels: list[Label]
) -> None:
    backend.put_trace(corpus_clm)
    for lbl in corpus_labels:
        if lbl.trace_id == "CLM-4471":
            backend.put_score(lbl)
    span_scoped = backend.get_scores("CLM-4471", span_id="CLM-4471/intake")
    assert len(span_scoped) == 1
    assert span_scoped[0].span_defect is True


# --- absent id raises; empty query returns [] (negative + edge) -----------


def test_get_trace_on_absent_id_raises_backend_error(backend: LocalBackend) -> None:
    with pytest.raises(BackendError, match="no trace with trace_id"):
        backend.get_trace("does-not-exist")


def test_local_backend_is_a_context_manager_that_closes_its_connection(tmp_path) -> None:
    # The backend owns a sqlite connection; `with` must open and reliably close it.
    with LocalBackend(root=tmp_path / "store") as backend:
        backend.put_trace(
            make_trace(trace_id="C-1", spans=[make_span(trace_id="C-1", span_id="C-1/r")])
        )
        assert backend.get_trace("C-1").trace_id == "C-1"
    # After __exit__ the connection is closed: using it raises sqlite's ProgrammingError.
    import sqlite3

    with pytest.raises(sqlite3.ProgrammingError):
        backend.get_trace("C-1")


def test_query_matching_nothing_returns_empty_list(backend: LocalBackend) -> None:
    backend.put_trace(
        make_trace(
            trace_id="D-1",
            frame="discovery",
            spans=[make_span(trace_id="D-1", span_id="D-1/root")],
        )
    )
    assert backend.query(frame="production-representative") == []
    assert backend.get_scores("D-1") == []


# --- empty trace_id refused, store unchanged (negative) -------------------


def test_put_trace_with_empty_trace_id_raises_and_persists_nothing(
    backend: LocalBackend,
) -> None:
    good = make_trace(trace_id="OK-1", spans=[make_span(trace_id="OK-1", span_id="OK-1/root")])
    backend.put_trace(good)
    before = len(backend.all_traces())
    bad = replace(good, trace_id="   ")
    with pytest.raises(SchemaError, match="trace_id"):
        backend.put_trace(bad)
    assert len(backend.all_traces()) == before  # row count unchanged


def test_put_score_with_empty_trace_id_raises_and_persists_nothing(
    backend: LocalBackend,
) -> None:
    ok = make_trace(trace_id="OK-1", spans=[make_span(trace_id="OK-1", span_id="OK-1/r")])
    backend.put_trace(ok)
    before = len(backend.all_scores())
    with pytest.raises(SchemaError, match="trace_id"):
        backend.put_score(make_label(trace_id=""))
    assert len(backend.all_scores()) == before


# --- no backend configured → factory yields LocalBackend, full sequence ---


def test_unconfigured_factory_yields_local_backend_and_full_sequence(tmp_path) -> None:
    be = get_backend(local_root=tmp_path / "store")
    assert isinstance(be, LocalBackend)
    trace = make_trace(trace_id="L-1", spans=[make_span(trace_id="L-1", span_id="L-1/root")])
    be.put_trace(trace)
    be.put_score(make_label(trace_id="L-1", outcome="PASS", critique="fine"))
    assert be.get_trace("L-1") == trace
    assert [t.trace_id for t in be.query(frame="discovery")] == ["L-1"]
    assert len(be.get_scores("L-1")) == 1


# --- configured backend not overridden; bad path raises (negative/edge) --


def test_configured_backend_is_returned_and_not_overridden_by_local(tmp_path) -> None:
    configured = LocalBackend(root=tmp_path / "explicit")
    assert get_backend(configured, local_root=tmp_path / "ignored") is configured


def test_unwritable_local_root_raises_rather_than_silently_no_op(tmp_path) -> None:
    # A regular file where the store directory should be → mkdir fails loudly (no silent no-op).
    clash = tmp_path / "not-a-dir"
    clash.write_text("i am a file, not a directory")
    with pytest.raises(OSError, match="not-a-dir"):
        get_backend(local_root=clash)


# --- local backend works with the network hard-disabled -------------------


def test_full_local_sequence_succeeds_with_sockets_blocked(monkeypatch, tmp_path) -> None:
    # Hard-disable the network: any attempt to construct a socket raises. The full local
    # put→get→query→dump→load sequence must still succeed touching only the filesystem.
    import socket as socket_mod

    from calibrated_agent_evaluation.dumpload import dump, load

    def _no_network(*args, **kwargs):
        raise OSError("network disabled for the offline test")

    monkeypatch.setattr(socket_mod, "socket", _no_network)

    backend = LocalBackend(root=tmp_path / "store")
    trace = make_trace(trace_id="OFF-1", spans=[make_span(trace_id="OFF-1", span_id="OFF-1/r")])
    backend.put_trace(trace)
    backend.put_score(make_label(trace_id="OFF-1", outcome="PASS", critique="ok offline"))
    assert backend.get_trace("OFF-1") == trace
    assert [t.trace_id for t in backend.query(frame="discovery")] == ["OFF-1"]
    dump_dir = tmp_path / "dump"
    dump(backend, dump_dir, fmt="json")
    backend.wipe()
    load(backend, dump_dir, fmt="json")
    assert backend.get_trace("OFF-1") == trace


# --- re-putting a label must not double it -----------------------------------


def test_put_score_twice_with_the_same_label_reads_back_once(tmp_path: Path) -> None:
    """Both score-write paths are append-only, so a re-put left two rows for one label.

    `labels.rowid_pk` is a plain autoincrement primary key — nothing keyed a label by identity — so
    re-labelling a trace (a dump→load round trip, a warm-cache judge re-run, a re-scored slice)
    doubled its labels. Every rate computed over them moved toward whichever verdict was written
    twice, and a CI gate reading that rate flipped with no error anywhere.
    """
    backend = LocalBackend(root=str(tmp_path / "store"))
    label = make_label(trace_id="D-1", kind="human", outcome="FAIL", critique="wrong tool")

    backend.put_score(label)
    backend.put_score(label)

    assert len(backend.get_scores("D-1")) == 1
    assert len(backend.all_scores()) == 1, "dump must not carry the duplicate either"


def test_labels_that_genuinely_differ_are_all_kept(tmp_path: Path) -> None:
    """The boundary the dedup must not cross, and the one a coarse key got wrong.

    Identity is the label's full content, not `(trace_id, span_id, kind)`. A key that coarse
    collapsed 120 distinct human labels on one trace into one — trading a double-count for a silent
    drop, which is the same defect wearing the fix's clothes.
    """
    backend = LocalBackend(root=str(tmp_path / "store"))
    for i in range(5):
        backend.put_score(
            make_label(trace_id="D-2", kind="human", outcome="FAIL", critique=f"reason {i}")
        )

    assert len(backend.get_scores("D-2")) == 5


def test_a_changed_verdict_for_the_same_target_is_kept(tmp_path: Path) -> None:
    """Two different verdicts are two labels, not a duplicate — the store has no basis to pick."""
    backend = LocalBackend(root=str(tmp_path / "store"))
    backend.put_score(make_label(trace_id="D-3", kind="human", outcome="PASS", critique="fine"))
    backend.put_score(make_label(trace_id="D-3", kind="human", outcome="FAIL", critique="fine"))

    outcomes = sorted(lbl.outcome for lbl in backend.get_scores("D-3"))
    assert outcomes == ["FAIL", "PASS"]


def test_a_dump_load_round_trip_does_not_grow_the_label_count(tmp_path: Path) -> None:
    """The concrete path the ticket named: `dumpload.load` re-puts every label it reads."""
    backend = LocalBackend(root=str(tmp_path / "store"))
    label = make_label(trace_id="D-4", kind="human", outcome="FAIL", critique="wrong tool")
    backend.put_score(label)

    for round_trip in backend.all_scores():
        backend.put_score(round_trip)

    assert len(backend.get_scores("D-4")) == 1


# --- a label can be retracted from the store ------------------------------


def test_retract_score_removes_exactly_that_label(tmp_path: Path) -> None:
    backend = LocalBackend(root=str(tmp_path / "store"))
    keep = make_label(trace_id="U-1", kind="human", outcome="FAIL", critique="kept")
    gone = make_label(trace_id="U-1", kind="human", outcome="PASS", critique="retracted")
    backend.put_score(keep)
    backend.put_score(gone)
    backend.put_score(gone)  # an idempotent re-put is still one label, and retracts as one

    backend.retract_score(gone)

    assert backend.get_scores("U-1") == [keep]
    assert backend.all_scores() == [keep]


def test_a_label_re_put_after_its_retraction_is_stored_again(tmp_path: Path) -> None:
    backend = LocalBackend(root=str(tmp_path / "store"))
    label = make_label(trace_id="U-2", kind="human", outcome="PASS", critique="same words")
    backend.put_score(label)
    backend.retract_score(label)
    backend.put_score(label)
    assert backend.get_scores("U-2") == [label]


def test_retracting_a_label_that_is_not_stored_changes_nothing(tmp_path: Path) -> None:
    """Parity with Langfuse, whose asynchronous ingestion cannot tell absent from not-yet-read."""
    backend = LocalBackend(root=str(tmp_path / "store"))
    stored = make_label(trace_id="U-3", critique="stored")
    backend.put_score(stored)
    backend.retract_score(make_label(trace_id="U-3", critique="never written"))
    assert backend.get_scores("U-3") == [stored]
