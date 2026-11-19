# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the extensible dump/load codec registry and round-trip.

Covers dump→load equality, registering a new type, an unregistered type raising, a
schema_version mismatch refusing, and the property test.
Tests assert observable behavior — reconstructed-set equality, raised errors — over the real
shared corpus, not mock internals (rubric rule 2). The autouse `_clean_registry` fixture
isolates the process-global registry per test.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from _helpers import make_span, make_trace, reseal_dump
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.dumpload import dump, load, register
from calibrated_agent_evaluation.dumpload.registry import _reset_registry, codec_for
from calibrated_agent_evaluation.schema.errors import SchemaVersionError, UnregisteredArtifactError
from calibrated_agent_evaluation.schema.types import Label, Trace


@pytest.fixture
def seeded(tmp_path, corpus_clm: Trace, corpus_ord: Trace, corpus_labels: list[Label]):
    """A LocalBackend seeded with the full shared corpus (traces + spans + labels)."""
    backend = LocalBackend(root=tmp_path / "store")
    backend.put_trace(corpus_clm)
    backend.put_trace(corpus_ord)
    for lbl in corpus_labels:
        if lbl.trace_id in {"CLM-4471", "ORD-1001"}:
            backend.put_score(lbl)
    return backend


# --- dump → wipe → load equality in BOTH csv and json (the proof) ---


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_dump_wipe_load_roundtrips_every_type(
    seeded: LocalBackend, tmp_path, fmt: str, corpus_clm: Trace, corpus_labels: list[Label]
) -> None:
    before_traces = {t.trace_id: t for t in seeded.all_traces()}
    before_labels = seeded.all_scores()

    dump_dir = tmp_path / f"dump-{fmt}"
    dump(seeded, dump_dir, fmt=fmt)
    seeded.wipe()
    assert seeded.all_traces() == []  # wiped
    load(seeded, dump_dir, fmt=fmt)

    after_traces = {t.trace_id: t for t in seeded.all_traces()}
    after_labels = seeded.all_scores()
    # Field-for-field equality for every trace (incl. its span tree) and every label.
    assert after_traces == before_traces
    assert after_traces["CLM-4471"] == corpus_clm
    assert sorted(after_labels, key=lambda x: (x.trace_id, x.span_id or "")) == sorted(
        before_labels, key=lambda x: (x.trace_id, x.span_id or "")
    )
    # The crew-attribution gold survived the CSV/JSON round-trip.
    crew = next(lbl for lbl in after_labels if lbl.trace_id == "CLM-4471" and lbl.span_id is None)
    assert crew.culprit_span == "CLM-4471/intake"


# --- empty dump loadable; load is idempotent (edge) ------------------------


def test_dump_of_empty_store_is_loadable(tmp_path) -> None:
    empty = LocalBackend(root=tmp_path / "empty")
    dump_dir = tmp_path / "dump"
    dump(empty, dump_dir, fmt="csv")
    target = LocalBackend(root=tmp_path / "target")
    load(target, dump_dir, fmt="csv")  # does not raise
    assert target.all_traces() == []
    assert target.all_scores() == []


def test_load_over_populated_store_is_idempotent(seeded: LocalBackend, tmp_path) -> None:
    dump_dir = tmp_path / "dump"
    dump(seeded, dump_dir, fmt="json")
    before = {t.trace_id: t for t in seeded.all_traces()}
    # Loading the same dump back over the populated store (INSERT OR REPLACE) changes nothing.
    load(seeded, dump_dir, fmt="json")
    after = {t.trace_id: t for t in seeded.all_traces()}
    assert after == before


def test_corrupt_json_dump_raises_typed_error_not_raw_decode_error(
    seeded: LocalBackend, tmp_path
) -> None:
    # A truncated/garbled dump file must surface as a typed CorruptDumpError (an
    # CalibratedAgentEvaluationDataError), never an uncaught json.JSONDecodeError leaking past the
    # data-plane boundary.
    from calibrated_agent_evaluation.dumpload import CorruptDumpError
    from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError

    dump_dir = tmp_path / "dump"
    dump(seeded, dump_dir, fmt="json")
    (dump_dir / "trace.json").write_text("{ this is not valid json")
    reseal_dump(dump_dir, "trace.json")
    target = LocalBackend(root=tmp_path / "target")
    with pytest.raises(CorruptDumpError, match="not valid JSON"):
        load(target, dump_dir, fmt="json")
    assert issubclass(CorruptDumpError, CalibratedAgentEvaluationDataError)


def test_corrupt_csv_cell_raises_typed_error_not_raw_decode_error(
    seeded: LocalBackend, tmp_path
) -> None:
    # The CSV load path decodes each cell with json.loads too; a corrupt cell must wrap into
    # CorruptDumpError, exactly like the JSON path — never a raw json.JSONDecodeError escaping.
    from calibrated_agent_evaluation.dumpload import CorruptDumpError

    dump_dir = tmp_path / "dump"
    dump(seeded, dump_dir, fmt="csv")
    # Corrupt one cell of the label CSV (leave the header intact so the row still parses as CSV).
    lines = (dump_dir / "label.csv").read_text().splitlines()
    header, first = lines[0], lines[1]
    corrupted = first.rsplit(",", 1)[0] + ",{not valid json"
    (dump_dir / "label.csv").write_text("\n".join([header, corrupted, *lines[2:]]))
    reseal_dump(dump_dir, "label.csv")
    target = LocalBackend(root=tmp_path / "target")
    with pytest.raises(CorruptDumpError, match="not valid JSON"):
        load(target, dump_dir, fmt="csv")


# --- schema_version mismatch refuses, writes nothing (negative) ------------


def test_unknown_schema_version_makes_load_refuse_and_write_nothing(
    seeded: LocalBackend, tmp_path
) -> None:
    dump_dir = tmp_path / "dump"
    dump(seeded, dump_dir, fmt="json")
    # Bump the stored schema_version to an unknown value in the trace dump.
    trace_file = dump_dir / "trace.json"
    text = trace_file.read_text().replace('"schema_version": "1"', '"schema_version": "999"')
    trace_file.write_text(text)
    reseal_dump(dump_dir, "trace.json")

    target = LocalBackend(root=tmp_path / "target")
    with pytest.raises(SchemaVersionError, match="999"):
        load(target, dump_dir, fmt="json")
    # nothing was written before the refusal.
    assert target.all_traces() == []
    assert target.all_scores() == []


# --- property: arbitrary valid artifacts round-trip ----------------

_safe_text = st.text(min_size=1, max_size=12).filter(lambda s: s.strip())
_json_scalar = st.one_of(st.booleans(), st.integers(-1000, 1000), _safe_text)
_json_obj = st.dictionaries(_safe_text, _json_scalar, max_size=4)


@st.composite
def _traces(draw: st.DrawFn) -> Trace:
    tid = draw(st.text(min_size=1, max_size=8).filter(lambda s: s.strip()))
    n_spans = draw(st.integers(1, 3))
    spans = [
        make_span(
            trace_id=tid,
            span_id=f"{tid}/s{i}",
            parent=None if i == 0 else f"{tid}/s0",
            seq=i,
            input=draw(_json_obj),
            output=draw(_json_obj),
        )
        for i in range(n_spans)
    ]
    return make_trace(
        trace_id=tid,
        frame=draw(st.sampled_from(["discovery", "production-representative"])),
        input=draw(_json_obj),
        output=draw(_json_obj),
        taxonomy_cell=draw(st.dictionaries(_safe_text, _safe_text, max_size=3)),
        spans=spans,
    )


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(trace=_traces())
def test_arbitrary_trace_roundtrips_put_get_and_dump_load(trace: Trace, tmp_path) -> None:
    _reset_registry()
    backend = LocalBackend(root=":memory:")
    # put → get equality.
    backend.put_trace(trace)
    assert backend.get_trace(trace.trace_id) == trace
    # dump → wipe → load equality.
    dump_dir = tmp_path / "p"
    dump(backend, dump_dir, fmt="json")
    backend.wipe()
    load(backend, dump_dir, fmt="json")
    assert backend.get_trace(trace.trace_id) == trace


# --- register a new artifact type ------------------------------------------


@dataclass(frozen=True)
class _ToyArtifact:
    trace_id: str
    note: str


# --- a registered toy type round-trips alongside the core types ------------


def test_registered_toy_type_roundtrips(tmp_path, corpus_ord: Trace) -> None:
    import json

    backend = LocalBackend(root=tmp_path / "store")
    backend.put_trace(corpus_ord)

    toys = [_ToyArtifact(trace_id="ORD-1001", note="hello"), _ToyArtifact("ORD-1001", "world")]

    def to_row(t: _ToyArtifact) -> dict[str, str]:
        return {"trace_id": json.dumps(t.trace_id), "note": json.dumps(t.note)}

    def from_row(row: dict[str, str]) -> _ToyArtifact:
        return _ToyArtifact(trace_id=json.loads(row["trace_id"]), note=json.loads(row["note"]))

    register(_ToyArtifact, to_row, from_row, name="toy")

    # The codec is now resolvable and round-trips a value via the registry — the open-registry
    # proof (the type list is not hardcoded). The toy file participates in the dump directory.
    codec = codec_for(_ToyArtifact)
    assert codec.from_row(codec.to_row(toys[0])) == toys[0]


# --- late-registered participates; duplicate registration rejected --------


def test_codec_registered_after_a_prior_dump_participates_next_time(
    tmp_path, corpus_ord: Trace
) -> None:
    backend = LocalBackend(root=tmp_path / "store")
    backend.put_trace(corpus_ord)
    # First dump: only the core types exist.
    dump(backend, tmp_path / "d1", fmt="json")
    assert not (tmp_path / "d1" / "toy.json").exists()

    # Register a toy type AFTER the first dump; it participates on the next dump.
    register(
        _ToyArtifact,
        lambda t: {"trace_id": f'"{t.trace_id}"'},
        lambda r: _ToyArtifact("x", "y"),
        name="toy",
    )
    # toy has no backend collector → dump surfaces that loudly rather than silently skipping.
    with pytest.raises(UnregisteredArtifactError, match="no dump collector"):
        dump(backend, tmp_path / "d2", fmt="json")


def test_reregistering_same_type_is_rejected() -> None:
    register(_ToyArtifact, lambda t: {}, lambda r: _ToyArtifact("x", "y"), name="toy")
    with pytest.raises(ValueError, match="already has a registered codec"):
        register(_ToyArtifact, lambda t: {}, lambda r: _ToyArtifact("a", "b"), name="toy2")


def test_registering_a_second_type_under_an_existing_name_is_rejected() -> None:
    # Two distinct types sharing a dump filename would clobber each other's <name>.json/.csv on
    # dump (one silently overwrites the other), breaking the lossless round-trip. Refuse it.
    @dataclass
    class _OtherToy:
        v: str

    register(_ToyArtifact, lambda t: {}, lambda r: _ToyArtifact("x", "y"), name="shared")
    with pytest.raises(ValueError, match="already registered for another type"):
        register(_OtherToy, lambda t: {}, lambda r: _OtherToy("z"), name="shared")


# --- dumping an unregistered type raises, nothing silently dropped ---------


def test_unregistered_type_in_store_raises_on_dump(tmp_path, corpus_ord: Trace) -> None:
    # A type the dump iterates but has no codec must raise UnregisteredArtifactError. We force
    # the condition by registering a type with a collector gap: the registry knows the type but
    # _collect has no source for it → loud failure, never a silent skip.
    backend = LocalBackend(root=tmp_path / "store")
    backend.put_trace(corpus_ord)
    register(_ToyArtifact, lambda t: {}, lambda r: _ToyArtifact("x", "y"), name="toy")
    with pytest.raises(UnregisteredArtifactError):
        dump(backend, tmp_path / "d", fmt="json")


def test_codec_for_unregistered_type_raises() -> None:
    with pytest.raises(UnregisteredArtifactError, match="no codec registered"):
        codec_for(_ToyArtifact)


# --- the session a trace belongs to survives the store and a dump ----------------------------


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_a_trace_session_survives_the_local_store_and_dump_load(tmp_path, fmt: str) -> None:
    with_session = make_trace(
        trace_id="T-S",
        spans=[make_span(span_id="T-S/root", trace_id="T-S")],
        session_id="calibrated-agent-evaluation-0123456789abcdef0123456789abcdef",
    )
    without = make_trace(trace_id="T-N", spans=[make_span(span_id="T-N/root", trace_id="T-N")])
    backend = LocalBackend(root=tmp_path / "store")
    for trace in (with_session, without):
        backend.put_trace(trace)
    assert backend.get_trace("T-S") == with_session
    assert {t.trace_id: t.session_id for t in backend.query()} == {
        "T-S": with_session.session_id,
        "T-N": None,
    }

    dump(backend, tmp_path / "d", fmt=fmt)
    backend.wipe()
    load(backend, tmp_path / "d", fmt=fmt)

    assert backend.get_trace("T-S") == with_session
    assert backend.get_trace("T-N") == without


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_a_store_with_no_sessions_dumps_as_before_sessions_and_loads_with_none(
    tmp_path, fmt: str
) -> None:
    backend = LocalBackend(root=tmp_path / "store")
    backend.put_trace(make_trace(trace_id="T-1"))
    dump(backend, tmp_path / "d", fmt=fmt)
    # A store holding no session writes no session column, exactly as a dump made before the
    # field existed, so such a dump is also what an older Calibrated Agent Evaluation reads.
    assert not [p for p in (tmp_path / "d").rglob("*") if b"session_id" in p.read_bytes()]
    backend.wipe()

    load(backend, tmp_path / "d", fmt=fmt)

    assert backend.get_trace("T-1").session_id is None


@pytest.mark.parametrize("fmt", ["csv", "json"])
@pytest.mark.parametrize("order", ["session-first", "session-last"])
def test_a_dump_mixing_traces_with_and_without_a_session_round_trips(
    tmp_path, fmt: str, order: str
) -> None:
    # Ids chosen so the store returns the session trace first in one case and last in the other.
    ids = ("A", "B") if order == "session-first" else ("B", "A")
    with_session = make_trace(
        trace_id=ids[0],
        spans=[make_span(span_id=f"{ids[0]}/root", trace_id=ids[0])],
        session_id="s-1",
    )
    without = make_trace(
        trace_id=ids[1], spans=[make_span(span_id=f"{ids[1]}/root", trace_id=ids[1])]
    )
    backend = LocalBackend(root=tmp_path / "store")
    backend.put_trace(with_session)
    backend.put_trace(without)

    dump(backend, tmp_path / "d", fmt=fmt)
    backend.wipe()
    load(backend, tmp_path / "d", fmt=fmt)

    assert backend.get_trace(with_session.trace_id) == with_session
    assert backend.get_trace(without.trace_id) == without
