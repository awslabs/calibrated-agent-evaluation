# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The nine lower-severity silent-degrade / swallowed-error / lifecycle items.

Each is small on its own, which is why they were one checklist rather than nine tickets — but each
is a real silent behaviour change, a swallowed error, or a resource/lifecycle fault. One test per
item, named for the item, so the checklist is verified rather than asserted to be done.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

# --- item 1: multi-block assistant text joins with a newline, not nothing -------------------------


def test_item1_multi_block_text_joins_with_a_newline() -> None:
    """`"".join` fused separate blocks: "...ordersHere is..." — a word in no source document.

    That string reached the judge prompt and the review UI, so a human or a model read text the
    agent never produced.
    """
    from calibrated_agent_evaluation.normalize.content import as_object

    blocks = [{"text": "Checked your orders"}, {"text": "Here is the refund"}]
    assert as_object(blocks) == {"answer": "Checked your orders\nHere is the refund"}
    assert "ordersHere" not in str(as_object(blocks)), "the fused word must be gone"


def test_item1_a_single_block_gains_no_separator() -> None:
    """The join must not add a trailing or leading newline to the common one-block case."""
    from calibrated_agent_evaluation.normalize.content import as_object

    assert as_object([{"text": "just one"}]) == {"answer": "just one"}


# --- item 2: an absent schema_version is absent, not "the running one" ----------------------------


def test_item2_a_trace_dict_without_schema_version_is_refused() -> None:
    """`data.get("schema_version", SCHEMA_VERSION)` stamped the running version onto a bare dict.

    Which contradicts `types.SCHEMA_VERSION`'s own contract — "`load` refuses on mismatch (no silent
    coercion)" — by treating "no version at all" as "already matching".
    """
    from calibrated_agent_evaluation.schema.errors import SchemaError
    from calibrated_agent_evaluation.schema.serde import trace_from_dict

    doc = {
        "trace_id": "T-1",
        "frame": "discovery",
        "ownership": "probed",
        "input": {},
        "output": {},
        "taxonomy_cell": {},
        "spans": [],
    }
    with pytest.raises(SchemaError, match="no schema_version"):
        trace_from_dict(doc)


def test_item2_a_trace_dict_with_a_schema_version_round_trips() -> None:
    """Regression guard: every writer stamps it, so the normal path must be untouched."""
    from calibrated_agent_evaluation.schema.serde import to_dict, trace_from_dict
    from calibrated_agent_evaluation.schema.types import SCHEMA_VERSION, Trace

    original = Trace(
        trace_id="T-2",
        frame="discovery",
        ownership="probed",
        input={},
        output={},
        taxonomy_cell={},
    )
    restored = trace_from_dict(to_dict(original))
    assert restored.schema_version == SCHEMA_VERSION
    assert restored == original


# --- item 3: a parent with a higher seq than its child must not be re-parented silently -----------


def test_item3_out_of_order_seq_does_not_reparent_the_child(monkeypatch: Any) -> None:
    """`sorted(spans, key=seq)` assumed a parent's seq is always lower than its children's.

    Nothing enforces that — `seq` is execution order, not depth — so an out-of-order seq left the
    parent's Langfuse id unknown when the child was written, and the child was silently re-parented
    to the trace root. A changed topology, and the topology is what crew attribution reads.
    """
    from calibrated_agent_evaluation.backends.langfuse import _parents_first
    from calibrated_agent_evaluation.schema.types import Span

    def _span(span_id: str, parent: str | None, seq: int) -> Span:
        return Span(
            span_id=span_id,
            trace_id="T",
            parent=parent,
            seq=seq,
            agent=span_id,
            role="specialist",
            status="completed",
            input={},
        )

    # The child has the LOWER seq, so seq order would write it before its parent.
    child = _span("T/child", "T/root", 0)
    root = _span("T/root", None, 1)

    order = [s.span_id for s in _parents_first([child, root])]
    assert order == ["T/root", "T/child"], "a parent must be written before its child"


def test_item3_a_dangling_parent_is_refused_not_flattened() -> None:
    """The unresolvable case: refuse, rather than quietly attach it to the trace root."""
    from calibrated_agent_evaluation.backends.langfuse import (
        _parents_first,
        _refuse_unresolved_parents,
    )
    from calibrated_agent_evaluation.schema.errors import BackendError
    from calibrated_agent_evaluation.schema.types import Span

    orphan = Span(
        span_id="T/orphan",
        trace_id="T",
        parent="T/missing",
        seq=0,
        agent="orphan",
        role="specialist",
        status="completed",
        input={},
    )
    with pytest.raises(BackendError, match="silently change the topology"):
        _refuse_unresolved_parents("T", _parents_first([orphan]))


# --- item 4: re-publishing a dashboard run must not duplicate every trend point -------------------


def test_item4_duplicate_trend_points_are_collapsed_before_publishing() -> None:
    """`create_score` is append-only and the trace id is seeded on run_id.

    So a re-publish appended a second copy of every point and the chart showed each value twice.
    """
    from calibrated_agent_evaluation.backends.dashboard import _dedup_points
    from calibrated_agent_evaluation.run.report import NumericPoint

    points = [
        NumericPoint(name="corrected_rate", value=0.12, kind="rate", run_id="r1"),
        NumericPoint(name="corrected_rate", value=0.12, kind="rate", run_id="r1"),
        NumericPoint(name="drift", value=0.03, kind="drift", run_id="r1"),
    ]
    deduped = _dedup_points(points)
    assert [(p.name, p.kind) for p in deduped] == [("corrected_rate", "rate"), ("drift", "drift")]


def test_item4_distinct_metrics_are_all_kept() -> None:
    """The boundary: two differently-named points are two points, not a duplicate."""
    from calibrated_agent_evaluation.backends.dashboard import _dedup_points
    from calibrated_agent_evaluation.run.report import NumericPoint

    points = [
        NumericPoint(name="rate_a", value=0.1, kind="rate", run_id="r1"),
        NumericPoint(name="rate_b", value=0.2, kind="rate", run_id="r1"),
    ]
    assert len(_dedup_points(points)) == 2


# --- item 5: a recorded response must not be lost after the paid call ----------------------------


def test_item5_an_unserializable_response_raises_typed_not_raw_typeerror(tmp_path: Path) -> None:
    """`_flush` used `json.dumps` with no `default=`; `fingerprint` passes `default=str`.

    So a recordable-but-not-serializable response raised a bare `TypeError` **after** the paid live
    call, and the response was never written — the next run pays again. `default=str` matches the
    encoder the fingerprint already uses, so the value is recorded rather than lost.
    """
    from calibrated_agent_evaluation.execute.cassette import Cassette, Mode

    path = tmp_path / "c.json"
    cassette = Cassette(path, mode=Mode.RECORD)
    cassette.record("fp1", {"req": 1}, {"when": object()})

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert len(saved["entries"]) == 1, "the paid response was recorded, not dropped"


def test_item5_a_write_failure_surfaces_as_cassette_error(tmp_path: Path) -> None:
    """`_load` wrapped OSError; the write path did not, so a full disk escaped the boundary."""
    from calibrated_agent_evaluation.execute.cassette import Cassette, CassetteError, Mode

    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")

    cassette = Cassette(blocker / "sub" / "c.json", mode=Mode.RECORD)
    with pytest.raises(CassetteError, match="failed to write cassette"):
        cassette.record("fp1", {"req": 1}, {"ok": True})


# --- item 6: a permanent config fault must not be retried as a transient model failure ------------


def test_item6_calling_the_judge_inside_an_event_loop_refuses_immediately() -> None:
    """`asyncio.run` raises RuntimeError in a running loop, which became a retried ExecutionError.

    `judges/runtime._judge_one` treats any `ExecutionError` as transient, so a permanent
    configuration fault burned the retry budget and then reported "model failed after 4 attempts",
    pointing the reader at the model instead of the call site.
    """
    pytest.importorskip("inspect_ai")
    from calibrated_agent_evaluation.execute.errors import JudgeConfigurationError
    from calibrated_agent_evaluation.execute.judge_model import InspectJudgeModel

    model = InspectJudgeModel(model="mockllm/model")

    async def _inside_a_loop() -> None:
        model.generate_structured(system="s", user="u", schema={"type": "object"})

    with pytest.raises(JudgeConfigurationError, match="running event loop"):
        asyncio.run(_inside_a_loop())


def test_item6_the_config_error_is_not_retried() -> None:
    """The half that matters: the retry loop must let it through on the first attempt."""
    from _judge_helpers import certified_record, judge
    from calibrated_agent_evaluation.execute.errors import JudgeConfigurationError
    from calibrated_agent_evaluation.judges.errors import JudgeRuntimeError
    from calibrated_agent_evaluation.judges.runtime import JudgeTarget, run_judge

    attempts = {"n": 0}

    class _AlwaysMisconfigured:
        def supports_response_schema(self) -> bool:
            return True

        def generate_structured(self, **_: Any) -> Any:
            attempts["n"] += 1
            raise JudgeConfigurationError("called from inside a running event loop")

    j = judge()
    with pytest.raises(JudgeConfigurationError):
        run_judge(
            j,
            [JudgeTarget(trace_id="T", span_id=None, content="c")],
            calib_record=certified_record(j),
            model_backend=_AlwaysMisconfigured(),  # type: ignore[arg-type]
            max_retries=3,
        )
    assert attempts["n"] == 1, "a permanent fault must not consume the retry budget"
    assert JudgeRuntimeError is not JudgeConfigurationError


# --- item 7: get_backend's connection lifetime, and an empty agreement pairing --------------------


def test_item7_backend_session_closes_the_backend_it_created(tmp_path: Path) -> None:
    """`get_backend()` returns a LocalBackend holding an open sqlite connection.

    A caller that neither uses it as a context manager nor calls close() leaks one per call, so a
    block-scoped form is provided and documented.
    """
    from calibrated_agent_evaluation.backends import backend_session
    from calibrated_agent_evaluation.backends.local import LocalBackend

    with backend_session(local_root=str(tmp_path / "store")) as backend:
        assert isinstance(backend, LocalBackend)
        backend.query()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        backend.query()


def test_item7_backend_session_does_not_close_a_caller_supplied_backend(tmp_path: Path) -> None:
    """Closing someone else's handle is the mirror-image bug of leaking your own."""
    from calibrated_agent_evaluation.backends import backend_session
    from calibrated_agent_evaluation.backends.local import LocalBackend

    caller_owned = LocalBackend(root=str(tmp_path / "store"))
    with backend_session(caller_owned) as backend:
        assert backend is caller_owned
    assert caller_owned.query() == [], "still usable: the caller still owns it"
    caller_owned.close()


def test_item7_empty_agreement_pairing_raises_typed_not_zerodivisionerror() -> None:
    """`agreement_display` is public and divided by `len(gold)` unguarded.

    An empty pairing raised a bare `ZeroDivisionError` out of the enforcement plane. There is also
    no honest number: 0.0 and 1.0 are both claims about agreement that was never measured.
    """
    from calibrated_agent_evaluation.enforce.display import agreement_display
    from calibrated_agent_evaluation.enforce.errors import EnforcementError

    with pytest.raises(EnforcementError, match="empty"):
        agreement_display([], [])


# --- item 8: two runs sharing a log dir must not resolve to each other's logs ---------------------


def test_item8_an_untitled_strands_run_gets_a_unique_export_name(tmp_path: Path) -> None:
    """`task.trace_id or "strands-run"` made two untitled runs overwrite each other's raw export.

    And a run cannot be repeated to recover what was replaced.
    """
    pytest.importorskip("strands")
    from calibrated_agent_evaluation.execute.strands_executor import StrandsExecutor, StrandsTask

    executor = StrandsExecutor(model_id="m", region="r")
    task = StrandsTask(build=lambda _m: object(), input="x", mode="latest")

    first = executor._write_export([], tmp_path, task)
    second = executor._write_export([], tmp_path, task)
    assert first != second, "two untitled runs must not share an export path"
    assert first.exists()
    assert second.exists()


def test_item8_an_ambiguous_inspect_log_dir_is_refused(tmp_path: Path) -> None:
    """The mtime fallback could resolve a DIFFERENT run's log when a log_dir holds several.

    The caller would then normalize the wrong transcript with no error, so ambiguity is refused.
    """
    pytest.importorskip("inspect_ai")
    from calibrated_agent_evaluation.execute.errors import ExecutionError
    from calibrated_agent_evaluation.execute.inspect_executor import InspectExecutor

    (tmp_path / "a.eval").touch()
    (tmp_path / "b.eval").touch()

    executor = InspectExecutor(model="mockllm/model", log_dir=tmp_path)

    class _NoLocation:
        location = None

    with pytest.raises(ExecutionError, match="ambiguous"):
        executor._resolve_log_path(_NoLocation(), tmp_path)


def test_item8_a_single_inspect_log_still_resolves(tmp_path: Path) -> None:
    """Regression guard: the fallback is refused only when it would be a guess."""
    pytest.importorskip("inspect_ai")
    from calibrated_agent_evaluation.execute.inspect_executor import InspectExecutor

    only = tmp_path / "only.eval"
    only.touch()
    executor = InspectExecutor(model="mockllm/model", log_dir=tmp_path)

    class _NoLocation:
        location = None

    assert executor._resolve_log_path(_NoLocation(), tmp_path) == only


# --- item 9: two codebook entries must not share a name ------------------------------------------


def test_item9_a_duplicate_codebook_entry_name_is_refused() -> None:
    """`_validate_entry` checks an entry in isolation, so two could share a name in a vocabulary.

    A label's `failure_mode` IS a name, so a duplicate makes the category it refers to ambiguous,
    and the codebook is the thing a labeler and a judge are supposed to agree on.
    """
    from calibrated_agent_evaluation.attribution.codebook import register_entry
    from calibrated_agent_evaluation.attribution.errors import CodebookError
    from calibrated_agent_evaluation.attribution.types import Codebook, CodebookEntry

    entry = CodebookEntry(
        name="over_refund",
        definition="refunded outside the policy window",
        positive_examples=("refunded on day 40",),
        boundary_examples=("refunded on the last eligible day",),
    )
    book = register_entry(Codebook(version="v1"), "failure_modes", entry)

    with pytest.raises(CodebookError, match="already has an entry named"):
        register_entry(book, "failure_modes", entry)


def test_item9_the_same_name_in_the_other_vocabulary_is_allowed() -> None:
    """The refusal is per vocabulary: the two namespaces are independent."""
    from calibrated_agent_evaluation.attribution.codebook import register_entry
    from calibrated_agent_evaluation.attribution.types import Codebook, CodebookEntry

    entry = CodebookEntry(
        name="shared_name",
        definition="d",
        positive_examples=("p",),
        boundary_examples=("b",),
    )
    book = register_entry(Codebook(version="v1"), "failure_modes", entry)
    book = register_entry(book, "responsibility_boundaries", entry)

    assert [e.name for e in book.failure_modes] == ["shared_name"]
    assert [e.name for e in book.responsibility_boundaries] == ["shared_name"]
