# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the run_judge batch fan-out, write-back, retry."""

from __future__ import annotations

import pytest

from _judge_helpers import FakeJudgeModel, certified_record, judge
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.execute import StructuredReply
from calibrated_agent_evaluation.judges.errors import JudgeRuntimeError
from calibrated_agent_evaluation.judges.runtime import JudgeTarget, run_judge
from calibrated_agent_evaluation.schema.errors import BackendError
from calibrated_agent_evaluation.schema.types import Label, Trace


def _targets() -> list[JudgeTarget]:
    return [
        JudgeTarget(trace_id="ORD-1001", span_id=None, content="issued a refund out of window"),
        JudgeTarget(trace_id="ORD-1002", span_id=None, content="issued an in-window refund"),
    ]


def test_batch_fan_out_returns_one_label_per_trace() -> None:
    """Run over 2 traces returns 2 llm-judge Labels, each keyed on its trace_id."""
    j = judge()
    result = run_judge(
        j, _targets(), calib_record=certified_record(j), model_backend=FakeJudgeModel()
    )
    assert len(result.labels) == 2
    assert [lab.trace_id for lab in result.labels] == ["ORD-1001", "ORD-1002"]
    for lab in result.labels:
        assert lab.kind == "llm-judge"
        assert lab.outcome in ("PASS", "FAIL")
        assert lab.critique.strip()  # non-empty critique
    assert result.record.n_computed == 2
    assert result.record.n_cached == 0


def test_run_emits_reproducibility_record() -> None:
    """The emitted JudgeRunRecord pins the frozen (judge_id, model, prompt_version), the
    cleared calib_id, and exactly the scored trace ids (the reproducibility anchor)."""
    j = judge()
    record = certified_record(j)
    result = run_judge(j, _targets(), calib_record=record, model_backend=FakeJudgeModel())
    assert result.record.judge_id == j.judge_id
    assert result.record.model == j.model
    assert result.record.prompt_version == j.prompt_version
    assert result.record.calib_id == record.calib_id
    assert result.record.trace_ids == ("ORD-1001", "ORD-1002")
    assert result.record.span_ids == (None, None)


def test_verdicts_written_back_and_readable() -> None:
    """Each verdict Label is persisted and re-readable via get_scores from the backend."""
    j = judge()
    with LocalBackend(":memory:") as backend:
        run_judge(
            j,
            _targets(),
            calib_record=certified_record(j),
            model_backend=FakeJudgeModel(),
            store=backend,
        )
        scores = backend.get_scores("ORD-1001")
        assert len(scores) == 1
        assert scores[0].kind == "llm-judge"
        assert scores[0].critique.strip()


def test_empty_target_list_yields_no_labels() -> None:
    """An empty batch returns zero Labels and does not crash."""
    j = judge()
    result = run_judge(j, [], calib_record=certified_record(j), model_backend=FakeJudgeModel())
    assert result.labels == ()
    assert result.record.trace_ids == ()


def test_transient_failure_retried_then_succeeds() -> None:
    """A transient model failure is retried up to the bound, then succeeds."""
    j = judge()
    model = FakeJudgeModel(fail_times=2)  # fail twice, succeed on the 3rd (max_retries=2)
    result = run_judge(
        j,
        [_targets()[0]],
        calib_record=certified_record(j),
        model_backend=model,
        max_retries=2,
    )
    assert len(result.labels) == 1
    assert model.call_count == 3  # 2 failures + 1 success


def test_retry_exhaustion_raises_typed_error() -> None:
    """Failures beyond the retry bound raise JudgeRuntimeError, chained."""
    j = judge()
    model = FakeJudgeModel(fail_times=5)
    with pytest.raises(JudgeRuntimeError, match="after"):
        run_judge(
            j,
            [_targets()[0]],
            calib_record=certified_record(j),
            model_backend=model,
            max_retries=2,
        )
    assert model.call_count == 3  # max_retries + 1 attempts, then gives up


class _UnreachableBackend:
    """A trace-store stub whose put_score raises BackendError (the unreachable-store case).

    Mirrors the ``TraceStoreBackend`` protocol surface so it is a faithful double; only the
    ``put_score`` write-back is exercised. The read methods raise/return to satisfy the types.
    """

    def put_trace(self, trace: Trace) -> None:  # pragma: no cover - unused
        raise BackendError("unreachable")

    def get_trace(self, trace_id: str) -> Trace:  # pragma: no cover - unused
        raise BackendError("unreachable")

    def query(
        self,
        frame: object = None,
        cell: dict[str, str] | None = None,
        ownership: object = None,
    ) -> list[Trace]:  # pragma: no cover - unused
        return []

    def put_score(self, label: Label) -> None:
        raise BackendError("Langfuse put_score failed: backend unreachable or auth rejected")

    def get_scores(
        self, trace_id: str, span_id: str | None = None
    ) -> list[Label]:  # pragma: no cover - unused
        return []

    def retract_score(self, label: Label) -> None:  # pragma: no cover - unused here
        raise NotImplementedError


def test_backend_unreachable_propagates_un_swallowed() -> None:
    """A BackendError on write-back propagates un-swallowed — the run is not
    reported as durably persisted."""
    j = judge()
    with pytest.raises(BackendError, match="unreachable"):
        run_judge(
            j,
            [_targets()[0]],
            calib_record=certified_record(j),
            model_backend=FakeJudgeModel(),
            store=_UnreachableBackend(),
        )


def test_reply_via_tool_call_path_builds_label() -> None:
    """A tool-call reply (non-Claude path) still yields a structured Label."""
    j = judge(model="bedrock/amazon.titan-text")
    model = FakeJudgeModel(
        reply=StructuredReply(fields={"critique": "wrong", "verdict": "fail"}, via="tool_call"),
        supports_schema=False,
    )
    result = run_judge(j, [_targets()[0]], calib_record=certified_record(j), model_backend=model)
    assert result.labels[0].outcome == "FAIL"


# --- a warm-cache re-run must not grow the stored label count ----------------


def test_a_warm_cache_rerun_does_not_write_the_label_again() -> None:
    """A cache hit is already durably stored, so writing it back added a second row.

    `run_judge` wrote back unconditionally, once per target, hit or miss — and both score-write
    paths are append-only. So re-running a judge over the same targets with a warm cache doubled
    every label it touched, and any rate computed over them moved toward the doubled verdict.
    """
    from calibrated_agent_evaluation.judges.cache import VerdictCache

    j = judge()
    cache = VerdictCache()
    with LocalBackend(":memory:") as backend:
        first = run_judge(
            j,
            _targets(),
            calib_record=certified_record(j),
            model_backend=FakeJudgeModel(),
            store=backend,
            cache=cache,
        )
        assert first.record.n_computed == 2
        assert first.record.n_cached == 0
        after_first = len(backend.all_scores())

        second = run_judge(
            j,
            _targets(),
            calib_record=certified_record(j),
            model_backend=FakeJudgeModel(),
            store=backend,
            cache=cache,
        )
        assert second.record.n_cached == 2, "precondition: the second run was served from cache"
        assert second.record.n_computed == 0

        assert len(backend.all_scores()) == after_first, "a cache hit must not write again"
        assert len(backend.get_scores("ORD-1001")) == 1


def test_a_cold_rerun_still_reads_back_one_label_per_target() -> None:
    """Without a cache the judge recomputes and re-writes, and read dedup keeps the count honest.

    This is the other half: the write-back skip fixes the cached path, and `dedup_labels` fixes the
    uncached one. Both are needed — a re-run with no cache still calls `put_score` again.
    """
    j = judge()
    with LocalBackend(":memory:") as backend:
        for _ in range(3):
            run_judge(
                j,
                _targets(),
                calib_record=certified_record(j),
                model_backend=FakeJudgeModel(),
                store=backend,
            )
        assert len(backend.get_scores("ORD-1001")) == 1
        assert len(backend.get_scores("ORD-1002")) == 1
