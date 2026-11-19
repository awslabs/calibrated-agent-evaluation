# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the VerdictCache.

The cache-hit / no-model-call behavior and stale-version re-computation at the
runtime level live in test_judges_runtime.py; here we test the cache datastructure's key semantics
and round-trip in isolation.
"""

from __future__ import annotations

from pathlib import Path

from _judge_helpers import FakeJudgeModel, certified_record
from calibrated_agent_evaluation.judges.cache import VerdictCache
from calibrated_agent_evaluation.judges.executable import ExecutableJudge
from calibrated_agent_evaluation.judges.runtime import JudgeTarget, run_judge
from calibrated_agent_evaluation.schema.types import Label


def _judge(**over: object) -> ExecutableJudge:
    base: dict[str, object] = {
        "judge_id": "orderbot-refund-resolved",
        "model": "bedrock/global.anthropic.claude-sonnet-4-6",
        "prompt_version": "v1",
        "level": "workflow",
        "prompt_template": "t",
    }
    base.update(over)
    return ExecutableJudge(**base)  # type: ignore[arg-type]


def _label(trace_id: str = "ORD-1001", span_id: str | None = None) -> Label:
    return Label(
        trace_id=trace_id,
        span_id=span_id,
        kind="llm-judge",
        outcome="FAIL",
        critique="out-of-window refund wrongly issued",
    )


def test_cache_hit_returns_stored_label() -> None:
    """A put then get for the same judge version + target returns the stored Label."""
    judge = _judge()
    cache = VerdictCache()
    cache.put(judge, _label())
    got = cache.get(judge, "ORD-1001", None)
    assert got is not None
    assert got.outcome == "FAIL"


def test_cache_miss_on_prompt_version_change() -> None:
    """A judge with a bumped prompt_version does NOT match a stale entry."""
    v1 = _judge(prompt_version="v1")
    cache = VerdictCache()
    cache.put(v1, _label())
    v2 = _judge(prompt_version="v2")
    assert cache.get(v2, "ORD-1001", None) is None  # stale-version non-serve
    # The v1 entry is still there for v1 (the key is version-scoped, not clobbered).
    assert cache.get(v1, "ORD-1001", None) is not None


def test_cache_miss_on_model_change() -> None:
    """A judge with a different model does NOT match a stale entry."""
    claude = _judge(model="bedrock/global.anthropic.claude-sonnet-4-6")
    cache = VerdictCache()
    cache.put(claude, _label())
    titan = _judge(model="bedrock/amazon.titan-text")
    assert cache.get(titan, "ORD-1001", None) is None


def test_cache_round_trips(tmp_path: Path) -> None:
    """Dump→load reconstructs the cache and its Labels field-for-field."""
    judge = _judge()
    cache = VerdictCache()
    cache.put(judge, _label(span_id="ORD-1001/orderbot"))
    cache.dump(tmp_path)
    reloaded = VerdictCache.load(tmp_path)
    got = reloaded.get(judge, "ORD-1001", "ORD-1001/orderbot")
    assert got is not None
    assert got.critique == "out-of-window refund wrongly issued"
    assert got.span_id == "ORD-1001/orderbot"


def test_runtime_second_run_serves_cache_without_model_call() -> None:
    """Run twice against a shared cache — the model is called on run 1, NOT on run 2, and
    both runs return an equal Label."""
    j = _judge()
    cache = VerdictCache()
    model = FakeJudgeModel()
    target = JudgeTarget(trace_id="ORD-1001", span_id=None, content="issued a refund out of window")

    r1 = run_judge(j, [target], calib_record=certified_record(j), model_backend=model, cache=cache)
    assert model.call_count == 1
    assert r1.record.n_computed == 1
    assert r1.record.n_cached == 0

    r2 = run_judge(j, [target], calib_record=certified_record(j), model_backend=model, cache=cache)
    assert model.call_count == 1  # unchanged — served from cache
    assert r2.record.n_cached == 1
    assert r2.record.n_computed == 0
    assert r1.labels[0] == r2.labels[0]


def test_runtime_reversioned_judge_recomputes() -> None:
    """After bumping prompt_version, the prior cache entry does NOT match and the
    model IS re-invoked (no stale verdict served)."""
    cache = VerdictCache()
    model = FakeJudgeModel()
    target = JudgeTarget(trace_id="ORD-1001", span_id=None, content="issued a refund out of window")

    v1 = _judge(prompt_version="v1")
    run_judge(v1, [target], calib_record=certified_record(v1), model_backend=model, cache=cache)
    assert model.call_count == 1

    v2 = _judge(prompt_version="v2")
    run_judge(v2, [target], calib_record=certified_record(v2), model_backend=model, cache=cache)
    assert model.call_count == 2  # re-versioned judge re-scored, did not serve v1's cached verdict
