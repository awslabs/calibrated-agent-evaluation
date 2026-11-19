# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the orchestration core.

These exercise the pure orchestration engine with lightweight injected stage plans (the corpus
binding is injected by design), so they verify the *engine's* contract -- version gate, completeness
invariant, oracle-driven verdict, world parity, live-unreachable refusal, and no-swallowed-error --
without needing the full corpus. The end-to-end proof over the real corpus lives in
test_flagship_orderbot.py / test_flagship_claimscrew.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.backends import TraceStoreBackend, get_backend
from calibrated_agent_evaluation.flagship.errors import FlagshipError
from calibrated_agent_evaluation.flagship.orchestrator import (
    AgentPlan,
    StageContext,
    StagePlan,
    StageResult,
    assert_worlds_equal,
    prove,
    run_agent_pipeline,
)
from calibrated_agent_evaluation.flagship.report import CANONICAL_STAGES, Agent, FlagshipConfig
from calibrated_agent_evaluation.schema import SCHEMA_VERSION


def _seed_fixtures(tmp_path: Path, version: str = SCHEMA_VERSION) -> Path:
    """A minimal seeded corpus root: just a VERSION file (the orchestrator checks it)."""
    root = tmp_path / "fixtures"
    root.mkdir()
    (root / "VERSION").write_text(version)
    return root


def _oracle(tmp_path: Path, name: str, expected: dict[str, Any]) -> str:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"expected": expected}))
    return str(path)


def _all_pass_plan(agent: Agent, oracle_path: str) -> AgentPlan:
    """A plan whose eight stages all reproduce a trivial oracle (matched)."""

    def _ok(_ctx: StageContext) -> StageResult:
        return StageResult(produced={"ok": True}, oracle_path=oracle_path, note="stub pass")

    return AgentPlan(
        agent=agent,
        stages=tuple(StagePlan(stage=s, run=_ok) for s in CANONICAL_STAGES),
        comparable=lambda _ctx: {"trace": "same-in-every-world"},
    )


def _backend_factory(tmp_path: Path):
    """world -> LocalBackend factory; both worlds map to one local root for the engine tests."""

    def factory(_world: str) -> TraceStoreBackend:
        return get_backend(None, local_root=tmp_path / "backend")

    return factory


def test_stale_corpus_version_refuses_before_any_stage(tmp_path: Path) -> None:
    """A seeded VERSION != running SCHEMA_VERSION raises and runs no stage."""
    root = _seed_fixtures(tmp_path, version="999")
    oracle = _oracle(tmp_path, "o", {"ok": True})
    config = FlagshipConfig(
        agents=("orderbot",), worlds=("local",), model_plane="cassette", fixtures_root=str(root)
    )
    with pytest.raises(FlagshipError, match="SCHEMA_VERSION"):
        prove(
            config, [_all_pass_plan("orderbot", oracle)], backend_factory=_backend_factory(tmp_path)
        )


def test_incomplete_plan_cannot_be_proven(tmp_path: Path) -> None:
    """A plan covering fewer than all eight canonical stages is refused (completeness)."""
    oracle = _oracle(tmp_path, "o", {"ok": True})

    def _ok(_ctx: StageContext) -> StageResult:
        return StageResult(produced={"ok": True}, oracle_path=oracle)

    short = AgentPlan(
        agent="orderbot",
        stages=(StagePlan(stage="probe", run=_ok), StagePlan(stage="run", run=_ok)),
        comparable=lambda _ctx: {},
    )
    ctx = StageContext(
        agent="orderbot",
        world="local",
        model_plane="cassette",
        backend=get_backend(None, local_root=tmp_path / "b"),
        fixtures_root=tmp_path,
    )
    with pytest.raises(FlagshipError, match="all"):
        run_agent_pipeline(short, ctx)


def test_all_stages_matched_yields_proven(tmp_path: Path) -> None:
    """Every stage clearing its oracle for every (agent, world) -> verdict 'proven'."""
    root = _seed_fixtures(tmp_path)
    oracle = _oracle(tmp_path, "o", {"ok": True})
    config = FlagshipConfig(
        agents=("orderbot",), worlds=("local",), model_plane="cassette", fixtures_root=str(root)
    )
    report = prove(
        config, [_all_pass_plan("orderbot", oracle)], backend_factory=_backend_factory(tmp_path)
    )
    assert report.verdict == "proven"
    assert report.reasons == ()
    assert len(report.stages) == len(CANONICAL_STAGES)


def test_a_diverging_stage_flips_verdict_to_failed(tmp_path: Path) -> None:
    """One stage diverging from its oracle -> verdict 'failed' naming that stage."""
    root = _seed_fixtures(tmp_path)
    good = _oracle(tmp_path, "good", {"ok": True})
    bad = _oracle(tmp_path, "bad", {"ok": False})  # the stage will produce ok=True -> divergence

    def _ok(_ctx: StageContext) -> StageResult:
        return StageResult(produced={"ok": True}, oracle_path=good)

    def _diverge(_ctx: StageContext) -> StageResult:
        return StageResult(produced={"ok": True}, oracle_path=bad)

    stages = [StagePlan(stage=s, run=_ok) for s in CANONICAL_STAGES]
    stages[6] = StagePlan(stage="run", run=_diverge)  # the "run" stage diverges
    plan = AgentPlan(agent="orderbot", stages=tuple(stages), comparable=lambda _c: {})
    config = FlagshipConfig(
        agents=("orderbot",), worlds=("local",), model_plane="cassette", fixtures_root=str(root)
    )
    report = prove(config, [plan], backend_factory=_backend_factory(tmp_path))
    assert report.verdict == "failed"
    assert any("run" in r for r in report.reasons)


def test_world_divergence_is_a_blocker(tmp_path: Path) -> None:
    """A result that differs between worlds -> equal=False, verdict failed."""
    root = _seed_fixtures(tmp_path)
    oracle = _oracle(tmp_path, "o", {"ok": True})

    def _ok(_ctx: StageContext) -> StageResult:
        return StageResult(produced={"ok": True}, oracle_path=oracle)

    # comparable returns a per-world value, so the two worlds diverge.
    plan = AgentPlan(
        agent="orderbot",
        stages=tuple(StagePlan(stage=s, run=_ok) for s in CANONICAL_STAGES),
        comparable=lambda ctx: {"trace": f"world-specific-{ctx.world}"},
    )
    config = FlagshipConfig(
        agents=("orderbot",),
        worlds=("local", "langfuse"),
        model_plane="cassette",
        fixtures_root=str(root),
    )
    report = prove(config, [plan], backend_factory=_backend_factory(tmp_path))
    assert report.verdict == "failed"
    assert any("divergence" in r for r in report.reasons)
    assert any(not p.equal for p in report.parity)


def test_equal_worlds_hold_parity() -> None:
    """Identical per-world comparables -> equal True; a perturbation -> False."""
    equal = assert_worlds_equal("orderbot", {"local": {"trace": "x"}, "langfuse": {"trace": "x"}})
    assert equal.equal is True
    diverged = assert_worlds_equal(
        "orderbot", {"local": {"trace": "x"}, "langfuse": {"trace": "y"}}
    )
    assert diverged.equal is False
    assert diverged.divergence is not None


def test_live_unreachable_raises_no_silent_cassette_fallback(tmp_path: Path) -> None:
    """A stage that re-raises an unreachable --live model as FlagshipError propagates.

    The orchestrator never swallows a FlagshipError into a 'failed' outcome and continues -- a
    proof-integrity failure (an unreachable live model wrapped as FlagshipError) propagates so the
    harness can never silently replay the cassette and report a live pass.
    """
    oracle = _oracle(tmp_path, "o", {"ok": True})

    def _ok(_ctx: StageContext) -> StageResult:
        return StageResult(produced={"ok": True}, oracle_path=oracle)

    def _live_unreachable(ctx: StageContext) -> StageResult:
        if ctx.model_plane == "live":
            raise FlagshipError("live model endpoint unreachable (no silent cassette fallback)")
        return _ok(ctx)

    stages = [StagePlan(stage=s, run=_ok) for s in CANONICAL_STAGES]
    stages[5] = StagePlan(stage="judge", run=_live_unreachable)
    plan = AgentPlan(agent="orderbot", stages=tuple(stages), comparable=lambda _c: {})
    ctx = StageContext(
        agent="orderbot",
        world="local",
        model_plane="live",
        backend=get_backend(None, local_root=tmp_path / "b"),
        fixtures_root=tmp_path,
    )
    with pytest.raises(FlagshipError, match="unreachable"):
        run_agent_pipeline(plan, ctx)


def test_stage_domain_error_fails_proof_no_partial_pass(tmp_path: Path) -> None:
    """A stage's typed domain error is recorded (stage + error type) and fails the proof."""
    oracle = _oracle(tmp_path, "o", {"ok": True})

    class _StageDomainError(Exception):
        pass

    def _ok(_ctx: StageContext) -> StageResult:
        return StageResult(produced={"ok": True}, oracle_path=oracle)

    def _raise(_ctx: StageContext) -> StageResult:
        raise _StageDomainError("run_ci_gate refused over empty input")

    stages = [StagePlan(stage=s, run=_ok) for s in CANONICAL_STAGES]
    stages[6] = StagePlan(stage="run", run=_raise)
    plan = AgentPlan(agent="orderbot", stages=tuple(stages), comparable=lambda _c: {})
    ctx = StageContext(
        agent="orderbot",
        world="local",
        model_plane="cassette",
        backend=get_backend(None, local_root=tmp_path / "b"),
        fixtures_root=tmp_path,
    )
    outcomes = run_agent_pipeline(plan, ctx)
    run_outcome = next(o for o in outcomes if o.stage == "run")
    assert run_outcome.ok is False
    assert run_outcome.error_type == "_StageDomainError"
    # the other stages still recorded (no crash), but the run stage is not a partial pass
    assert len(outcomes) == len(CANONICAL_STAGES)
