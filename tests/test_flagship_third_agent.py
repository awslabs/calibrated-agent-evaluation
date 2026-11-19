# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The flagship harness run against an agent outside the two reference agents.

The agent id is an open string checked against the plans the caller registers, not a closed
``Literal``; and a plan that opts out of oracle pinning gets its absent-oracle stages recorded as
unpinned instead of raising, so a new agent with no hand-authored oracle corpus still produces a
report. The reference agents keep the hard refusal on a missing oracle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends import TraceStoreBackend, get_backend
from calibrated_agent_evaluation.flagship import dump_flagship_reports, load_flagship_reports
from calibrated_agent_evaluation.flagship.errors import FlagshipError
from calibrated_agent_evaluation.flagship.orchestrator import (
    AgentPlan,
    StageContext,
    StagePlan,
    StageResult,
    prove,
)
from calibrated_agent_evaluation.flagship.report import CANONICAL_STAGES, FlagshipConfig
from calibrated_agent_evaluation.schema import SCHEMA_VERSION


def _fixtures(tmp_path: Path) -> Path:
    root = tmp_path / "fixtures"
    root.mkdir()
    (root / "VERSION").write_text(SCHEMA_VERSION)
    return root


def _factory(tmp_path: Path):
    def factory(_world: str) -> TraceStoreBackend:
        return get_backend(None, local_root=tmp_path / "backend")

    return factory


def _plan(
    agent: str, oracle_path: str, *, oracle_required: bool = True, diverge: str | None = None
) -> AgentPlan:
    def _make(stage: str):
        def _run(_ctx: StageContext) -> StageResult:
            produced = {"ok": stage != diverge}
            return StageResult(produced=produced, oracle_path=oracle_path, note=f"{stage} ran")

        return _run

    return AgentPlan(
        agent=agent,
        stages=tuple(StagePlan(stage=s, run=_make(s)) for s in CANONICAL_STAGES),
        comparable=lambda _ctx: {"trace": "x"},
        oracle_required=oracle_required,
    )


def _config(root: Path, *agents: str) -> FlagshipConfig:
    return FlagshipConfig(
        agents=agents, worlds=("local",), model_plane="cassette", fixtures_root=str(root)
    )


def test_unregistered_agent_with_no_oracles_reports_unpinned_rows(tmp_path: Path) -> None:
    """A third agent id with no oracle files runs end to end; its stages are unpinned."""
    root = _fixtures(tmp_path)
    absent = str(tmp_path / "expected" / "probe" / "autogen.json")
    plan = _plan("autogen", absent, oracle_required=False)

    report = prove(_config(root, "autogen"), [plan], backend_factory=_factory(tmp_path))

    assert report.agents == ("autogen",)
    assert report.verdict == "unpinned"
    assert report.reasons == ()
    assert len(report.stages) == len(CANONICAL_STAGES)
    for outcome in report.stages:
        assert outcome.agent == "autogen"
        assert outcome.ok is True
        assert outcome.oracle.pinned is False
        assert outcome.oracle.matched is False
        assert outcome.oracle.oracle_path == absent


def test_an_unpinned_run_still_fails_on_a_real_divergence(tmp_path: Path) -> None:
    """Unpinned is not a free pass: diverging from a present oracle still fails the proof."""
    root = _fixtures(tmp_path)
    present = tmp_path / "o.json"
    present.write_text(json.dumps({"expected": {"ok": True}}))
    plan = _plan("autogen", str(present), oracle_required=False, diverge="run")

    report = prove(_config(root, "autogen"), [plan], backend_factory=_factory(tmp_path))

    assert report.verdict == "failed"
    assert any("autogen/local/run" in r for r in report.reasons)
    assert all(o.oracle.pinned for o in report.stages)


def test_a_pinned_plan_still_refuses_a_missing_oracle(tmp_path: Path) -> None:
    """The reference-agent default keeps the refusal: a missing oracle is a proof-integrity
    failure."""
    root = _fixtures(tmp_path)
    plan = _plan("orderbot", str(tmp_path / "gone.json"))

    with pytest.raises(FlagshipError, match="missing"):
        prove(_config(root, "orderbot"), [plan], backend_factory=_factory(tmp_path))


def test_an_agent_with_no_registered_plan_is_refused(tmp_path: Path) -> None:
    """An open id is still validated: a requested agent with no plan is refused, not skipped."""
    root = _fixtures(tmp_path)
    oracle = str(tmp_path / "unused.json")
    with pytest.raises(FlagshipError, match="no AgentPlan"):
        prove(
            _config(root, "gptresearcher"),
            [_plan("autogen", oracle, oracle_required=False)],
            backend_factory=_factory(tmp_path),
        )


def test_two_plans_claiming_one_agent_id_are_refused(tmp_path: Path) -> None:
    """A duplicate registration would silently shadow one plan; refuse it."""
    root = _fixtures(tmp_path)
    oracle = str(tmp_path / "unused.json")
    plans = [_plan("autogen", oracle, oracle_required=False) for _ in range(2)]
    with pytest.raises(FlagshipError, match="more than one AgentPlan"):
        prove(_config(root, "autogen"), plans, backend_factory=_factory(tmp_path))


@pytest.mark.parametrize("bad_id", ["", "   ", "auto gen", "a/b"])
def test_a_malformed_agent_id_is_refused_at_construction(bad_id: str, tmp_path: Path) -> None:
    """The id becomes a path segment and a report key, so a blank or spaced id is refused."""
    with pytest.raises(FlagshipError, match="agent id"):
        _plan(bad_id, str(tmp_path / "o.json"))


def test_an_unpinned_report_round_trips_through_the_store(tmp_path: Path) -> None:
    """The unpinned verdict and per-stage pinned flag survive dump/load."""
    root = _fixtures(tmp_path)
    plan = _plan("autogen", str(tmp_path / "absent.json"), oracle_required=False)
    report = prove(_config(root, "autogen"), [plan], backend_factory=_factory(tmp_path))

    dump_flagship_reports([report], tmp_path / "out", fmt="json")
    (loaded,) = load_flagship_reports(tmp_path / "out", fmt="json")

    assert loaded.verdict == "unpinned"
    assert all(o.oracle.pinned is False for o in loaded.stages)
    assert loaded.stages == report.stages
