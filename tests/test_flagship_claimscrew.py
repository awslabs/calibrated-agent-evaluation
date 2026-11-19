# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end hermetic proof of ClaimsCrew through all eight stages.

Cassette-replay + local backend, no network. The crew pipeline runs and the attribution stage
records culprit != manifestation (CLM-4471 intake vs payout, matching gold.json), the cluster stage
recovers the ratified (origin agent, failure type) taxonomy, and the improve stage catches the
workflow-rate relocation. A single-agent-rate-gated improvement would fail.
"""

from __future__ import annotations

from pathlib import Path

from calibrated_agent_evaluation.backends import TraceStoreBackend, get_backend
from calibrated_agent_evaluation.flagship import (
    FlagshipConfig,
    StageContext,
    World,
    prove,
    run_agent_pipeline,
)
from calibrated_agent_evaluation.flagship.report import CANONICAL_STAGES

_REPO = Path(__file__).resolve().parent.parent
_FIXTURES = _REPO / "fixtures"


def _backend_factory(tmp_path: Path):
    def factory(_world: World) -> TraceStoreBackend:
        return get_backend(None, local_root=tmp_path / "backend")

    return factory


def _crew_ctx(tmp_path: Path) -> StageContext:
    return StageContext(
        agent="claimscrew",
        world="local",
        model_plane="cassette",
        backend=get_backend(None, local_root=tmp_path / "backend"),
        fixtures_root=_FIXTURES,
    )


def test_claimscrew_attribution_records_culprit_not_manifestation(tmp_path: Path) -> None:
    """The crew attribution stage records culprit != manifestation, matching the oracle."""
    from _flagship_helpers import claimscrew_plan

    outcomes = run_agent_pipeline(claimscrew_plan(), _crew_ctx(tmp_path))
    assert tuple(o.stage for o in outcomes) == CANONICAL_STAGES
    attribution = next(o for o in outcomes if o.stage == "judge")  # crew judge stage = attribution
    assert attribution.ok is True  # equals fixtures/expected/attribution/claimscrew.json
    cluster = next(o for o in outcomes if o.stage == "cluster")
    assert cluster.ok is True  # recovers the 4-mode ratified taxonomy


def test_claimscrew_improve_catches_relocation(tmp_path: Path) -> None:
    """The crew improve stage gates on the workflow outcome rate and catches relocation."""
    from _flagship_helpers import claimscrew_plan

    config = FlagshipConfig(
        agents=("claimscrew",),
        worlds=("local",),
        model_plane="cassette",
        fixtures_root=str(_FIXTURES),
    )
    report = prove(config, [claimscrew_plan()], backend_factory=_backend_factory(tmp_path))
    assert report.verdict == "proven", report.reasons
    # the crew improve stage ran evaluate_improvement (relocation folded via relocation_check)
    improve = next(o for o in report.stages if o.stage == "improve")
    assert improve.ok is True


def test_attribution_oracle_divergence_fails(tmp_path: Path) -> None:
    """A walk-back that does not match the attribution oracle fails the stage."""
    from _flagship_helpers import claimscrew_plan, corpus
    from calibrated_agent_evaluation._model import replace
    from calibrated_agent_evaluation.flagship import attribution_stage
    from calibrated_agent_evaluation.flagship.orchestrator import StagePlan
    from calibrated_agent_evaluation.flagship.reference import claimscrew_definition

    crew = claimscrew_definition()
    # Swap the CLM-4471 culprit/manifestation (payout vs intake) -> diverges from the oracle.
    wrong = attribution_stage(
        crew,
        [(corpus().trace("CLM-4471"), ("CLM-4471/payout",), "CLM-4471/intake", None)],
        oracle_path=str(_FIXTURES / "expected" / "attribution" / "claimscrew.json"),
    )
    plan = claimscrew_plan()
    stages = list(plan.stages)
    stages[5] = StagePlan("judge", wrong)  # the crew attribution stage
    perturbed = replace(plan, stages=tuple(stages))
    outcomes = run_agent_pipeline(perturbed, _crew_ctx(tmp_path))
    attribution = next(o for o in outcomes if o.stage == "judge")
    # Either the walk-back validator rejects the swapped culprit>manifestation, or the oracle
    # comparison flags the divergence -- either way the stage is not ok (no false proof).
    assert attribution.ok is False
