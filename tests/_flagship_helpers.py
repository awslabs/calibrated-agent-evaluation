# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reference corpus and illustrative admission records for the flagship-proof tests.

The records are illustrative: a CERTIFIED calibration and a CLEAR bias battery pinned to a judge, so
a test can drive the admission chain without running a calibration study.
"""

from __future__ import annotations

from pathlib import Path

from calibrated_agent_evaluation._provenance import capture_provenance
from calibrated_agent_evaluation.backends import TraceStoreBackend, get_backend
from calibrated_agent_evaluation.bias import BiasBatteryRecord
from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord
from calibrated_agent_evaluation.flagship import AgentPlan, ReferenceCorpus, World
from calibrated_agent_evaluation.judges import ExecutableJudge

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def corpus() -> ReferenceCorpus:
    return ReferenceCorpus(FIXTURES)


def certified_record(judge: ExecutableJudge) -> CalibrationRecord:
    """A CERTIFIED calibration record pinned to ``judge``'s current version and fingerprint."""
    return CalibrationRecord(
        provenance=capture_provenance(),
        calib_id=f"calib-{judge.judge_id}",
        judge_id=judge.judge_id,
        model=judge.model,
        prompt_version=judge.prompt_version,
        judge_fingerprint=judge.fingerprint,
        hunted_set_id=f"hunted-{judge.judge_id}",
        n_pos=80,
        n_neg=80,
        tpr=Bound(point=0.95, lb95=0.90, n=80),
        tnr=Bound(point=0.95, lb95=0.90, n=80),
        youden_j=0.90,
        result="CERTIFIED",
    )


def clear_battery(judge: ExecutableJudge) -> BiasBatteryRecord:
    """A CLEAR bias battery pinned to ``judge``."""
    return BiasBatteryRecord(
        provenance=capture_provenance(),
        battery_id=f"bias-{judge.judge_id}-clear",
        judge_id=judge.judge_id,
        model=judge.model,
        prompt_version=judge.prompt_version,
        judge_fingerprint=judge.fingerprint,
        is_pairwise=False,
        results=(),
        verdict="CLEAR",
        failed_biases=(),
    )


def orderbot_plan() -> AgentPlan:
    """The OrderBot plan, its judge admitted on illustrative records."""
    reference = corpus()
    judge = reference.orderbot_judge()
    return reference.orderbot_plan(
        calibration=certified_record(judge), battery=clear_battery(judge)
    )


def claimscrew_plan() -> AgentPlan:
    return corpus().claimscrew_plan()


def local_backend_factory(root: Path):
    """Every world maps to one local store under ``root``."""

    def factory(_world: World) -> TraceStoreBackend:
        return get_backend(None, local_root=root)

    return factory
