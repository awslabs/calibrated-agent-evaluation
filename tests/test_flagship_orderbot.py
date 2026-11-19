# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end hermetic proof of OrderBot through all eight stages.

Cassette-replay + local backend, no network. The whole OrderBot pipeline runs and each stage clears
its committed corpus oracle; the probe capability map equals fixtures/expected/probe/, the run gate
reproduces expected/run/gate_decision.json, and the overall verdict is 'proven'. A perturbed
stage flips the verdict to 'failed' (the oracle-driven negative).
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


def _orderbot_ctx(tmp_path: Path) -> StageContext:
    return StageContext(
        agent="orderbot",
        world="local",
        model_plane="cassette",
        backend=get_backend(None, local_root=tmp_path / "backend"),
        fixtures_root=_FIXTURES,
    )


def test_orderbot_runs_all_eight_stages(tmp_path: Path) -> None:
    """The OrderBot pipeline runs all eight stages; probe reproduces its oracle."""
    from _flagship_helpers import orderbot_plan

    outcomes = run_agent_pipeline(orderbot_plan(), _orderbot_ctx(tmp_path))
    assert tuple(o.stage for o in outcomes) == CANONICAL_STAGES
    probe = next(o for o in outcomes if o.stage == "probe")
    assert probe.ok is True  # capability map == fixtures/expected/probe/orderbot_capability.json


def test_orderbot_run_stage_reproduces_gate_oracle(tmp_path: Path) -> None:
    """The OrderBot run stage reproduces the gate oracle; verdict 'proven'."""
    from _flagship_helpers import orderbot_plan

    config = FlagshipConfig(
        agents=("orderbot",),
        worlds=("local",),
        model_plane="cassette",
        fixtures_root=str(_FIXTURES),
    )
    report = prove(config, [orderbot_plan()], backend_factory=_backend_factory(tmp_path))
    assert report.verdict == "proven", report.reasons
    run_stage = next(o for o in report.stages if o.stage == "run")
    assert run_stage.ok is True  # reproduces fixtures/expected/run/gate_decision.json (gate fails)
    judge_stage = next(o for o in report.stages if o.stage == "judge")
    assert judge_stage.ok is True  # the certified+clear judge was admitted and wrote Scores back


def test_perturbed_probe_oracle_fails_naming_probe(tmp_path: Path) -> None:
    """A capability map != its oracle fails naming the probe."""
    from _flagship_helpers import orderbot_plan
    from calibrated_agent_evaluation.flagship import probe_stage
    from calibrated_agent_evaluation.flagship.orchestrator import StagePlan
    from calibrated_agent_evaluation.probe import ConnectionConfig

    def _wrong_transport(url, headers, body, timeout):
        import json

        # A probe that observes a DIFFERENT tool than the oracle's ["lookup_order"].
        return 200, [
            b"data: " + json.dumps({"type": "tool_use", "tool_name": "delete_everything"}).encode(),
            b"data: " + json.dumps({"type": "response", "response": "x"}).encode(),
        ]

    plan = orderbot_plan()
    bad_probe = probe_stage(
        ConnectionConfig(endpoint="https://stub.local/invoke"),
        _wrong_transport,
        str(_FIXTURES / "expected" / "probe" / "orderbot_capability.json"),
    )
    stages = list(plan.stages)
    stages[0] = StagePlan("probe", bad_probe)
    from calibrated_agent_evaluation._model import replace

    perturbed = replace(plan, stages=tuple(stages))
    outcomes = run_agent_pipeline(perturbed, _orderbot_ctx(tmp_path))
    probe = next(o for o in outcomes if o.stage == "probe")
    assert probe.ok is False
    assert probe.oracle.divergence is not None
    assert probe.oracle.stage == "probe"
