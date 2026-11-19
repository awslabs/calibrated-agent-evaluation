# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The flagship orchestrator -- sequence the eight stages, oracle-check, assemble the proof.

This is the pure orchestration core: it composes shipped ``calibrated_agent_evaluation.*`` surfaces
and adds only sequencing, oracle comparison, world/model selection, cross-world parity, and the
:class:`~calibrated_agent_evaluation.flagship.report.FlagshipReport` assembly. It computes no
estimator, gate, judge, or statistic. The corpus-specific binding (which trace/cassette/judge/crew
per agent) is **injected** by the caller as an :class:`AgentPlan`, so this module imports only
``calibrated_agent_evaluation.*`` -- never
``scripts``/``examples`` (not in the wheel) -- and no vendor SDK type. The one place a
vendor is touched is world/model selection, which goes through the backend factory
(:func:`calibrated_agent_evaluation.backends.get_backend`) and the caller-supplied executor/model,
never a raw vendor object.

Flow (per :func:`prove`): assert the seeded fixture version matches the running ``SCHEMA_VERSION``;
for each requested ``(agent, world)`` run the eight stages via
:func:`run_agent_pipeline`; assert cross-world parity per agent; assemble the
overall verdict -- ``proven`` iff every stage cleared its oracle for every ``(agent, world)`` AND
every world-parity is equal AND every judge used to score was fully admitted, else ``failed`` with
``reasons``. A run with no failures but at least one unpinned stage (a plan with
``oracle_required=False`` and no oracle file) is ``unpinned``: it ran, but proved nothing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation._provenance import capture_provenance
from calibrated_agent_evaluation.backends import TraceStoreBackend
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.flagship.errors import FlagshipError
from calibrated_agent_evaluation.flagship.oracle import compare_oracle
from calibrated_agent_evaluation.flagship.report import (
    CANONICAL_STAGES,
    Agent,
    FlagshipConfig,
    FlagshipReport,
    GateDecisionRef,
    ModelPlane,
    OracleCheck,
    StageName,
    StageOutcome,
    World,
    WorldParity,
    validate_agent_id,
)
from calibrated_agent_evaluation.schema import SCHEMA_VERSION


@dataclass(frozen=True)
class StageContext:
    """What a stage callable is handed: the world's backend + run coordinates.

    A stage reads/writes through ``backend`` (the trace-store protocol -- local or Langfuse, chosen
    by the orchestrator so the stage never branches on world), and consults ``model_plane`` to pick
    cassette-replay vs live for the model-touching stages.
    """

    agent: Agent
    world: World
    model_plane: ModelPlane
    backend: TraceStoreBackend
    fixtures_root: Path


@dataclass(frozen=True)
class StageResult:
    """What a stage callable returns -- the produced fields the oracle pins + optional headline.

    ``produced`` is the subset of fields the committed oracle constrains (compared semantically);
    ``oracle_path`` is ``None`` for an oracle-less stage proven structurally (the ``note`` records
    the structural check). ``headline`` is the calibration ``Interval`` where the stage emits a
    rate. ``gate`` carries the three-gate judge-trust decision for the judge stage.
    """

    produced: dict[str, Any]
    oracle_path: str | None = None
    headline: Interval | None = None
    artifact_ids: tuple[str, ...] = ()
    gate: GateDecisionRef | None = None
    note: str = ""


# A stage callable composes a shipped surface and returns a StageResult. The corpus binding lives in
# the caller's plan; the orchestrator only sequences + oracle-checks.
StageCallable = Callable[[StageContext], StageResult]


@dataclass(frozen=True)
class StagePlan:
    """One stage's binding: the canonical stage name + the callable that runs it."""

    stage: StageName
    run: StageCallable


@dataclass(frozen=True)
class AgentPlan:
    """The injected per-agent corpus binding the orchestrator drives.

    ``stages`` is the eight ``StagePlan``s in canonical order; the orchestrator refuses a plan that
    does not cover all eight canonical stages (the completeness invariant). ``comparable``
    reads the world-invariant artifacts (the normalized trace/label/score dicts) the cross-world
    parity assertion compares.

    ``oracle_required`` (default True) keeps the hard refusal: a stage naming an oracle that is
    absent is a proof-integrity failure. A plan for an agent with no hand-authored oracle corpus
    sets it False, and each absent-oracle stage is recorded as unpinned instead, which makes the
    report's verdict ``unpinned`` rather than ``proven``. A present but corrupt oracle still raises.
    """

    agent: Agent
    stages: tuple[StagePlan, ...]
    comparable: Callable[[StageContext], dict[str, Any]]
    oracle_required: bool = True

    def __post_init__(self) -> None:
        validate_agent_id(self.agent)


def _assert_fixture_version(fixtures_root: Path) -> str:
    """Refuse to run against a corpus whose VERSION != the running SCHEMA_VERSION."""
    version_file = fixtures_root / "VERSION"
    try:
        fixture_version = version_file.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise FlagshipError(
            f"cannot read fixture version at {version_file}: {exc} -- refusing to prove against "
            "an unreadable corpus"
        ) from exc
    if fixture_version != SCHEMA_VERSION:
        raise FlagshipError(
            f"seeded fixture version {fixture_version!r} != running SCHEMA_VERSION "
            f"{SCHEMA_VERSION!r}; refusing to prove against a stale/mismatched corpus"
        )
    return fixture_version


def _assert_plan_complete(plan: AgentPlan) -> None:
    """Refuse a plan that does not cover all eight canonical stages (the completeness invariant)."""
    covered = tuple(sp.stage for sp in plan.stages)
    if covered != CANONICAL_STAGES:
        raise FlagshipError(
            f"agent {plan.agent!r} plan covers stages {covered} but the flagship proof needs all "
            f"eight canonical stages in order {CANONICAL_STAGES} -- a proof that skips a stage "
            "cannot be 'proven'"
        )


def run_agent_pipeline(plan: AgentPlan, ctx: StageContext) -> tuple[StageOutcome, ...]:
    """Run the eight canonical stages for one ``(agent, world)`` cell.

    Each stage callable composes a shipped surface (never re-implemented) and returns a
    :class:`StageResult`; this compares the produced fields against the committed oracle
    and records a :class:`StageOutcome`. A stage whose shipped feature raises a typed domain error
    is not swallowed -- the harness records the stage + the error type on the outcome and marks it
    failed, never a catch-and-continue partial pass.
    """
    _assert_plan_complete(plan)
    outcomes: list[StageOutcome] = []
    for sp in plan.stages:
        try:
            result = sp.run(ctx)
        except FlagshipError:
            raise  # a proof-integrity failure (missing oracle, unreachable --live) propagates
        except Exception as exc:  # noqa: BLE001 -- a stage's typed domain error fails the proof
            outcomes.append(
                StageOutcome(
                    agent=ctx.agent,
                    world=ctx.world,
                    stage=sp.stage,
                    ok=False,
                    oracle=OracleCheck(
                        stage=sp.stage,
                        oracle_path=None,
                        matched=False,
                        divergence=f"{type(exc).__name__}: {exc}",
                    ),
                    error_type=type(exc).__name__,
                    note=f"stage raised {type(exc).__name__} -- proof failed",
                )
            )
            continue

        if (
            result.oracle_path is not None
            and not plan.oracle_required
            and not Path(result.oracle_path).exists()
        ):
            # Nothing to compare against: record the row as unpinned, never as a match.
            oracle = OracleCheck(
                stage=sp.stage, oracle_path=result.oracle_path, matched=False, pinned=False
            )
        elif result.oracle_path is not None:
            oracle = compare_oracle(sp.stage, result.produced, result.oracle_path)
        else:
            # Oracle-less stage proven structurally; the callable's note records the check made.
            oracle = OracleCheck(stage=sp.stage, oracle_path=None, matched=True, divergence=None)

        gate_ok = result.gate is None or result.gate.admitted
        outcomes.append(
            StageOutcome(
                agent=ctx.agent,
                world=ctx.world,
                stage=sp.stage,
                ok=(oracle.matched or not oracle.pinned) and gate_ok,
                oracle=oracle,
                headline=result.headline,
                artifact_ids=result.artifact_ids,
                note=result.note,
            )
        )
    return tuple(outcomes)


def assert_worlds_equal(
    agent: Agent, comparable_by_world: Mapping[World, dict[str, Any]]
) -> WorldParity:
    """Assert the normalized results are field-for-field equal across worlds.

    Compares the world-invariant artifacts (normalized trace/label/score dicts) each world produced;
    ``equal=False`` names the first divergent field -- a divergence is a blocker that flips
    the proof to ``failed``. With fewer than two worlds there is nothing to compare, so
    parity holds vacuously (a single-world run is not a both-worlds claim).
    """
    worlds = tuple(comparable_by_world.keys())
    if len(worlds) < 2:
        only: World = worlds[0] if worlds else "local"
        return WorldParity(agent=agent, compared=(only, only), equal=True, divergence=None)
    a, b = worlds[0], worlds[1]
    doc_a, doc_b = comparable_by_world[a], comparable_by_world[b]
    # Canonicalize to JSON so ordering/tuple-vs-list differences do not read as divergences.
    if json.dumps(doc_a, sort_keys=True) == json.dumps(doc_b, sort_keys=True):
        return WorldParity(agent=agent, compared=(a, b), equal=True, divergence=None)
    divergence = _first_divergent_key(doc_a, doc_b)
    return WorldParity(agent=agent, compared=(a, b), equal=False, divergence=divergence)


def _first_divergent_key(a: Any, b: Any, prefix: str = "") -> str:
    """Human path to the first field where two comparable docs diverge (for the parity message)."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{prefix}{key} (present in one world only)"
            sub = _first_divergent_key(a[key], b[key], prefix=f"{prefix}{key}.")
            if sub:
                return sub
        return ""
    if a != b:
        return f"{prefix.rstrip('.') or '<root>'}: {a!r} != {b!r}"
    return ""


def prove(
    config: FlagshipConfig,
    plans: Sequence[AgentPlan],
    *,
    backend_factory: Callable[[World], TraceStoreBackend],
    gates: Sequence[GateDecisionRef] = (),
    flagship_id: str = "flagship",
) -> FlagshipReport:
    """Run the whole flagship proof and return the reconstructable :class:`FlagshipReport`.

    ``plans`` supplies the per-agent corpus binding (one :class:`AgentPlan` per requested agent);
    ``backend_factory`` returns the backend for a world (local or Langfuse) so the
    orchestrator never handles a vendor object. ``gates`` carries the judge-trust decisions
    (:func:`calibrated_agent_evaluation.enforce.admission.admit_judge` outputs) so the report
    records whether every scoring judge was fully admitted.

    The overall verdict is ``proven`` iff every ``StageOutcome.ok`` is True for every configured
    ``(agent, world)`` AND every ``WorldParity.equal`` is True AND every ``GateDecisionRef`` is
    ``admitted``; otherwise ``failed`` with ``reasons`` naming the offending stage/parity/gate.
    With no failures but an unpinned stage, the verdict is ``unpinned``. Every requested agent
    needs exactly one registered plan; an unregistered or doubly-registered id is refused.
    """
    fixtures_root = Path(config.fixtures_root)
    fixture_version = _assert_fixture_version(fixtures_root)  # before any stage runs

    plan_by_agent: dict[Agent, AgentPlan] = {}
    for p in plans:
        if p.agent in plan_by_agent:
            raise FlagshipError(
                f"more than one AgentPlan registered for agent {p.agent!r}; one would silently "
                "shadow the other"
            )
        plan_by_agent[p.agent] = p
    missing = [a for a in config.agents if a not in plan_by_agent]
    if missing:
        raise FlagshipError(
            f"no AgentPlan supplied for requested agent(s) {missing}; a proof cannot run an agent "
            "it was given no plan for"
        )

    all_stages: list[StageOutcome] = []
    all_parity: list[WorldParity] = []
    reasons: list[str] = []

    for agent in config.agents:
        plan = plan_by_agent[agent]
        comparable_by_world: dict[World, dict[str, Any]] = {}
        for world in config.worlds:
            backend = backend_factory(world)
            ctx = StageContext(
                agent=agent,
                world=world,
                model_plane=config.model_plane,
                backend=backend,
                fixtures_root=fixtures_root,
            )
            outcomes = run_agent_pipeline(plan, ctx)
            all_stages.extend(outcomes)
            for oc in outcomes:
                if not oc.ok:
                    reasons.append(
                        f"{agent}/{world}/{oc.stage} failed: "
                        f"{oc.oracle.divergence or oc.error_type or 'oracle mismatch'}"
                    )
            comparable_by_world[world] = plan.comparable(ctx)

        parity = assert_worlds_equal(agent, comparable_by_world)
        all_parity.append(parity)
        if not parity.equal:
            reasons.append(
                f"{agent} both-worlds divergence at {parity.divergence} "
                f"({parity.compared[0]} != {parity.compared[1]}); the two worlds must agree"
            )

    for g in gates:
        if not g.admitted:
            reasons.append(f"judge {g.judge_id} not admitted: {g.refusal_reason}")

    if reasons:
        verdict = "failed"
    elif any(not oc.oracle.pinned for oc in all_stages):
        verdict = "unpinned"
    else:
        verdict = "proven"
    return FlagshipReport(
        flagship_id=flagship_id,
        verdict=verdict,
        agents=tuple(config.agents),
        worlds=tuple(config.worlds),
        model_plane=config.model_plane,
        fixture_version=fixture_version,
        stages=tuple(all_stages),
        gates=tuple(gates),
        parity=tuple(all_parity),
        reasons=tuple(reasons),
        inputs={
            "agents": ",".join(config.agents),
            "worlds": ",".join(config.worlds),
            "model_plane": config.model_plane,
            "fixtures_root": str(config.fixtures_root),
        },
        provenance=capture_provenance(),
    )
