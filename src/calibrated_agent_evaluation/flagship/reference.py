# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reference-corpus plans: OrderBot and ClaimsCrew bound to a checkout's ``fixtures/`` tree.

:func:`~calibrated_agent_evaluation.flagship.prove` drives injected agent plans. This module builds
the two the flagship proof runs, from the committed corpus under a ``fixtures_root`` the caller
names (the wheel ships no corpus, so this needs a source checkout). Each plan composes the shipped
stage builders with the corpus inputs (which cassette, judge, crew and oracle); it computes no
statistic.

The OrderBot judge is admitted on records the caller supplies, so whether a calibration and a
bias battery are real or illustrative is the caller's statement, never this module's.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.attribution import (
    Attribution,
    CrewDefinition,
    CrewOutcomes,
    ExpectedProcess,
    MandatedRule,
    build_crew_definition,
    validate_walkback,
)
from calibrated_agent_evaluation.bias import BiasBatteryRecord
from calibrated_agent_evaluation.calibrate.record import CalibrationRecord, Interval
from calibrated_agent_evaluation.execute import CassetteJudgeModel
from calibrated_agent_evaluation.execute.cassette import Cassette, Mode
from calibrated_agent_evaluation.flagship.errors import FlagshipError
from calibrated_agent_evaluation.flagship.orchestrator import AgentPlan, StageContext, StagePlan
from calibrated_agent_evaluation.flagship.stages import (
    attribution_stage,
    cluster_stage,
    improve_stage,
    judge_stage,
    probe_stage,
    run_gate_stage,
    structural_stage,
)
from calibrated_agent_evaluation.improve import (
    GuardSlice,
    ImproveRequest,
    PairedObs,
    PreRegEntry,
    PreRegLedger,
    register_prereg,
)
from calibrated_agent_evaluation.judges import ExecutableJudge, workflow_target
from calibrated_agent_evaluation.probe import ConnectionConfig
from calibrated_agent_evaluation.run import ModeKey
from calibrated_agent_evaluation.schema import BackendError, CalibratedAgentEvaluationDataError
from calibrated_agent_evaluation.schema.serde import label_from_dict, trace_from_dict
from calibrated_agent_evaluation.schema.types import Label, Trace

__all__ = [
    "ORDERBOT_JUDGE_FILE",
    "ReferenceCorpus",
    "claimscrew_definition",
    "claimscrew_improve_request",
    "walkback_from_gold",
]

ORDERBOT_JUDGE_FILE = Path("judge-files") / "orderbot-refund-resolved.json"
"""The OrderBot judge file, relative to the fixtures root; ``judge`` reads the same one."""
_ORDERBOT_CASSETTE = Path("raw") / "cassettes" / "judge_orderbot_resolved.cassette.json"
_READ_ATTEMPTS = 15
_READ_WAIT_SECONDS = 3.0


def claimscrew_definition() -> CrewDefinition:
    """The ClaimsCrew Define artifact: its agents, handoffs, the vacancy scenario and two rules."""
    vacancy = ExpectedProcess(
        scenario="S2_vacancy_exclusion",
        expected_agents=("supervisor", "intake", "policy-checker", "fraud-scorer", "payout"),
        terminal_decision="deny",
        payout_allowed=False,
        escalation_required=False,
    )
    return build_crew_definition(
        agents=("supervisor", "intake", "policy-checker", "fraud-scorer", "payout", "escalation"),
        handoffs=(
            "intake->policy-checker",
            "policy-checker->fraud-scorer",
            "fraud-scorer->payout",
            "fraud-scorer->escalation",
        ),
        scenarios={"S2_vacancy_exclusion": vacancy},
        rules=(
            MandatedRule("R2", "fraud flag raised", "escalation", deterministic=False),
            MandatedRule("R3", "occupancy narrative present", "intake"),
        ),
    )


def walkback_from_gold(trace: Trace, crew: CrewDefinition, gold: Label) -> Attribution:
    """Validate a human gold attribution against the trace's real span tree.

    A ``"disputed"`` culprit stands for the two defensible culprits, intake and policy-checker.
    """
    if gold.culprit_span == "disputed":
        culprits: tuple[str, ...] = (
            f"{trace.trace_id}/intake",
            f"{trace.trace_id}/policy-checker",
        )
    else:
        culprits = (gold.culprit_span,) if gold.culprit_span else ()
    return validate_walkback(
        trace,
        crew,
        culprit_spans=culprits,
        manifestation_span=gold.manifestation_span,
        handoff_implicated=gold.handoff_implicated,
    )


def _illustrative(point: float) -> Interval:
    return Interval(
        point=point,
        lo=point - 0.05,
        hi=point + 0.05,
        refusal_reason=None,
        method="illustrative",
        level=0.95,
        sided="two-sided",
        n=100,
    )


def claimscrew_improve_request() -> ImproveRequest:
    """A ClaimsCrew fix that relocates: intake failures fall, the workflow rate stays flat."""
    crew = CrewDefinition(
        agents=("intake", "policy-checker"),
        handoffs=("intake->policy-checker",),
        scenarios={
            "clm": ExpectedProcess(
                scenario="clm",
                expected_agents=("intake", "policy-checker"),
                terminal_decision="approve",
                payout_allowed=True,
                escalation_required=False,
            )
        },
        rules=(),
    )
    ledger = register_prereg(
        PreRegLedger(),
        PreRegEntry(
            prereg_id="p-clm",
            target_metric="workflow_outcome_rate",
            success_threshold=0.10,
            is_crew=True,
        ),
    )
    pairs = tuple(
        PairedObs(f"CLM-conf-{i}", before_fail=(i < 18), after_fail=(i < 4)) for i in range(40)
    )
    return ImproveRequest(
        ledger=ledger,
        target_metric="workflow_outcome_rate",
        pairs=pairs,
        confirmation_ids=tuple(f"CLM-conf-{i}" for i in range(40)),
        development_ids=("CLM-4471", "CLM-4473"),
        slices=(
            GuardSlice(
                name="policy-checker",
                kind="agent",
                regression_p=0.8,
                rate_before=_illustrative(0.15),
                rate_after=_illustrative(0.15),
            ),
        ),
        crew=crew,
        crew_before=CrewOutcomes(
            workflow_failures=60, workflow_total=200, per_agent_failures={"intake": 60}
        ),
        crew_after=CrewOutcomes(
            workflow_failures=60, workflow_total=200, per_agent_failures={"intake": 20}
        ),
        touched_agent="intake",
    )


def _orderbot_improve_request() -> ImproveRequest:
    """A held-out paired slice: 8 of 20 failing before, 3 after, all five changes fail to pass."""
    pairs = tuple(
        PairedObs(f"ORD-conf-{i}", before_fail=i < 8, after_fail=i < 3) for i in range(20)
    )
    # 8/20 -> 3/20 has an effect interval whose upper bound clears a two-point reduction.
    ledger = register_prereg(
        PreRegLedger(),
        PreRegEntry(
            prereg_id="p-ob", target_metric="trusted_broken_policy_tool", success_threshold=0.02
        ),
    )
    return ImproveRequest(
        ledger=ledger,
        target_metric="trusted_broken_policy_tool",
        pairs=pairs,
        confirmation_ids=tuple(f"ORD-conf-{i}" for i in range(20)),
        development_ids=("ORD-1001", "ORD-1002"),
        slices=(
            GuardSlice(
                name="misread_amount",
                kind="other",
                regression_p=0.7,
                rate_before=Interval(
                    point=0.06,
                    lo=0.01,
                    hi=0.11,
                    refusal_reason=None,
                    method="illustrative",
                    level=0.95,
                    sided="two-sided",
                    n=100,
                ),
                rate_after=Interval(
                    point=0.06,
                    lo=0.01,
                    hi=0.11,
                    refusal_reason=None,
                    method="illustrative",
                    level=0.95,
                    sided="two-sided",
                    n=100,
                ),
            ),
        ),
    )


def _run_gate_inputs() -> tuple[list[Label], dict[tuple[str, str | None], object]]:
    """Verdicts and gold modes for the run-gate oracle: 48 of 50 and 10 of 20 passing."""
    over_refund = ModeKey(failure_type="trusted_broken_policy_tool")
    dropped = ModeKey(failure_type="dropped_material_fact", agent="intake")
    labels: list[Label] = []
    gold: dict[tuple[str, str | None], object] = {}
    for prefix, total, passing, mode in (("OR", 50, 48, over_refund), ("DR", 20, 10, dropped)):
        for i in range(total):
            label = Label(
                trace_id=f"{prefix}{i}",
                span_id=None,
                kind="llm-judge",
                outcome="PASS" if i < passing else "FAIL",
                critique="c",
            )
            labels.append(label)
            gold[(label.trace_id, label.span_id)] = mode
    return labels, gold


def _deterministic_embed(texts: list[str]) -> list[list[float]]:
    """An offline embedder, so the cluster stage needs no network."""
    return [[b / 255.0 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts]


def _probe_transport(
    url: str,  # noqa: ARG001  # the transport signature, which this stub ignores
    headers: dict[str, str],  # noqa: ARG001
    body: bytes,  # noqa: ARG001
    timeout: float,  # noqa: ARG001
) -> tuple[int, list[bytes]]:
    """An offline server-sent-events transport for the probe stage."""
    return 200, [
        b"data: " + json.dumps({"type": "tool_use", "tool_name": "lookup_order"}).encode(),
        b"data: " + json.dumps({"type": "response", "response": "I help"}).encode(),
    ]


def _trace_doc(trace: Trace) -> dict[str, Any]:
    """The world-invariant fields the cross-world parity check compares."""
    return {
        "trace_id": trace.trace_id,
        "frame": trace.frame,
        "ownership": trace.ownership,
        "taxonomy_cell": dict(trace.taxonomy_cell),
        "n_spans": len(trace.spans),
        "agents": sorted({span.agent for span in trace.spans}),
    }


def _put_and_read(ctx: StageContext, trace: Trace) -> Trace:
    """Persist a trace and read it back, waiting out a backend that ingests asynchronously."""
    ctx.backend.put_trace(trace)
    last: BackendError | None = None
    for attempt in range(_READ_ATTEMPTS):
        try:
            return ctx.backend.get_trace(trace.trace_id)
        except BackendError as exc:
            last = exc
            if attempt + 1 < _READ_ATTEMPTS:
                time.sleep(_READ_WAIT_SECONDS)
    raise BackendError(
        f"trace {trace.trace_id!r} was written but never became readable "
        f"after {_READ_ATTEMPTS} attempts: {last}"
    ) from last


class ReferenceCorpus:
    """The committed reference corpus under one ``fixtures/`` directory, and its two plans."""

    def __init__(self, fixtures_root: str | Path) -> None:
        """Bind to a corpus.

        Args:
            fixtures_root: The checkout's ``fixtures/`` directory.

        Raises:
            FlagshipError: The directory does not hold the corpus (no ``normalized/``).
        """
        self.root = Path(fixtures_root)
        if not (self.root / "normalized").is_dir():
            raise FlagshipError(
                f"{self.root} is not the reference corpus (no normalized/ directory); the "
                "flagship proof runs from a source checkout's fixtures/ tree"
            )
        self._gold_rows: list[dict[str, Any]] | None = None

    def _read_json(self, relative: str | Path) -> Any:
        path = self.root / relative
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FlagshipError(f"cannot read corpus file {path}: {exc}") from exc

    def trace(self, trace_id: str) -> Trace:
        """A normalized corpus trace."""
        try:
            return trace_from_dict(self._read_json(Path("normalized") / f"{trace_id}.json"))
        except CalibratedAgentEvaluationDataError as exc:
            raise FlagshipError(f"corpus trace {trace_id!r} is invalid: {exc}") from exc

    def gold_rows(self) -> list[dict[str, Any]]:
        """The committed gold label rows, as stored."""
        rows = self._gold_rows
        if rows is None:
            rows = self._gold_rows = self._read_json(Path("labels") / "gold.json")
        return rows

    def gold_labels(self) -> dict[tuple[str, str | None], Label]:
        """The gold labels keyed by ``(trace_id, span_id)``."""
        labels = [label_from_dict(row) for row in self.gold_rows()]
        return {(label.trace_id, label.span_id): label for label in labels}

    def orderbot_judge(self) -> ExecutableJudge:
        """The OrderBot judge, read from the same judge file ``judge --judge`` reads."""
        from calibrated_agent_evaluation.cli.errors import CliError
        from calibrated_agent_evaluation.cli.judgefile import load_judge_file

        try:
            return load_judge_file(self.root / ORDERBOT_JUDGE_FILE).judge
        except CliError as exc:
            raise FlagshipError(f"cannot read the OrderBot judge: {exc}") from exc

    def _trace_outcome(self, trace_id: str) -> str:
        for row in self.gold_rows():
            if row["trace_id"] == trace_id and row.get("span_id") is None:
                return str(row["outcome"])
        raise FlagshipError(f"the corpus has no trace-level gold label for {trace_id!r}")

    def _fail_rows(self, prefix: str) -> list[dict[str, Any]]:
        return [
            row
            for row in self.gold_rows()
            if row["trace_id"].startswith(prefix) and row["outcome"] == "FAIL"
        ]

    def _cluster(self, prefix: str, oracle: str) -> StagePlan:
        fails = self._fail_rows(prefix)
        return StagePlan(
            "cluster",
            cluster_stage(
                [row["failure_mode"] for row in fails if row.get("failure_mode")],
                [row["critique"] for row in fails],
                _deterministic_embed,
                threshold=0.5,
                oracle_path=str(self.root / "expected" / "cluster" / oracle),
            ),
        )

    def _label(self, trace_id: str) -> StagePlan:
        return StagePlan(
            "label",
            structural_stage(
                "human gold read from fixtures/labels/gold.json (kind=human)",
                lambda _ctx: {"gold_outcome": self._trace_outcome(trace_id)},
            ),
        )

    def _run_gate(self) -> StagePlan:
        labels, gold = _run_gate_inputs()
        return StagePlan(
            "run",
            run_gate_stage(
                labels,
                gold,
                threshold=0.80,
                oracle_path=str(self.root / "expected" / "run" / "gate_decision.json"),
            ),
        )

    def orderbot_plan(
        self, *, calibration: CalibrationRecord, battery: BiasBatteryRecord
    ) -> AgentPlan:
        """The eight stages for OrderBot, its judge admitted on ``calibration`` and ``battery``.

        Args:
            calibration: The calibration record the judge stage admits the judge on.
            battery: The bias battery the judge stage admits the judge on.
        """
        judge = self.orderbot_judge()
        model = CassetteJudgeModel(
            Cassette(self.root / _ORDERBOT_CASSETTE, mode=Mode.REPLAY), supports_schema=True
        )
        expected = self.root / "expected"
        stages = (
            StagePlan(
                "probe",
                probe_stage(
                    ConnectionConfig(endpoint="https://stub.local/invoke"),
                    _probe_transport,
                    str(expected / "probe" / "orderbot_capability.json"),
                ),
            ),
            StagePlan(
                "define",
                structural_stage(
                    "crew/taxonomy define is a no-op for the single-agent OrderBot (one span)",
                    lambda _ctx: {"defined": True},
                ),
            ),
            StagePlan(
                "generate",
                structural_stage(
                    "inputs-only synthesis is agent-agnostic; the proof reuses the corpus trace",
                    lambda _ctx: {"generated": True},
                ),
            ),
            self._label("ORD-1001"),
            self._cluster("ORD", "orderbot.json"),
            StagePlan(
                "judge",
                judge_stage(
                    judge,
                    calibration,
                    battery,
                    (workflow_target(self.trace("ORD-1001")),),
                    model,
                ),
            ),
            self._run_gate(),
            # The paired oracle is compared end to end by the tests, not field-pinned here.
            StagePlan("improve", improve_stage(_orderbot_improve_request(), oracle_path=None)),
        )
        return AgentPlan(
            agent="orderbot",
            stages=stages,
            comparable=lambda ctx: _trace_doc(_put_and_read(ctx, self.trace("ORD-1001"))),
        )

    def claimscrew_plan(self) -> AgentPlan:
        """The eight stages for ClaimsCrew; its judge stage is the attribution walk-back."""
        walkbacks = [
            (
                self.trace("CLM-4471"),
                ("CLM-4471/intake",),
                "CLM-4471/payout",
                "intake->policy-checker",
            ),
            (self.trace("CLM-4473"), ("CLM-4473/policy-checker",), "CLM-4473/payout", None),
            (
                self.trace("CLM-4477"),
                ("CLM-4477/intake", "CLM-4477/policy-checker"),
                "CLM-4477/payout",
                "intake->policy-checker",
            ),
        ]
        stages = (
            StagePlan(
                "probe",
                structural_stage(
                    "probing is the entry for an agent you do not own; the crew is normalized",
                    lambda _ctx: {"owned": True},
                ),
            ),
            StagePlan(
                "define",
                structural_stage(
                    "crew Define artifact built (agents, handoffs, expected process)",
                    lambda _ctx: {"defined": True},
                ),
            ),
            StagePlan(
                "generate",
                structural_stage(
                    "inputs-only synthesis is agent-agnostic; the proof reuses the crew traces",
                    lambda _ctx: {"generated": True},
                ),
            ),
            self._label("CLM-4471"),
            self._cluster("CLM", "claimscrew.json"),
            StagePlan(
                "judge",
                attribution_stage(
                    claimscrew_definition(),
                    walkbacks,
                    oracle_path=str(self.root / "expected" / "attribution" / "claimscrew.json"),
                ),
            ),
            self._run_gate(),
            # The relocation verdict is compared end to end by the tests, not field-pinned here.
            StagePlan("improve", improve_stage(claimscrew_improve_request(), oracle_path=None)),
        )
        return AgentPlan(
            agent="claimscrew",
            stages=stages,
            comparable=lambda ctx: _trace_doc(_put_and_read(ctx, self.trace("CLM-4471"))),
        )
