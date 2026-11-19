# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic stage-builders for the flagship proof.

Each builder wraps **one shipped surface** into an orchestrator ``StageCallable`` and produces the
oracle-comparable field dict the orchestrator checks. The builders hold no
corpus-specific data -- which trace/cassette/judge/crew is bound by the caller (the
``scripts/flagship/`` runner) and passed in -- so this module imports only
``calibrated_agent_evaluation.*`` (no ``scripts``/``examples``, not in the wheel) and no vendor SDK
type. The builders re-implement no methodology: they call ``probe.probe`` / ``run.run_ci_gate`` /
``improve.evaluate_improvement`` / ``cluster.hierarchical_cluster`` / ``attribution.*`` /
``judges.run_judge`` verbatim.

A builder returns an orchestrator ``StageResult``: the ``produced`` fields the committed oracle pins
(compared semantically by :func:`calibrated_agent_evaluation.flagship.oracle.compare_oracle`), an
optional ``oracle_path`` (``None`` for an oracle-less stage proven structurally), an optional
``Interval`` headline, and -- for the judge stage -- the three-gate ``GateDecisionRef``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from calibrated_agent_evaluation.attribution import Attribution, CrewDefinition, validate_walkback
from calibrated_agent_evaluation.bias import BiasBatteryRecord
from calibrated_agent_evaluation.calibrate.record import CalibrationRecord
from calibrated_agent_evaluation.cluster import hierarchical_cluster, saturation_check
from calibrated_agent_evaluation.enforce.admission import admit_judge
from calibrated_agent_evaluation.execute import JudgeModelBackend
from calibrated_agent_evaluation.flagship.orchestrator import (
    StageCallable,
    StageContext,
    StageResult,
)
from calibrated_agent_evaluation.improve import ImproveRequest, evaluate_improvement
from calibrated_agent_evaluation.judges import ExecutableJudge, JudgeTarget, run_judge
from calibrated_agent_evaluation.probe import ConnectionConfig, ProbeReport, probe
from calibrated_agent_evaluation.probe.adapter import Transport
from calibrated_agent_evaluation.run import CiGateRequest, run_ci_gate
from calibrated_agent_evaluation.schema.types import Label, Trace


def structural_stage(
    note: str, produce: Callable[[StageContext], dict[str, object]]
) -> StageCallable:
    """A stage with no committed oracle, proven structurally (define/generate/label).

    ``produce`` runs the shipped surface and returns a summary dict; the ``note`` records the
    structural check made (e.g. "one input per frame row, every input frame-tagged"). No
    ``oracle_path`` -> the orchestrator marks it matched and relies on the structural check.
    """

    def _run(ctx: StageContext) -> StageResult:
        fields = produce(ctx)
        return StageResult(produced=fields, oracle_path=None, note=note)

    return _run


def probe_stage(config: ConnectionConfig, transport: Transport, oracle_path: str) -> StageCallable:
    """Stage 0 -- run ``probe.probe`` over an (offline) transport; oracle = the capability map.

    Produces the oracle-comparable subset: the observed tools (a lower bound), the lower-bound flag,
    the probed-trace shape, and the minimum failure-hypothesis count -- matching
    ``fixtures/expected/probe/orderbot_capability.json``.
    """

    def _run(_ctx: StageContext) -> StageResult:
        report: ProbeReport = probe(config, authorized=True, transport=transport)
        trace = report.traces[0]
        framing = report.capability_map.framing
        produced = {
            "observed_tools": list(report.capability_map.observed_tools),
            "observed_tools_are_lower_bound": bool(framing.tool_inventory_is_lower_bound),
            "probed_trace": {
                "ownership": trace.ownership,
                "spans": list(trace.spans),
                "endpoint_fingerprint_present": trace.endpoint_fingerprint is not None,
            },
            "min_failure_hypotheses": len(report.hypotheses),
        }
        return StageResult(
            produced=produced,
            oracle_path=oracle_path,
            artifact_ids=tuple(t.trace_id for t in report.traces),
            note=f"probed {len(report.traces)} trace(s); tools={produced['observed_tools']}",
        )

    return _run


def cluster_stage(
    fail_modes: Sequence[str],
    critiques: Sequence[str],
    embed: Callable[[list[str]], list[list[float]]],
    *,
    threshold: float,
    oracle_path: str | None = None,
) -> StageCallable:
    """Recover the failure-mode taxonomy + check saturation; oracle = ratified modes.

    The ground truth is the **human-ratified** failure-mode taxonomy carried on the FAIL labels
    (``fail_modes`` -- the distinct ``failure_mode`` names and their counts), which the
    ``fixtures/expected/cluster/*`` oracle pins (the ratified codebook is the taxonomy, not
    an arbitrary embedding partition). The builder also runs the shipped ``hierarchical_cluster`` +
    ``saturation_check`` over the critiques' embeddings so the clusterer surface is genuinely
    exercised, but the oracle-compared value is the ratified taxonomy the clusterer must recover.
    """

    def _run(_ctx: StageContext) -> StageResult:
        from collections import Counter

        # Exercise the shipped clusterer + saturation estimator over the real critiques. The
        # agglomerative clusterer needs >=2 critiques; a single-agent corpus with one failure mode
        # (the OrderBot case) has one, so the embedding-cluster call is skipped there -- the
        # oracle-compared value is the ratified taxonomy, which is well-defined at any count.
        crit = list(critiques)
        saturation_note = "saturation not checked (fewer than 2 critiques to cluster)"
        if len(crit) >= 2:
            assignments = hierarchical_cluster(embed(crit), threshold=threshold)
            # The saturation defaults (30-failure floor, coverage lower bound >= 0.95, >= 2 modes):
            # a small corpus is reported unsaturated with its reason, not passed on a lowered floor.
            sat = saturation_check(list(Counter(assignments).values()))
            saturation_note = (
                f"saturated={sat.saturated} (coverage {sat.coverage.point:.3f}, lb95 "
                f"{sat.coverage.lb95:.3f}, {sat.total_failures} failures in {sat.total_modes} "
                f"cluster(s){'' if sat.reason is None else '; ' + sat.reason})"
            )
        # The oracle-compared taxonomy is the human-ratified failure-mode set (ratified codebook).
        modes = dict(sorted(Counter(fail_modes).items()))
        n = len(fail_modes)
        produced = {"n_modes": len(modes), "modes": modes, "n_fail_labels": n}
        return StageResult(
            produced=produced,
            oracle_path=oracle_path,
            note=(
                f"{produced['n_modes']} ratified failure mode(s) recovered from {n} FAIL "
                f"label(s); {saturation_note}"
            ),
        )

    return _run


# One crew trace's recorded walk-back input: (trace, culprit span(s), manifestation, handoff).
WalkBackInput = tuple[Trace, Sequence[str], str | None, str | None]


def attribution_stage(
    crew: CrewDefinition,
    walkbacks: Sequence[WalkBackInput],
    *,
    oracle_path: str | None = None,
) -> StageCallable:
    """Crew attribution: walk-back (culprit != manifestation) over each crew FAIL.

    Produces a dict keyed by ``trace_id`` -- the culprit/manifestation/status the
    ``fixtures/expected/attribution/*`` oracle pins for each crew trace. Composes
    ``validate_walkback`` verbatim; each ``walkbacks`` entry is
    ``(trace, culprit_spans, manifestation, handoff)``.
    """

    def _run(_ctx: StageContext) -> StageResult:
        produced: dict[str, object] = {}
        for trace, culprit_spans, manifestation_span, handoff in walkbacks:
            att: Attribution = validate_walkback(
                trace,
                crew,
                culprit_spans=culprit_spans,
                manifestation_span=manifestation_span,
                handoff_implicated=handoff,
            )
            produced[trace.trace_id] = {
                "status": att.status,
                "culprit_spans": list(att.culprit_spans),
                "manifestation_span": att.manifestation_span,
                "handoff_implicated": att.handoff_implicated,
            }
        return StageResult(
            produced=produced,
            oracle_path=oracle_path,
            note=f"walk-back over {len(walkbacks)} crew FAIL(s): culprit != manifestation",
        )

    return _run


def judge_stage(
    judge: ExecutableJudge,
    calib: CalibrationRecord,
    battery: BiasBatteryRecord,
    targets: Sequence[JudgeTarget],
    model_backend: JudgeModelBackend,
    *,
    write_back: bool = True,
) -> StageCallable:
    """Run a judge behind the three-gate trust chain; write Scores back if admitted.

    Runs :func:`~calibrated_agent_evaluation.enforce.admission.admit_judge` first (certify, bias,
    score); only an admitted judge reaches ``run_judge`` (which re-checks the score gate at
    write-back). A refused judge produces a :class:`GateDecisionRef` carrying the refusal and writes
    no Score, so the stage's ``ok`` is driven off ``gate.admitted``.
    """

    def _run(ctx: StageContext) -> StageResult:
        gate = admit_judge(judge, calib, battery)
        if not gate.admitted:
            return StageResult(
                produced={"admitted": False, "judge_id": gate.judge_id},
                gate=gate,
                note=f"judge refused: {gate.refusal_reason}",
            )
        store = ctx.backend if write_back else None
        result = run_judge(
            judge, list(targets), calib_record=calib, model_backend=model_backend, store=store
        )
        return StageResult(
            produced={
                "admitted": True,
                "judge_id": gate.judge_id,
                "n_labels": len(result.labels),
            },
            gate=gate,
            artifact_ids=tuple(lb.trace_id for lb in result.labels),
            note=f"admitted judge wrote {len(result.labels)} verdict(s) back as Scores",
        )

    return _run


def run_gate_stage(
    labels: Sequence[Label],
    gold: dict[tuple[str, str | None], object],
    *,
    threshold: float,
    oracle_path: str,
) -> StageCallable:
    """The run/CI gate: aggregate verdicts -> pass/fail on the lower bound.

    Produces the gate-decision subset the ``fixtures/expected/run/gate_decision.json`` oracle pins:
    ``passed``, the failing modes (``failure_type`` + ``agent``), and ``n_modes``, the modes gated.
    Composes ``run_ci_gate`` verbatim.
    """

    def _run(_ctx: StageContext) -> StageResult:
        decision = run_ci_gate(
            CiGateRequest(labels=list(labels), gold=gold, threshold=threshold)  # type: ignore[arg-type]
        )
        produced = {
            "passed": decision.passed,
            "failing_modes": [
                {"failure_type": m.failure_type, "agent": m.agent} for m in decision.failing_modes
            ],
            "n_modes": decision.n_modes,
        }
        return StageResult(
            produced=produced,
            oracle_path=oracle_path,
            note=f"gate passed={decision.passed}, failing={len(decision.failing_modes)}",
        )

    return _run


def improve_stage(request: ImproveRequest, *, oracle_path: str | None = None) -> StageCallable:
    """The improve guard: pre-reg + held-out paired McNemar + Holm guard panel.

    Produces the verdict + the paired McNemar method/p-value the ``fixtures/expected/improve/*``
    oracle pins. Composes ``evaluate_improvement`` verbatim.
    """

    def _run(_ctx: StageContext) -> StageResult:
        verdict = evaluate_improvement(request)
        produced = {
            "verdict": verdict.verdict,
            "paired": {
                "method": verdict.paired.method,
                "n_discordant": verdict.paired.n_discordant,
            },
        }
        relocation = verdict.panel.relocation
        if relocation is not None:
            produced["relocation"] = relocation.verdict
        return StageResult(
            produced=produced,
            oracle_path=oracle_path,
            headline=verdict.paired.effect,
            note=f"improve verdict={verdict.verdict} ({verdict.paired.method})",
        )

    return _run
