# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-stage data types -- the frozen artifacts every run stage reads and the store persists.

Pure, frozen data holders (the "typed functional core" house style): the numeric work lives
in :mod:`calibrated_agent_evaluation.run.gate` / ``trend`` / ``drift_report`` / ``multi`` and the
shared :mod:`calibrated_agent_evaluation.stats`, never in ``__init__``. No vendor SDK type appears
here -- the methodology- plane boundary. The corrected-rate CI is the shared
:class:`~calibrated_agent_evaluation.calibrate.Interval` (reused, never re-modelled) so a refused
correction carries ``point=None`` + ``refusal_reason`` rather than a fabricated float.

  * :class:`ModeKey` / :class:`ModeCounts` -- a failure mode and its per-run confusion counts.
  * :class:`TrendPoint` -- one per failure mode: the Rogan-Gladen corrected rate + CI.
  * :class:`DriftMetric` / :class:`InputDriftReport` / :class:`JudgeDriftReport` -- drift reports.
  * :class:`MultiplicityResult` -- a corrected family's method + comparison count ``m``.
  * :class:`RunDecision` -- a single tiered decision (staleness / flag); ``inputs`` non-empty.
  * :class:`ModeGate` -- one gated mode's counts, lower bound, threshold and result.
  * :class:`RunGateDecision` -- the hard CI pass/fail verdict.
  * :class:`RunReport` -- the full cadence-run rollup (trend + drift + decisions).
  * :class:`RunRecord` -- the persisted, version-pinned audit artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, Self

from pydantic import ConfigDict, model_validator

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._provenance import ProvenanceStamp, capture_provenance
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.run.errors import RunError
from calibrated_agent_evaluation.schema.types import Frame

# Stamped on every persisted RunRecord; load refuses on mismatch (no coercion). Homed here (not
# __init__) so the store + types share one constant without a cycle.
# "2" provenance; "3" Interval kind fields; "4" TrendPoint.raw_rate; "5" drift p-values + method;
# "6" the CI gate no longer records a Holm "ci_gate" family; "7" the provenance stamp names
# calibrated_agent_evaluation_version (records from before the rename are refused); "8" the gate's
# per-mode counts, bounds and failing modes
RUN_SCHEMA_VERSION = "8"

RunMode = Literal["ci", "cadence"]
# Mirrors enforce.decision.Tier; a run decision is warning-tier (staleness/flags) or hard_refusal
# (the gate raises rather than emitting a hard_refusal decision, so in practice run decisions warn).
Tier = Literal["hard_refusal", "warning", "display_default"]
_TIERS: frozenset[str] = frozenset({"hard_refusal", "warning", "display_default"})
MultMethod = Literal["holm", "fdr_by"]


class ModeKey(CalibratedAgentEvaluationModel):
    """A failure-mode key: ``failure_type`` alone, or ``(agent, failure_type)`` for a crew.

    ``agent`` is ``None`` for single-agent and workflow-level modes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = RunError

    failure_type: str
    agent: str | None = None

    def __hash__(self) -> int:
        """Hash on the key's fields, so a mode keys dicts and sets like the tuple it names."""
        return hash((self.failure_type, self.agent))


@dataclass(frozen=True)
class DiscoveryEnqueue:
    """A CI-failing trace routed into the labeling queue tagged as discovery.

    ``frame`` is always ``"discovery"``: a trace the CI gate caught is non-random augmentation (not
    a random draw), so it grows the discovery set but must never enter production-frame estimation
    (the ``assert_production_frame`` guard enforces that downstream).
    """

    trace_id: str
    frame: Frame = "discovery"


class ModeCounts(CalibratedAgentEvaluationModel):
    """Per-mode confusion counts aggregated from the flat verdict/scorer ``Label``s.

    ``n_judged_fail`` is the judge/scorer FAIL count over the ``n_total`` items of this mode; the
    gate turns it into a pass-rate lower bound and the trend turns it into a corrected rate.
    ``calib_id`` names the :class:`~calibrated_agent_evaluation.calibrate.CalibrationRecord` that
    corrects this mode (its own judge).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = RunError

    mode: ModeKey
    n_total: int
    n_judged_fail: int
    calib_id: str | None = None


class TrendPoint(CalibratedAgentEvaluationModel):
    """One per failure mode: the Rogan-Gladen corrected rate + CI, beside the raw rate.

    ``corrected_rate`` is the shared :class:`~calibrated_agent_evaluation.calibrate.Interval`: a
    refused correction (J<0.5 or non-identifiable) carries ``point=None`` + ``refusal_reason`` and
    the mode is still present with its widening interval -- never dropped, never a fabricated point.

    ``raw_rate`` is the judge's uncorrected FAIL fraction on the same rate sample with its
    two-sided 95% Wilson interval: what a reader would conclude without calibration. It is
    reported so the correction's effect is visible, never as a substitute for ``corrected_rate``;
    it stays populated when the correction is refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = RunError

    mode: ModeKey
    n_total: int
    corrected_rate: Interval
    calib_id: str | None = None
    raw_rate: Interval


class DriftMetric(CalibratedAgentEvaluationModel):
    """One input-drift statistic, its permutation p-value, and whether it was flagged.

    ``value`` is the statistic as ``method`` computes it (PSI over baseline-quantile bins, JSD
    over the same bins, Wasserstein divided by the pooled IQR). ``p_value`` is its permutation
    test against the same-distribution null. For PSI and Wasserstein, ``exceeded`` is True only
    when that p-value survives the BY-FDR correction across those two AND, where ``threshold`` is
    set, ``value`` is above it; ``threshold=None`` means significance alone decides. JSD is
    reported only: its ``threshold`` is None and ``exceeded`` is always False. Records written
    before JSD left the gate may carry a JSD flag from the older three-metric family.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = RunError

    name: Literal["psi", "jsd", "wasserstein"]
    value: float
    threshold: float | None
    exceeded: bool
    p_value: float
    method: str


class InputDriftReport(CalibratedAgentEvaluationModel):
    """Input-distribution drift: PSI + JSD + Wasserstein over one input dimension.

    ``multiplicity`` is the BY-FDR family the gated metrics' p-values were corrected in (PSI and
    Wasserstein, in that order; JSD is reported outside it), and
    ``n_permutations`` / ``seed`` say how those p-values were drawn, so the flags can be recomputed
    from the record. ``n_baseline`` / ``n_fresh`` are the two sample sizes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = RunError

    metrics: tuple[DriftMetric, ...]
    dimension: str
    multiplicity: MultiplicityResult | None = None
    n_permutations: int | None = None
    seed: int | None = None
    n_baseline: int | None = None
    n_fresh: int | None = None

    @property
    def any_exceeded(self) -> bool:
        """True when any metric crossed its threshold (the staleness input)."""
        return any(m.exceeded for m in self.metrics)


class JudgeDriftReport(CalibratedAgentEvaluationModel):
    """Judge drift vs a frozen baseline: ΔTPR / ΔTNR, each with a CI.

    Each delta is an :class:`~calibrated_agent_evaluation.calibrate.Interval` (``point`` = the
    delta, ``lo``/``hi`` the CI) so the confidence treatment travels with the number. There is no
    Δκ: TPR and TNR describe a binary judge completely, and κ on a hunted calibration set also moves
    with the hunt's class mix, which says nothing about the judge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = RunError

    judge_id: str
    delta_tpr: Interval
    delta_tnr: Interval


class MultiplicityResult(CalibratedAgentEvaluationModel):
    """A corrected family's method + comparison count ``m`` + reject vector.

    ``m`` is stated per run. ``family`` names the correction site ("drift_flags",
    "improve_guard", ...); ``method`` is Holm (a hard-reject guardrail) or fdr_by (exploratory
    flags).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = RunError

    family: str
    method: MultMethod
    m: int
    reject: tuple[bool, ...]


class RunDecision(CalibratedAgentEvaluationModel):
    """A single tiered run decision -- a staleness warning or an exploratory flag.

    ``inputs`` is validated non-empty at construction: a recorded decision with no inputs is refused
    rather than silently accepted (the ``EnforcementDecision`` precedent). Staleness decisions
    are always ``warning``-tier and never flip the gate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = RunError

    rule: str
    tier: Tier
    inputs: dict[str, str]
    outcome: str
    message: str

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        self._post_init()
        return self

    def _post_init(self) -> None:
        # Reject an out-of-set tier loudly rather than persisting an un-auditable value.
        if self.tier not in _TIERS:
            raise RunError(f"RunDecision.tier {self.tier!r} is not one of {sorted(_TIERS)}")
        if not self.inputs:
            raise RunError(
                f"RunDecision(rule={self.rule!r}) requires non-empty inputs "
                "(a recorded decision with no inputs is not auditable --)"
            )


class ModeGate(CalibratedAgentEvaluationModel):
    """One gated mode: its verdict counts, its pass-rate 95% lower bound and the bar it must clear.

    ``lower_bound`` is the one-sided Wilson bound on ``(n_total - n_fail) / n_total``; ``passed``
    is ``lower_bound >= threshold``. ``threshold`` is this mode's own bar (a per-mode override, or
    the gate's default).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = RunError

    mode: ModeKey
    n_total: int
    n_fail: int
    lower_bound: float
    threshold: float
    passed: bool


class RunGateDecision(CalibratedAgentEvaluationModel):
    """The hard CI pass/fail verdict.

    ``passed`` is True iff every gated mode's pass-rate 95% lower bound cleared ``threshold``;
    ``failing_modes`` names the modes that did not and ``n_modes`` counts the modes gated. ``modes``
    holds each gated mode's counts and bound, in the order the gate checked them. Requiring
    every mode to pass is an intersection-union test, so the false-pass rate is capped at the
    per-mode alpha without a correction across modes, and none is recorded. ``decisions`` carries
    attached non-gating warnings (e.g. staleness): they are recorded but never flip ``passed``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = RunError

    passed: bool
    failing_modes: tuple[ModeKey, ...]
    threshold: float
    n_modes: int
    decisions: tuple[RunDecision, ...] = ()
    modes: tuple[ModeGate, ...] = ()


class RunReport(CalibratedAgentEvaluationModel):
    """The full cadence-run rollup -- the vendor-neutral artifact the dashboard sink publishes.

    Holds the per-mode trend, the input/judge drift, every corrected
    family's multiplicity result, and the tiered decisions (staleness + flags). It carries
    no confidence-treatment-free headline number: the only rates it holds are the
    per-``TrendPoint`` corrected-rate intervals.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = RunError

    run_id: str
    mode: RunMode
    trend: tuple[TrendPoint, ...]
    input_drift: InputDriftReport | None = None
    judge_drift: tuple[JudgeDriftReport, ...] = ()
    multiplicity: tuple[MultiplicityResult, ...] = ()
    decisions: tuple[RunDecision, ...] = ()
    gate: RunGateDecision | None = None


class NumericPoint(CalibratedAgentEvaluationModel):
    """One numeric dashboard series point flattened from a run report.

    The vendor-neutral unit the dashboard sinks publish: a named ``value`` for a ``run_id``, with
    its confidence treatment. ``kind`` distinguishes a per-mode corrected rate ("rate") from a
    drift metric ("drift"). A rate carries its interval (``lo``/``hi``/``interval_method``); a
    drift metric carries its permutation ``p_value``. A refused correction is a row too, with
    ``value=None`` and its ``refusal_reason`` and the widened interval, so a refusal is visible in
    the series instead of missing from it. The local sink round-trips these to CSV+JSON
    (authoritative in both worlds); the Langfuse sink maps each to a score.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = RunError

    name: str
    value: float | None
    kind: Literal["rate", "drift"]
    run_id: str
    lo: float | None = None
    hi: float | None = None
    interval_method: str | None = None
    p_value: float | None = None
    refusal_reason: str | None = None


def numeric_series(report: RunReport) -> tuple[NumericPoint, ...]:
    """Flatten a :class:`RunReport` into the numeric dashboard series.

    One point per mode with its corrected rate and interval (a refused correction is a point with
    ``value=None`` and the refusal reason -- no fabricated number, no silent gap), plus one point
    per input-drift metric with its p-value. Vendor-neutral; the sinks consume exactly this.
    """
    points: list[NumericPoint] = []
    for tp in report.trend:
        label = tp.mode.failure_type
        if tp.mode.agent is not None:
            label = f"{tp.mode.agent}/{label}"
        rate = tp.corrected_rate
        points.append(
            NumericPoint(
                name=f"rate.{label}",
                value=rate.point,
                kind="rate",
                run_id=report.run_id,
                lo=rate.lo,
                hi=rate.hi,
                interval_method=rate.method,
                refusal_reason=rate.refusal_reason if rate.point is None else None,
            )
        )
    if report.input_drift is not None:
        for metric in report.input_drift.metrics:
            points.append(
                NumericPoint(
                    name=f"drift.{report.input_drift.dimension}.{metric.name}",
                    value=metric.value,
                    kind="drift",
                    run_id=report.run_id,
                    p_value=metric.p_value,
                )
            )
    return tuple(points)


class RunRecord(CalibratedAgentEvaluationModel):
    """The persisted, version-pinned audit artifact.

    Records the run mode, the gate outcome, how many modes it gated, which failed and each mode's
    counts and lower bound, the judge/calibration ids
    and scorer lanes used, the per-mode corrected rates, the drift, every corrected family + its
    ``m``, the tiered decisions, and the inputs that produced the decision. ``inputs`` is validated
    non-empty. Round-trips through :mod:`calibrated_agent_evaluation.run.store`; ``schema_version``
    refuses on mismatch (no coercion).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = RunError

    run_id: str
    mode: RunMode
    gate_passed: bool | None
    judge_ids: tuple[str, ...]
    calib_ids: tuple[str, ...]
    scorer_lanes: tuple[str, ...]
    trend: tuple[TrendPoint, ...]
    input_drift: InputDriftReport | None
    judge_drift: tuple[JudgeDriftReport, ...]
    multiplicity: tuple[MultiplicityResult, ...]
    decisions: tuple[RunDecision, ...]
    inputs: dict[str, str]
    # How many modes the CI gate required to pass (None for a cadence run with no gate).
    gate_n_modes: int | None = None
    # The modes that did not clear the gate, and every gated mode's counts and lower bound.
    gate_failing_modes: tuple[ModeKey, ...] = ()
    gate_modes: tuple[ModeGate, ...] = ()
    schema_version: str = RUN_SCHEMA_VERSION
    # Seed/resamples of the trend's corrected rates and the gate threshold applied.
    provenance: ProvenanceStamp

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        self._post_init()
        return self

    def _post_init(self) -> None:
        if not self.inputs:
            raise RunError(
                f"RunRecord(run_id={self.run_id!r}) requires non-empty inputs "
                "(a run record with no inputs is not reproducible --)"
            )


def run_record_of(
    report: RunReport,
    *,
    inputs: dict[str, str],
    judge_ids: tuple[str, ...] = (),
    scorer_lanes: tuple[str, ...] = (),
    seed: int | None = None,
    n_resamples: int | None = None,
) -> RunRecord:
    """Build the persisted :class:`RunRecord` from a :class:`RunReport`.

    ``gate_passed`` comes from the report's gate (``None`` for a cadence run with no gate), and the
    gate threshold that produced it is stamped on ``provenance.thresholds["gate"]`` so the record
    keeps it. ``calib_ids`` are the trend points' calibration ids, in order, without repeats.
    ``seed`` / ``n_resamples`` are what the trend's corrected rates were computed with; pass the
    values given to :func:`calibrated_agent_evaluation.run.trend_report`. The gate's attached
    decisions are recorded alongside the report's own.
    """
    gate = report.gate
    thresholds = {"gate": gate.threshold} if gate is not None else {}
    decisions = list(report.decisions)
    if gate is not None:
        decisions.extend(d for d in gate.decisions if d not in decisions)
    calib_ids = tuple(dict.fromkeys(tp.calib_id for tp in report.trend if tp.calib_id))
    return RunRecord(
        run_id=report.run_id,
        mode=report.mode,
        gate_passed=gate.passed if gate is not None else None,
        gate_n_modes=gate.n_modes if gate is not None else None,
        gate_failing_modes=gate.failing_modes if gate is not None else (),
        gate_modes=gate.modes if gate is not None else (),
        judge_ids=judge_ids,
        calib_ids=calib_ids,
        scorer_lanes=scorer_lanes,
        trend=report.trend,
        input_drift=report.input_drift,
        judge_drift=report.judge_drift,
        multiplicity=report.multiplicity,
        decisions=tuple(decisions),
        inputs=inputs,
        provenance=capture_provenance(seed=seed, n_resamples=n_resamples, thresholds=thresholds),
    )
