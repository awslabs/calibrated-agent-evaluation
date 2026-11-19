# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CI pass/fail gate.

The hard gate of the run stage. Given the per-trace verdict/scorer ``Label``s a change produced,
plus the judges + their calibration records and the CI-lane scorers used, it:

  1. **admits inputs** -- refuses a non-CERTIFIED / version-drifted judge (via ``certifies`` through
     ``enforce_score_gate`` -> ``UncalibratedScoreError``) and an exploration-lane
     scorer (via ``enforce_ci_gate``, re-raised as ``RunGateError``);
  2. **aggregates** the flat labels into per-failure-mode counts (``aggregate_by_mode``);
  3. **gates on the lower bound** -- each mode passes iff its pass-rate 95% one-sided lower bound
     (Wilson, via ``wilson_lower``) clears the threshold, never the point estimate;
  4. **routes failing traces** to the labeling queue tagged ``frame="discovery"``.

The gate passes only if every mode passes, which makes it an intersection-union test: a build with
any mode truly below its threshold passes with probability at most the per-mode alpha (5%), however
many modes are gated. No multiplicity correction is applied or recorded: one across modes could only
make the gate stricter than a false-pass cap it already meets. The
number of gated modes is recorded as ``n_modes``.

Empty input (no traces / no verdicts) is a refusal, never a vacuous ``passed=True``.
Pure methodology plane -- no vendor type. Returns a
:class:`~calibrated_agent_evaluation.run.report.RunGateDecision`; the CLI maps ``passed`` to the
process exit code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from calibrated_agent_evaluation.calibrate.engine import Judge
from calibrated_agent_evaluation.calibrate.record import CalibrationRecord
from calibrated_agent_evaluation.enforce import enforce_score_gate
from calibrated_agent_evaluation.run.aggregate import ModeSource, aggregate_by_mode
from calibrated_agent_evaluation.run.errors import RunGateError
from calibrated_agent_evaluation.run.report import (
    DiscoveryEnqueue,
    ModeCounts,
    ModeGate,
    ModeKey,
    RunDecision,
    RunGateDecision,
)
from calibrated_agent_evaluation.schema.types import Label
from calibrated_agent_evaluation.scorers import Scorer, ScorerLaneError, enforce_ci_gate
from calibrated_agent_evaluation.stats import wilson_lower

# The default pass-rate lower-bound each gated mode must clear. A per-mode override may be supplied
# via CiGateRequest.thresholds; this is the fallback bar (the calibration floor, reused as the
# CI pass-rate floor -- an Owner tunes it per suite).
DEFAULT_PASSRATE_THRESHOLD = 0.80


@dataclass(frozen=True)
class CiGateRequest:
    """The inputs to one CI-gate run.

    ``labels`` are the per-trace verdict/scorer results; ``gold`` maps each ``(trace_id, span_id)``
    to its failure mode, or is a function giving each label's mode (see
    :func:`~calibrated_agent_evaluation.run.aggregate_by_mode`; use the function when several modes'
    judges score one trace). ``judges`` + ``calib_records`` are the (judge, its CalibrationRecord)
    pairs whose certification is checked before gating; ``scorers`` are the code scorers whose
    lane is checked. ``threshold`` is the default pass-rate lower-bound bar; ``thresholds``
    overrides it per mode. ``run_id`` labels the resulting decision/record.
    """

    labels: Sequence[Label]
    gold: ModeSource
    judges: Sequence[tuple[Judge, CalibrationRecord]] = ()
    scorers: Sequence[Scorer] = ()
    threshold: float = DEFAULT_PASSRATE_THRESHOLD
    thresholds: Mapping[ModeKey, float] = field(default_factory=dict)
    run_id: str = "run"
    decisions: Sequence[RunDecision] = ()


def _admit_inputs(request: CiGateRequest) -> None:
    """Refuse an uncalibrated judge or an exploration-lane scorer BEFORE gating.

    Judge certification is delegated to ``enforce_score_gate`` (which reads
    ``certifies`` and raises the typed ``UncalibratedScoreError``); scorer lane to
    ``enforce_ci_gate`` (whose ``ScorerLaneError`` is re-raised as ``RunGateError`` naming it).
    """
    for judge, record in request.judges:
        enforce_score_gate(record, judge)  # raises UncalibratedScoreError
    for scorer in request.scorers:
        try:
            enforce_ci_gate(scorer)  # raises ScorerLaneError
        except ScorerLaneError as exc:
            raise RunGateError(f"CI gate refuses scorer {type(scorer).__name__!r}: {exc}") from exc


def run_ci_gate(request: CiGateRequest) -> RunGateDecision:
    """Run the CI gate and return a :class:`RunGateDecision`.

    Refuses an empty input rather than passing vacuously. Admits inputs, aggregates, and gates each
    mode on its pass-rate 95% lower bound. ``passed`` is True iff every gated mode clears its
    threshold (an intersection-union test, so no correction across modes); ``failing_modes`` names
    those that do not and ``n_modes`` counts the modes gated. Attached ``decisions`` (e.g. a
    staleness warning) are recorded but never flip ``passed``.

    Args:
        request: The verdicts, thresholds and options for the gate.

    Returns:
        Whether every mode passed, which failed, and any attached decisions.

    Raises:
        RunGateError: The input is empty, or a scorer is not allowed to gate.
        UncalibratedScoreError: A judge verdict comes from a judge with no certifying record.
    """
    if not request.labels:
        raise RunGateError(
            "run_ci_gate: no verdicts to gate on -- refusing a vacuous pass over an empty run "
            "(a gate over zero traces is not a green build)"
        )

    _admit_inputs(request)

    counts = aggregate_by_mode(request.labels, request.gold)
    if not counts:
        raise RunGateError(
            "run_ci_gate: no verdict mapped to a measured failure mode (check the gold map); "
            "refusing a vacuous pass"
        )

    return gate_mode_counts(
        counts,
        threshold=request.threshold,
        thresholds=request.thresholds,
        decisions=request.decisions,
    )


def gate_mode_counts(
    counts: Mapping[ModeKey, ModeCounts],
    *,
    threshold: float = DEFAULT_PASSRATE_THRESHOLD,
    thresholds: Mapping[ModeKey, float] | None = None,
    decisions: Sequence[RunDecision] = (),
) -> RunGateDecision:
    """Gate per-mode counts on each mode's pass-rate 95% lower bound.

    The gating step of :func:`run_ci_gate`, for a caller that has already reduced its verdicts to
    one observation per trace and mode (for example human labels, where one mode-free PASS is a
    pass for every mode). Admits nothing: judge and scorer admission is :func:`run_ci_gate`'s.

    Args:
        counts: Each mode's verdict total and FAIL count.
        threshold: The default bar every mode's lower bound must clear.
        thresholds: Per-mode overrides of ``threshold``.
        decisions: Non-gating decisions to attach (recorded, never flip ``passed``).

    Returns:
        Whether every mode passed, which failed, and each mode's counts and bound.

    Raises:
        RunGateError: ``counts`` is empty, or a mode has no verdicts.
    """
    if not counts:
        raise RunGateError(
            "no verdict mapped to a measured failure mode; refusing a vacuous pass (a gate over "
            "zero verdicts is not a green build)"
        )
    overrides = thresholds or {}
    ordered = sorted(counts.values(), key=lambda c: (c.mode.failure_type, c.mode.agent or ""))
    gated: list[ModeGate] = []
    for c in ordered:
        if c.n_total <= 0:
            raise RunGateError(f"mode {c.mode!r} has no verdicts to gate on")
        bar = overrides.get(c.mode, threshold)
        # The bound, not the point. Every mode must clear its own bar; requiring all of them is
        # what caps P(false pass) at the per-mode alpha, so the per-mode bound is used as is.
        lb95 = wilson_lower(c.n_total - c.n_judged_fail, c.n_total)
        gated.append(
            ModeGate(
                mode=c.mode,
                n_total=c.n_total,
                n_fail=c.n_judged_fail,
                lower_bound=lb95,
                threshold=bar,
                passed=lb95 >= bar,
            )
        )
    failing = tuple(g.mode for g in gated if not g.passed)
    return RunGateDecision(
        passed=not failing,
        failing_modes=failing,
        threshold=threshold,
        n_modes=len(gated),
        decisions=tuple(decisions),
        modes=tuple(gated),
    )


def discovery_traces(request: CiGateRequest) -> tuple[DiscoveryEnqueue, ...]:
    """Return the CI-failing traces as discovery-tagged labeling-queue entries.

    A trace whose verdict is ``FAIL`` is CI-augmentation, not a random draw: it is enqueued tagged
    ``frame="discovery"`` (the caller does the enqueue; the gate supplies the frame-tagged set). The
    frame tag is what keeps it out of production-frame rate estimation (enforced by the
    ``assert_production_frame`` guard on the trend side). A passing trace is not enqueued.
    """
    failing_ids = dict.fromkeys(
        label.trace_id for label in request.labels if label.outcome == "FAIL"
    )
    return tuple(DiscoveryEnqueue(trace_id=tid) for tid in failing_ids)
