# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CI pass/fail gate.

Behavioural: the gate passes/fails on the pass-rate lower bound (never the point), passes only when
every mode does (an intersection-union test, so no correction across modes) and counts the modes,
refuses an uncalibrated judge and an exploration-lane scorer, routes
failing traces as discovery, refuses an empty run, and never lets a staleness warning flip the gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from _helpers import TEST_STAMP, make_label
from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord
from calibrated_agent_evaluation.enforce import UncalibratedScoreError
from calibrated_agent_evaluation.run.errors import RunGateError
from calibrated_agent_evaluation.run.gate import CiGateRequest, discovery_traces, run_ci_gate
from calibrated_agent_evaluation.run.report import ModeKey, RunDecision
from calibrated_agent_evaluation.schema.types import Label
from calibrated_agent_evaluation.scorers import Scorer, ScorerLane


@dataclass(frozen=True)
class _Judge:
    """A minimal Judge-shaped object (the calibrate.engine.Judge structural protocol)."""

    judge_id: str
    model: str
    prompt_version: str


class _Scorer:
    """A scorer stub mirroring the Scorer protocol (writable ``lane`` + ``score`` signature).

    Not frozen: the real ``Scorer`` protocol declares ``lane`` as a writable attribute, so a stub
    that satisfies the protocol (checked below) must too. The gate only reads ``.lane`` and never
    calls ``score``.
    """

    def __init__(self, lane: ScorerLane) -> None:
        self.lane = lane

    def score(
        self, output: dict[str, Any], expected: dict[str, Any] | None, *, trace_id: str
    ) -> Label:
        raise NotImplementedError


# Fail fast at type-check time if the stub drifts from the real Scorer protocol surface.
_STUB_MATCHES_PROTOCOL: Scorer = _Scorer(lane=ScorerLane.CI_GATE)


def _record(judge: _Judge, *, result: str = "CERTIFIED") -> CalibrationRecord:
    return CalibrationRecord(
        provenance=TEST_STAMP,
        calib_id=f"calib-{judge.judge_id}",
        judge_id=judge.judge_id,
        model=judge.model,
        prompt_version=judge.prompt_version,
        hunted_set_id="hunted",
        n_pos=80,
        n_neg=80,
        tpr=Bound(point=0.95, lb95=0.90, n=80),
        tnr=Bound(point=0.95, lb95=0.90, n=80),
        youden_j=0.90,
        result=result,  # type: ignore[arg-type]
    )


def _lbl(trace_id: str, outcome: str) -> Label:
    return make_label(trace_id=trace_id, span_id=None, kind="llm-judge", outcome=outcome)


def _mode_labels(mode_prefix: str, n_pass: int, n_fail: int) -> list[Label]:
    passes = [_lbl(f"{mode_prefix}p{i}", "PASS") for i in range(n_pass)]
    fails = [_lbl(f"{mode_prefix}f{i}", "FAIL") for i in range(n_fail)]
    return passes + fails


def _gold_for(labels: list[Label], mode: ModeKey) -> dict[tuple[str, str | None], ModeKey]:
    return {(lb.trace_id, lb.span_id): mode for lb in labels}


def test_passes_when_every_mode_lower_bound_clears_threshold() -> None:
    """A mode whose pass-rate 95% lower bound clears the threshold passes."""
    mode = ModeKey(failure_type="over_refund")
    labels = _mode_labels("A", n_pass=50, n_fail=0)  # perfect pass-rate, lb95 well above 0.80
    decision = run_ci_gate(
        CiGateRequest(labels=labels, gold=_gold_for(labels, mode), threshold=0.80)
    )
    assert decision.passed is True
    assert decision.failing_modes == ()


def test_fails_on_lower_bound_even_when_point_estimate_clears() -> None:
    """A point estimate above the bar but a lower bound below it does NOT pass.

    9/10 = 0.90 point pass-rate is above 0.80, but wilson_lower(9, 10) ~= 0.61 < 0.80 -- the gate
    reads the bound, not the point, so this mode fails and is named.
    """
    mode = ModeKey(failure_type="over_refund")
    labels = _mode_labels("A", n_pass=9, n_fail=1)
    decision = run_ci_gate(
        CiGateRequest(labels=labels, gold=_gold_for(labels, mode), threshold=0.80)
    )
    assert decision.passed is False
    assert decision.failing_modes == (mode,)


def test_gate_counts_modes_and_one_failing_mode_fails_it() -> None:
    """The gate counts the modes it gated and fails on any one of them.

    45/50 has wilson_lower ~= 0.807 >= 0.80 and passes alone; 44/50 does not. The decision records
    n_modes and carries no multiplicity family.
    """
    a, b = ModeKey(failure_type="over_refund"), ModeKey(failure_type="dropped_fact")
    la = _mode_labels("A", 45, 5)
    lb = _mode_labels("B", 44, 6)
    gold = {**_gold_for(la, a), **_gold_for(lb, b)}
    decision = run_ci_gate(CiGateRequest(labels=la + lb, gold=gold, threshold=0.80))
    assert decision.n_modes == 2
    assert decision.passed is False
    assert decision.failing_modes == (b,)
    assert not hasattr(decision, "multiplicity")


def test_many_modes_at_the_bar_are_not_tightened() -> None:
    """Twenty modes each at 45/50 all pass; gating more modes adds no correction."""
    labels: list[Any] = []
    gold: dict[tuple[str, str | None], ModeKey] = {}
    for i in range(20):
        mode_labels = _mode_labels(f"M{i}-", 45, 5)
        labels += mode_labels
        gold |= _gold_for(mode_labels, ModeKey(failure_type=f"mode_{i:02d}"))
    decision = run_ci_gate(CiGateRequest(labels=labels, gold=gold, threshold=0.80))
    assert decision.n_modes == 20
    assert decision.passed is True


def test_intersection_union_caps_the_false_pass_rate() -> None:
    """The all-modes-pass rule caps P(false pass) at the per-mode alpha, without Holm.

    Exact from the binomial at n = 50 per mode and tau = 0.80: a mode whose true pass rate is
    exactly tau clears the gate iff it has >= 45 passes, probability 0.048027. Two such modes pass
    together with probability 0.048027**2 = 0.002307, and the worst case (one mode at tau, the rest
    truly good) is the single-mode 0.048027, which is under 0.05.
    """
    from scipy.stats import binom

    from calibrated_agent_evaluation.stats import wilson_lower

    n, tau = 50, 0.80
    clearing = [k for k in range(n + 1) if wilson_lower(k, n) >= tau]
    assert clearing[0] == 45
    p_single = float(sum(binom.pmf(k, n, tau) for k in clearing))
    assert p_single == pytest.approx(0.048027219, abs=1e-8)
    assert p_single**2 == pytest.approx(0.002306614, abs=1e-8)
    assert p_single < 0.05


def test_exploration_lane_scorer_is_refused() -> None:
    """An EXPLORATION-lane scorer presented to the gate raises RunGateError."""
    mode = ModeKey(failure_type="over_refund")
    labels = _mode_labels("A", 50, 0)
    request = CiGateRequest(
        labels=labels,
        gold=_gold_for(labels, mode),
        scorers=[_Scorer(lane=ScorerLane.EXPLORATION)],
    )
    with pytest.raises(RunGateError):
        run_ci_gate(request)


def test_ci_gate_lane_scorer_is_admitted() -> None:
    """A CI_GATE-lane scorer is admitted and the gate runs."""
    mode = ModeKey(failure_type="over_refund")
    labels = _mode_labels("A", 50, 0)
    request = CiGateRequest(
        labels=labels,
        gold=_gold_for(labels, mode),
        scorers=[_Scorer(lane=ScorerLane.CI_GATE)],
    )
    assert run_ci_gate(request).passed is True


def test_uncertified_and_version_drifted_judge_are_refused() -> None:
    """A non-CERTIFIED record and a version-drifted judge each raise."""
    mode = ModeKey(failure_type="over_refund")
    labels = _mode_labels("A", 50, 0)
    gold = _gold_for(labels, mode)
    judge = _Judge("j1", "bedrock/claude", "v1")

    refused = _record(judge, result="REFUSED")
    with pytest.raises(UncalibratedScoreError):
        run_ci_gate(CiGateRequest(labels=labels, gold=gold, judges=[(judge, refused)]))

    certified_v1 = _record(judge)
    drifted = _Judge("j1", "bedrock/claude", "v2")  # prompt_version drifted from the record
    with pytest.raises(UncalibratedScoreError):
        run_ci_gate(CiGateRequest(labels=labels, gold=gold, judges=[(drifted, certified_v1)]))


def test_certified_matching_judge_is_admitted() -> None:
    """A CERTIFIED record whose (model, prompt_version) matches is admitted."""
    mode = ModeKey(failure_type="over_refund")
    labels = _mode_labels("A", 50, 0)
    judge = _Judge("j1", "bedrock/claude", "v1")
    decision = run_ci_gate(
        CiGateRequest(labels=labels, gold=_gold_for(labels, mode), judges=[(judge, _record(judge))])
    )
    assert decision.passed is True


def test_failing_traces_are_enqueued_as_discovery() -> None:
    """A failing trace is enqueued tagged frame='discovery'; a passing trace is not."""
    mode = ModeKey(failure_type="over_refund")
    labels = _mode_labels("A", n_pass=3, n_fail=2)
    enqueued = discovery_traces(CiGateRequest(labels=labels, gold=_gold_for(labels, mode)))
    assert {e.trace_id for e in enqueued} == {"Af0", "Af1"}
    assert all(e.frame == "discovery" for e in enqueued)
    # A passing trace is not enqueued.
    assert "Ap0" not in {e.trace_id for e in enqueued}


def test_empty_run_refuses_rather_than_vacuous_pass() -> None:
    """A gate over zero verdicts raises RunGateError, never passed=True."""
    with pytest.raises(RunGateError):
        run_ci_gate(CiGateRequest(labels=[], gold={}))


def test_no_mapped_mode_refuses() -> None:
    """Verdicts that map to no measured mode refuse rather than pass vacuously."""
    labels = _mode_labels("A", 3, 0)
    with pytest.raises(RunGateError):
        run_ci_gate(CiGateRequest(labels=labels, gold={}))  # empty gold => no mapped mode


def test_staleness_decision_never_flips_a_passing_gate() -> None:
    """An attached staleness warning is recorded but never flips passed to False."""
    mode = ModeKey(failure_type="over_refund")
    labels = _mode_labels("A", 50, 0)
    stale = RunDecision(
        rule="RUN-4",
        tier="warning",
        inputs={"judge_id": "j1"},
        outcome="warned",
        message="judge version drifted",
    )
    decision = run_ci_gate(
        CiGateRequest(
            labels=labels, gold=_gold_for(labels, mode), threshold=0.80, decisions=[stale]
        )
    )
    assert decision.passed is True
    assert stale in decision.decisions
