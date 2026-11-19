# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The calibration gate -- certify a judge, correct a rate, enforce the disciplines.

Three public entry points, plus the anti-self-deception guards woven through them:

  * :func:`certify` -- run the locked-test gate. CERTIFIED iff the 95% one-sided lower bound on
    BOTH TPR and TNR clears 0.80. Refuses to certify from anything but the locked
    test and refuses to re-run an already-consumed locked test.
  * :func:`corrected_rate` -- Rogan-Gladen prevalence-correct a rate measured on a SEPARATE
    production-representative sample, with a Jeffreys-posterior CI, refusing
    the point when J<0.5 or the rate is non-identifiable, and refusing entirely
    for an uncertified judge or one certified on gold below the coverage floor.
  * :func:`certifies` -- does a stored record still certify a judge at its current
    ``(model, prompt_version)``? False after either changes -- re-calibrate.

This module does NOT run judges: it consumes paired (gold, verdict) observations produced upstream
by the judge runtime (or a synthetic set such as ``fixtures/calibration/synthetic_9090.json``).
Pure methodology plane; the numeric engine is :mod:`calibrated_agent_evaluation.stats`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation._provenance import capture_provenance
from calibrated_agent_evaluation.calibrate.errors import (
    FewShotEvaluationError,
    InfeasiblePlanError,
    NonIdentifiableRateError,
    RateSampleTooSmallError,
    RateUnsafeError,
    SampleMisuseError,
    UncalibratedJudgeError,
)
from calibrated_agent_evaluation.calibrate.ledger import LockedTestLedger
from calibrated_agent_evaluation.calibrate.pairing import pair_observations
from calibrated_agent_evaluation.calibrate.plan import (
    MIN_RATE_SAMPLE,
    CalibrationPlan,
    plan_calibration,
)
from calibrated_agent_evaluation.calibrate.record import (
    METHOD_RG_JEFFREYS,
    METHOD_RG_JEFFREYS_CLUSTERED,
    METHOD_WILSON,
    METHOD_WILSON_CLUSTERED,
    Bound,
    CalibrationRecord,
    CalibrationSet,
    Interval,
    LabelPair,
)
from calibrated_agent_evaluation.stats.proportions import wilson_lower
from calibrated_agent_evaluation.stats.rogan_gladen import (
    cluster_design_effect,
    rg_interval,
    rogan_gladen,
    youden_j,
)

# The certification threshold: the 95% one-sided lower bound on BOTH TPR and TNR must clear this.
# A point estimate at the threshold does not certify -- the bound does.
GATE_THRESHOLD = 0.80

# Youden's J below this refuses the Rogan-Gladen point correction.
YOUDEN_FLOOR = 0.50

# The bootstrap resample count rg_interval uses unless a caller asks otherwise.
DEFAULT_N_RESAMPLES = 9999
# The corrected-rate interval is two-sided at this level (rg_interval's ``conf``).
RG_LEVEL = 0.95


def _gate_thresholds() -> dict[str, float]:
    # Read at call time, so the stamp names the thresholds actually applied.
    return {
        "gate": GATE_THRESHOLD,
        "youden_floor": YOUDEN_FLOOR,
        "min_rate_sample": float(MIN_RATE_SAMPLE),
    }


class Judge(Protocol):
    """The minimal judge identity the gate pins to.

    A structural type: any object exposing ``judge_id`` / ``model`` / ``prompt_version`` satisfies
    it (the judge object itself is owned by the judge runtime; the gate reads only these three
    fields to pin the calibration to an exact model+prompt version).
    """

    @property
    def judge_id(self) -> str:
        """The judge's stable id."""
        ...

    @property
    def model(self) -> str:
        """The model the judge runs on."""
        ...

    @property
    def prompt_version(self) -> str:
        """The version of the judge's prompt."""
        ...


# Below this many traces in a class, the design effect estimated from the traces is too noisy to
# credit. At true TPR 0.80 with Beta-distributed per-trace accuracy, the Korn-Graubard size
# certified 7.0-7.7% of the time at 10 traces of 8 spans and 5.4-5.9% at 15 traces, against
# 5.2-5.3% at 20 or more, the Wilson gate's own 5.6% at 80 independent items (6000 replicates
# per cell, seed 13). Below it each trace counts as one draw.
MIN_TRACES_FOR_DESIGN_EFFECT = 20


def _class_design_effect(pairs: Sequence[LabelPair], outcome: str) -> float:
    """The design effect of one gold class's judge-correct rate, clustered by trace.

    1.0 when every trace contributes one target. Several spans of one trace share whatever made
    the trace easy or hard to judge, so they are not independent evidence. The class is then
    taken at the Korn and Graubard (1998) effective size ``n / deff * (t_{n-1} / t_{C-1})^2``,
    with ``deff`` from :func:`~calibrated_agent_evaluation.stats.rogan_gladen.cluster_design_effect`
    (each trace one cluster, the worst case ``N / C`` when the judge was right or wrong throughout)
    and the degrees-of-freedom factor for estimating ``deff`` from only ``C`` traces. With fewer
    than :data:`MIN_TRACES_FOR_DESIGN_EFFECT` traces each trace counts as one draw (``N / C``). The
    returned value is ``n`` over the effective size.
    """
    members = [p for p in pairs if p.gold.outcome == outcome]
    traces = [p.gold.trace_id for p in members]
    n, n_traces = len(members), len(set(traces))
    if n_traces == n:
        return 1.0
    if n_traces < MIN_TRACES_FOR_DESIGN_EFFECT:
        return n / n_traces
    correct = [int(p.verdict.outcome == p.gold.outcome) for p in members]
    deff = cluster_design_effect(correct, traces)
    from scipy.stats import t as student_t

    df_factor = (student_t.ppf(0.95, n_traces - 1) / student_t.ppf(0.95, n - 1)) ** 2
    return min(n / n_traces, deff * float(df_factor))


def _bound(right: int, n: int, deff: float) -> Bound:
    """A class's rate with its 95% one-sided Wilson lower bound at the effective size."""
    return Bound(
        point=right / n if n else 0.0,
        lb95=wilson_lower(right / deff, n / deff) if n else 0.0,
        n=n,
        method=METHOD_WILSON if deff == 1.0 else METHOD_WILSON_CLUSTERED,
    )


def certify(
    judge: Judge,
    locked: CalibrationSet,
    *,
    ledger: LockedTestLedger,
    calib_id: str | None = None,
    plan: CalibrationPlan | None = None,
    sampling: Mapping[str, float] | None = None,
    agreement_alpha: Interval | None = None,
) -> CalibrationRecord:
    """Run the locked-test gate and return a CERTIFIED/REFUSED :class:`CalibrationRecord`.

    Refuses (never certifies) unless ``locked.split == "locked_test"`` -- a dev set never certifies
    and the few-shot pool is never evaluated. Consumes ``locked`` from ``ledger`` and refuses a set
    whose id, content or any trace was already consumed (single-use). Gold must be human, one
    label per target (enforced in pairing). CERTIFIED iff both lower bounds clear
    :data:`GATE_THRESHOLD`.

    ``plan`` -- the :class:`~calibrated_agent_evaluation.calibrate.plan.CalibrationPlan` the gold
    was hunted against. Its per-class size and coverage floor are recorded, and gold below the floor
    is marked (``CalibrationRecord.below_coverage_floor``) without changing the result: the judge
    still certifies, and :func:`corrected_rate` refuses an interval from the record.
    A certified judge's floor is also planned at the TPR and TNR it showed, with or without a
    ``plan``, and the larger of the floors applies; a judge whose accuracy differs from the plan
    can need more gold.

    Each judge's certification stands alone. Certifying several judges in one run is not a joint
    claim that all of them are good, so no correction across judges is applied and none is
    recorded.

    Promotion is an intersection-union test: a judge deficient in TPR or in TNR is certified with
    probability at most about 5% (slightly more at some ``n``, since Wilson coverage oscillates),
    with no per-judge correction needed for using two bounds. The two bounds are not a joint 95%
    region: at TPR = TNR = 0.80 and 80 per class their simultaneous coverage is 0.890.

    Args:
        judge: The judge identity (id, model, prompt version) the record pins to.
        locked: The locked test set of (judge verdict, human gold) pairs.
        ledger: The record of locked sets already used; ``locked`` is consumed from it.
        calib_id: The record id; generated when omitted.
        plan: The calibration plan the gold was collected against.
        sampling: The sampling parameters the judge's verdicts were drawn with, recorded on the
            provenance stamp (empty: none were sent).
        agreement_alpha: The gold raters' agreement, recorded on the record when measured.

    A REFUSED set short of ``plan``'s per-class size is opened in ``ledger`` for a top-up by the
    same judge: a later set holding all its traces plus fresh ones may be tested, and the record
    counts the looks (``looks``) so the retest is disclosed.

    Returns:
        A CERTIFIED or REFUSED calibration record.

    Raises:
        FewShotEvaluationError: ``locked`` is the few-shot pool or the dev set.
        LockedTestReuseError: ``locked``, its content or one of its traces was already consumed.
        UnratifiedGoldError: A gold label is not human.
        UnpairedObservationError: A verdict and its gold label name different targets.
        DuplicateGoldError: A target carries more than one gold label.
    """
    if locked.split == "few_shot":
        raise FewShotEvaluationError(
            f"refusing to evaluate judge {judge.judge_id!r} on the few-shot pool "
            f"(set_id={locked.set_id!r}); the few-shot pool is never an evaluation set"
        )
    if locked.split == "dev":
        raise FewShotEvaluationError(
            f"refusing to certify judge {judge.judge_id!r} from the dev set "
            f"(set_id={locked.set_id!r}); certification derives only from the locked test"
        )
    counts = pair_observations(locked.pairs)  # human-gold, exact pairing, one label per target
    fingerprint = fingerprint_of(judge)
    looks = ledger.consume(locked, judge_fingerprint=fingerprint)

    deff_pos = _class_design_effect(locked.pairs, "FAIL")
    deff_neg = _class_design_effect(locked.pairs, "PASS")
    tpr = _bound(counts.tp, counts.n_pos, deff_pos)
    tnr = _bound(counts.tn, counts.n_neg, deff_neg)
    j = youden_j(tpr.point, tnr.point)
    certified = tpr.lb95 >= GATE_THRESHOLD and tnr.lb95 >= GATE_THRESHOLD
    trace_ids = tuple(dict.fromkeys(p.gold.trace_id for p in locked.pairs))
    # The observed floor applies to every certified record, planned or not: a library caller who
    # passes no plan must not get a corrected rate from gold below the floor.
    observed_floor, unmeasured = (
        _observed_floor(tpr.point, tnr.point) if certified else (None, False)
    )
    if (
        not certified
        and fingerprint is not None
        and plan is not None
        and min(counts.n_pos, counts.n_neg) < plan.per_class
    ):
        ledger.allow_top_up(locked.set_id, judge_fingerprint=fingerprint)
    return CalibrationRecord(
        calib_id=calib_id or f"calib-{judge.judge_id}-{locked.set_id}",
        judge_id=judge.judge_id,
        judge_fingerprint=fingerprint,
        model=judge.model,
        prompt_version=judge.prompt_version,
        hunted_set_id=locked.set_id,
        n_pos=counts.n_pos,
        n_neg=counts.n_neg,
        tpr=tpr,
        tnr=tnr,
        tpr_design_effect=deff_pos,
        tnr_design_effect=deff_neg,
        youden_j=j,
        result="CERTIFIED" if certified else "REFUSED",
        locked_test_consumed=True,
        trace_ids=trace_ids,
        planned_per_class=None if plan is None else plan.per_class,
        planned_coverage_floor=None if plan is None else plan.coverage_floor,
        observed_coverage_floor=observed_floor,
        observed_floor_unmeasured=unmeasured,
        looks=looks,
        agreement_alpha=agreement_alpha,
        provenance=capture_provenance(thresholds=_gate_thresholds(), sampling=sampling),
    )


def _rate_unsafe_message(record: CalibrationRecord) -> str:
    """Why :func:`corrected_rate` refuses ``record``: its sizes, both floors and the plan."""
    head = (
        f"judge {record.judge_id!r} is certified by {record.calib_id!r} on {record.n_pos} FAIL "
        f"and {record.n_neg} PASS gold label(s)"
    )
    if record.tpr_design_effect > 1.0 or record.tnr_design_effect > 1.0:
        head += (
            f", worth {record.effective_n_pos:.0f} and {record.effective_n_neg:.0f} independent "
            "label(s) because several come from one trace"
        )
    observed = f"the observed TPR {record.tpr.point:.3f} and TNR {record.tnr.point:.3f}"
    planned = f"planned {record.planned_per_class} per class"
    stands = "the certification stands, but the corrected rate's interval is not known to cover "
    if record.observed_floor_unmeasured:
        return (
            f"{head}, and at {observed} the corrected rate's interval was measured and did not "
            f"cover nominally at any scanned gold size ({planned}); {stands}nominally for this "
            "judge, so none is built from this record. Label more gold and calibrate on the new "
            "set"
        )
    floors = []
    if record.planned_coverage_floor is not None:
        floors.append(f"{record.planned_coverage_floor} at the planned accuracy")
    if record.observed_coverage_floor is not None:
        floors.append(f"{record.observed_coverage_floor} at {observed}")
    which = f"the larger of {' and '.join(floors)}" if len(floors) > 1 else floors[0]
    if record.planned_per_class is not None:
        which += f"; {planned}"
    return (
        f"{head}, below the coverage floor of {record.coverage_floor} per class ({which}); "
        f"{stands}nominally on gold that small, so none is "
        f"built from this record. Label up to {record.coverage_floor} per "
        "class and calibrate on the new gold"
    )


def _observed_floor(tpr: float, tnr: float) -> tuple[int | None, bool]:
    """The coverage floor at the observed rates, and whether none qualified there.

    The floor depends on accuracy, and not monotonically (it peaks near 0.99 and falls at the
    boundary), so it is planned again at the TPR and TNR the judge showed. Called for a certified
    judge only, whose rates both exceed the gate threshold, so the
    certification half of the plan always has an answer and an InfeasiblePlanError can only mean
    that no scanned size covered nominally at the higher rate.
    """
    try:
        return plan_calibration(tpr, GATE_THRESHOLD, tnr_hat=tnr).coverage_floor, False
    except InfeasiblePlanError:
        return None, True


def corrected_rate(
    record: CalibrationRecord,
    rate_sample_verdicts: Sequence[int],
    rate_set_id: str,
    *,
    rate_trace_ids: Sequence[str],
    seed: int = 0,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    rate_cluster_ids: Sequence[str] | None = None,
) -> CalibrationRecord:
    """Return a copy of ``record`` with the Rogan-Gladen corrected rate + CI attached.

    ``seed`` and ``n_resamples`` (the Monte Carlo draw count) drive the interval and are stamped
    on the returned record's ``provenance``, so the interval can be recomputed from the stored
    record alone.

    Refuses for an uncertified judge, and with :class:`RateUnsafeError` for a record whose gold is
    below its coverage floor, the larger of the planned and the observed one,
    or whose observed rates have no measured floor, and with :class:`RateSampleTooSmallError` for a
    rate sample of fewer than :data:`~calibrated_agent_evaluation.calibrate.plan.MIN_RATE_SAMPLE`
    traces (distinct traces when ``rate_cluster_ids`` is given). When J<0.5 the corrected *point* is
    refused (the interval is still returned as the signal); when the rate is non-identifiable the
    point is refused and the bound reported. When more than one tail's share of the interval's draws
    have no corrected rate, the interval is refused too: ``(lo, hi) = (0, 1)`` with a
    ``refusal_reason`` starting ``"non-identifiable interval:"``. The interval always stays finite
    and in [0,1].

    Args:
        record: A CERTIFIED calibration record for the judge.
        rate_sample_verdicts: The judge's verdicts (1 = FAIL, 0 = PASS) on a separate
            production-representative sample; their mean is the raw rate.
        rate_set_id: The id of that sample; it must differ from ``record.hunted_set_id``.
        rate_trace_ids: The trace id of each verdict, in the same order. Required, because the
            rate sample must be checked against the calibration gold: hunted traces re-entering
            it break the independence the interval assumes.
        seed: Fixes the Monte Carlo draws behind the interval.
        n_resamples: The number of Monte Carlo draws.
        rate_cluster_ids: The trace id of each verdict, when one trace contributes several and
            the interval should account for the clustering.

    Returns:
        A copy of ``record`` with the corrected rate and its interval attached.

    Raises:
        UncalibratedJudgeError: ``record`` is not CERTIFIED.
        RateUnsafeError: The record's gold is below its coverage floor.
        SampleMisuseError: The rate sample is empty, is the hunted calibration set, shares
            traces with the calibration gold, or its trace ids do not match its verdicts.
        RateSampleTooSmallError: The rate sample has fewer than ``MIN_RATE_SAMPLE`` traces.
    """
    if record.result != "CERTIFIED":
        raise UncalibratedJudgeError(
            f"judge {record.judge_id!r} has no CERTIFIED calibration (result={record.result!r}); "
            "refusing to compute or display a corrected rate"
        )
    if record.below_coverage_floor:
        raise RateUnsafeError(_rate_unsafe_message(record))
    if rate_set_id == record.hunted_set_id:
        raise SampleMisuseError(
            f"rate_set_id {rate_set_id!r} equals the hunted_set_id; the true rate must come from a "
            "SEPARATE production-representative sample, never the hunted fail-heavy set"
        )
    if not rate_sample_verdicts:
        raise SampleMisuseError("corrected_rate: the rate sample must be non-empty")
    if len(rate_trace_ids) != len(rate_sample_verdicts) or not all(
        isinstance(t, str) and t for t in rate_trace_ids
    ):
        raise SampleMisuseError(
            f"corrected_rate: rate_trace_ids must name the trace of each of the "
            f"{len(rate_sample_verdicts)} verdict(s), in order, got {len(rate_trace_ids)} id(s); "
            "without them the rate sample cannot be checked against the calibration gold"
        )
    _refuse_calibration_overlap(record, rate_trace_ids, rate_cluster_ids)
    n_traces = len(rate_sample_verdicts) if rate_cluster_ids is None else len(set(rate_cluster_ids))
    if n_traces < MIN_RATE_SAMPLE:
        raise RateSampleTooSmallError(
            f"the rate sample has {n_traces} trace(s); a corrected rate needs at least "
            f"{MIN_RATE_SAMPLE}, so score {MIN_RATE_SAMPLE - n_traces} more "
            "production-representative trace(s) and rerun. Below that the interval was measured "
            "to cover the true rate less often than it claims.",
            n=n_traces,
            minimum=MIN_RATE_SAMPLE,
        )

    measured = sum(1 for v in rate_sample_verdicts if v) / len(rate_sample_verdicts)
    tpr_point, tnr_point = record.tpr.point, record.tnr.point

    # Reconstruct the hunted per-class verdict arrays from the confusion counts so the double
    # bootstrap can resample them alongside the eval sample. The arrays are 1/0 draws whose
    # means are exactly TPR (positive class) and 1-TNR (negative class) -- verdict==FAIL indicators.
    n_pos, n_neg = record.n_pos, record.n_neg
    tp = round(tpr_point * n_pos)
    fp = round((1.0 - tnr_point) * n_neg)
    cal_pos = [1] * tp + [0] * (n_pos - tp)  # judge-FAIL on the true-FAIL class -> mean = TPR
    cal_neg = [1] * fp + [0] * (n_neg - fp)  # judge-FAIL on the true-PASS class -> mean = 1-TNR
    try:
        drawn = rg_interval(
            cal_pos,
            cal_neg,
            list(rate_sample_verdicts),
            conf=RG_LEVEL,
            n_resamples=n_resamples,
            seed=seed,
            eval_cluster_ids=rate_cluster_ids,
            cal_pos_design_effect=record.tpr_design_effect,
            cal_neg_design_effect=record.tnr_design_effect,
        )
    except NonIdentifiableRateError as exc:
        # Too many draws without a corrected rate to place a tail:
        # refuse the interval as well as the point. [0, 1] claims nothing and keeps the mode
        # visible with the reason, rather than an interval over a mixture of estimands.
        lo, hi, interval_refusal = 0.0, 1.0, f"non-identifiable interval: {exc}"
    else:
        lo, hi, interval_refusal = drawn.lo, drawn.hi, None

    j = youden_j(tpr_point, tnr_point)
    n_rate = len(rate_sample_verdicts)

    def _interval(point: float | None, reason: str | None) -> Interval:
        return Interval(
            point=point,
            lo=lo,
            hi=hi,
            refusal_reason=reason,
            method=METHOD_RG_JEFFREYS if rate_cluster_ids is None else METHOD_RG_JEFFREYS_CLUSTERED,
            level=RG_LEVEL,
            sided="two-sided",
            n=n_rate,
        )

    if interval_refusal is not None:
        interval = _interval(None, interval_refusal)
    elif j < YOUDEN_FLOOR:
        interval = _interval(None, f"youden_j={j:.3f}<0.5")
    else:
        try:
            point = rogan_gladen(measured, tpr_point, tnr_point)
        except NonIdentifiableRateError:
            # Outside the identifiable region -- refuse the point, report the bound.
            interval = _interval(None, "non-identifiable")
        else:
            interval = _interval(point, None)

    stamp = capture_provenance(
        seed=seed,
        n_resamples=n_resamples,
        thresholds=_gate_thresholds(),
        sampling=record.provenance.sampling,
    )
    return replace(record, corrected_rate=interval, rate_set_id=rate_set_id, provenance=stamp)


_SHOWN_OVERLAP = 5


def _refuse_calibration_overlap(
    record: CalibrationRecord,
    rate_trace_ids: Sequence[str] | None,
    rate_cluster_ids: Sequence[str] | None,
) -> None:
    """Refuse a rate sample that reuses traces of the gold the judge was calibrated on.

    The interval treats the calibration gold and the rate sample as independent samples. Hunted,
    fail-heavy traces re-entering the rate sample break that and push the measured rate up.
    """
    rate_traces = {*(rate_trace_ids or ()), *(rate_cluster_ids or ())}
    shared = sorted(rate_traces & set(record.trace_ids))
    if not shared:
        return
    more = len(shared) - _SHOWN_OVERLAP
    raise SampleMisuseError(
        f"{len(shared)} trace(s) of the rate sample are in the gold judge {record.judge_id!r} was "
        f"calibrated on ({record.calib_id!r}): {', '.join(shared[:_SHOWN_OVERLAP])}"
        + (f" and {more} more" if more > 0 else "")
        + ". The corrected rate needs a rate sample independent of the calibration gold; leave "
        "those traces out of the rate sample and rerun"
    )


def fingerprint_of(judge: object) -> str | None:
    """The judge's content fingerprint, or ``None`` for a judge that carries none."""
    fingerprint = getattr(judge, "fingerprint", None)
    return fingerprint if isinstance(fingerprint, str) else None


def pin_mismatch(record: object, judge: Judge) -> str | None:
    """Why ``record`` is not pinned to ``judge`` as it is now, or ``None`` when it is.

    ``record`` is any record pinning a judge (calibration, bias battery): it carries ``judge_id``,
    ``model``, ``prompt_version`` and ``judge_fingerprint``. A record made for another judge id
    never pins this one. A judge with a fingerprint matches only a record stamped with the same
    one, so an in-place prompt edit under an unchanged ``prompt_version`` is caught, and a record
    stamped before fingerprints existed proves nothing about the current prompt. A record stamped
    with a fingerprint never pins a judge that carries none, since nothing shows that judge's
    content is the one measured. A judge without a fingerprint matches an unstamped record on
    ``(judge_id, model, prompt_version)`` alone.
    """
    recorded_id = getattr(record, "judge_id", None)
    if recorded_id != judge.judge_id:
        return (
            f"this record was made for judge {recorded_id!r}, not {judge.judge_id!r}; a judge is "
            "certified on its own locked test"
        )
    if getattr(record, "model", None) != judge.model:
        return (
            f"the judge's model changed since this record was made "
            f"({getattr(record, 'model', None)} to {judge.model}); re-certify on a fresh locked set"
        )
    if getattr(record, "prompt_version", None) != judge.prompt_version:
        return (
            f"the judge's prompt version changed since this record was made "
            f"({getattr(record, 'prompt_version', None)} to {judge.prompt_version}); re-certify "
            "on a fresh locked set"
        )
    current = fingerprint_of(judge)
    stamped = getattr(record, "judge_fingerprint", None)
    if current is None:
        if stamped is None:
            return None
        return (
            "this record was made for a judge whose content was fingerprinted, and this judge "
            "carries no fingerprint, so nothing shows it is the judge that was measured; certify "
            "it on its own locked test"
        )
    if stamped is None:
        return (
            "this record was made before judge prompts were fingerprinted, so it cannot show the "
            "prompt is unchanged; re-certify on a fresh locked set"
        )
    if stamped != current:
        return (
            "the judge's prompt changed since it was certified (same prompt version, different "
            "content); re-certify on a fresh locked set"
        )
    return None


def certifies(record: CalibrationRecord, judge: Judge) -> bool:
    """Return whether ``record`` still certifies ``judge`` as it is now.

    False when the record is not CERTIFIED or is not pinned to the judge (:func:`pin_mismatch`):
    a model, prompt version or prompt content change requires re-calibration.
    """
    return record.result == "CERTIFIED" and pin_mismatch(record, judge) is None
