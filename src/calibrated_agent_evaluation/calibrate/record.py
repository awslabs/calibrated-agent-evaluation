# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibration data types -- the gate evidence and its inputs.

Pure, frozen data holders (the "typed functional core" house style): validation and the
gate logic live in :mod:`calibrated_agent_evaluation.calibrate.engine` /
:mod:`calibrated_agent_evaluation.calibrate.store`, never in ``__init__``. No vendor SDK type
appears here -- the methodology-plane boundary.

  * :class:`Bound`    -- a gated rate (TPR/TNR) with its 95% one-sided lower bound.
  * :class:`Interval` -- the two-sided CI for the corrected rate; ``point`` is ``None`` when the
    correction is refused (J<0.5 / non-identifiable), and :meth:`Interval.require_point` is the
    accessor that raises the typed refusal so a ``None`` never silently leaks into a report.
  * :class:`LabelPair` -- one paired (human gold, judge verdict) observation.
  * :class:`CalibrationSet` -- a split-tagged set of pairs (few-shot / dev / locked-test); the
    engine refuses to certify from anything but the locked test, and the ledger keeps the locked
    test single-use by its id, its content and its traces.
  * :class:`CalibrationRecord` -- the reproducible gate evidence, version-pinned.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import ConfigDict, Field, field_validator

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._provenance import ProvenanceStamp
from calibrated_agent_evaluation.calibrate.errors import (
    TRANSFER_REFUSAL,
    CalibrationError,
    ErrorRatesDoNotTransferError,
    NonIdentifiableRateError,
    WeakJudgeError,
)
from calibrated_agent_evaluation.schema.types import Label

# Stamped on every persisted CalibrationRecord; load refuses on mismatch. "2" added `provenance`;
# "3" added method/level/sided/n to Bound and Interval; "4" dropped m_comparisons and gate_family
# (each judge's certification stands alone); "5" added
# planned_per_class and coverage_floor; "6" renamed coverage_floor to
# planned_coverage_floor and added observed_coverage_floor and observed_floor_unmeasured;
# "7" the provenance stamp names calibrated_agent_evaluation_version (records from before the rename
# are refused); "8" added tpr_design_effect and tnr_design_effect (span-level gold clustered by
# trace); "9" added looks (a refused-short set grown and tested again) and agreement_alpha.
CALIB_SCHEMA_VERSION = "9"

# The interval method of the gold's inter-rater agreement (Krippendorff's alpha, unit bootstrap).
METHOD_ALPHA_BOOTSTRAP = "krippendorff_alpha_unit_bootstrap_percentile"

CalibResult = Literal["CERTIFIED", "REFUSED"]
Split = Literal["few_shot", "dev", "locked_test"]
# Which tail(s) an interval bounds. A Bound is always one-sided "lower"; an Interval says which.
Sided = Literal["two-sided", "lower", "upper"]

# The interval methods the shipped estimators produce, named once so records and the results table
# agree on the spelling.
METHOD_WILSON = "wilson"
# The Wilson bound taken at the effective size n / deff, for gold whose items cluster by trace.
METHOD_WILSON_CLUSTERED = "wilson_cluster_neff"
METHOD_RG_JEFFREYS = "rogan_gladen_jeffreys_posterior_percentile"
# The same interval with the eval sample taken at its effective size n / deff (clustered verdicts).
METHOD_RG_JEFFREYS_CLUSTERED = "rogan_gladen_jeffreys_posterior_percentile_cluster_neff"
METHOD_NEWCOMBE_PAIRED = "newcombe_paired_hybrid_score"
METHOD_NEWCOMB_DIFF = "newcomb_diff_2indep"


def _refuse_bool_or_text(value: Any) -> Any:
    """Refuse a bool or a string where a gate number belongs; pydantic would coerce either."""
    if isinstance(value, bool | str | bytes):
        # pydantic reports only a ValueError as a validation failure; a TypeError would escape.
        raise ValueError(f"must be a number, got {type(value).__name__} {value!r}")  # noqa: TRY004
    return value


class Bound(CalibratedAgentEvaluationModel):
    """A gated rate with its 95% one-sided lower confidence bound.

    The gate reads ``lb95``, never ``point`` -- the point estimate at the threshold does
    not certify. Carrying both in one object is how the "no bare number" rule is made structural.

    ``lb95`` is a **one-sided** 95% Wilson lower bound (``sided="lower"``), not the lower edge of a
    two-sided 95% interval; ``n`` is the class size it was computed over. For gold whose items
    cluster (several spans of one trace) the bound is taken at the effective size and ``method``
    says so.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = CalibrationError

    point: float
    lb95: float
    n: int
    method: str = METHOD_WILSON
    level: float = 0.95
    sided: Literal["lower"] = "lower"

    @field_validator("point", "lb95", "n", "level", mode="before")
    @classmethod
    def _numbers_only(cls, value: Any) -> Any:
        return _refuse_bool_or_text(value)


class Interval(CalibratedAgentEvaluationModel):
    """A two-sided confidence interval for the Rogan-Gladen corrected rate.

    ``point`` is ``None`` when the correction was refused (J<0.5, or non-identifiable);
    ``refusal_reason`` records which. The ``(lo, hi)`` bound is always populated and is the
    honest signal even when the point is refused.

    ``method``, ``level``, ``sided`` and ``n`` say what kind of interval this is, so a 95%
    two-sided interval and a 95% one-sided bound can never be confused downstream. ``n`` is the
    sample the interval was computed over (for a two-sample difference, both samples together).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = CalibrationError

    point: float | None
    lo: float
    hi: float
    refusal_reason: str | None = None
    method: str
    level: float
    sided: Sided
    n: int

    @field_validator("point", "lo", "hi", "level", "n", mode="before")
    @classmethod
    def _numbers_only(cls, value: Any) -> Any:
        return _refuse_bool_or_text(value)

    def require_point(self) -> float:
        """Return the corrected point rate, or raise the typed refusal that produced ``None``.

        The point-rate accessor: a caller wanting the scalar must go through this,
        so a refused correction surfaces as a typed :class:`WeakJudgeError` /
        :class:`NonIdentifiableRateError` rather than a silent ``None`` leaking into a report. The
        ``(lo, hi)`` bound stays readable as the signal.
        """
        if self.point is not None:
            return self.point
        reason = self.refusal_reason or "the correction was refused"
        if self.refusal_reason is not None and TRANSFER_REFUSAL in self.refusal_reason:
            raise ErrorRatesDoNotTransferError(f"corrected point rate is unavailable: {reason}")
        if self.refusal_reason is not None and self.refusal_reason.startswith("non-identifiable"):
            raise NonIdentifiableRateError(
                f"corrected point rate is unavailable: {reason}; the interval "
                f"[{self.lo:.4f}, {self.hi:.4f}] is the honest signal"
            )
        raise WeakJudgeError(
            f"corrected point rate is unavailable: {reason}; the widening interval "
            f"[{self.lo:.4f}, {self.hi:.4f}] is the honest signal"
        )


def interval_from_dict(doc: Mapping[str, Any]) -> Interval:
    """Rebuild an :class:`Interval` from its dumped form; every kind field is required."""
    return Interval.model_validate(dict(doc))


@dataclass(frozen=True)
class LabelPair:
    """One paired observation: a human gold label and the judge's verdict for the same item.

    Both reference the same ``(trace_id, span_id)``; the engine derives the 2x2 confusion counts
    from a set of these, treating ``outcome == "FAIL"`` as the positive class.
    """

    gold: Label
    verdict: Label


@dataclass(frozen=True)
class CalibrationSet:
    """A split-tagged set of paired observations.

    ``split`` tags the provenance: ``"few_shot"`` (never evaluated), ``"dev"`` (iterate
    freely, never certifies), ``"locked_test"`` (the only source of a CERTIFIED verdict,
    consumed exactly once). ``set_id`` names the set; the ledger also keys on its content, so a
    renamed copy is still a used set.
    """

    pairs: tuple[LabelPair, ...]
    split: Split
    set_id: str


class CalibrationRecord(CalibratedAgentEvaluationModel):
    """The reproducible gate evidence.

    Pins the judge's exact ``(model, prompt_version)`` -- a change to either invalidates the record.
    Carries the gated bounds, Youden's J, the corrected-rate interval, and the CERTIFIED/REFUSED
    result. Certification is an intersection-union test per judge (both lower bounds must clear),
    and certifying several judges is not a joint claim, so the record carries no multiplicity
    family. References its sample members by ``trace_id`` even though the
    record lives outside the trace store.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = CalibrationError

    calib_id: str
    judge_id: str
    model: str  # pinned judge model id     ┐ a change to either
    prompt_version: str  # pinned judge prompt sha  ┘ invalidates this record
    hunted_set_id: str  # fail-heavy TPR/TNR set
    n_pos: int  # hunted positives (FAIL class)
    n_neg: int  # hunted negatives (PASS class)
    tpr: Bound  # gate on tpr.lb95
    tnr: Bound  # gate on tnr.lb95
    # How much the clustering of each class's gold by trace inflates its variance; 1.0 when every
    # trace contributes one target. The bounds above and the corrected rate's calibration
    # posteriors are taken at n / design_effect.
    tpr_design_effect: float = 1.0
    tnr_design_effect: float = 1.0
    youden_j: float  # RG refused if < 0.5
    result: CalibResult
    rate_set_id: str | None = None  # SEPARATE production-representative sample; None until scored
    corrected_rate: Interval | None = None  # None until a rate sample is scored
    locked_test_consumed: bool = False  # single-use
    bias_battery_ref: str | None = None  # slot for the bias battery; not run here
    schema_version: str = CALIB_SCHEMA_VERSION
    trace_ids: tuple[str, ...] = Field(default_factory=tuple)  # members by trace_id
    # The plan the gold was graded against: the per-class size to hunt and the
    # coverage floor at the planned accuracy. None when the judge was certified without a plan.
    planned_per_class: int | None = None
    planned_coverage_floor: int | None = None
    # The coverage floor at the TPR and TNR the judge showed on the locked test.
    # The floor depends on accuracy, so a judge whose accuracy differs from the plan can need more
    # gold than planned. Set for every CERTIFIED record, with or without a plan;
    # ``observed_floor_unmeasured`` is True when no floor was measured at the observed rates, and
    # the floor is then None.
    observed_coverage_floor: int | None = None
    observed_floor_unmeasured: bool = False
    # Hash of the judge's verdict-deciding content (prompt, model, fields, sampling) when the judge
    # carries one; None for records made before fingerprints, which never certify such a judge.
    judge_fingerprint: str | None = None
    # How many times the judge was tested on this gold: 1, plus one per refused-short set this
    # one grew. A count above 1 is disclosed, never corrected for.
    looks: int = 1
    # The human raters' agreement on the gold (Krippendorff's alpha with its interval), when the
    # gold came from two or more raters who overlapped; None when it was not measured.
    agreement_alpha: Interval | None = None
    # What produced the numbers above: seed/resamples of the corrected-rate bootstrap,
    # the gate thresholds, and the library versions. Required, keyword-only.
    provenance: ProvenanceStamp

    @property
    def coverage_floor(self) -> int | None:
        """The floor that applies: the larger of the planned and the observed floor, if any."""
        floors = [
            f for f in (self.planned_coverage_floor, self.observed_coverage_floor) if f is not None
        ]
        return max(floors) if floors else None

    @property
    def effective_n_pos(self) -> float:
        """The FAIL-class gold size the bounds were taken at: ``n_pos / tpr_design_effect``."""
        return self.n_pos / self.tpr_design_effect

    @property
    def effective_n_neg(self) -> float:
        """The PASS-class gold size the bounds were taken at: ``n_neg / tnr_design_effect``."""
        return self.n_neg / self.tnr_design_effect

    @property
    def below_coverage_floor(self) -> bool:
        """Whether either class of the gold is smaller than the coverage floor that applies.

        Such a record certifies (the Wilson bound is valid at any n), but the corrected rate's
        interval is not known to cover nominally at that size, so ``corrected_rate`` refuses it.
        A record whose observed rates have no measured floor is below it at any size. Clustered
        gold is compared at its effective size.
        """
        if self.observed_floor_unmeasured:
            return True
        floor = self.coverage_floor
        return floor is not None and min(self.effective_n_pos, self.effective_n_neg) < floor
