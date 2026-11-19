# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error family for the judge-calibration engine.

Every refusal the calibration gate makes is a typed, explained error -- never a silent no-op or a
raw ``scipy``/``statsmodels``/``OSError`` escaping a public surface (conventions: "no silent
failure", "wrap external errors at the boundary"). All errors subclass
:class:`CalibrationError`, which subclasses the shared
:class:`~calibrated_agent_evaluation.schema.errors.CalibratedAgentEvaluationDataError`
base so a caller can catch the whole methodology-plane family at the plane boundary.

Each subclass maps to exactly one refusal (one refusal -> one typed error -> one negative test):

  * :class:`UncalibratedJudgeError`   -- no CERTIFIED calibration -> refuse to correct/promote.
  * :class:`WeakJudgeError`           -- Youden's J < 0.5 -> refuse the corrected *point* rate.
  * :class:`NonIdentifiableRateError` -- measured rate outside the RG-identifiable region.
  * :class:`UnratifiedGoldError`      -- a gold set contains a non-human label.
  * :class:`FewShotEvaluationError`   -- metrics requested over the few-shot pool.
  * :class:`LockedTestReuseError`     -- an already-consumed locked test is re-evaluated.
  * :class:`SampleMisuseError`        -- the true rate is read off the hunted set.
  * :class:`RateUnsafeError`          -- a corrected rate from gold below the coverage floor.
  * :class:`RateSampleTooSmallError`  -- a corrected rate from too few rate-sample traces.
  * :class:`UnpairedObservationError` -- an observation lacks exactly one gold + one verdict.
  * :class:`DuplicateGoldError`       -- one target carries more than one gold label.
  * :class:`ErrorRatesDoNotTransferError` -- the rate sample contradicts the judge's measured error
    rates, so no corrected-rate interval can be read from it.
  * :class:`InfeasiblePlanError`      -- the planner is asked for an unreachable target.
  * :class:`CalibrationStoreError`    -- a store read/write failed (wraps ``OSError``).
  * :class:`CalibrationVersionError`  -- a stored record's schema_version != the running one.
  * :class:`StatisticsInputError`     -- a pure stats primitive got an out-of-domain numeric input
    (n <= 0, k outside [0, n], a probability outside [0, 1]); guarded so a bad count surfaces as a
    typed error rather than a NaN leaking out of scipy/statsmodels ("no silent failure").
"""

from __future__ import annotations

from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError

# The plain-language core of the refusal for a rate sample that contradicts the judge's error rates.
TRANSFER_REFUSAL = "judge error rates don't transfer to this sample"


class CalibrationError(CalibratedAgentEvaluationDataError):
    """Base class for every calibration-engine refusal."""


class UncalibratedJudgeError(CalibrationError):
    """A corrected rate or promotion was requested for a judge with no CERTIFIED calibration."""


class WeakJudgeError(CalibrationError):
    """Youden's J < 0.5 -- the Rogan-Gladen point correction is refused.

    The widening interval is still returned as the honest signal; only the *point* rate accessor
    raises, so a near-random judge cannot be "rescued" by a correction that divides by ~0.
    """


class NonIdentifiableRateError(CalibrationError):
    """The measured rate lies outside the Rogan-Gladen identifiable region.

    ``measured < 1 - TNR`` or the denominator ``TPR - (1 - TNR)) <= 0``; the corrected point is
    refused and the bound is reported rather than silently clamping to 0/1.
    """


class UnratifiedGoldError(CalibrationError):
    """A calibration gold set contains a label not produced/ratified by a human.

    Certifying a judge against another judge turns every confidence bound into theater -- refused.
    """


class FewShotEvaluationError(CalibrationError):
    """Certification/TPR/TNR metrics were requested over the few-shot pool.

    The few-shot pool is never an evaluation set; evaluating on it inflates the numbers.
    """


class LockedTestReuseError(CalibrationError):
    """A locked test set that was already consumed once was re-evaluated (single-use)."""


class SampleMisuseError(CalibrationError):
    """The prevalence-corrected rate was requested from the hunted fail-heavy set.

    The true rate is measured on a *separate* production-representative sample -- two samples, two
    jobs. Reading it off the hunted set would report a wildly inflated rate. Raised too when the
    rate sample shares traces with the calibration gold.
    """


class RateUnsafeError(CalibrationError):
    """A corrected rate was requested from a record certified on gold below the coverage floor.

    The certification stands (the Wilson bound is valid at any n), but below the coverage floor
    the corrected rate's interval was measured to cover the true rate less often than 95%, so an
    interval built from smaller gold is refused. At or above the floor coverage is close to 95%
    but not everywhere: a class with about 2 to 3 errors still under-covers slightly, a known
    limit of the current interval.
    """


class RateSampleTooSmallError(CalibrationError):
    """A corrected rate was requested from a rate sample smaller than the measured minimum.

    The interval was measured to under-cover with too few evaluation traces at low prevalence, so
    a rate sample below :data:`~calibrated_agent_evaluation.calibrate.plan.MIN_RATE_SAMPLE` is
    refused. ``n`` is the sample's size (distinct traces when verdicts are clustered) and
    ``minimum`` the size needed.
    """

    def __init__(self, message: str, *, n: int, minimum: int) -> None:
        super().__init__(message)
        self.n = n
        self.minimum = minimum


class UnpairedObservationError(CalibrationError):
    """An observation could not be paired to exactly one human gold + one judge verdict.

    Raised rather than silently dropping the observation (which would bias TPR/TNR).
    """


class DuplicateGoldError(UnpairedObservationError):
    """One ``(trace_id, span_id)`` target carries more than one gold label.

    Two raters' labels on one target are one piece of evidence, not two: counting both doubles the
    sample size behind the bounds and can certify a judge the single-label set refuses.
    ``duplicates`` names the repeated targets.
    """

    def __init__(self, message: str, *, duplicates: tuple[tuple[str, str | None], ...]) -> None:
        super().__init__(message)
        self.duplicates = duplicates


class ErrorRatesDoNotTransferError(NonIdentifiableRateError):
    """The rate sample contradicts the judge's calibrated error rates.

    Almost every draw puts the measured rate below the judge's false-positive floor (or above its
    true-positive ceiling), so the corrected rate clips to a single point. A zero-width interval
    there would claim the true rate is exactly 0 (or 1); what the data show is that the error rates
    measured on the calibration gold do not hold on this sample.
    """


class InfeasiblePlanError(CalibrationError):
    """The calibration planner was asked for a target no finite sample size can reach.

    ``tau >= p_hat``: the expected judge accuracy never clears the threshold, so the lower bound
    cannot either -- refused rather than searching unboundedly.
    """


class CalibrationStoreError(CalibrationError):
    """A calibration-record file read or write failed.

    Wraps the underlying ``OSError`` so a filesystem failure surfaces inside the typed family
    rather than as a raw OS exception escaping the store's public surface.
    """


class CalibrationVersionError(CalibrationError):
    """A stored calibration record's ``schema_version`` != the running ``CALIB_SCHEMA_VERSION``.

    Refuse-loudly rather than coercing the data (the no-silent-coercion precedent).
    """


class StatisticsInputError(CalibrationError):
    """A pure statistics primitive received an out-of-domain numeric input.

    ``n <= 0``, ``k`` outside ``[0, n]``, or a probability outside ``[0, 1]`` -- guarded so a bad
    count surfaces as a typed error rather than a NaN leaking out of scipy/statsmodels.
    """
