# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-failure-mode trend report.

Turns the per-mode :class:`~calibrated_agent_evaluation.run.report.ModeCounts` into one
:class:`~calibrated_agent_evaluation.run.report.TrendPoint` per failure mode, each carrying the
**Rogan-Gladen prevalence-corrected rate + CI** (never a raw judge fraction, never a single blended
aggregate accuracy number). The correction is ``corrected_rate`` (reused, never
re-implemented): a mode whose judge has Youden's J < 0.5, or whose measured rate is outside the
identifiable region, comes back as an :class:`~calibrated_agent_evaluation.calibrate.Interval` with
``point=None`` + ``refusal_reason`` -- and the trend keeps that mode **present with its widening
interval**, never dropped and never a fabricated point. For a crew each ``(agent, failure_type)``
mode is corrected by its OWN judge's calibration record.

The production-frame guard lives here too: :func:`assert_production_rate_frame` refuses a
discovery-frame (CI-augmented) row from entering a headline rate, reusing
``assert_production_frame`` -- a hunted trace is not a random draw.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from calibrated_agent_evaluation.calibrate import (
    CalibrationRecord,
    RateSampleTooSmallError,
    RateUnsafeError,
    corrected_rate,
)
from calibrated_agent_evaluation.calibrate.engine import DEFAULT_N_RESAMPLES, RG_LEVEL
from calibrated_agent_evaluation.calibrate.record import METHOD_RG_JEFFREYS, METHOD_WILSON, Interval
from calibrated_agent_evaluation.run.report import ModeCounts, ModeKey, TrendPoint
from calibrated_agent_evaluation.sampling import FrameRow, assert_production_frame
from calibrated_agent_evaluation.stats.proportions import wilson_lower, wilson_upper


def trend_report(
    counts: Mapping[ModeKey, ModeCounts],
    calib_by_mode: Mapping[ModeKey, CalibrationRecord],
    rate_samples: Mapping[ModeKey, Sequence[int]],
    *,
    seed: int = 0,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    rate_trace_ids: Mapping[ModeKey, Sequence[str]],
    rate_cluster_ids: Mapping[ModeKey, Sequence[str]] | None = None,
) -> list[TrendPoint]:
    """One :class:`TrendPoint` per mode: the corrected rate + CI, keyed per failure mode.

    ``counts`` is the per-mode aggregation (keyed by
    :class:`~calibrated_agent_evaluation.run.report.ModeKey`); ``calib_by_mode`` gives each mode its
    own CalibrationRecord; ``rate_samples`` gives each mode the judge verdicts (1=FAIL, 0=PASS) over
    its SEPARATE production-representative sample
    that ``corrected_rate`` corrects. A mode absent from ``calib_by_mode`` or ``rate_samples`` is
    skipped (it cannot be corrected without its own judge + rate sample) rather than mis-reported.

    ``rate_trace_ids`` gives, per mode, the trace id of each rate-sample verdict, which
    ``corrected_rate`` checks against the calibration gold; a mode without them is refused there.
    ``rate_cluster_ids`` gives, per mode, the trace id of each rate-sample verdict when a trace
    contributes several, so that mode's interval accounts for the clustering.
    A mode absent from it is taken as independent verdicts.

    A refused correction (J < 0.5, non-identifiable, a rate sample below the minimum, or a record
    whose gold is below the coverage floor) surfaces as an ``Interval`` with ``point=None`` +
    ``refusal_reason``; the mode is still emitted, so the gate and the other modes still report.
    The report contains no single blended aggregate accuracy scalar:
    the only rates are the per-mode corrected-rate intervals.
    """
    points: list[TrendPoint] = []
    for key, mode_counts in counts.items():
        record = calib_by_mode.get(key)
        verdicts = rate_samples.get(key)
        if record is None or verdicts is None:
            continue
        # Reuse corrected_rate; it returns a copy of the record carrying the
        # Interval (point + CI, or point=None + refusal_reason when refused). We surface that
        # Interval verbatim -- the widening-bound-on-refusal behavior is the correction's contract.
        rate_set_id = f"rate-{mode_counts.mode.failure_type}-{mode_counts.mode.agent or 'workflow'}"
        try:
            scored = corrected_rate(
                record,
                verdicts,
                rate_set_id=rate_set_id,
                rate_trace_ids=rate_trace_ids.get(key, ()),
                seed=seed,
                n_resamples=n_resamples,
                rate_cluster_ids=None if rate_cluster_ids is None else rate_cluster_ids.get(key),
            )
        except (RateSampleTooSmallError, RateUnsafeError) as exc:
            # The mode stays in the report with the reason; [0, 1] claims nothing.
            why = (
                "rate sample too small"
                if isinstance(exc, RateSampleTooSmallError)
                else ("gold below the coverage floor")
            )
            interval: Interval | None = Interval(
                point=None,
                lo=0.0,
                hi=1.0,
                refusal_reason=f"{why}: {exc}",
                method=METHOD_RG_JEFFREYS,
                level=RG_LEVEL,
                sided="two-sided",
                n=len(verdicts),
            )
        else:
            interval = scored.corrected_rate
        if interval is None:  # defensive: corrected_rate always attaches one; never leak a None
            interval = Interval(
                point=None,
                lo=0.0,
                hi=1.0,
                refusal_reason="no interval produced",
                method=METHOD_RG_JEFFREYS,
                level=RG_LEVEL,
                sided="two-sided",
                n=len(verdicts),
            )
        points.append(
            TrendPoint(
                mode=mode_counts.mode,
                n_total=mode_counts.n_total,
                corrected_rate=interval,
                calib_id=record.calib_id,
                raw_rate=raw_rate_interval(verdicts),
            )
        )
    return points


def raw_rate_interval(verdicts: Sequence[int]) -> Interval:
    """The uncorrected FAIL fraction of ``verdicts`` (1 = FAIL) with its two-sided 95% Wilson CI.

    Raises :class:`~calibrated_agent_evaluation.calibrate.errors.StatisticsInputError` on an empty
    sample.
    """
    n = len(verdicts)
    k = sum(1 for v in verdicts if v)
    alpha = 1.0 - RG_LEVEL
    lo, hi = wilson_lower(k, n, alpha=alpha), wilson_upper(k, n, alpha=alpha)
    return Interval(
        point=k / n,
        lo=lo,
        hi=hi,
        method=METHOD_WILSON,
        level=RG_LEVEL,
        sided="two-sided",
        n=n,
    )


def assert_production_rate_frame(rows: Sequence[FrameRow]) -> None:
    """Refuse a discovery-frame row from entering a headline (production) rate.

    Reuses :func:`calibrated_agent_evaluation.sampling.assert_production_frame`: a CI-augmented
    (discovery) trace is non-random augmentation, so it grows the discovery set but must never be
    counted toward a production-representative headline rate (raises ``FrameContaminationError``).
    The run stage does not re-implement the guard -- it wires its rate path through the existing
    one.
    """
    assert_production_frame(rows)
