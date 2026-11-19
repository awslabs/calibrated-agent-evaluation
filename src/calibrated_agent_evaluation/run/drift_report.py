# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input-distribution and judge drift reports.

Drift is split in two:

  * **input-distribution drift** -- :func:`input_drift` measures PSI + Jensen-Shannon divergence +
    IQR-normalized Wasserstein distance (reusing :mod:`calibrated_agent_evaluation.stats.drift`)
    between a frozen baseline and a fresh sample, each with a permutation p-value. PSI and
    Wasserstein are the gated pair: a metric is flagged when its p-value survives BY-FDR across
    those two and it clears its severity bar. JSD is reported only and never flags, because on the
    same bins its p-value tracks PSI's almost exactly, so gating both counts one test twice. NOT the
    KS test.
  * **judge drift** -- ΔTPR / ΔTNR against the frozen baseline, each with a CI, so a judge that
    has silently degraded surfaces before it reports a wrong rate. :func:`judge_drift` compares two
    calibration records on independent samples (``stats.drift.delta_proportion_ci``) and refuses
    records that share items; :func:`judge_drift_paired` compares two runs over the same items with
    Newcombe's paired interval.

Both are pure computations over a caller-supplied frozen baseline + fresh slice -- the continuous
gold-label trickle that feeds judge drift in production is a later active-cadence feature, not here.
Empty inputs raise the typed :class:`StatisticsInputError` rather than returning a spurious zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from calibrated_agent_evaluation.calibrate.errors import SampleMisuseError, StatisticsInputError
from calibrated_agent_evaluation.calibrate.record import (
    METHOD_NEWCOMB_DIFF,
    METHOD_NEWCOMBE_PAIRED,
    CalibrationRecord,
    Interval,
    LabelPair,
)
from calibrated_agent_evaluation.run.flags import flag_slices
from calibrated_agent_evaluation.run.report import DriftMetric, InputDriftReport, JudgeDriftReport
from calibrated_agent_evaluation.stats.drift import (
    DEFAULT_N_PERMUTATIONS,
    delta_proportion_ci,
    jensen_shannon,
    jsd_of_samples,
    permutation_p,
    psi,
    psi_permutation_p,
    quantile_histogram,
    wasserstein_normalized,
    wasserstein_normalized_arrays,
)
from calibrated_agent_evaluation.stats.proportions import newcombe_paired_diff

# Severity bars a flagged metric must also clear. A gated metric is flagged only when its
# permutation p-value survives BY-FDR across the gated pair; the bar then keeps a statistically
# detectable but negligible shift at large n from flagging (the reason KS is not used). PSI's 0.25
# is the conventional "major shift" band for decile bins. Wasserstein's 0.10 is a shift of a tenth
# of the pooled interquartile range (about 0.13 standard deviations for a normal input): below it a
# shift is too small to act on however many traces detect it. A 0.03 SD shift at 20,000 traces per
# side is significant but stays under the bar.
DEFAULT_PSI_THRESHOLD = 0.25
DEFAULT_WASSERSTEIN_THRESHOLD: float | None = 0.10
DEFAULT_DRIFT_BINS = 10

METHOD_PSI = "psi_baseline_quantile_bins_add_half"
METHOD_JSD_BINNED = "jsd_baseline_quantile_bins"
METHOD_JSD_COUNTS = "jsd_category_counts"
METHOD_WASSERSTEIN = "wasserstein_iqr_normalized"


@dataclass(frozen=True)
class DriftThresholds:
    """Severity bars for the gated metrics; ``None`` means significance alone flags the metric.

    JSD has no bar: it is reported only and never flags.
    """

    psi: float | None = DEFAULT_PSI_THRESHOLD
    wasserstein: float | None = DEFAULT_WASSERSTEIN_THRESHOLD


def _expand_counts(counts: Sequence[float], *, name: str) -> list[float]:
    """Turn a per-category count vector into one category index per observation (to permute)."""
    out: list[float] = []
    for i, c in enumerate(counts):
        if c < 0 or float(c) != int(c):
            raise StatisticsInputError(
                f"drift: {name} must be non-negative integer counts to permutation-test, "
                f"element {i} is {c!r}"
            )
        out.extend([float(i)] * int(c))
    return out


def _flag(value: float, threshold: float | None, significant: bool) -> bool:
    return significant and (threshold is None or value > threshold)


def input_drift(
    baseline: Sequence[float],
    fresh: Sequence[float],
    *,
    dimension: str = "input",
    thresholds: DriftThresholds | None = None,
    jsd_baseline: Sequence[float] | None = None,
    jsd_fresh: Sequence[float] | None = None,
    bins: int = DEFAULT_DRIFT_BINS,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 0,
) -> InputDriftReport:
    """Report PSI + JSD + Wasserstein input drift between ``baseline`` and ``fresh``.

    ``baseline``/``fresh`` are raw-value samples. PSI bins them on the baseline's quantiles;
    Wasserstein is divided by the pooled IQR. JSD reads ``jsd_baseline``/``jsd_fresh`` when supplied
    (per-category counts, permuted as category labels); absent those it uses the PSI bins. Each
    metric gets a permutation p-value (``n_permutations`` relabellings, ``seed``). PSI and
    Wasserstein are corrected together with BY-FDR at q = 0.10 via
    :func:`calibrated_agent_evaluation.run.flag_slices` (family ``"drift_flags"``, ``m = 2``), and
    either is ``exceeded`` when it survives that correction and clears its severity bar. JSD is
    reported with its p-value but sits outside the family and is never ``exceeded``. Empty inputs,
    or fewer than ``5 * bins`` values per side, raise :class:`StatisticsInputError` -- never a
    spurious zero-drift.
    """
    thresholds = thresholds or DriftThresholds()
    psi_value = psi(baseline, fresh, bins=bins)
    psi_p = psi_permutation_p(baseline, fresh, bins=bins, n_perm=n_permutations, seed=seed)

    if jsd_baseline is not None and jsd_fresh is not None:
        jsd_value = jensen_shannon(jsd_baseline, jsd_fresh)
        k = max(len(jsd_baseline), len(jsd_fresh))

        def _jsd_counts(b: np.ndarray, f: np.ndarray) -> float:
            return jensen_shannon(
                np.bincount(b.astype(int), minlength=k).tolist(),
                np.bincount(f.astype(int), minlength=k).tolist(),
            )

        jsd_p = permutation_p(
            _jsd_counts,
            _expand_counts(jsd_baseline, name="jsd_baseline"),
            _expand_counts(jsd_fresh, name="jsd_fresh"),
            n_perm=n_permutations,
            seed=seed,
        )
        jsd_method = METHOD_JSD_COUNTS
    else:
        jsd_value = jensen_shannon(*quantile_histogram(baseline, fresh, bins=bins))
        jsd_p = permutation_p(
            lambda b, f: jsd_of_samples(b, f, bins=bins),
            baseline,
            fresh,
            n_perm=n_permutations,
            seed=seed,
        )
        jsd_method = METHOD_JSD_BINNED

    wass_value = wasserstein_normalized(baseline, fresh)
    wass_p = permutation_p(
        wasserstein_normalized_arrays, baseline, fresh, n_perm=n_permutations, seed=seed
    )

    family = flag_slices([psi_p, wass_p], family="drift_flags")
    rej_psi, rej_wass = family.reject
    metrics = (
        DriftMetric(
            name="psi",
            value=psi_value,
            threshold=thresholds.psi,
            exceeded=_flag(psi_value, thresholds.psi, rej_psi),
            p_value=psi_p,
            method=METHOD_PSI,
        ),
        DriftMetric(
            name="jsd",
            value=jsd_value,
            threshold=None,
            exceeded=False,
            p_value=jsd_p,
            method=jsd_method,
        ),
        DriftMetric(
            name="wasserstein",
            value=wass_value,
            threshold=thresholds.wasserstein,
            exceeded=_flag(wass_value, thresholds.wasserstein, rej_wass),
            p_value=wass_p,
            method=METHOD_WASSERSTEIN,
        ),
    )
    return InputDriftReport(
        metrics=metrics,
        dimension=dimension,
        multiplicity=family,
        n_permutations=n_permutations,
        seed=seed,
        n_baseline=len(baseline),
        n_fresh=len(fresh),
    )


def _delta(count_after: int, nobs_after: int, count_before: int, nobs_before: int) -> Interval:
    """A change-in-proportion delta as an Interval (point = after-before, lo/hi = its CI)."""
    point, lo, hi = delta_proportion_ci(count_after, nobs_after, count_before, nobs_before)
    # n is both samples together: the delta is a two-sample difference.
    return Interval(
        point=point,
        lo=lo,
        hi=hi,
        method=METHOD_NEWCOMB_DIFF,
        level=0.95,
        sided="two-sided",
        n=nobs_after + nobs_before,
    )


def judge_drift(before: CalibrationRecord, after: CalibrationRecord) -> JudgeDriftReport:
    """Report ΔTPR / ΔTNR between two calibration records on independent samples, each with a CI.

    Reconstructs the TPR/TNR counts from each record (``round(tpr.point * n_pos)`` etc.) and takes
    the difference-of-proportions CI for two independent samples
    (``stats.drift.delta_proportion_ci``). A judge whose TPR/TNR is unchanged yields deltas
    straddling 0; a degraded judge yields a delta whose CI sits below 0 -- the signal that the judge
    needs re-calibration. ``before``/``after`` are the two frozen calibration records the caller
    supplies (the trickle that produces them in production is a later active-cadence feature).

    The independent interval is wrong when the two records share items, because the same item
    judged twice makes the comparison paired. Sharing is refused with
    :class:`SampleMisuseError` when the records name the same hunted set or any member trace id in
    common; use :func:`judge_drift_paired` with both runs' verdicts instead. A record without
    member trace ids (built outside :func:`~calibrated_agent_evaluation.calibrate.certify`) is taken
    to be on its own sample.
    """
    shared = set(before.trace_ids) & set(after.trace_ids)
    if before.hunted_set_id == after.hunted_set_id or shared:
        what = (
            f"the same hunted set {after.hunted_set_id!r}"
            if before.hunted_set_id == after.hunted_set_id
            else f"{len(shared)} shared trace id(s)"
        )
        raise SampleMisuseError(
            f"judge_drift: the before and after records share items ({what}), so the comparison "
            "is paired and an independent-samples interval is wrong; call judge_drift_paired with "
            "both runs' verdicts on the shared items"
        )
    tp_before = round(before.tpr.point * before.n_pos)
    tp_after = round(after.tpr.point * after.n_pos)
    tn_before = round(before.tnr.point * before.n_neg)
    tn_after = round(after.tnr.point * after.n_neg)
    return JudgeDriftReport(
        judge_id=after.judge_id,
        delta_tpr=_delta(tp_after, after.n_pos, tp_before, before.n_pos),
        delta_tnr=_delta(tn_after, after.n_neg, tn_before, before.n_neg),
    )


def _paired_delta(before_ok: list[bool], after_ok: list[bool]) -> Interval:
    """After-minus-before rate of ``ok`` over the same items, with Newcombe's paired interval."""
    a = sum(1 for b, x in zip(before_ok, after_ok, strict=True) if b and x)
    b_only = sum(1 for b, x in zip(before_ok, after_ok, strict=True) if x and not b)
    c_only = sum(1 for b, x in zip(before_ok, after_ok, strict=True) if b and not x)
    d = len(before_ok) - a - b_only - c_only
    # newcombe_paired_diff(a, b, c, d) returns (a+b)/n - (a+c)/n: after minus before.
    point, lo, hi = newcombe_paired_diff(a, b_only, c_only, d)
    return Interval(
        point=point,
        lo=lo,
        hi=hi,
        method=METHOD_NEWCOMBE_PAIRED,
        level=0.95,
        sided="two-sided",
        n=len(before_ok),
    )


def judge_drift_paired(
    before: Sequence[LabelPair], after: Sequence[LabelPair], *, judge_id: str
) -> JudgeDriftReport:
    """Report ΔTPR / ΔTNR between two runs of a judge over the same items, each with a CI.

    ``before`` and ``after`` are (human gold, judge verdict) pairs for the same items, matched on
    ``(trace_id, span_id)``. Within each gold class, an item judged correctly by one run and not the
    other is a discordant pair; the change in TPR (over gold FAIL items) and TNR (over gold PASS
    items) gets Newcombe's (1998) paired hybrid-score interval, which uses that pairing instead of
    treating the two runs as independent samples.

    Raises :class:`SampleMisuseError` when the two runs do not cover exactly the same items or an
    item's gold label differs between them, and :class:`StatisticsInputError` when a gold class is
    empty.
    """

    def _index(pairs: Sequence[LabelPair], side: str) -> dict[tuple[str, str | None], LabelPair]:
        out: dict[tuple[str, str | None], LabelPair] = {}
        for pair in pairs:
            key = (pair.gold.trace_id, pair.gold.span_id)
            if key in out:
                raise SampleMisuseError(f"judge_drift_paired: {side} has item {key} twice")
            out[key] = pair
        return out

    b_items, a_items = _index(before, "before"), _index(after, "after")
    if b_items.keys() != a_items.keys():
        missing = len(b_items.keys() ^ a_items.keys())
        raise SampleMisuseError(
            f"judge_drift_paired: the runs must cover the same items; {missing} item(s) are in "
            "only one of them"
        )
    by_class: dict[str, tuple[list[bool], list[bool]]] = {"FAIL": ([], []), "PASS": ([], [])}
    for key, b_pair in b_items.items():
        a_pair = a_items[key]
        gold = b_pair.gold.outcome
        if a_pair.gold.outcome != gold:
            raise SampleMisuseError(
                f"judge_drift_paired: item {key} has gold {gold!r} before and "
                f"{a_pair.gold.outcome!r} after"
            )
        if gold not in by_class:
            raise StatisticsInputError(
                f"judge_drift_paired: gold outcome must be FAIL or PASS, got {gold!r}"
            )
        before_ok, after_ok = by_class[gold]
        before_ok.append(b_pair.verdict.outcome == gold)
        after_ok.append(a_pair.verdict.outcome == gold)
    for gold, (ok, _) in by_class.items():
        if not ok:
            raise StatisticsInputError(f"judge_drift_paired: no gold {gold} items to compare")
    return JudgeDriftReport(
        judge_id=judge_id,
        delta_tpr=_paired_delta(*by_class["FAIL"]),
        delta_tnr=_paired_delta(*by_class["PASS"]),
    )
