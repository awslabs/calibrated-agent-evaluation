# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The results table: one flat row per reported estimate, refusals included.

Every record store writes one CSV row per *record* with JSON-blob cells, which is an archive, not a
table a reader can load into pandas or R. :func:`results_table` flattens persisted records into
:class:`Estimate` rows (one per interval or bound the record reports) and :func:`write_results_csv`
writes them. A refused estimate keeps its row: ``point`` is empty and ``refusal_reason`` says why,
so an honest refusal never vanishes from the table the way it does from the dashboard series.

Each row carries the interval's method, level, sidedness and ``n`` (from the
:class:`~calibrated_agent_evaluation.calibrate.Interval` /
:class:`~calibrated_agent_evaluation.calibrate.Bound` it came from) and the record's provenance, so
a row can be traced to the computation that produced it.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation._provenance import ProvenanceStamp
from calibrated_agent_evaluation._recordstore import atomic_write_text
from calibrated_agent_evaluation.agentcore.audit_types import (
    AbTestClaimRecord,
    ArmAssignment,
    AuditRecord,
    RecommendationClaimRecord,
    RunProvenance,
)
from calibrated_agent_evaluation.bias.record import BiasBatteryRecord
from calibrated_agent_evaluation.calibrate.record import (
    METHOD_WILSON,
    Bound,
    CalibrationRecord,
    Interval,
    Sided,
)
from calibrated_agent_evaluation.flagship.report import FlagshipReport
from calibrated_agent_evaluation.improve.report import ImproveRecord
from calibrated_agent_evaluation.ingest.sample import SampleRecord
from calibrated_agent_evaluation.judges.run_record import JudgeRunRecord
from calibrated_agent_evaluation.probe.policy import POLICY_FINDING_KIND
from calibrated_agent_evaluation.probe.record import ProbeBatteryRecord
from calibrated_agent_evaluation.review.record import (
    AgreementRecord,
    AgreementStat,
    CodebookRecord,
    ProposedModeRecord,
)
from calibrated_agent_evaluation.run.judge_check import JudgeCheckRecord
from calibrated_agent_evaluation.run.report import ModeKey, RunRecord
from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError
from calibrated_agent_evaluation.subjective.store import PreferenceRecord

Record = (
    CalibrationRecord
    | RunRecord
    | ImproveRecord
    | BiasBatteryRecord
    | FlagshipReport
    | PreferenceRecord
    | JudgeRunRecord
    | AgreementRecord
    | CodebookRecord
    | ProposedModeRecord
    | JudgeCheckRecord
    | SampleRecord
    | ProbeBatteryRecord
    | AbTestClaimRecord
    | ArmAssignment
    | RecommendationClaimRecord
    | AuditRecord
    | RunProvenance
)

METHOD_UNIT_BOOTSTRAP = "percentile_bootstrap_over_units"


class ResultsTableError(CalibratedAgentEvaluationDataError):
    """A results table could not be built (unknown record type) or written (filesystem failure)."""


@dataclass(frozen=True)
class Estimate:
    """One reported estimate with everything needed to read it correctly.

    ``estimand`` names the quantity (``"tpr"``, ``"corrected_rate"``, ``"bias_magnitude"``, ...);
    ``slice`` names what it is about (a judge, a failure mode, a bias, a stage). ``point`` is
    ``None`` exactly when the estimate was refused, and then ``refusal_reason`` is set. ``lo`` /
    ``hi`` are the interval's edges; a one-sided bound puts the open side at the support's edge
    (``hi=1.0`` for a lower bound, ``lo=0.0`` for an upper one) and says so in ``sided``.
    ``seed`` / ``n_resamples`` are set only for a bootstrapped estimate. ``source`` is the record
    kind and id the row came from.
    """

    estimand: str
    slice: str
    point: float | None
    lo: float
    hi: float
    level: float
    sided: Sided
    method: str
    n: int
    n_resamples: int | None
    seed: int | None
    refusal_reason: str | None
    provenance: ProvenanceStamp
    source: str


def _from_interval(
    iv: Interval,
    *,
    estimand: str,
    slice_: str,
    provenance: ProvenanceStamp,
    source: str,
    bootstrapped: bool,
) -> Estimate:
    return Estimate(
        estimand=estimand,
        slice=slice_,
        point=iv.point,
        lo=iv.lo,
        hi=iv.hi,
        level=iv.level,
        sided=iv.sided,
        method=iv.method,
        n=iv.n,
        n_resamples=provenance.n_resamples if bootstrapped else None,
        seed=provenance.seed if bootstrapped else None,
        refusal_reason=iv.refusal_reason,
        provenance=provenance,
        source=source,
    )


def _from_bound(
    b: Bound, *, estimand: str, slice_: str, provenance: ProvenanceStamp, source: str
) -> Estimate:
    return Estimate(
        estimand=estimand,
        slice=slice_,
        point=b.point,
        lo=b.lb95,
        hi=1.0,
        level=b.level,
        sided=b.sided,
        method=b.method,
        n=b.n,
        n_resamples=None,
        seed=None,
        refusal_reason=None,
        provenance=provenance,
        source=source,
    )


def _mode_label(mode: ModeKey) -> str:
    return f"{mode.agent}/{mode.failure_type}" if mode.agent else mode.failure_type


def _calibration_rows(r: CalibrationRecord) -> list[Estimate]:
    src = f"calibration:{r.calib_id}"
    rows = [
        _from_bound(r.tpr, estimand="tpr", slice_=r.judge_id, provenance=r.provenance, source=src),
        _from_bound(r.tnr, estimand="tnr", slice_=r.judge_id, provenance=r.provenance, source=src),
    ]
    if r.agreement_alpha is not None:
        rows.append(
            _from_interval(
                r.agreement_alpha,
                estimand="gold_agreement_alpha",
                slice_=r.judge_id,
                provenance=r.provenance,
                source=src,
                bootstrapped=False,
            )
        )
    if r.corrected_rate is not None:
        rows.append(
            _from_interval(
                r.corrected_rate,
                estimand="corrected_rate",
                slice_=r.judge_id,
                provenance=r.provenance,
                source=src,
                bootstrapped=True,
            )
        )
    return rows


def _run_rows(r: RunRecord) -> list[Estimate]:
    src = f"run:{r.run_id}"
    rows: list[Estimate] = []
    for tp in r.trend:
        rows.append(
            _from_interval(
                tp.corrected_rate,
                estimand="corrected_rate",
                slice_=_mode_label(tp.mode),
                provenance=r.provenance,
                source=src,
                bootstrapped=True,
            )
        )
        rows.append(
            _from_interval(
                tp.raw_rate,
                estimand="raw_rate",
                slice_=_mode_label(tp.mode),
                provenance=r.provenance,
                source=src,
                bootstrapped=False,
            )
        )
    for jd in r.judge_drift:
        deltas = [("delta_tpr", jd.delta_tpr), ("delta_tnr", jd.delta_tnr)]
        rows.extend(
            _from_interval(
                iv,
                estimand=name,
                slice_=jd.judge_id,
                provenance=r.provenance,
                source=src,
                bootstrapped=False,
            )
            for name, iv in deltas
        )
    return rows


def _improve_rows(r: ImproveRecord) -> list[Estimate]:
    src = f"improve:{r.improve_id}"
    effect = r.paired.effect
    if r.paired.undefined_reason is not None and effect.refusal_reason is None:
        # McNemar had nothing to test; the effect row must not read as a measured change.
        effect = Interval(
            point=None,
            lo=effect.lo,
            hi=effect.hi,
            refusal_reason=r.paired.undefined_reason,
            method=effect.method,
            level=effect.level,
            sided=effect.sided,
            n=effect.n,
        )
    rows = [
        _from_interval(
            effect,
            estimand="paired_effect",
            slice_=r.prereg.target_metric,
            provenance=r.provenance,
            source=src,
            bootstrapped=False,
        )
    ]
    for gs in r.panel.slices:
        for name, iv in (
            ("guard_rate_before", gs.rate_before),
            ("guard_rate_after", gs.rate_after),
        ):
            rows.append(
                _from_interval(
                    iv,
                    estimand=name,
                    slice_=gs.name,
                    provenance=r.provenance,
                    source=src,
                    bootstrapped=False,
                )
            )
    return rows


def _bias_rows(r: BiasBatteryRecord) -> list[Estimate]:
    src = f"bias:{r.battery_id}"
    rows = []
    for pr in r.results:
        ran = pr.status == "run"
        rows.append(
            Estimate(
                estimand="bias_magnitude",
                slice=f"{r.judge_id}/{pr.bias}",
                point=pr.magnitude if ran else None,
                lo=0.0,
                hi=pr.magnitude_ub95,
                level=0.95,
                sided="upper",
                method=METHOD_WILSON,
                n=pr.n,
                n_resamples=None,
                seed=None,
                refusal_reason=None if ran else (pr.detail or pr.status),
                provenance=r.provenance,
                source=src,
            )
        )
    return rows


def _flagship_rows(r: FlagshipReport) -> list[Estimate]:
    src = f"flagship:{r.flagship_id}"
    return [
        _from_interval(
            oc.headline,
            estimand=f"{oc.stage}_headline",
            slice_=f"{oc.agent}/{oc.world}",
            provenance=r.provenance,
            source=src,
            bootstrapped=False,
        )
        for oc in r.stages
        if oc.headline is not None
    ]


def _agreement_row(
    stat: AgreementStat, *, estimand: str, slice_: str, r: AgreementRecord
) -> Estimate:
    return Estimate(
        estimand=estimand,
        slice=slice_,
        point=stat.point,
        lo=stat.lower,
        hi=stat.upper,
        level=stat.confidence,
        sided="two-sided",
        method=METHOD_UNIT_BOOTSTRAP,
        n=stat.n_units,
        n_resamples=stat.n_resamples,
        seed=stat.seed,
        refusal_reason=None,
        provenance=r.provenance,
        source=f"agreement:{r.agreement_id}",
    )


def _agreement_rows(r: AgreementRecord) -> list[Estimate]:
    version = r.codebook_version if r.codebook_version is not None else "unversioned"
    scope = f"{r.axis}/{version}/{'+'.join(r.raters)}"
    rows = [_agreement_row(r.alpha, estimand="krippendorff_alpha", slice_=scope, r=r)]
    if r.cohen_kappa is not None:
        rows.append(_agreement_row(r.cohen_kappa, estimand="cohen_kappa", slice_=scope, r=r))
    rows.extend(
        _agreement_row(stat, estimand="specific_agreement", slice_=f"{scope}/{category}", r=r)
        for category, stat in sorted(r.per_class.items())
    )
    return rows


def _judge_check_rows(r: JudgeCheckRecord) -> list[Estimate]:
    src = f"judge-check:{r.check_id}"
    p = r.provenance
    rows = [
        _from_bound(r.tpr, estimand="check_tpr", slice_=r.judge_id, provenance=p, source=src),
        _from_bound(r.tnr, estimand="check_tnr", slice_=r.judge_id, provenance=p, source=src),
    ]
    for name, iv in (("delta_tpr", r.drift.delta_tpr), ("delta_tnr", r.drift.delta_tnr)):
        rows.append(
            _from_interval(
                iv, estimand=name, slice_=r.judge_id, provenance=p, source=src, bootstrapped=False
            )
        )
    return rows


def _sample_rows(r: SampleRecord) -> list[Estimate]:
    """One row for the sampling fraction: a known quantity, so its interval is a point."""
    stratum = ", ".join(f"{k}={v}" for k, v in sorted(r.cell.items())) or "whole population"
    fraction = r.sample_size / r.population_size if r.population_size else 0.0
    return [
        Estimate(
            estimand="sampling_fraction",
            slice=f"{r.source}/{stratum}",
            point=fraction,
            lo=fraction,
            hi=fraction,
            level=1.0,
            sided="two-sided",
            method="counted",
            n=r.population_size,
            n_resamples=None,
            seed=r.provenance.seed,
            refusal_reason=None,
            provenance=r.provenance,
            source=f"sample:{r.sample_id}",
        )
    ]


def _probe_rows(r: ProbeBatteryRecord) -> list[Estimate]:
    """One row per benchmark: the one-sided upper bound on its break rate, with no point.

    ``point`` stays empty because a black-box target has no calibration set, which is the same
    reason :meth:`calibrated_agent_evaluation.probe.UpperBoundRate.require_point` refuses one; the
    refusal reason says so rather than leaving the cell unexplained.
    """
    rows: list[Estimate] = []
    for finding in r.findings:
        payload = finding.payload
        if finding.kind == POLICY_FINDING_KIND:
            rows.extend(_policy_rows(r, finding.endpoint_fingerprint, payload))
            continue
        if "break_rate_ub95" not in payload:
            continue
        rows.append(
            Estimate(
                estimand="break_rate",
                slice=f"{finding.endpoint_fingerprint}/{payload.get('benchmark', finding.kind)}",
                point=None,
                lo=0.0,
                hi=float(payload["break_rate_ub95"]),
                level=0.95,
                sided="upper",
                method="clopper_pearson_one_sided_upper",
                n=int(payload.get("k", 0)),
                n_resamples=None,
                seed=None,
                refusal_reason=(
                    "no calibrated point rate exists for an endpoint you do not own; the "
                    "judge-biased upper bound is the honest signal"
                ),
                provenance=r.provenance,
                source=f"probe:{r.probe_id}",
            )
        )
    return rows


_POLICY_COUNTS = ("observed_allow", "gap", "denied", "over_deny")


def _policy_rows(
    r: ProbeBatteryRecord, fingerprint: str, payload: dict[str, Any]
) -> list[Estimate]:
    """Four rows per policy case, one per count, each the one-sided upper bound on that rate.

    The rate is over the case's scored trials (those with a Policy decision on a call). A case
    with no scored trial has no bound, so its rows run to 1 with the reason; the flip counts and
    the two-sided intervals stay in the probe record.
    """
    bounds = payload.get("bounds")
    scored = int(payload.get("scored", 0))
    unbounded = bounds is None or scored == 0
    reason = (
        "no trial of this case held a Policy decision on a tool call, so the rate is not bounded"
        if unbounded
        else "a probe battery reports each trial rate as its one-sided upper bound; the count and "
        "its two-sided interval are in the probe record"
    )
    return [
        Estimate(
            estimand=f"policy_{count}_rate",
            slice=f"{fingerprint}/{payload.get('case_id')}/{payload.get('mode')}",
            point=None,
            lo=0.0,
            hi=1.0 if bounds is None or unbounded else float(bounds[count]["ub95"]),
            level=0.95,
            sided="upper",
            method="clopper_pearson_one_sided_upper",
            n=0 if unbounded else scored,
            n_resamples=None,
            seed=None,
            refusal_reason=reason,
            provenance=r.provenance,
            source=f"probe:{r.probe_id}",
        )
        for count in _POLICY_COUNTS
    ]


def results_table(records: Iterable[Record]) -> list[Estimate]:
    """Flatten persisted records into one :class:`Estimate` per reported interval or bound.

    Raises :class:`ResultsTableError` on a record type it does not know, rather than silently
    leaving its estimates out.
    """
    rows: list[Estimate] = []
    for record in records:
        if isinstance(record, CalibrationRecord):
            rows.extend(_calibration_rows(record))
        elif isinstance(record, RunRecord):
            rows.extend(_run_rows(record))
        elif isinstance(record, ImproveRecord):
            rows.extend(_improve_rows(record))
        elif isinstance(record, BiasBatteryRecord):
            rows.extend(_bias_rows(record))
        elif isinstance(record, FlagshipReport):
            rows.extend(_flagship_rows(record))
        elif isinstance(record, PreferenceRecord):
            # One preference is an observation, not an estimate; ratings come from all of them.
            continue
        elif isinstance(record, JudgeRunRecord):
            # A judge run records which verdicts were made and by which judge; it reports no
            # estimate, so it adds no row.
            continue
        elif isinstance(record, AgreementRecord):
            rows.extend(_agreement_rows(record))
        elif isinstance(record, CodebookRecord | ProposedModeRecord):
            continue  # a vocabulary or a proposed mode, with no estimate to report
        elif isinstance(record, JudgeCheckRecord):
            rows.extend(_judge_check_rows(record))
        elif isinstance(record, SampleRecord):
            rows.extend(_sample_rows(record))
        elif isinstance(record, ProbeBatteryRecord):
            rows.extend(_probe_rows(record))
        elif isinstance(record, AbTestClaimRecord | RecommendationClaimRecord):
            # AgentCore's A/B and recommendation figures are a vendor claim, never a Calibrated
            # Agent Evaluation estimate, so they add no row.
            continue
        elif isinstance(record, ArmAssignment | RunProvenance | AuditRecord):
            # Bookkeeping: an arm, a replay's bundle, or what a re-test rested on. The re-test's
            # estimate is the ImproveRecord it names, which adds its own rows.
            continue
        else:
            raise ResultsTableError(
                f"results_table: no estimate extractor for {type(record).__name__!r}"
            )
    return rows


RESULTS_COLUMNS: tuple[str, ...] = (
    "estimand",
    "slice",
    "point",
    "lo",
    "hi",
    "level",
    "sided",
    "method",
    "n",
    "n_resamples",
    "seed",
    "refusal_reason",
    "source",
    "run_at",
    "calibrated_agent_evaluation_version",
    "git_sha",
    "python_version",
    "dep_versions",
    "thresholds",
)


def _cell(value: Any) -> str:
    # Empty for None so a refused point reads as missing, not as the string "None".
    return "" if value is None else str(value)


def _row(e: Estimate) -> dict[str, str]:
    p = e.provenance
    return {
        "estimand": e.estimand,
        "slice": e.slice,
        "point": _cell(e.point),
        "lo": _cell(e.lo),
        "hi": _cell(e.hi),
        "level": _cell(e.level),
        "sided": e.sided,
        "method": e.method,
        "n": _cell(e.n),
        "n_resamples": _cell(e.n_resamples),
        "seed": _cell(e.seed),
        "refusal_reason": _cell(e.refusal_reason),
        "source": e.source,
        "run_at": p.run_at,
        "calibrated_agent_evaluation_version": p.calibrated_agent_evaluation_version,
        "git_sha": _cell(p.git_sha),
        "python_version": p.python_version,
        "dep_versions": json.dumps(p.dep_versions, sort_keys=True),
        "thresholds": json.dumps(p.thresholds, sort_keys=True),
    }


def write_results_csv(rows: Sequence[Estimate] | Iterable[Record], path: str | Path) -> int:
    """Write the results table to ``path`` and return the number of rows written.

    Accepts either :class:`Estimate` rows or records (flattened with :func:`results_table`).
    Raises :class:`ResultsTableError` on a filesystem failure.
    """
    items = list(rows)
    n_estimates = sum(isinstance(r, Estimate) for r in items)
    if n_estimates == len(items):
        estimates = [r for r in items if isinstance(r, Estimate)]
    elif n_estimates == 0:
        estimates = results_table(r for r in items if not isinstance(r, Estimate))
    else:
        raise ResultsTableError("write_results_csv: pass Estimate rows or records, not a mix")
    out = Path(path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(RESULTS_COLUMNS))
        writer.writeheader()
        writer.writerows(_row(e) for e in estimates)
        atomic_write_text(out, buffer.getvalue())
    except OSError as exc:
        raise ResultsTableError(f"failed to write the results table to {out}: {exc}") from exc
    return len(estimates)
