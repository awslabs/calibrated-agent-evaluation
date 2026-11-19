# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agreement gate on a judge's gold: do the human raters behind it agree with each other?

``calibrated-agent-evaluation calibrate`` certifies a judge against human gold, and a gold set whose
raters disagree certifies against noise. Before any model call the command measures the raters'
agreement on the units the judge is graded on, with
:func:`calibrated_agent_evaluation.review.agreement.inter_rater_agreement`:

* gold from two or more raters who coded some units in common is measured, and calibration is
  refused unless the verdict is ``usable`` (alpha's lower bound clears the gate);
* gold from one rater, from raters who never coded a unit in common, or with no rater recorded
  cannot be measured; calibration goes ahead and says agreement was not measured.

The units are every target of the judge's level that carries human labels, including targets the
gold leaves out because their raters disagree: leaving those out would raise the number. Each
rater's newest label on a unit counts. Agreement is measured within one codebook version, and a
label with none is read as the version ``unversioned``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.cli.errors import CliError
from calibrated_agent_evaluation.schema.types import Label, Trace

if TYPE_CHECKING:
    from calibrated_agent_evaluation.calibrate import Interval
    from calibrated_agent_evaluation.cli.judgefile import JudgeFile
    from calibrated_agent_evaluation.review.agreement import InterRaterReport

__all__ = ["UNVERSIONED", "GoldAgreement", "GoldAgreementError", "measure_gold_agreement"]

UNVERSIONED = "unversioned"


class GoldAgreementError(CliError):
    """The gold's raters do not agree well enough to certify a judge against it.

    ``verdict`` is the agreement gate's verdict (``top_up_overlap``, ``revise_codebook`` or
    ``codebook_broken``).
    """

    def __init__(self, message: str, *, verdict: str) -> None:
        super().__init__(message)
        self.verdict = verdict


@dataclass(frozen=True)
class GoldAgreement:
    """What the agreement check found: a report, or why agreement was not measured."""

    report: InterRaterReport | None
    raters: tuple[str, ...]
    not_measured: str | None = None
    n_unattributed: int = 0

    def alpha_interval(self) -> Interval | None:
        """Alpha as a calibration ``Interval``, for the record; ``None`` when not measured."""
        if self.report is None:
            return None
        from calibrated_agent_evaluation.calibrate import Interval
        from calibrated_agent_evaluation.calibrate.record import METHOD_ALPHA_BOOTSTRAP

        alpha = self.report.alpha
        return Interval(
            point=alpha.point,
            lo=alpha.lower,
            hi=alpha.upper,
            method=METHOD_ALPHA_BOOTSTRAP,
            level=alpha.confidence,
            sided="two-sided",
            n=self.report.n_pairable_units,
        )

    def detail(self) -> list[str]:
        """The ``--explain`` lines."""
        if self.report is None:
            return [f"  rater agreement on the gold: not measured ({self.not_measured})"]
        r = self.report
        lines = [
            f"  rater agreement on the gold ({', '.join(r.raters)}; codebook {r.codebook_version}; "
            f"{r.n_pairable_units} of {r.n_units} unit(s) coded by two or more): "
            f"Krippendorff's alpha {_iv(r.alpha)}; verdict {r.gate}"
        ]
        if r.cohen_kappa is not None:
            lines.append(f"    Cohen's kappa {_iv(r.cohen_kappa)}")
        lines += [f"    specific agreement on {c}: {_iv(iv)}" for c, iv in r.per_class.items()]
        if self.n_unattributed:
            lines.append(
                f"    {self.n_unattributed} label(s) with no rater recorded were left out of it"
            )
        return lines


def _iv(iv: Any) -> str:
    return f"{iv.point:.3f} (95% interval {iv.lower:.3f} to {iv.upper:.3f})"


def measure_gold_agreement(
    jf: JudgeFile, traces: list[Trace], labels_by_trace: dict[str, list[Label]]
) -> GoldAgreement:
    """Measure the raters' agreement on ``jf``'s units; refuse gold that is not usable.

    Raises:
        GoldAgreementError: Two or more raters overlapped and the verdict is not ``usable``.
        CliError: The labels span more than one codebook version.
    """
    from calibrated_agent_evaluation.review.agreement import inter_rater_agreement
    from calibrated_agent_evaluation.stats.agreement import UndefinedAgreementError

    newest: dict[tuple[str, str, str], Label] = {}
    unattributed = 0
    for label in _unit_labels(jf, traces, labels_by_trace):
        if label.rater_id is None:
            unattributed += 1
            continue
        newest[(label.trace_id, label.span_id or "", label.rater_id)] = label
    labels = [
        lb.model_copy(update={"codebook_version": lb.codebook_version or UNVERSIONED})
        for lb in newest.values()
    ]
    raters = tuple(sorted({lb.rater_id for lb in labels if lb.rater_id is not None}))
    if len(raters) < 2:
        why = (
            "the gold has one rater"
            if raters
            else "the gold's labels record no rater, so who labeled what is unknown"
        )
        return GoldAgreement(None, raters, why, unattributed)
    versions = sorted({lb.codebook_version or UNVERSIONED for lb in labels})
    if len(versions) > 1:
        raise CliError(
            f"the gold for judge {jf.judge.judge_id!r} was labeled under codebook versions "
            f"{', '.join(versions)}; rater agreement is measured within one version, so relabel "
            "the older labels under the current codebook (or narrow the scope with --cell) first"
        )
    try:
        report = inter_rater_agreement(labels, raters=raters, codebook_version=versions[0])
    except UndefinedAgreementError as exc:
        return GoldAgreement(None, raters, f"{exc}", unattributed)
    if report.n_pairable_units == 0:
        return GoldAgreement(None, raters, "the raters never labeled the same target", unattributed)
    if report.gate != "usable":
        raise GoldAgreementError(_refusal(jf, report), verdict=report.gate)
    return GoldAgreement(report, raters, None, unattributed)


def _refusal(jf: JudgeFile, report: InterRaterReport) -> str:
    from calibrated_agent_evaluation.review.agreement import ALPHA_LOWER_BOUND_GATE
    from calibrated_agent_evaluation.review.gate import MINIMUM_OVERLAP

    short = report.n_pairable_units < MINIMUM_OVERLAP
    conclusive = (
        f" (alpha could be at most {report.alpha.upper:.3f}, so double-coding more of the gold "
        "would not change this)"
        if short and report.gate != "top_up_overlap"
        else ""
    )
    next_step = {
        "top_up_overlap": (
            f"have a second rater label more of this gold until at least {MINIMUM_OVERLAP} of "
            f"its traces are coded by both ({MINIMUM_OVERLAP - report.n_pairable_units} more), "
            "then calibrate again; below that the gate cannot trust alpha"
            if short
            else "have a second rater label more of this gold, then calibrate again (the "
            "interval is too wide to tell)"
        ),
        "revise_codebook": "revise the codebook where the raters split, relabel those traces "
        "under the new version, then calibrate again",
        "codebook_broken": "rewrite the failure-mode definitions with the raters before any more "
        "labeling; at this agreement the gold measures the codebook, not the judge",
    }[report.gate]
    return (
        f"the raters behind judge {jf.judge.judge_id!r}'s gold do not agree well enough to "
        f"certify against it: Krippendorff's alpha {_iv(report.alpha)} over "
        f"{report.n_pairable_units} unit(s) coded by two or more of {', '.join(report.raters)}, "
        f"and the gold is usable only when the interval's lower end reaches "
        f"{ALPHA_LOWER_BOUND_GATE:.2f} (verdict {report.gate}{conclusive}). Next: {next_step}. "
        "No judge "
        "was run and nothing was recorded"
    )


def _unit_labels(
    jf: JudgeFile, traces: list[Trace], labels_by_trace: dict[str, list[Label]]
) -> list[Label]:
    """The human labels at the judge's level: trace labels, or span labels on its agent."""
    out: list[Label] = []
    for trace in traces:
        human = [lb for lb in labels_by_trace.get(trace.trace_id, []) if lb.kind == "human"]
        if jf.judge.level == "workflow":
            out += [lb for lb in human if lb.span_id is None]
        else:
            mine = {s.span_id for s in trace.spans if s.agent == jf.agent}
            out += [lb for lb in human if lb.span_id in mine]
    return out
