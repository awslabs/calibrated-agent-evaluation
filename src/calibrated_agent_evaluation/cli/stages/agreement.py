# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``label --agreement``: the gold-set agreement gate over double-coded human labels.

Reads the human labels the named raters wrote on the traces in scope (``--trace`` / ``--frame`` /
``--cell``), within one codebook version (``--codebook-version``; without it the unversioned
labels), and runs :func:`calibrated_agent_evaluation.review.inter_rater_agreement` on ``--axis``. A
rater who relabeled a unit counts once, with the newest label. The result is stored as an
``agreement`` record under ``<root>/records/agreement/`` whatever the verdict; the command exits 0
only on ``usable``, so a script can stop before calibrating on gold the raters do not agree on.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError

if TYPE_CHECKING:
    from calibrated_agent_evaluation.review.agreement import AgreementAxis

_NEXT_STEP = {
    "top_up_overlap": "the raters agree, but the interval is still too wide to be sure: have "
    "the second rater label more of the calibration gold and run this again",
    "short_overlap": "too few shared traces for the gate to trust alpha: have the second rater "
    "label {more} more traces of the calibration gold (no traces beyond the gold are needed) "
    "and run this again",
    "revise_codebook": "the raters disagree often enough that the definitions are ambiguous: "
    "tighten the codebook where they split (the per-class lines under --explain), bump "
    "--codebook-version, and relabel the shared traces",
    "codebook_broken": "the raters agree little beyond chance: rewrite the codebook with "
    "examples of each class before labeling more, then relabel under a new --codebook-version",
}


def parse_raters(raw: str | None) -> tuple[str, ...]:
    """``"a, b,c"`` -> ``("a", "b", "c")``; refuses fewer than two distinct ids."""
    raters = tuple(dict.fromkeys(r.strip() for r in (raw or "").split(",") if r.strip()))
    if len(raters) < 2:
        raise UsageError(
            "`calibrated-agent-evaluation label --agreement` needs --raters with at least two "
            "rater ids, e.g. "
            "--raters alice,bob (the ids they labeled under with `label --serve --rater`)"
        )
    return raters


def _in_scope(traces: list[Any], args: argparse.Namespace) -> list[Any]:
    from calibrated_agent_evaluation.cli.scope import parse_cells

    frame = getattr(args, "frame", None)
    cell = parse_cells(getattr(args, "cell", None)) or {}
    return [
        t
        for t in traces
        if (frame is None or t.frame == frame)
        and all(t.taxonomy_cell.get(k) == v for k, v in cell.items())
    ]


def agreement_handler(args: argparse.Namespace) -> StageOutcome:
    """Measure the named raters' agreement, store the record, exit 0 only when ``usable``."""
    if getattr(args, "serve", False):
        raise UsageError("--agreement and --serve are separate runs: pick one")
    raters = parse_raters(getattr(args, "raters", None))
    version: str | None = getattr(args, "codebook_version", None)
    axis: AgreementAxis = (
        "failure_mode" if getattr(args, "axis", None) == "failure_mode" else "outcome"
    )

    from calibrated_agent_evaluation.cli.backends import local_root, open_backend
    from calibrated_agent_evaluation.cli.records import append_records
    from calibrated_agent_evaluation.cli.scope import parse_cells, read_scope
    from calibrated_agent_evaluation.review import agreement_record_of, inter_rater_agreement

    with open_backend(args) as backend:
        traces = _in_scope(read_scope(backend, args).traces, args)
        if not traces:
            raise CliError("no stored trace matches --trace/--frame/--cell; nothing to compare")
        # Oldest first within each trace, the order the newest-label-per-rater rule reads.
        labels = [lb for t in traces for lb in backend.get_scores(t.trace_id)]
    report = inter_rater_agreement(labels, raters=raters, codebook_version=version, axis=axis)
    named = set(raters)
    trace_ids = {
        lb.trace_id for lb in labels if lb.rater_id in named and lb.codebook_version == version
    }
    record = agreement_record_of(
        report,
        trace_ids=trace_ids,
        cell=parse_cells(getattr(args, "cell", None)),
        trace_id=getattr(args, "trace_id", None),
        frame=getattr(args, "frame", None),
    )
    root = local_root(args)
    append_records(root, "agreement", [record])
    path = root / "records" / "agreement"
    return StageOutcome(
        stage="label",
        exit_code=EXIT_OK if report.gate == "usable" else EXIT_DOMAIN_ERROR,
        summary="\n".join(_summary(report)),
        detail="\n".join(_detail(report, record.agreement_id, str(path))),
    )


def _version(version: str | None) -> str:
    return "unversioned labels" if version is None else f"codebook version {version!r}"


def _summary(report: Any) -> list[str]:
    what = "PASS/FAIL" if report.axis == "outcome" else "failure mode"
    head = (
        f"{' and '.join(report.raters)} agree on {what} with alpha {report.alpha.point:.2f} "
        f"(could be as low as {report.alpha.lower:.2f}) over {report.n_pairable_units} shared "
        f"unit(s), {_version(report.codebook_version)}"
    )
    if report.gate == "usable":
        lines = [head + ": the gold set is usable for calibration."]
    else:
        below_minimum = report.n_pairable_units < report.minimum_overlap
        short = below_minimum and report.gate == "top_up_overlap"
        lines = [head + f": not usable yet ({report.gate.replace('_', ' ')})."]
        step = _NEXT_STEP["short_overlap" if short else report.gate]
        lines.append(
            "Next: " + step.format(more=report.minimum_overlap - report.n_pairable_units) + "."
        )
        if short:
            lines.append(
                f"The gate needs at least {report.minimum_overlap} traces labeled by two raters "
                "before it can say usable (--explain says how that was worked out)."
            )
        elif below_minimum:
            lines.append(
                f"This holds even though fewer than {report.minimum_overlap} traces are shared: "
                f"alpha could be at most {report.alpha.upper:.2f}, so more shared traces would "
                "not change it."
            )
    if report.n_superseded:
        lines.append(f"{report.n_superseded} older label(s) were replaced by a rater's relabel.")
    return lines


def _stat(name: str, s: Any) -> str:
    return (
        f"{name} {s.point:.3f}, {s.confidence:.0%} bootstrap interval [{s.lower:.3f}, "
        f"{s.upper:.3f}] ({s.n_resamples} resamples over {s.n_units} unit(s), seed {s.seed}"
        + (f", {s.n_undefined} undefined" if s.n_undefined else "")
        + ")"
    )


def _detail(report: Any, record_id: str, path: str) -> list[str]:
    from calibrated_agent_evaluation.review import (
        ALPHA_LOWER_BOUND_GATE,
        ALPHA_REVISE_FLOOR,
        MINIMUM_OVERLAP,
    )
    from calibrated_agent_evaluation.review.overlap import (
        DEFAULT_EXPECTED_ALPHA,
        DEFAULT_PREVALENCE,
        GATE_RESAMPLES,
        SCAN_LIMIT,
    )

    lines = [
        f"raters {', '.join(report.raters)}; axis {report.axis}; "
        f"{_version(report.codebook_version)}; {report.n_units} unit(s) coded, "
        f"{report.n_pairable_units} coded by two or more",
        _stat("Krippendorff's alpha", report.alpha),
    ]
    if report.cohen_kappa is not None:
        lines.append(_stat("Cohen's kappa", report.cohen_kappa))
    lines += ["  " + _stat(f"specific agreement on {c!r}", s) for c, s in report.per_class.items()]
    lines.append(
        f"gate: {report.gate} (under {report.minimum_overlap} units coded by two or more it is "
        f"codebook_broken when alpha's upper bound < {ALPHA_REVISE_FLOOR}, revise_codebook when "
        f"it is < {ALPHA_LOWER_BOUND_GATE}, top_up_overlap otherwise; from there usable when "
        f"alpha's lower bound >= "
        f"{ALPHA_LOWER_BOUND_GATE}; a point below {ALPHA_REVISE_FLOOR} means the codebook is "
        f"broken, between that and {ALPHA_LOWER_BOUND_GATE} that it needs revising, at or above "
        f"{ALPHA_LOWER_BOUND_GATE} with a lower bound below it that the overlap is too small)"
    )
    if report.minimum_overlap == MINIMUM_OVERLAP:
        lines.append(
            f"minimum overlap: {MINIMUM_OVERLAP} units, the smallest size from which two raters "
            f"agreeing at the expected alpha {DEFAULT_EXPECTED_ALPHA} on gold that is "
            f"{DEFAULT_PREVALENCE:.0%} FAIL (the calibration gold, balanced per class) keep "
            f"alpha's lower bound at or above {ALPHA_LOWER_BOUND_GATE} at that size and every "
            f"larger one checked up to {SCAN_LIMIT} (expected counts rounded against the raters, "
            f"exact bootstrap, a margin of two standard errors for the gate's {GATE_RESAMPLES} "
            "resamples); double-code the calibration gold itself, no extra traces"
        )
    lines.append(f"recorded as {record_id} in {path}")
    return lines
