# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``run --drift-baseline STORE --drift-feature F``: input drift against a baseline store.

Both sides are production-representative traces: the baseline store's, and the ones in scope that
the gate reads. Comparing a hunted set with a random sample would flag drift by construction.
``--drift-feature`` is compared with PSI and an IQR-normalized Wasserstein distance (the two that
can flag); how the traces fall across taxonomy cells is compared with Jensen-Shannon divergence on
the cell counts, always reported and never flagging
(``calibrated_agent_evaluation.run.input_drift``). Drift is passive: it never changes the gate. With
``--fail-on-drift`` a flag gives the run its own exit code.

Too few traces on a side for the bins is a named note, never a "no drift".
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from calibrated_agent_evaluation.cli.errors import CliError, UsageError

if TYPE_CHECKING:
    from calibrated_agent_evaluation.run import InputDriftReport, RunDecision
    from calibrated_agent_evaluation.schema.types import Trace

_PRODUCTION = "production-representative"


class DriftOptions:
    """The validated drift flags of one ``calibrated-agent-evaluation run``."""

    def __init__(self, baseline: Path | None, feature: str | None, fail_on_drift: bool) -> None:
        self.baseline = baseline
        self.feature = feature
        self.fail_on_drift = fail_on_drift


def drift_options(args: argparse.Namespace) -> DriftOptions:
    """Read and check ``--drift-baseline`` / ``--drift-feature`` / ``--fail-on-drift``.

    The feature has no default, so the two come together; ``--fail-on-drift`` without a baseline
    could never flag and is refused rather than accepted as a no-op.
    """
    raw = getattr(args, "drift_baseline", None)
    feature = getattr(args, "drift_feature", None)
    fail = bool(getattr(args, "fail_on_drift", False))
    if raw is None:
        if feature is not None:
            raise UsageError(
                "--drift-feature needs --drift-baseline STORE, the local store to compare against"
            )
        if fail:
            raise UsageError(
                "--fail-on-drift needs --drift-baseline STORE and --drift-feature; without a "
                "baseline there is no drift to fail on"
            )
        return DriftOptions(None, None, fail_on_drift=False)
    if feature is None:
        raise UsageError(
            "--drift-baseline needs --drift-feature input-chars|turns|tool-calls: the input "
            "feature to compare (there is no default, since which one matters depends on the agent)"
        )
    path = Path(str(raw))
    if not path.is_dir():
        raise CliError(f"--drift-baseline {path} is not a directory (a local store root)")
    return DriftOptions(path, str(feature), fail_on_drift=fail)


class DriftResult:
    """The drift report (``None`` when it could not be measured) and how to say what it found."""

    def __init__(
        self, report: InputDriftReport | None, detail: list[str], plain: list[str]
    ) -> None:
        self.report = report
        self.detail = detail
        self.plain = plain

    @property
    def flagged(self) -> bool:
        """Whether a drift metric was flagged."""
        return self.report is not None and self.report.any_exceeded


def measure_drift(
    baseline_root: Path, feature: str, fresh: list[Trace], *, seed: int
) -> DriftResult:
    """Compare ``feature`` and the cell mix between the store at ``baseline_root`` and ``fresh``."""
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.calibrate.errors import StatisticsInputError
    from calibrated_agent_evaluation.run import input_drift
    from calibrated_agent_evaluation.run.drift_report import DEFAULT_DRIFT_BINS
    from calibrated_agent_evaluation.run.features import cell_counts, feature_values
    from calibrated_agent_evaluation.stats.drift import MIN_BIN_COUNT

    with LocalBackend(root=baseline_root) as backend:
        stored = backend.query()
    baseline = [t for t in stored if t.frame == _PRODUCTION]
    if not baseline:
        what = f"{len(stored)} trace(s), none of them" if stored else "no traces, so none"
        raise CliError(
            f"the drift baseline {baseline_root} holds {what} production-representative; "
            "drift compares two random samples, so the baseline needs the random sample of an "
            "earlier run (for example a `calibrated-agent-evaluation dump` of it)"
        )
    fresh = [t for t in fresh if t.frame == _PRODUCTION]
    base_values, base_missing = feature_values(baseline, feature)
    new_values, new_missing = feature_values(fresh, feature)
    cells, base_cells, new_cells = cell_counts(baseline, fresh)
    detail: list[str] = []
    plain: list[str] = []
    if base_missing or new_missing:
        detail.append(
            f"  drift: {base_missing} baseline and {new_missing} current trace(s) have no "
            f"{feature} (probed traces record no agent steps) and are left out of that comparison"
        )
    try:
        report = input_drift(
            base_values,
            new_values,
            dimension=feature,
            jsd_baseline=base_cells,
            jsd_fresh=new_cells,
            seed=seed,
        )
    except StatisticsInputError as exc:
        detail.insert(
            0,
            f"input drift on {feature}: not measured ({exc})",
        )
        plain.append(
            f"Input drift on {feature}: not measured, too few traces to compare "
            f"({len(base_values)} in the baseline, {len(new_values)} now; each side needs at "
            f"least {MIN_BIN_COUNT * DEFAULT_DRIFT_BINS}). Add traces to the smaller side and run "
            "again."
        )
        return DriftResult(None, detail, plain)
    detail[:0] = _detail(report, baseline_root, len(cells))
    plain += _plain(report, feature)
    return DriftResult(report, detail, plain)


def _detail(report: InputDriftReport, baseline: Path, n_cells: int) -> list[str]:
    lines = [
        f"input drift on {report.dimension}: {report.n_baseline} baseline trace(s) from "
        f"{baseline} vs {report.n_fresh} in scope, {report.n_permutations} permutations, seed "
        f"{report.seed}"
    ]
    for m in report.metrics:
        if m.name == "jsd":
            lines.append(
                f"  taxonomy-cell JSD {m.value:.4f} over {n_cells} cell(s), permutation p = "
                f"{m.p_value:.4f} ({m.method}; reported only, never flags)"
            )
            continue
        bar = "none" if m.threshold is None else f"{m.threshold:.2f}"
        lines.append(
            f"  {m.name} {m.value:.4f}, permutation p = {m.p_value:.4f}, bar {bar}: "
            + ("FLAGGED" if m.exceeded else "not flagged")
            + f" ({m.method})"
        )
    if report.multiplicity is not None:
        lines.append(
            f"  flags corrected together with {report.multiplicity.method} across "
            f"m = {report.multiplicity.m} metrics"
        )
    return lines


def _plain(report: InputDriftReport, feature: str) -> list[str]:
    jsd = next(m for m in report.metrics if m.name == "jsd")
    if report.any_exceeded:
        head = (
            f"Input drift on {feature}: the current traces differ from the baseline by more than "
            "chance and by enough to matter. A judge's measured accuracy may not hold on them; "
            "check it against fresh human labels before trusting its rate."
        )
    else:
        head = (
            f"Input drift on {feature}: none found ({report.n_fresh} current traces against "
            f"{report.n_baseline} in the baseline)."
        )
    return [
        head,
        f"  (The mix of taxonomy cells moved by {jsd.value:.3f} on a 0 to 1 scale; shown for "
        "reference, it never raises a flag.)",
    ]


def staleness_lines(decisions: list[RunDecision]) -> tuple[list[str], list[str]]:
    """The detail and plain lines for the staleness warnings of the admitted judges."""
    detail = [f"  staleness: {d.message}" for d in decisions]
    plain = [
        f"Judge {d.inputs['judge_id']!r} may need recalibrating: "
        + (
            "its inputs have drifted since the baseline"
            if d.inputs.get("drift_exceeded") == "True"
            else "it no longer matches its calibration record"
        )
        + ". Recalibrate it on current traces (`calibrated-agent-evaluation calibrate`)."
        for d in decisions
    ]
    return detail, plain
