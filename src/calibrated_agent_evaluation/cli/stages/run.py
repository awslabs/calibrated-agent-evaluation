# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``run``: the CI gate, the per-mode corrected-rate trend, and the persisted run record.

Reuses the ``calibrated_agent_evaluation.run`` surface:

  * the **gate**: the stored verdicts that carry a failure mode are gated by ``gate_mode_counts``
    at ``--threshold``, which passes only if every mode's pass-rate 95% lower bound clears it.
    ``--verdicts`` names the source: ``judge`` (the default: the LLM-judge verdicts, the
    production measurement), ``scorer`` (the code-scorer verdicts) or ``human`` (the human
    labels). A source with no verdicts in scope is refused with a typed error naming it; another
    source is never gated in its place. The source is recorded on the run record's inputs as
    ``verdict_source``. ``--ci`` turns a failing gate into exit 1 so a regression blocks CI.

    The gate reads the production-representative traces only: its bound assumes a random sample,
    and a hunted discovery trace is not one. Each verdict counts under the mode it carries (and the
    agent of the span it is about), a judge re-run counts once (``latest_verdicts``), and a span
    defect is never gated. A human verdict is per trace and rater, read with the rule
    ``calibrated-agent-evaluation improve`` pairs on (``cli.pairing.verdicts_for``): a FAIL for a
    mode fails it, a PASS with no mode passes every gated mode whose agent ran in the trace, and a
    FAIL for another mode alone says nothing about this one.
  * the **trend**: for each ``--judge FILE``, that judge's own verdicts (matched on the
    ``rater_id`` ``calibrated-agent-evaluation judge`` stamps) on production-representative traces
    are corrected by ``trend_report`` with the newest calibration record under
    ``<root>/records/calibration/`` that certifies the judge, seeded by ``--seed`` and
    ``--n-resamples``. The judge must be admitted as ``calibrated-agent-evaluation judge`` admits
    it: certified, and its bias battery CLEAR. A mode whose correction is refused keeps its interval
    and its reason. With no ``--judge``, an unadmitted judge, or no verdicts of its own, there is no
    trend, and the default output says why. With ``--judge`` given, ``--verdicts judge`` gates those
    judges' verdicts only.
  * the **record**: ``run_record_of`` builds the ``RunRecord``, stamped with the seed, resamples
    and threshold, the scorer lanes gated and the pre-registration ledger head, and it is appended
    to ``<root>/records/run/``.
  * the **checks around the gate**: ``--atoms FILE`` refuses a gated mode that is not an objective
    atom; under ``--ci`` a code verdict from a generic generation-quality scorer is refused;
    ``--drift-baseline`` / ``--drift-feature`` add input drift and a staleness warning per
    admitted judge (``calibrated_agent_evaluation.cli.drift``), passive unless ``--fail-on-drift``
    (exit 3).
  * the **publish**: ``--publish local|langfuse`` sends the recorded run's numeric series to a
    dashboard sink; a failure exits 1 and names the run that was recorded.

Imports only ``calibrated_agent_evaluation.*``, never ``scripts/``, so it works for an installed
user.
"""

from __future__ import annotations

import argparse
import contextlib
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.cli.dispatch import (
    EXIT_DOMAIN_ERROR,
    EXIT_DRIFT,
    EXIT_OK,
    StageOutcome,
)
from calibrated_agent_evaluation.cli.errors import CliError, UsageError
from calibrated_agent_evaluation.schema.serde import latest_verdicts
from calibrated_agent_evaluation.schema.types import Label, Trace

if TYPE_CHECKING:
    from calibrated_agent_evaluation.attribution.labels import SpanAgents
    from calibrated_agent_evaluation.calibrate import CalibrationRecord, Interval
    from calibrated_agent_evaluation.calibrate.engine import Judge
    from calibrated_agent_evaluation.cli.judgefile import JudgeFile
    from calibrated_agent_evaluation.run import (
        ModeCounts,
        ModeKey,
        RunDecision,
        RunGateDecision,
        TrendPoint,
    )
    from calibrated_agent_evaluation.subjective import AtomSet

_PRODUCTION = "production-representative"

# What `--verdicts` selects: the label kinds gated, how the output names them, and how to get
# some. A code scorer is its own source: `judge` gates LLM-judge verdicts only.
_SOURCES: dict[str, tuple[tuple[str, ...], str, str]] = {
    "judge": (
        ("llm-judge",),
        "judge verdicts",
        "score the traces with `calibrated-agent-evaluation judge` first",
    ),
    "scorer": (
        ("code",),
        "code-scorer verdicts",
        "score the traces with `calibrated-agent-evaluation score` first (or "
        "`calibrated-agent-evaluation load` a dump with code verdicts)",
    ),
    "human": (
        ("human",),
        "human verdicts",
        "label them with `calibrated-agent-evaluation label` first",
    ),
}


def run_handler(args: argparse.Namespace) -> StageOutcome:
    """Gate the stored verdicts, compute the trend, persist the run record."""
    from calibrated_agent_evaluation.cli.stages.calibrate_agentcore import (
        refuse_flags_without_agentcore,
    )

    refuse_flags_without_agentcore(args, getattr(args, "judges", None) or [])
    # Holds the AWS clients an AgentCore judge file opens to read its evaluator, closed however
    # the run ends.
    with contextlib.ExitStack() as held:
        return _run(args, held)


def _run(args: argparse.Namespace, held: contextlib.ExitStack) -> StageOutcome:
    from calibrated_agent_evaluation.cli.backends import local_root, open_backend
    from calibrated_agent_evaluation.cli.config import require_extras

    # The run gate lives in the [stats] extra (scipy/statsmodels), NOT the default tree;
    # probe first so a bare install gets a typed hint, not a raw ModuleNotFoundError.
    require_extras("scipy", "statsmodels")
    from calibrated_agent_evaluation.cli.scope import read_scope, scope_note, skip_lines
    from calibrated_agent_evaluation.run import RunReport, gate_mode_counts

    threshold = float(getattr(args, "threshold", 0.80))
    if not 0.0 < threshold < 1.0:
        raise UsageError(f"--threshold must be in (0, 1), got {threshold}")
    seed = int(getattr(args, "seed", 0))
    n_resamples = _n_resamples(args)
    root = local_root(args)
    from calibrated_agent_evaluation.cli.drift import drift_options

    drift_opts = drift_options(args)
    publish = getattr(args, "publish", None)
    # A missing key or extra is known before the run; only a failure at publish time comes after
    # the record.
    credentials = _publish_credentials(publish)
    atoms = _atoms(args)
    # After the local checks, so a bad flag is refused before an AgentCore judge file's evaluator
    # is read.
    judge_files = _judge_files(args, held)
    ci_mode = bool(getattr(args, "ci", False))

    with open_backend(args) as backend:
        scan = read_scope(backend, args)
        traces = scan.traces
        labels_by_trace = {
            t.trace_id: latest_verdicts(backend.get_scores(t.trace_id)) for t in traces
        }

    all_labels = [lb for labels in labels_by_trace.values() for lb in labels]
    verdict_source = str(getattr(args, "verdicts", None) or "judge")
    if verdict_source == "judge" and judge_files:
        mine = _from_judges(all_labels, judge_files)
        if any(lb.kind == "llm-judge" for lb in all_labels) and not any(
            lb.kind == "llm-judge" for lb in mine
        ):
            names = ", ".join(repr(jf.judge.judge_id) for jf in judge_files)
            raise CliError(
                f"none of the judge verdicts on {scope_note(args, len(traces))} were made by the "
                f"--judge file(s) as they are now ({names}); the stored verdicts came from another "
                "judge or another version. Score the traces with this judge "
                "(`calibrated-agent-evaluation judge`) first"
            )
        all_labels = mine
    from calibrated_agent_evaluation.attribution.labels import span_agents

    agents = span_agents(traces)
    gate_input = _gate_input(
        all_labels, traces, verdict_source, scope_note(args, len(traces)), agents
    )
    source = gate_input.name
    if atoms is not None:
        _refuse_subjective(gate_input.counts, atoms)
    lanes = _scorer_lanes(all_labels, traces) if verdict_source == "scorer" else ()
    lane_decisions = _lint_lanes(lanes, ci=ci_mode)
    # gate_mode_counts refuses an empty input rather than passing vacuously (exit 1).
    gate = gate_mode_counts(gate_input.counts, threshold=threshold)

    trend, trend_notes, admitted = _trend(
        judge_files, traces, labels_by_trace, root, agents, seed=seed, n_resamples=n_resamples
    )
    drift = None
    if drift_opts.baseline is not None:
        from calibrated_agent_evaluation.cli.drift import measure_drift

        drift = measure_drift(drift_opts.baseline, drift_opts.feature or "", traces, seed=seed)
    stale = _staleness(admitted, drift.report if drift is not None else None)
    report = RunReport(
        run_id=_run_id(),
        mode="ci" if ci_mode else "cadence",
        trend=tuple(trend),
        input_drift=drift.report if drift is not None else None,
        decisions=(*lane_decisions, *stale),
        gate=gate,
    )
    record_id = _persist(
        report,
        root,
        args=args,
        source=verdict_source,
        n_gated=gate_input.n,
        judge_ids=tuple(jf.judge.judge_id for jf in judge_files),
        scorer_lanes=lanes,
        seed=seed,
        n_resamples=n_resamples,
    )
    publish_error = _publish(publish, report, root, credentials)

    lines = [
        _gate_line(gate, gate_input.n, source, threshold, scope_note(args, len(traces))),
        *_mode_lines(gate),
        *gate_input.notes,
        *skip_lines(scan),
        *_trend_lines(trend, trend_notes.detail, root),
        *_lane_lines(lanes, lane_decisions),
    ]
    from calibrated_agent_evaluation.cli.drift import staleness_lines

    stale_detail, stale_plain = staleness_lines(list(stale))
    if drift is not None:
        lines += drift.detail
    lines += stale_detail
    lines.append(f"recorded run {record_id} in {root / 'records' / 'run'}")
    if publish is not None and publish_error is None:
        from calibrated_agent_evaluation.run.report import numeric_series

        n_points = len(numeric_series(report))
        lines.append(
            f"published {n_points} rate and drift point(s) to "
            f"{_published_to(publish, root, record_id)}"
            + ("" if n_points else " (none to chart: pass --judge or --drift-baseline)")
        )
    failed = not gate.passed
    drift_flagged = (drift is not None and drift.flagged) or bool(stale)
    from calibrated_agent_evaluation.cli import plain

    summary = plain.run(
        gate, trend, _mode_name, threshold=threshold, ci=ci_mode, notes=trend_notes.plain
    )
    summary += gate_input.plain
    summary += plain.skipped(scan)
    summary += [d.message for d in lane_decisions]
    if drift is not None:
        summary += drift.plain
    summary += stale_plain
    if trend:
        lines.append(f"  {plain.KNOWN_LIMIT}")
    # --ci makes a failing gate exit 1 so it blocks CI; without it a failing gate is reported with
    # exit 0. A drift flag gets its own code only when asked for, and never masks a gate failure.
    if publish is not None and publish_error is None:
        summary.append(f"Published the run to {_published_to(publish, root, record_id)}.")
    if publish_error is not None:
        summary.append(f"Run recorded as {record_id}; publish failed: {publish_error}")
        exit_code = EXIT_DOMAIN_ERROR
    elif failed and ci_mode:
        exit_code = EXIT_DOMAIN_ERROR
    elif drift_flagged and drift_opts.fail_on_drift:
        summary.append(
            "Exit code 3: drift or a stale judge was flagged and --fail-on-drift is set."
        )
        exit_code = EXIT_DRIFT
    else:
        exit_code = EXIT_OK
    return StageOutcome(
        stage="run",
        exit_code=exit_code,
        summary="\n".join(summary),
        detail="\n".join(lines),
    )


def _publish_credentials(publish: str | None) -> tuple[str, str, str] | None:
    if publish != "langfuse":
        return None
    from calibrated_agent_evaluation.cli.backends import langfuse_credentials

    return langfuse_credentials()


def _published_to(publish: str, root: Path, run_id: str) -> str:
    where = root / "dashboard" / run_id
    return f"{where}" if publish == "local" else f"{where} and Langfuse"


def _publish(
    publish: str | None,
    report: Any,
    root: Path,
    credentials: tuple[str, str, str] | None,
) -> str | None:
    """Publish the recorded run to its dashboard; the failure message, or ``None`` on success.

    Runs after the record is appended, so a failed publish never loses the run. The local series
    is written in both cases: it is the authoritative copy, and Langfuse gets it as scores too.
    """
    if publish is None:
        return None
    from calibrated_agent_evaluation.backends.dashboard import (
        LangfuseDashboardSink,
        LocalDashboardSink,
    )
    from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError

    local = LocalDashboardSink(root / "dashboard" / report.run_id)
    try:
        if publish == "local":
            local.publish(report)
        else:
            assert credentials is not None  # noqa: S101
            host, public_key, secret_key = credentials
            LangfuseDashboardSink.from_credentials(
                public_key, secret_key, host, local=local
            ).publish(report)
    except CalibratedAgentEvaluationDataError as exc:
        return str(exc)
    return None


def _atoms(args: argparse.Namespace) -> AtomSet | None:
    path = getattr(args, "atoms_file", None)
    if not path:
        return None
    from calibrated_agent_evaluation.cli.atoms import load_atom_set

    return load_atom_set(path)


def _refuse_subjective(counts: dict[ModeKey, Any], atoms: AtomSet) -> None:
    """Refuse a gate over a mode the AtomSet declares subjective, or does not declare at all."""
    from calibrated_agent_evaluation.subjective import validate_gateable

    for failure_type in sorted({m.failure_type for m in counts}):
        validate_gateable(failure_type, atoms)


_UNRECORDED_LANE = "unrecorded"


def _scorer_lanes(labels: list[Label], traces: list[Trace]) -> tuple[str, ...]:
    """The scorers behind the code verdicts the gate reads, as each verdict's ``rater_id``."""
    production = {t.trace_id for t in traces if t.frame == _PRODUCTION}
    lanes = {
        lb.rater_id or _UNRECORDED_LANE
        for lb in labels
        if lb.kind == "code"
        and lb.failure_mode is not None
        and not lb.span_defect
        and lb.trace_id in production
    }
    return tuple(sorted(lanes))


def _scorer_name(lane: str) -> str:
    """``"scorer:<name>@<params>"`` -> ``<name>``; a lane in another form is its own name."""
    return lane.removeprefix("scorer:").split("@", 1)[0]


def _lint_lanes(lanes: tuple[str, ...], *, ci: bool) -> tuple[RunDecision, ...]:
    """Refuse a generic generation-quality scorer under ``--ci``; outside it, say it can't gate."""
    from calibrated_agent_evaluation.enforce import EnforcementContext, lint_scorer_lanes
    from calibrated_agent_evaluation.run import RunDecision

    names = [_scorer_name(lane) for lane in lanes if lane != _UNRECORDED_LANE]
    return tuple(
        RunDecision(rule=d.rule, tier=d.tier, inputs=d.inputs, outcome=d.outcome, message=d.message)
        for d in lint_scorer_lanes(names, EnforcementContext(ci=ci))
    )


def _lane_lines(lanes: tuple[str, ...], decisions: tuple[RunDecision, ...]) -> list[str]:
    if not lanes:
        return []
    lines = [f"scorer lanes gated: {', '.join(lanes)}"]
    if _UNRECORDED_LANE in lanes:
        lines.append(
            "  note: some scorer verdicts record no scorer, so whether a generic metric made them "
            "cannot be checked"
        )
    return lines + [f"  note: {d.message}" for d in decisions]


def _staleness(
    admitted: list[tuple[Judge, CalibrationRecord]], drift: Any
) -> tuple[RunDecision, ...]:
    """One passive staleness warning per admitted judge whose inputs drifted or version moved."""
    from calibrated_agent_evaluation.run import check_staleness

    out = []
    for judge, record in admitted:
        decision = check_staleness(judge, record, drift)
        if decision is not None:
            out.append(decision)
    return tuple(out)


def _n_resamples(args: argparse.Namespace) -> int:
    from calibrated_agent_evaluation.calibrate.engine import DEFAULT_N_RESAMPLES

    n = getattr(args, "n_resamples", None)
    value = DEFAULT_N_RESAMPLES if n is None else int(n)
    if value < 1:
        raise UsageError(f"--n-resamples must be positive, got {value}")
    return value


def _judge_files(args: argparse.Namespace, held: contextlib.ExitStack) -> list[JudgeFile]:
    from calibrated_agent_evaluation.cli.stages.calibrate_agentcore import judge_file_for_identity

    files = [judge_file_for_identity(args, p, held) for p in getattr(args, "judges", None) or []]
    seen: set[tuple[str, str | None]] = set()
    for jf in files:
        key = (jf.failure_mode, jf.agent)
        if key in seen:
            raise UsageError(
                f"two --judge files measure the same mode {key}; a mode is corrected by one judge"
            )
        seen.add(key)
    return files


def _from_judges(labels: list[Label], judge_files: list[JudgeFile]) -> list[Label]:
    """The labels, with LLM-judge verdicts kept only when one of ``judge_files`` made them.

    With ``--judge`` given, the gate reads the same verdicts the trend corrects; a verdict made by
    another judge or another version of one is not that judge's evidence.
    """
    from calibrated_agent_evaluation.judges import verdict_rater_id

    mine = {verdict_rater_id(jf.judge) for jf in judge_files}
    return [lb for lb in labels if lb.kind != "llm-judge" or lb.rater_id in mine]


class _GateInput:
    """What the gate counts: each mode's observations, their total, and what was left out."""

    def __init__(self, name: str, counts: dict[ModeKey, ModeCounts], n_hunted: int) -> None:
        self.name = name
        self.counts = counts
        self.n = sum(c.n_total for c in counts.values())
        self.notes: list[str] = []
        self.plain: list[str] = []
        if n_hunted:
            self.notes.append(
                f"  note: {n_hunted} {name} on discovery-frame traces left out of the gate (it "
                "reads the production-representative sample only)"
            )
            self.plain.append(
                f"{n_hunted} {name} on hand-picked traces were left out of the gate; it is "
                "measured on the random sample only."
            )


def _gate_input(
    labels: list[Label],
    traces: list[Trace],
    source: str,
    scope: str,
    agents: dict[tuple[str, str], str],
) -> _GateInput:
    """The per-mode counts the gate reads from ``source``'s verdicts on the random sample.

    A verdict with no failure mode is not gated (it cannot be keyed to a mode; never folded into a
    wrong one), and neither is a span defect, which is attribution evidence, not a verdict on the
    trace. A source with nothing to gate is refused rather than replaced by the other one: a gate
    over human gold that was asked for judge verdicts would report a number nobody chose.
    """
    if source not in _SOURCES:
        raise UsageError(f"--verdicts must be one of {sorted(_SOURCES)}, got {source!r}")
    kinds, name, hint = _SOURCES[source]
    production = {t.trace_id for t in traces if t.frame == _PRODUCTION}
    in_source = [
        lb
        for lb in labels
        if lb.kind in kinds
        and lb.failure_mode is not None
        and not lb.span_defect
        and (source != "human" or lb.span_id is None)
    ]
    sampled = [lb for lb in in_source if lb.trace_id in production]
    if not sampled:
        others = " or ".join(f"--verdicts {s}" for s in _SOURCES if s != source)
        hunted = (
            f" ({len(in_source)} are on discovery-frame traces, which the gate does not read: a "
            "hunted trace is not a random draw)"
            if in_source
            else ""
        )
        raise CliError(
            f"no {name} with a failure mode on {scope} to gate (--verdicts {source}){hunted}; "
            f"{hint}, or gate another source with {others}"
        )
    n_hunted = sum(
        1
        for lb in labels
        if lb.kind in kinds and not lb.span_defect and lb.trace_id not in production
    )
    if source == "human":
        return _GateInput(name, _human_counts(labels, traces, production, agents), n_hunted)
    from calibrated_agent_evaluation.run import aggregate_by_mode

    return _GateInput(name, aggregate_by_mode(sampled, lambda lb: _mode_for(lb, agents)), n_hunted)


def _human_counts(
    labels: list[Label],
    traces: list[Trace],
    production: set[str],
    agents: dict[tuple[str, str], str],
) -> dict[ModeKey, ModeCounts]:
    """One observation per trace, rater and gated mode, from the trace-level human labels.

    The gated modes are the ones a human FAIL names. Each rater's labels on a trace are read by
    ``verdicts_for`` over that rater's PASS labels and the FAILs keyed to the mode (agent
    included), so a FAIL for another mode or agent is no evidence either way. A mode-free PASS
    counts for a crew mode only on a trace where that agent ran. One rater saying both FAIL and
    PASS for a mode on one trace is refused: there is no verdict to count until one is retracted.
    """
    from calibrated_agent_evaluation.cli.pairing import verdicts_for
    from calibrated_agent_evaluation.run import ModeCounts

    human = [
        lb
        for lb in labels
        if lb.kind == "human"
        and lb.span_id is None
        and not lb.span_defect
        and lb.trace_id in production
    ]
    modes = sorted(
        {_mode_for(lb, agents) for lb in human if lb.outcome == "FAIL" and lb.failure_mode},
        key=lambda m: (m.failure_type, m.agent or ""),
    )
    ran = {t.trace_id: {s.agent for s in t.spans} for t in traces}
    by_rater: dict[tuple[str, str | None], list[Label]] = {}
    for lb in human:
        by_rater.setdefault((lb.trace_id, lb.rater_id), []).append(lb)
    totals: dict[ModeKey, int] = {}
    fails: dict[ModeKey, int] = {}
    conflicts: set[str] = set()
    for (trace_id, _rater), mine in by_rater.items():
        for mode in modes:
            evidence = [lb for lb in mine if lb.outcome == "PASS" or _mode_for(lb, agents) == mode]
            fail, conflicting = verdicts_for(evidence, mode.failure_type, "human")
            if conflicting:
                conflicts.add(trace_id)
                continue
            if fail is None or (not fail and mode.agent and mode.agent not in ran[trace_id]):
                continue
            totals[mode] = totals.get(mode, 0) + 1
            fails[mode] = fails.get(mode, 0) + int(fail)
    if conflicts:
        shown = ", ".join(repr(t) for t in sorted(conflicts)[:5])
        raise CliError(
            f"one rater both failed and passed a gated mode on {len(conflicts)} trace(s) "
            f"({shown}), so the gate has no verdict to count there; retract the stale label "
            "(`calibrated-agent-evaluation label --serve --relabel`) and run again"
        )
    return {
        m: ModeCounts(mode=m, n_total=n, n_judged_fail=fails.get(m, 0)) for m, n in totals.items()
    }


class _Notes:
    """What the trend left out and why: the detail line and the plain line for each."""

    def __init__(self) -> None:
        self.detail: list[str] = []
        self.plain: list[str] = []

    def add(self, detail: str, plain: str) -> None:
        self.detail.append(detail)
        self.plain.append(plain)


def _trend(
    judge_files: list[JudgeFile],
    traces: list[Trace],
    labels_by_trace: dict[str, list[Label]],
    root: Path,
    agents: dict[tuple[str, str], str],
    *,
    seed: int,
    n_resamples: int,
) -> tuple[list[TrendPoint], _Notes, list[tuple[Judge, CalibrationRecord]]]:
    """Each ``--judge`` mode's corrected rate over its production-representative verdicts.

    A mode is corrected only with its judge admitted (certified, and its bias battery CLEAR, the
    chain ``calibrated-agent-evaluation judge`` enforces) and only over that judge's own verdicts,
    matched on the ``rater_id`` the verdicts carry: another judge's or version's verdicts are never
    corrected with this judge's error rates.
    """
    from calibrated_agent_evaluation.cli.records import latest_for_judge, load_records_of
    from calibrated_agent_evaluation.judges import verdict_rater_id
    from calibrated_agent_evaluation.run import ModeKey, aggregate_by_mode, trend_report

    notes = _Notes()
    admitted: list[tuple[Judge, CalibrationRecord]] = []
    if not judge_files:
        notes.add(
            "no trend: pass --judge FILE for each judge whose verdicts to correct",
            "No failure rate estimated: pass --judge FILE to correct a judge's verdicts.",
        )
        return [], notes, admitted
    calibrations = load_records_of(root, "calibration")
    batteries = load_records_of(root, "bias")
    production = {t.trace_id for t in traces if t.frame == "production-representative"}
    counts: dict[ModeKey, Any] = {}
    calib_by_mode: dict[ModeKey, CalibrationRecord] = {}
    samples: dict[ModeKey, list[int]] = {}
    clusters: dict[ModeKey, list[str]] = {}
    for jf in judge_files:
        mode = ModeKey(failure_type=jf.failure_mode, agent=jf.agent)
        name = _mode_name(mode)
        judge_id = jf.judge.judge_id
        judge = jf.judge
        record = latest_for_judge(calibrations, judge)
        battery = latest_for_judge(batteries, judge)
        refusal = _admission_refusal(judge, record, battery)
        if refusal is not None or record is None:
            detail, plain_why = refusal or ("no calibration record", "this judge isn't certified")
            notes.add(
                f"{name}: no trend, judge {judge_id!r} is not admitted ({detail}); "
                "no corrected rate",
                f"{name} failure rate: not estimated ({plain_why}).",
            )
            continue
        admitted.append((judge, record))
        mine = verdict_rater_id(judge)
        for_mode = [
            lb
            for labels in labels_by_trace.values()
            for lb in labels
            if lb.kind == "llm-judge"
            and lb.failure_mode == jf.failure_mode
            and _agent_matches(lb, jf, agents)
        ]
        verdicts = [lb for lb in for_mode if lb.rater_id == mine]
        if len(verdicts) < len(for_mode):
            other = len(for_mode) - len(verdicts)
            notes.add(
                f"{name}: {other} verdict(s) for this mode from another judge, another version of "
                f"this one, or no recorded judge left out (this judge's verdicts carry {mine!r})",
                f"{name}: {other} verdict(s) made by another judge or judge version were left out; "
                f"only this judge's verdicts are corrected with its error rates.",
            )
        rate = [lb for lb in verdicts if lb.trace_id in production]
        if len(rate) < len(verdicts):
            notes.add(
                f"{name}: {len(verdicts) - len(rate)} verdict(s) on discovery-frame "
                "traces left out of the rate (a hunted trace is not a random draw)",
                f"{name}: {len(verdicts) - len(rate)} verdict(s) on hand-picked traces were left "
                "out of the rate; it is measured on the random sample only.",
            )
        # The interval treats the gold and the rate sample as independent, so a production trace
        # the judge was calibrated on cannot also be measured by it. A rate-unsafe record refuses
        # the rate before any trace is read, so its refusal is reported instead.
        gold_ids = set(record.trace_ids)
        in_gold = sorted({lb.trace_id for lb in rate if lb.trace_id in gold_ids})
        if in_gold and not record.below_coverage_floor:
            n_left = sum(1 for lb in rate if lb.trace_id in gold_ids)
            rate = [lb for lb in rate if lb.trace_id not in gold_ids]
            notes.add(
                f"{name}: {n_left} verdict(s) on {len(in_gold)} production trace(s) in the "
                f"calibration gold of {record.calib_id!r} left out of the rate (the rate sample "
                f"must be independent of the gold): {', '.join(in_gold)}",
                f"{name}: {n_left} verdict(s) on {len(in_gold)} trace(s) this judge was "
                "calibrated on were left out of the rate; it is measured on traces the judge "
                "has not seen as gold.",
            )
            if not rate:
                notes.add(
                    f"{name}: no trend, every production-representative verdict of judge "
                    f"{judge_id!r} is on its calibration gold",
                    f"{name} failure rate: not estimated (every random-sample trace this judge "
                    "scored is in its calibration gold; score other production traces with "
                    f"`calibrated-agent-evaluation judge --judge {jf.path} --frame "
                    "production-representative`).",
                )
                continue
        if not rate:
            notes.add(
                f"{name}: no trend, judge {judge_id!r} has no verdicts on "
                "production-representative traces (run `calibrated-agent-evaluation judge` over "
                "them first)",
                f"{name} failure rate: not estimated (this judge hasn't scored any random-sample "
                f"traces yet; run `calibrated-agent-evaluation judge --judge {jf.path} --frame "
                "production-representative`).",
            )
            continue
        gold = {(lb.trace_id, lb.span_id): mode for lb in rate}
        counts.update(aggregate_by_mode(rate, gold, calib_by_mode={mode: record.calib_id}))
        calib_by_mode[mode] = record
        samples[mode] = [1 if lb.outcome == "FAIL" else 0 for lb in rate]
        clusters[mode] = [lb.trace_id for lb in rate]
    # A trace contributing several verdicts (span-level judges) is one cluster; the rate interval
    # then uses its effective size. Independent verdicts need no cluster ids.
    cluster_ids = {m: ids for m, ids in clusters.items() if len(set(ids)) < len(ids)}
    trend = trend_report(
        counts,
        calib_by_mode,
        samples,
        seed=seed,
        n_resamples=n_resamples,
        rate_trace_ids=clusters,
        rate_cluster_ids=cluster_ids or None,
    )
    return trend, notes, admitted


def _admission_refusal(judge: Any, record: Any, battery: Any) -> tuple[str, str] | None:
    """Why ``judge`` may not have its verdicts corrected (detail, plain), or ``None`` if it may."""
    from calibrated_agent_evaluation.enforce.admission import admit_judge

    if record is None:
        return (
            "no calibration record",
            "this judge isn't certified; run `calibrated-agent-evaluation calibrate` on it first",
        )
    if battery is None:
        return (
            f"calibration {record.calib_id!r}, but no bias battery",
            "this judge hasn't been checked for bias; run `calibrated-agent-evaluation bias` on "
            "it first",
        )
    gate = admit_judge(judge, record, battery)
    if gate.admitted:
        return None
    if not gate.certified:
        plain_why = "this judge isn't certified as it is now; recalibrate it"
    else:
        plain_why = "this judge didn't pass its bias check, so its verdicts aren't trusted"
    return gate.refusal_reason or "refused", plain_why


def _agent_matches(label: Label, jf: JudgeFile, agents: dict[tuple[str, str], str]) -> bool:
    return jf.agent is None or _mode_for(label, agents).agent == jf.agent


def _persist(
    report: Any,
    root: Path,
    *,
    args: argparse.Namespace,
    source: str,
    n_gated: int,
    judge_ids: tuple[str, ...],
    scorer_lanes: tuple[str, ...],
    seed: int,
    n_resamples: int,
) -> str:
    from calibrated_agent_evaluation.cli.records import append_records, ledger_head
    from calibrated_agent_evaluation.run import run_record_of

    head, n_entries = ledger_head(root)

    inputs = {
        "backend": str(getattr(args, "backend", "local")),
        "root": str(root),
        "scope": str(getattr(args, "trace_id", None) or "all"),
        "verdict_source": source,
        "n_gated": str(n_gated),
        "judge_files": ",".join(str(p) for p in getattr(args, "judges", None) or []) or "none",
        "ledger_head": head,
        "ledger_entries": str(n_entries),
    }
    if getattr(args, "drift_baseline", None):
        inputs["drift_baseline"] = str(args.drift_baseline)
        inputs["drift_feature"] = str(args.drift_feature)
    if getattr(args, "atoms_file", None):
        inputs["atoms_file"] = str(args.atoms_file)
    record = run_record_of(
        report,
        inputs=inputs,
        judge_ids=judge_ids,
        scorer_lanes=scorer_lanes,
        seed=seed,
        n_resamples=n_resamples,
    )
    append_records(root, "run", [record])
    return record.run_id


def _run_id() -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{stamp}-{secrets.token_hex(3)}"


def _mode_lines(gate: RunGateDecision) -> list[str]:
    """One line per gated mode: its counts, its lower bound and the bar it was held to."""
    return [
        f"  {_mode_name(g.mode)}: {g.n_fail} FAIL of {g.n_total}, pass-rate 95% lower bound "
        f"{g.lower_bound:.3f} vs threshold {g.threshold:.2f}: "
        + ("cleared" if g.passed else "FAILED")
        for g in gate.modes
    ]


def _gate_line(gate: Any, n: int, source: str, threshold: float, scope: str) -> str:
    head = f"CI gate over {n} {source} on {scope}, threshold {threshold:.2f} on each mode's "
    head += "pass-rate 95% lower bound"
    if gate.passed:
        return f"{head}: PASSED, all {gate.n_modes} mode(s) cleared it"
    failing = ", ".join(_mode_name(m) for m in gate.failing_modes)
    return f"{head}: FAILED, {len(gate.failing_modes)} of {gate.n_modes} mode(s) did not: {failing}"


def _trend_lines(trend: list[TrendPoint], notes: list[str], root: Path) -> list[str]:
    lines = [f"  {_mode_name(tp.mode)}: {_trend_text(tp)}" for tp in trend]
    if not trend:
        lines.insert(
            0, f"trend: none (calibration records read from {root / 'records' / 'calibration'})"
        )
    else:
        lines.insert(0, f"trend ({len(trend)} mode(s), Rogan-Gladen corrected fail rate):")
    return lines + [f"  note: {n}" for n in notes]


def _trend_text(tp: TrendPoint) -> str:
    cr = tp.corrected_rate
    raw = tp.raw_rate
    shown = _interval_text(cr)
    return (
        f"corrected {shown} ({cr.method}, n={cr.n}, calibration {tp.calib_id}); "
        f"raw {_interval_text(raw)} ({raw.method})"
    )


def _interval_text(iv: Interval) -> str:
    pct = f"{iv.level:.0%} CI [{iv.lo:.3f}, {iv.hi:.3f}]"
    if iv.point is None:
        return f"REFUSED ({iv.refusal_reason}), {pct}"
    return f"{iv.point:.3f}, {pct}"


def _mode_name(mode: ModeKey) -> str:
    return mode.failure_type if mode.agent is None else f"{mode.agent}/{mode.failure_type}"


def _mode_for(label: Label, agents: SpanAgents | None = None) -> ModeKey:
    """The failure-mode key of a human-ratified label or verdict; a crew mode keys per agent.

    Called only from ``run_handler`` after ``require_extras`` has confirmed the [stats] engine, so
    the ``calibrated_agent_evaluation.run`` import here always resolves.
    """
    from calibrated_agent_evaluation.attribution.labels import attributed_agent
    from calibrated_agent_evaluation.run import ModeKey

    return ModeKey(failure_type=label.failure_mode or "", agent=attributed_agent(label, agents))
