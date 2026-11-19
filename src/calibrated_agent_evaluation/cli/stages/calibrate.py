# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation calibrate``: certify a judge against the human gold in the store.

The first of the judge-trust gates (``calibrate`` -> ``bias`` -> ``judge``). For the judge in
``--judge FILE``:

  * the **gold** is the human labels on the stored traces that grade the judge's failure mode
    (:func:`calibrated_agent_evaluation.cli.judging.gold_for`); labels that are not gold for it are
    counted and named, never dropped silently;
  * the **plan**: ``calibrated_agent_evaluation.calibrate.plan_calibration`` at
    ``--expected-accuracy`` (and ``--expected-tnr`` for a judge expected to be better on one class)
    says how many gold labels per class the judge needs, and the output says whether the gold set
    has them. The plan is recorded, and gold below its coverage floor is marked rate-unsafe: the
    judge can still certify, but ``calibrated-agent-evaluation run`` refuses a corrected rate from
    the record. The floor is the larger of the one at the planned accuracy and the one at the TPR
    and TNR the judge showed, since a judge more accurate than planned needs more gold for its rate.
    ``--rate-safe`` plans for that up front: the size at which every accuracy that would certify
    leaves the record rate-safe, 162 per class. Without it the plan and
    its output are the certification-only ones;
  * the **verdicts**: the judge runs over the gold targets through
    ``calibrated_agent_evaluation.judges.locked_test_set``, with its live model or a ``--cassette``
    replay. They are not scores and are not written to the store;
  * the **gate**: ``calibrated_agent_evaluation.calibrate.certify`` on that locked test, CERTIFIED
    only when the 95% lower bounds of both TPR and TNR reach 0.80.

The ``CalibrationRecord`` is appended to ``<root>/records/calibration/`` whether it certifies or
not, since a REFUSED record is the evidence ``calibrated-agent-evaluation judge`` must see. A
REFUSED judge exits 1. The gold set is the judge's locked test, and a locked test is single-use:
evaluating it uses it up, whatever the result. Gold that already tested any judge for the same
failure mode is refused, in whole or in part, whatever the judge file's id, because tuning a prompt
until it passes the test it is scored on is what the single-use rule prevents. The one exception is
growing a set that was refused for having fewer labels than its plan, with the judge unchanged: the
new set must hold every trace of the refused one, and the record counts the looks (``looks``) so the
retest is disclosed. ``--explain`` shows the count.

Before any model call the gold's raters are checked against each other
(:mod:`calibrated_agent_evaluation.cli.goldagreement`): gold from two or more raters who overlapped
is refused unless their agreement is usable, and gold from one rater certifies with a note that
agreement was not measured. The alpha is stored on the record. ``--explain`` adds the
judge-versus-gold agreement (Cohen's kappa with the confusion cells, or raw agreement on balanced
gold) and the change in TPR and TNR against the previous record for the same judge id.

``calibrated-agent-evaluation calibrate --check`` is the re-check of a certified judge on fresh gold
(:mod:`calibrated_agent_evaluation.cli.stages.calibrate_check`); it writes a ``judge-check`` record
and never a calibration record.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError

_DEFAULT_ACCURACY = 0.90


def calibrate_handler(args: argparse.Namespace) -> StageOutcome:
    """Run the judge over its human gold, certify it, and record the result."""
    if getattr(args, "check", False):
        from calibrated_agent_evaluation.cli.stages.calibrate_check import calibrate_check_handler

        return calibrate_check_handler(args)
    # Holds the AWS clients an AgentCore judge file opens, so they are closed however the run ends.
    with contextlib.ExitStack() as held:
        return _calibrate(args, held)


def _calibrate(args: argparse.Namespace, held: contextlib.ExitStack) -> StageOutcome:
    from calibrated_agent_evaluation.cli.backends import local_root, open_backend
    from calibrated_agent_evaluation.cli.config import require_extras

    require_extras("scipy")
    from calibrated_agent_evaluation.calibrate import InMemoryLockedTestLedger, plan_calibration
    from calibrated_agent_evaluation.calibrate.engine import GATE_THRESHOLD, certify
    from calibrated_agent_evaluation.cli.goldagreement import measure_gold_agreement
    from calibrated_agent_evaluation.cli.judgefile import (
        declares_agentcore_backend,
        load_judge_file,
    )
    from calibrated_agent_evaluation.cli.judging import gold_for, model_backend
    from calibrated_agent_evaluation.cli.records import (
        append_records,
        load_records_of,
        records_dir,
        records_lock,
        refuse_used_gold,
        same_mode_priors,
    )
    from calibrated_agent_evaluation.cli.scope import in_cell, parse_cells, read_scope, skip_lines
    from calibrated_agent_evaluation.judges import locked_test_set

    path = getattr(args, "judge_file", None)
    if not path:
        raise UsageError(
            "calibrated-agent-evaluation calibrate needs --judge FILE (the judge to certify)"
        )
    agentcore_run = None
    if declares_agentcore_backend(path):
        from calibrated_agent_evaluation.cli.stages.calibrate_agentcore import open_agentcore_judge

        agentcore_run = open_agentcore_judge(args, Path(path), held)
        jf = agentcore_run.judge_file
    else:
        from calibrated_agent_evaluation.cli.stages.calibrate_agentcore import (
            refuse_agentcore_flags,
        )

        refuse_agentcore_flags(args)
        jf = load_judge_file(path)
    judge = jf.judge
    accuracy = _accuracy(getattr(args, "expected_accuracy", None), "--expected-accuracy")
    tnr = getattr(args, "expected_tnr", None)
    tnr_hat = None if tnr is None else _accuracy(tnr, "--expected-tnr")
    # Planned before any model call: an expected accuracy the gate cannot certify at is refused
    # (InfeasiblePlanError, exit 1) without spending a token. --rate-safe sizes for the corrected
    # rate at every accuracy that would certify.
    rate_safe = bool(getattr(args, "rate_safe", False))
    raters_flag = _raters(getattr(args, "raters", None))
    planned = functools.partial(
        plan_calibration, accuracy, GATE_THRESHOLD, tnr_hat=tnr_hat, rate_safe=rate_safe
    )
    one_rater = planned()
    plan = planned(raters=raters_flag or 1)
    cell = parse_cells(getattr(args, "cell", None))
    root = local_root(args)

    with open_backend(args) as backend:
        scan = read_scope(backend, args)
        traces = [t for t in scan.traces if in_cell(t, cell)]
        labels = {t.trace_id: backend.get_scores(t.trace_id) for t in traces}
    gold = gold_for(jf, traces, labels)
    scope = f"{len(traces)} stored trace(s)"
    if cell:
        scope += " in cell " + ", ".join(f"{k}={v}" for k, v in sorted(cell.items()))
    left_out = [f"  left out: {n} {why}" for why, n in sorted(gold.left_out.items())]
    left_out += skip_lines(scan)
    if gold.n_fail == 0 or gold.n_pass == 0:
        raise CliError(
            f"no locked test for judge {judge.judge_id!r} on {scope}: the human gold has "
            f"{gold.n_fail} FAIL for {_mode(jf)} and {gold.n_pass} PASS, and certification needs "
            f"both classes (plan: {plan.per_class} per class). Label traces with "
            "`calibrated-agent-evaluation label` "
            "first." + ("\n" + "\n".join(left_out) if left_out else "")
        )

    # Before any model call: gold whose raters disagree is refused without spending a token.
    agreement = measure_gold_agreement(jf, traces, labels)
    # The flag says how many raters will code the gold; the gold says how many already have. The
    # larger is planned for, so gold a second rater already double-coded is never under-planned.
    plan = planned(raters=max(raters_flag or 1, len(agreement.raters)))
    raters_lines = _raters_lines(plan, one_rater, raters_flag, agreement.raters)

    from calibrated_agent_evaluation.calibrate.engine import fingerprint_of

    set_id = _set_id(jf, gold.pairs)
    # Held from reading the stored records to appending the new one, model calls included: a
    # second run on the same gold waits, then sees the first run's record and is refused before
    # it spends a token.
    with records_lock(root, "calibration"):
        priors = load_records_of(root, "calibration")
        # Seeded from every stored record for this mode, whatever judge made it, so the
        # single-use refusal holds across runs and across judge files, and a top-up of a
        # refused-short set is recognized and its looks counted. append_records applies the same
        # rule again.
        ledger = InMemoryLockedTestLedger.from_records(
            same_mode_priors(set_id, judge.judge_id, priors)
        )
        refuse_used_gold(
            set_id=set_id,
            trace_ids={g.trace_id for _, g in gold.pairs},
            judge_id=judge.judge_id,
            fingerprint=fingerprint_of(judge),
            priors=priors,
        )
        if agentcore_run is None:
            locked = locked_test_set(
                judge, gold.pairs, set_id=set_id, model_backend=model_backend(args, judge)
            )
        else:
            scored = agentcore_run.score(traces, gold.pairs, set_id=set_id)
            if isinstance(scored, StageOutcome):
                return scored  # the estimate was shown and not approved; nothing was sent
            locked = scored
        record = certify(
            judge,
            locked,
            ledger=ledger,
            plan=plan,
            sampling=None if agentcore_run is not None else judge.sampling(),
            agreement_alpha=agreement.alpha_interval(),
        )
        if agentcore_run is not None:
            # Stored before the record: a log with no record is harmless, while a record whose
            # sessions were never logged would let the online read-back count them as traffic.
            agentcore_run.record_sessions(root, record)
        append_records(root, "calibration", [record], locked=True)
    previous = _previous_record(priors, judge.judge_id)

    certified = record.result == "CERTIFIED"
    rater_ids = [g.rater_id or "" for _, g in gold.pairs]
    lines = [
        f"calibrated judge {judge.judge_id!r} ({judge.model}@{judge.prompt_version}) for "
        f"{_mode(jf)} on {_gold_kind(rater_ids)} from {scope}: {record.result}",
        f"  TPR {_bound(record.tpr, 'FAIL')}; TNR {_bound(record.tnr, 'PASS')}; "
        f"Youden's J {record.youden_j:.3f}",
        f"  gate: both 95% lower bounds must reach {GATE_THRESHOLD:.2f}",
        _plan_line(plan, accuracy, tnr_hat, gold.n_fail, gold.n_pass, rate_safe=rate_safe),
        *raters_lines,
        *_rate_unsafe_line(record, certified),
        *_looks_lines(record),
        *agreement.detail(),
        *_judge_agreement_lines(locked),
        *_drift_lines(previous, record),
        *left_out,
        f"  the {len(gold.pairs)} calibration verdict(s) were not written to the store: they "
        "certify the judge and are not scores",
        f"recorded {record.calib_id} in {records_dir(root, 'calibration')}",
    ]
    if agentcore_run is not None:
        lines[1:1] = agentcore_run.detail_lines(gold.pairs)
    from calibrated_agent_evaluation.cli import plain

    carried = plain.carried_flags(args)
    if certified and agentcore_run is None:
        bias_cmd = plain.command(
            *("bias", "--judge", str(jf.path), "--probes", "DIR", "--thresholds", "FILE"),
            **carried,
        )
        lines.append(f"next: `{bias_cmd}` runs its bias battery")
    from calibrated_agent_evaluation.review.gate import MINIMUM_OVERLAP

    summary = plain.calibration(
        record,
        plan_per_class=plan.per_class,
        judge_path=str(jf.path),
        gold=plain.gold_note(rater_ids),
        **carried,
    )
    if agentcore_run is not None:
        summary += agentcore_run.summary_lines(gold.pairs)
    if plan.per_class > one_rater.per_class:
        summary.append(
            f"The plan is {plan.per_class} labels per class, not {one_rater.per_class}: a "
            f"second rater double-codes this gold, and their agreement needs {MINIMUM_OVERLAP} "
            "shared traces."
        )
    if agreement.report is None:
        summary.append(
            f"Rater agreement on the gold was not measured ({agreement.not_measured}); have a "
            f"second rater label the same gold (at least {MINIMUM_OVERLAP} of its traces) so "
            "the gold itself is checked."
        )
    if record.looks > 1:
        summary.append(
            f"This judge has now been tested {record.looks} times on this gold as it grew; the "
            "count is recorded with the result."
        )
    n_left_out = sum(gold.left_out.values())
    if n_left_out:
        summary.append(
            f"{n_left_out} label(s) in scope weren't used as gold for this judge; --explain says "
            "why."
        )
    summary += plain.skipped(scan)
    return StageOutcome(
        stage="calibrate",
        exit_code=EXIT_OK if certified else EXIT_DOMAIN_ERROR,
        summary="\n".join(summary),
        detail="\n".join(lines),
    )


def _raters(value: Any) -> int | None:
    if value is None:
        return None
    if int(value) < 1:
        raise UsageError(f"--raters must be at least 1, got {value}")
    return int(value)


def _raters_lines(
    plan: Any, one_rater: Any, raters_flag: int | None, gold_raters: tuple[str, ...]
) -> list[str]:
    """How the rater count was decided, and what double-coding did to the plan."""
    from calibrated_agent_evaluation.review.gate import MINIMUM_OVERLAP

    in_gold = "in the gold: " + (", ".join(gold_raters) or "none recorded")
    source = in_gold if raters_flag is None else f"--raters {raters_flag}; {in_gold}"
    lines = []
    if raters_flag is not None and len(gold_raters) > raters_flag:
        lines.append(
            f"  --raters {raters_flag} is fewer than the raters already in the gold, so the plan "
            f"is for {len(gold_raters)}"
        )
    if plan.double_coded is None:
        return [f"  raters: one ({source}), so the gold needs no agreement overlap", *lines]
    head = (
        f"  raters: two or more ({source}): they double-code this gold, and the agreement check "
        f"needs {MINIMUM_OVERLAP} shared traces, {plan.double_coded} per class"
    )
    if plan.per_class > one_rater.per_class:
        head += f", which raises the plan from {one_rater.per_class} to {plan.per_class}"
    else:
        head += f", which the plan of {plan.per_class} already covers"
    return [head, *lines]


def _rate_unsafe_line(record: Any, certified: bool) -> list[str]:
    """The note for a judge certified on gold below the coverage floor; empty otherwise."""
    if not (certified and record.below_coverage_floor):
        return []
    sizes = f"{record.n_pos} FAIL and {record.n_neg} PASS"
    observed = f"TPR {record.tpr.point:.3f} and TNR {record.tnr.point:.3f}"
    if record.observed_floor_unmeasured:
        why = (
            f"at the observed {observed} the corrected rate's interval was measured and did not "
            f"cover nominally at any scanned gold size, so {sizes} is not known to be enough"
        )
    else:
        why = (
            f"{sizes} is below the coverage floor of {record.coverage_floor} per class "
            f"({record.planned_coverage_floor} at the planned accuracy, "
            f"{record.observed_coverage_floor} at the observed {observed})"
        )
    return [
        f"  rate-unsafe: {why}; the judge is certified, but `calibrated-agent-evaluation run` "
        "refuses a corrected rate "
        "from this record"
    ]


def _looks_lines(record: Any) -> list[str]:
    if record.looks <= 1:
        return ["  looks: 1 (the first test of this judge on this gold)"]
    return [
        f"  looks: {record.looks} (this set grew a set refused short of its plan, and the judge "
        "was tested on each; disclosed, not corrected for)"
    ]


def _judge_agreement_lines(locked: Any) -> list[str]:
    """Judge-versus-gold agreement on the locked test: kappa with its cells under imbalance."""
    from calibrated_agent_evaluation.enforce import agreement_display

    gold = [p.gold.outcome for p in locked.pairs]
    shown = agreement_display(gold, [p.verdict.outcome for p in locked.pairs])
    name = "Cohen's kappa" if shown.scalar_name == "cohen_kappa" else "raw agreement"
    if shown.value is None:
        value = f"undefined ({shown.undefined_reason})"
    else:
        iv = shown.value
        value = f"{iv.point:.3f} (95% interval {iv.lo:.3f} to {iv.hi:.3f})"
    line = f"  judge vs gold: {name} {value}"
    if shown.confusion_cells is not None:
        c = shown.confusion_cells
        line += (
            f"; cells TP {c['tp']}, FN {c['fn']}, FP {c['fp']}, TN {c['tn']} (raw agreement "
            "hidden: the gold is imbalanced)"
        )
    return [line]


def _previous_record(priors: list[Any], judge_id: str) -> Any:
    mine = [(i, r) for i, r in enumerate(priors) if r.judge_id == judge_id]
    if not mine:
        return None
    return max(mine, key=lambda ir: (ir[1].provenance.run_at, ir[0]))[1]


def _drift_lines(previous: Any, record: Any) -> list[str]:
    """TPR and TNR change against the previous record for the same judge id."""
    if previous is None:
        return ["  judge drift: no earlier record for this judge id to compare with"]
    shared = set(previous.trace_ids) & set(record.trace_ids)
    if shared:
        return [
            f"  judge drift: not compared with {previous.calib_id!r}: the two share "
            f"{len(shared)} gold trace(s), so the comparison would be paired, not independent"
        ]
    from calibrated_agent_evaluation.run import judge_drift

    drift = judge_drift(previous, record)
    return [
        f"  judge drift since {previous.calib_id!r} ({previous.model}@{previous.prompt_version}, "
        f"{previous.result}): change in TPR {_delta(drift.delta_tpr)}; change in TNR "
        f"{_delta(drift.delta_tnr)}"
    ]


def _delta(iv: Any) -> str:
    return f"{iv.point:+.3f} (95% interval {iv.lo:+.3f} to {iv.hi:+.3f})"


def _gold_kind(rater_ids: list[str]) -> str:
    """``N human gold label(s)`` for human gold; the plain note names synthetic gold instead."""
    from calibrated_agent_evaluation.cli import plain

    return plain.labeled_note(rater_ids, "gold label(s)")


def _accuracy(value: Any, flag: str) -> float:
    if value is None:
        return _DEFAULT_ACCURACY
    out = float(value)
    if not 0.0 < out <= 1.0:
        raise UsageError(f"{flag} must be in (0, 1], got {out}")
    return out


def _set_id(jf: Any, pairs: list[tuple[Any, Any]]) -> str:
    """The locked test's id: the mode it tests and exactly which gold it holds.

    Keyed on content, not on the judge, so the same gold handed to a copy of the judge file under
    a new id is recognized as used. The mode tag comes first so the ids of earlier tests of the
    same mode can be found again (``gold-<mode>-<content>``).
    """
    from calibrated_agent_evaluation.cli.records import gold_mode_tag

    tag = gold_mode_tag(jf.failure_mode, jf.agent)
    members = sorted(f"{g.trace_id}|{g.span_id or ''}|{g.outcome}" for _, g in pairs)
    digest = hashlib.sha256("\n".join([tag, *members]).encode()).hexdigest()[:12]
    return f"gold-{tag}-{digest}"


def _mode(jf: Any) -> str:
    return jf.failure_mode if jf.agent is None else f"{jf.agent}/{jf.failure_mode}"


def _bound(bound: Any, cls: str) -> str:
    return f"{bound.point:.3f} (95% lower bound {bound.lb95:.3f}, n={bound.n} {cls})"


def _plan_line(
    sizes: Any,
    accuracy: float,
    tnr_hat: float | None,
    n_fail: int,
    n_pass: int,
    *,
    rate_safe: bool = False,
) -> str:
    plan = sizes.per_class
    expected = f"expected accuracy {accuracy:.2f}"
    if tnr_hat is not None:
        expected = f"expected TPR {accuracy:.2f} and TNR {tnr_hat:.2f}"
    if rate_safe:
        head = (
            f"  rate-safe plan at {expected}: {plan} gold label(s) per class "
            f"({sizes.certification} certify the judge; {plan} keep the corrected rate's interval "
            "covering at every certifying accuracy)"
        )
    else:
        head = f"  plan at {expected}: {plan} gold label(s) per class"
    head += f"; this set has {n_fail} FAIL and {n_pass} PASS"
    short = [f"{plan - n} {cls}" for n, cls in ((n_fail, "FAIL"), (n_pass, "PASS")) if n < plan]
    if not short:
        return head + ", enough"
    return head + f", short by {' and '.join(short)}"
