# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrate --check``: re-check a certified judge on fresh human gold.

The judge in ``--judge FILE`` must hold a certifying calibration record (the newest one pinned to
it as it is now). Its gold for the re-check is the human gold in scope on traces no calibration
of its failure mode and no earlier re-check used, so the comparison with the certifying record is
between independent samples. The fresh gold needed per class comes from
:func:`calibrated_agent_evaluation.run.judge_check.check_size` on the certifying record; short gold
is refused before any model call, with the count still to label. The gold's raters are checked
against each other the way ``calibrated-agent-evaluation calibrate`` checks them.

The judge runs over the fresh gold, and :func:`calibrated_agent_evaluation.run.judge_drift` compares
its TPR and TNR with the certifying record's. The result is a ``judge-check`` record under
``<root>/records/judge-check/``. No calibration record is written or changed, and no locked test
is consumed. A flagged drop exits 1 and says to certify again on new gold.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, JudgeNotCertifiedError, UsageError

_FRESH_LEFT_OUT = "gold label(s) on traces a calibration or an earlier check of this mode used"


def calibrate_check_handler(args: argparse.Namespace) -> StageOutcome:
    """Measure a certified judge on fresh gold and record its drift from the certification."""
    # Holds the AWS clients an AgentCore judge file opens, so they are closed however the run ends.
    with contextlib.ExitStack() as held:
        return _check(args, held)


def _check(args: argparse.Namespace, held: contextlib.ExitStack) -> StageOutcome:
    from calibrated_agent_evaluation.cli.backends import local_root, open_backend
    from calibrated_agent_evaluation.cli.config import require_extras

    require_extras("scipy", "statsmodels")
    from calibrated_agent_evaluation._provenance import capture_provenance
    from calibrated_agent_evaluation.calibrate import InMemoryLockedTestLedger, certifies
    from calibrated_agent_evaluation.calibrate.engine import (
        GATE_THRESHOLD,
        certify,
        fingerprint_of,
        pin_mismatch,
    )
    from calibrated_agent_evaluation.cli.goldagreement import measure_gold_agreement
    from calibrated_agent_evaluation.cli.judgefile import (
        declares_agentcore_backend,
        load_judge_file,
    )
    from calibrated_agent_evaluation.cli.judging import gold_for, model_backend
    from calibrated_agent_evaluation.cli.records import (
        append_records,
        gold_mode_tag,
        latest_for_judge,
        load_records_of,
        records_dir,
        records_lock,
        same_mode_priors,
    )
    from calibrated_agent_evaluation.cli.scope import in_cell, parse_cells, read_scope
    from calibrated_agent_evaluation.judges import locked_test_set
    from calibrated_agent_evaluation.run import judge_drift
    from calibrated_agent_evaluation.run.judge_check import (
        JudgeCheckRecord,
        check_size,
        drift_flagged,
    )

    path = getattr(args, "judge_file", None)
    if not path:
        raise UsageError(
            "calibrated-agent-evaluation calibrate --check needs --judge FILE (the certified "
            "judge to check)"
        )
    for flag in ("rate_safe", "expected_accuracy", "expected_tnr"):
        if getattr(args, flag, None):
            raise UsageError(
                f"--{flag.replace('_', '-')} plans a certification; calibrated-agent-evaluation "
                "calibrate --check "
                "derives its gold size from the certifying record instead, so leave it out"
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
    root = local_root(args)
    calibrations = load_records_of(root, "calibration")
    record = latest_for_judge(calibrations, judge)
    if record is None or not certifies(record, judge):
        why = (
            "it has no calibration record"
            if record is None
            else (pin_mismatch(record, judge) or f"its newest record is {record.result}")
        )
        raise JudgeNotCertifiedError(
            f"judge {judge.judge_id!r} cannot be re-checked: {why}. A re-check measures a "
            "certified judge against its certification; run `calibrated-agent-evaluation "
            f"calibrate --judge {jf.path}` "
            "first",
            reason=why,
            judge_path=str(jf.path),
        )
    needed_fail, needed_pass = check_size(record, threshold=GATE_THRESHOLD)

    cell = parse_cells(getattr(args, "cell", None))
    with open_backend(args) as backend:
        scan = read_scope(backend, args)
        traces = [t for t in scan.traces if in_cell(t, cell)]
        labels = {t.trace_id: backend.get_scores(t.trace_id) for t in traces}
    tag = gold_mode_tag(jf.failure_mode, jf.agent)
    probe_id = f"gold-{tag}-check"
    used = {
        t for r in same_mode_priors(probe_id, judge.judge_id, calibrations) for t in r.trace_ids
    }
    checks = load_records_of(root, "judge-check")
    used |= {
        t
        for c in checks
        if c.check_set_id.startswith(f"check-{tag}-") or c.judge_id == judge.judge_id
        for t in c.trace_ids
    }
    fresh = [t for t in traces if t.trace_id not in used]
    gold = gold_for(jf, fresh, labels)
    left_out = dict(gold.left_out)
    n_used = sum(
        1 for t in traces if t.trace_id in used for lb in labels[t.trace_id] if lb.kind == "human"
    )
    if n_used:
        left_out[_FRESH_LEFT_OUT] = n_used
    left_lines = [f"  left out: {n} {why}" for why, n in sorted(left_out.items())]
    short = [
        f"{need - have} more {cls}"
        for have, need, cls in (
            (gold.n_fail, needed_fail, "FAIL"),
            (gold.n_pass, needed_pass, "PASS"),
        )
        if have < need
    ]
    if short:
        raise CliError(
            f"not enough fresh gold to re-check judge {judge.judge_id!r}: a drop to "
            f"{GATE_THRESHOLD:.2f} against {record.calib_id!r} is detected with {needed_fail} "
            f"FAIL and {needed_pass} PASS gold labels on traces no calibration of this mode used, "
            f"and the store has {gold.n_fail} FAIL and {gold.n_pass} PASS. Label "
            f"{' and '.join(short)} on new traces, then check again. No judge was run"
            + ("\n" + "\n".join(left_lines) if left_lines else "")
        )
    agreement = measure_gold_agreement(jf, fresh, labels)

    members = sorted(f"{g.trace_id}|{g.span_id or ''}|{g.outcome}" for _, g in gold.pairs)
    digest = hashlib.sha256("\n".join([tag, *members]).encode()).hexdigest()[:12]
    set_id = f"check-{tag}-{digest}"
    with records_lock(root, "judge-check"):
        if agentcore_run is None:
            locked = locked_test_set(
                judge, gold.pairs, set_id=set_id, model_backend=model_backend(args, judge)
            )
        else:
            scored = agentcore_run.score(fresh, gold.pairs, set_id=set_id)
            if isinstance(scored, StageOutcome):
                return scored
            locked = scored
        # Measured with the gate's own estimator on a ledger of its own: the check consumes no
        # locked test and the measurement is never stored as a calibration record.
        measured = certify(judge, locked, ledger=InMemoryLockedTestLedger())
        drift = judge_drift(record, measured)
        flagged = drift_flagged(drift)
        check = JudgeCheckRecord(
            check_id=f"check-{judge.judge_id}-{digest}",
            judge_id=judge.judge_id,
            model=judge.model,
            prompt_version=judge.prompt_version,
            judge_fingerprint=fingerprint_of(judge),
            against_calib_id=record.calib_id,
            check_set_id=set_id,
            n_pos=measured.n_pos,
            n_neg=measured.n_neg,
            tpr=measured.tpr,
            tnr=measured.tnr,
            drift=drift,
            drifted=flagged,
            needed_fail=needed_fail,
            needed_pass=needed_pass,
            agreement_alpha=agreement.alpha_interval(),
            trace_ids=measured.trace_ids,
            provenance=capture_provenance(
                thresholds={"gate": GATE_THRESHOLD},
                # An AgentCore evaluator's sampling is its own; the fingerprint pins it.
                sampling=judge.sampling() if agentcore_run is None else None,
            ),
        )
        append_records(root, "judge-check", [check], locked=True)

    detail = [
        f"re-checked judge {judge.judge_id!r} ({judge.model}@{judge.prompt_version}) for "
        f"{_mode(jf)} on {measured.n_pos} FAIL and {measured.n_neg} PASS fresh gold label(s)",
        f"  against {record.calib_id!r}: TPR {record.tpr.point:.3f} on {record.n_pos}, TNR "
        f"{record.tnr.point:.3f} on {record.n_neg}",
        f"  now: TPR {_bound(measured.tpr)}; TNR {_bound(measured.tnr)}",
        f"  change in TPR {_delta(drift.delta_tpr)}; change in TNR {_delta(drift.delta_tnr)} "
        "(difference of two independent proportions)",
        f"  size: {needed_fail} FAIL and {needed_pass} PASS, the smallest at which a judge "
        f"showing {GATE_THRESHOLD:.2f} would be flagged against this record",
        f"  drift flagged: {'yes' if flagged else 'no'} (a change whose interval lies wholly "
        "below zero)",
        *agreement.detail(),
        *left_lines,
        "  no calibration record was written or changed, and no locked test was used",
        f"recorded {check.check_id} in {records_dir(root, 'judge-check')}",
    ]
    from calibrated_agent_evaluation.cli import plain

    if flagged:
        summary = [
            f"Judge {judge.judge_id!r} has got worse since it was certified: on fresh labels it "
            f"now catches {plain.pct(measured.tpr.point)} of failures (was "
            f"{plain.pct(record.tpr.point)}) and clears {plain.pct(measured.tnr.point)} of "
            f"passes (was {plain.pct(record.tnr.point)}).",
            "Next: certify it again on new gold (`calibrated-agent-evaluation calibrate --judge "
            f"{jf.path}`); its "
            "current certification still stands until you do.",
        ]
    else:
        summary = [
            f"Judge {judge.judge_id!r} shows no drop since it was certified: on fresh labels it "
            f"catches {plain.pct(measured.tpr.point)} of failures and clears "
            f"{plain.pct(measured.tnr.point)} of passes.",
        ]
    if agreement.report is None:
        summary.append(
            f"Rater agreement on the fresh gold was not measured ({agreement.not_measured})."
        )
    return StageOutcome(
        stage="calibrate",
        exit_code=EXIT_DOMAIN_ERROR if flagged else EXIT_OK,
        summary="\n".join(summary),
        detail="\n".join(detail),
    )


def _mode(jf: Any) -> str:
    return jf.failure_mode if jf.agent is None else f"{jf.agent}/{jf.failure_mode}"


def _bound(bound: Any) -> str:
    return f"{bound.point:.3f} (95% lower bound {bound.lb95:.3f}, n={bound.n})"


def _delta(iv: Any) -> str:
    return f"{iv.point:+.3f} (95% interval {iv.lo:+.3f} to {iv.hi:+.3f})"
