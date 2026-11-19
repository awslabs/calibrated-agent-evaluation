# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``bias``: run a calibrated judge's bias battery and record the verdict.

The second judge-trust gate (``calibrate`` -> ``bias`` -> ``judge``). For the judge in
``--judge FILE`` the handler loads the probe set in ``--probes DIR`` and the per-model ceilings in
``--thresholds FILE`` (the ``fixtures/bias/`` format, read by
``calibrated_agent_evaluation.bias.load_probe_set`` / ``load_thresholds``), then runs
``calibrated_agent_evaluation.bias.run_battery`` through
``calibrated_agent_evaluation.judges.JudgeProbeRunner``: each probe input is judged by
``run_judge``, with the judge's live model or a ``--cassette`` replay.

``run_judge`` scores only a certified judge, so the newest calibration record for the judge under
``<root>/records/calibration/`` must certify it at its current version: an uncalibrated judge has
no bias battery to run, and is refused before any model call. The ``BiasBatteryRecord`` is appended
to ``<root>/records/bias/`` whether CLEAR or BIASED, and a BIASED judge exits 1. The battery runs on
the single-verdict path, so the pairwise positional probe is recorded as not applicable.

An ``agentcore`` judge file is certified the same way, then its probes are scored by its AgentCore
evaluator: every probe input in one run of billed ``Evaluate`` calls, shown as an estimate and sent
only with ``--yes``, up to ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_CONCURRENCY`` at a time with
progress on standard error. Its thresholds are the file's entry for the evaluator's ARN, or else for
the model it runs on, and the entry used is printed. The session ids the probes sent are stored with
the record, for the online read-back to skip.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import JudgeNotCertifiedError, UsageError


def bias_handler(args: argparse.Namespace) -> StageOutcome:
    """Run the bias battery for a certified judge and record the battery."""
    # Holds the AWS clients an AgentCore judge file opens, so they are closed however the run ends.
    with contextlib.ExitStack() as held:
        return _bias(args, held)


def _bias(args: argparse.Namespace, held: contextlib.ExitStack) -> StageOutcome:
    from calibrated_agent_evaluation.cli.backends import local_root
    from calibrated_agent_evaluation.cli.config import require_extras

    require_extras("scipy")
    from calibrated_agent_evaluation.bias import load_probe_set, load_thresholds, run_battery
    from calibrated_agent_evaluation.calibrate import certifies, pin_mismatch
    from calibrated_agent_evaluation.cli.judgefile import (
        declares_agentcore_backend,
        load_judge_file,
    )
    from calibrated_agent_evaluation.cli.judging import model_backend
    from calibrated_agent_evaluation.cli.records import (
        append_records,
        latest_for_judge,
        load_records_of,
        records_dir,
    )
    from calibrated_agent_evaluation.judges import JudgeProbeRunner

    path = getattr(args, "judge_file", None)
    probes_dir = getattr(args, "probes", None)
    thresholds_file = getattr(args, "thresholds", None)
    if not path or not probes_dir or not thresholds_file:
        raise UsageError(
            "calibrated-agent-evaluation bias needs --judge FILE, --probes DIR and --thresholds "
            "FILE (the probe set and "
            "the per-model ceilings; `calibrated-agent-evaluation quickstart --certify` writes an "
            "example of each)"
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
    calib = latest_for_judge(load_records_of(root, "calibration"), judge)
    if calib is None or not certifies(calib, judge):
        if calib is None:
            reason = "it has no calibration record"
        elif calib.result != "CERTIFIED":
            reason = f"its newest calibration record {calib.calib_id!r} is {calib.result}"
        else:
            reason = (
                f"the newest certified record {calib.calib_id!r} is for "
                f"{calib.model}@{calib.prompt_version}, not this judge file: "
                f"{pin_mismatch(calib, judge)}"
            )
        raise JudgeNotCertifiedError(
            f"judge {judge.judge_id!r} ({judge.model}@{judge.prompt_version}) is not certified: "
            f"{reason} (records under {records_dir(root, 'calibration')}); the bias battery runs "
            f"through the judge runtime, which scores only a certified judge. Run "
            f"`{_calibrate_command(args, str(jf.path))}`"
            " first.",
            reason=reason,
            judge_path=str(jf.path),
            root=_plain().carried_flags(args)["root"],
        )
    probes = load_probe_set(Path(probes_dir))
    thresholds = load_thresholds(Path(thresholds_file))
    if agentcore_run is None:
        runner = JudgeProbeRunner(judge, calib, model_backend(args, judge))
        record = run_battery(
            judge,
            runner,
            thresholds,
            probes,
            battery_id=getattr(args, "battery_id", None),
            sampling=judge.sampling(),
        )
    else:
        from calibrated_agent_evaluation.agentcore.bias import bias_thresholds_for

        # Before the estimate, so the ceilings that apply are known before anything is paid for;
        # a file with no entry for this judge is refused here, at no cost.
        _, key = bias_thresholds_for(judge, thresholds)
        print(f"bias thresholds: the {key!r} entry of {thresholds_file}", file=sys.stderr)
        scored = agentcore_run.run_battery(calib, thresholds, probes)
        if isinstance(scored, StageOutcome):
            return scored
        record = scored.record
        # Stored before the record, so a battery on record always has its sessions to skip.
        agentcore_run.record_sessions(root, record)
    append_records(root, "bias", [record])

    clear = record.verdict == "CLEAR"
    head = f"bias battery for judge {judge.judge_id!r} ({judge.model}@{judge.prompt_version}): "
    head += record.verdict
    if not clear:
        head += f", must-pass bias exceeded: {', '.join(record.failed_biases)}"
    lines = [
        head,
        "  each bias is how much more often the judge's verdicts disagree across the biasing "
        "dimension than across a placebo that varies nothing, gated on its one-sided 95% upper "
        "bound against the judge model's threshold; a bias with a direction is also gated on how "
        "much more often its verdicts move toward the favored one than the placebo's do",
        *(f"  {_result_line(r)}" for r in record.results),
        f"recorded {record.battery_id} in {records_dir(root, 'bias')}",
    ]
    from calibrated_agent_evaluation.cli import plain

    carried = plain.carried_flags(args)
    if clear:
        judge_cmd = plain.command("judge", "--judge", str(jf.path), **carried)
        lines.append(f"next: `{judge_cmd}` can score with it")

    return StageOutcome(
        stage="bias",
        exit_code=EXIT_OK if clear else EXIT_DOMAIN_ERROR,
        summary="\n".join(plain.bias(record, judge_path=str(jf.path), **carried)),
        detail="\n".join(lines),
    )


def _result_line(result: Any) -> str:
    kind = "must-pass" if result.is_must_pass else "advisory"
    if result.status != "run":
        return f"{result.bias} ({kind}): {result.status}, {result.detail}"
    shown = (
        f"{result.biased_count} of {result.n} trial(s) disagreed, {result.placebo_count} of "
        f"{result.n} placebo(s) did; excess {result.magnitude:.3f} "
        f"(95% upper bound {result.magnitude_ub95:.3f})"
    )
    if result.direction_ub95 is not None:
        within = "within" if result.direction_passed else "OVER"
        shown += (
            f"; toward the favored verdict {result.direction_count} of {result.direction_n} "
            f"trial(s) moved, {result.direction_placebo_count} placebo(s) did, excess "
            f"{result.direction_magnitude:.3f} (95% upper bound {result.direction_ub95:.3f}, "
            f"{within} {result.direction_threshold:.2f})"
        )
    if result.threshold is None:
        return f"{result.bias} ({kind}): {shown}, no threshold for this model, recorded only"
    verdict = "within" if result.passed else "OVER"
    return f"{result.bias} ({kind}): {shown}, {verdict} the {result.threshold:.2f} threshold"


def _plain() -> Any:
    from calibrated_agent_evaluation.cli import plain

    return plain


def _calibrate_command(args: argparse.Namespace, judge_path: str) -> str:
    return _plain().command("calibrate", "--judge", judge_path, **_plain().carried_flags(args))
