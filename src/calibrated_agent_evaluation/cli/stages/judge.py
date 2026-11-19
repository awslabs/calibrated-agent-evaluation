# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``judge``: score stored traces with a judge that has cleared the trust chain.

The judge stage is the one place the CLI emits a score, so it is where "no score without a
calibrated judge" is enforced. For the judge in ``--judge FILE`` the handler reads the newest
calibration record under ``<root>/records/calibration/`` and bias battery under
``<root>/records/bias/`` for that judge, then runs the shipped three-gate chain
``calibrated_agent_evaluation.enforce.admission.admit_judge`` (certification, then the bias gate,
then ``enforce_score_gate``). A refused judge exits 1 with the gate's reason and writes nothing. An
admitted judge runs through ``calibrated_agent_evaluation.judges.run_judge``, which checks
``enforce_score_gate`` again before any model call, and its verdicts are written back stamped with
the file's failure mode and with the judge that made them (``rater_id``
``"<judge_id>@<fingerprint>"``), so ``calibrated-agent-evaluation run --judge FILE`` gates and
corrects only that judge's verdicts.

A workflow judge scores each trace in scope. A span judge scores every span of the
judge file's agent, rendered with its ancestor path from the root
(``calibrated_agent_evaluation.judges.span_target``), and each verdict is keyed on
``(trace_id, span_id)``: the judged span is the one the verdict is about, which is how
``calibrated-agent-evaluation run`` assigns it to an agent.

``--frame production-representative`` scores only the random sample, so a store that also holds
a hunted calibration set gets verdicts where ``calibrated-agent-evaluation run`` reads its rate and
its gate.

The model is the judge's own (``[inspect]`` extra, live), or a recorded cassette replayed with
``--cassette``. A cassette miss is a typed error and nothing is written: the verdicts of one run
are written together or not at all. ``--cassette-mode record`` calls the live model for each reply
the cassette lacks and records it, so the run can be replayed offline afterwards.

An ``agentcore`` judge file is admitted through the same chain, then scored by its AgentCore
evaluator: one billed ``Evaluate`` call per trace, shown as an estimate and sent only with
``--yes``, up to ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_CONCURRENCY`` at a time with progress
on standard error. Its verdicts are stamped the same way, and the run is recorded the same way.

``calibrated-agent-evaluation judge --check FILE`` only checks a judge file: it runs the same
``load_judge_file`` and ``build_verdict_schema`` the scoring path runs, reads no store, calls no
model and writes nothing.
"""

from __future__ import annotations

import argparse
import contextlib
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from calibrated_agent_evaluation.agentcore.judge import AgentCoreJudge
    from calibrated_agent_evaluation.execute.judge_model import JudgeModelBackend
    from calibrated_agent_evaluation.judges.run_record import JudgeRunRecord
    from calibrated_agent_evaluation.schema.types import Label


def judge_handler(args: argparse.Namespace) -> StageOutcome:
    """Gate the judge, then score the traces in scope and write the verdicts back."""
    from calibrated_agent_evaluation.cli.judgefile import declares_agentcore_backend
    from calibrated_agent_evaluation.cli.stages.calibrate_agentcore import refuse_agentcore_flags

    check = getattr(args, "check_file", None)
    path = getattr(args, "judge_file", None)
    # The AWS flags apply only when a judge file the command names is an AgentCore one.
    if not any(f and declares_agentcore_backend(f) for f in (path, check)):
        refuse_agentcore_flags(args)
    if check:
        if path:
            raise UsageError(
                "calibrated-agent-evaluation judge --check FILE checks one judge file and scores "
                "nothing; give the file "
                "to --check alone, without --judge"
            )
        return check_judge_file(check)
    if not path:
        raise UsageError(
            "calibrated-agent-evaluation judge needs --judge FILE (the judge to score with), or "
            "--check FILE to check a "
            "judge file without scoring"
        )
    # Holds the AWS clients an AgentCore judge file opens, so they are closed however the run ends.
    with contextlib.ExitStack() as held:
        return _judge(args, path, held)


def _judge(args: argparse.Namespace, path: str, held: contextlib.ExitStack) -> StageOutcome:
    from calibrated_agent_evaluation.cli.backends import local_root, open_backend
    from calibrated_agent_evaluation.cli.config import require_extras

    require_extras("scipy")
    from calibrated_agent_evaluation.cli.judgefile import (
        declares_agentcore_backend,
        load_judge_file,
    )
    from calibrated_agent_evaluation.cli.judging import model_backend, targets_for
    from calibrated_agent_evaluation.cli.records import (
        latest_for_judge,
        load_records_of,
        records_dir,
    )
    from calibrated_agent_evaluation.cli.scope import read_scope, scope_note, skip_lines
    from calibrated_agent_evaluation.enforce.admission import admit_judge

    agentcore_run = None
    if declares_agentcore_backend(path):
        from calibrated_agent_evaluation.cli.stages.calibrate_agentcore import open_agentcore_judge

        agentcore_run = open_agentcore_judge(args, Path(path), held)
        jf = agentcore_run.judge_file
    else:
        jf = load_judge_file(path)
    judge = jf.judge
    root = local_root(args)

    calib = latest_for_judge(load_records_of(root, "calibration"), judge)
    if calib is None:
        raise CliError(
            f"no calibration record for judge {judge.judge_id!r} under "
            f"{records_dir(root, 'calibration')}: no calibration, no score. Run "
            f"`{_calibrate_command(args, str(jf.path))}`"
            " first."
        )
    battery = latest_for_judge(load_records_of(root, "bias"), judge)
    if battery is None:
        raise CliError(
            f"no bias battery for judge {judge.judge_id!r} under {records_dir(root, 'bias')}: a "
            "judge is admitted only after its bias battery runs CLEAR "
            f"(`{_bias_command(args, str(jf.path))}`)"
        )

    gate = admit_judge(judge, calib, battery)
    if not gate.admitted:
        return StageOutcome(
            stage="judge",
            exit_code=EXIT_DOMAIN_ERROR,
            summary=f"judge {judge.judge_id!r} refused, no verdict written: {gate.refusal_reason}",
        )

    # Built before the store is read, so a missing extra or cassette refuses at no cost.
    model = model_backend(args, judge) if agentcore_run is None else None
    with open_backend(args) as backend:
        scan = read_scope(backend, args)
        traces = scan.traces
        frame = getattr(args, "frame", None)
        if frame:
            traces = [t for t in traces if t.frame == frame]
        if not traces:
            where = f" in the {frame} frame" if frame else ""
            raise CliError(
                f"calibrated-agent-evaluation judge: the store holds no traces{where} to score"
            )
        targets = targets_for(jf, traces)
        if not targets:
            raise CliError(
                f"calibrated-agent-evaluation judge: no span of agent {jf.agent!r} on "
                f"{scope_note(args, len(traces))} "
                "for the span judge to score"
            )
        if agentcore_run is None:
            from calibrated_agent_evaluation.judges import run_judge

            # No store passed: the verdicts are stamped with the judge's mode, then written
            # together. A span judge's verdicts are per agent, which run_judge checks against its
            # level.
            result = run_judge(
                judge,
                targets,
                calib_record=calib,
                # Built above whenever there is no AgentCore run.
                model_backend=cast("JudgeModelBackend", model),
                per_agent_rate=judge.level == "span",
                # One record per invocation: the derived id repeats for the same batch, and a
                # re-run after the ledger moved would clash with the record of the first.
                run_id=_judge_run_id(),
            )
            labels, record = list(result.labels), result.record
        else:
            scored = agentcore_run.judge_traces(traces)
            if isinstance(scored, StageOutcome):
                return scored
            labels = _with_rater(scored.labels, judge)
            record = _agentcore_run_record(judge, labels, calib_id=calib.calib_id)
        verdicts = [replace(lb, failure_mode=jf.failure_mode) for lb in labels]
        for label in verdicts:
            backend.put_score(label)
    from calibrated_agent_evaluation.cli.records import append_records, ledger_head

    head, n_entries = ledger_head(root)
    run_record = replace(record, ledger_head=head, ledger_entries=n_entries)
    append_records(root, "judge", [run_record])

    outcomes = Counter(lb.outcome for lb in verdicts)
    mode = jf.failure_mode if jf.agent is None else f"{jf.agent}/{jf.failure_mode}"
    lines = [
        f"judge {judge.judge_id!r} admitted (calibration {calib.calib_id!r} certifies "
        f"{judge.model}@{judge.prompt_version}; bias battery {battery.battery_id!r} CLEAR)",
        f"wrote {len(verdicts)} verdict(s) for {mode} on {_targets_note(targets, traces, args)}: "
        f"{outcomes['FAIL']} FAIL, {outcomes['PASS']} PASS (run {record.run_id})",
        f"recorded judge run {run_record.run_id} in {records_dir(root, 'judge')}",
        *skip_lines(scan),
        f"these are raw verdicts, not a rate: `calibrated-agent-evaluation run --judge {jf.path}` "
        "corrects them with "
        "the calibration record",
    ]
    from calibrated_agent_evaluation.cli import plain

    summary = plain.judge(
        n_fail=outcomes["FAIL"],
        n_pass=outcomes["PASS"],
        n_traces=len({t.trace_id for t in targets}),
        mode=mode,
        judge_path=str(jf.path),
        root=plain.carried_flags(args)["root"],
    )
    summary += plain.skipped(scan)
    return StageOutcome(
        stage="judge", exit_code=EXIT_OK, summary="\n".join(summary), detail="\n".join(lines)
    )


def check_judge_file(path: str) -> StageOutcome:
    """``judge --check FILE``: load the file and build its verdict schema.

    This is what ``calibrated-agent-evaluation calibrate`` and ``calibrated-agent-evaluation judge``
    do before any model call. It reads no store and calls no model; a bad file raises the loader's
    or the schema's own typed error.
    """
    from calibrated_agent_evaluation.cli.judgefile import (
        declares_agentcore_backend,
        load_judge_file,
    )

    if declares_agentcore_backend(path):
        # Imported here, so checking a prompt judge never loads the AgentCore package.
        from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
        from calibrated_agent_evaluation.agentcore.judge import load_agentcore_judge_file

        try:
            judge_file = load_agentcore_judge_file(path)
        except AgentCoreError as exc:
            raise CliError(str(exc)) from exc
        return StageOutcome(
            stage="judge",
            exit_code=EXIT_OK,
            summary=(
                f"judge file ok: AgentCore judge {judge_file.judge_id!r} on evaluator "
                f"{judge_file.evaluator_id!r} (read when the judge is built; score it with "
                "`calibrated-agent-evaluation judge --judge FILE`)"
            ),
        )
    # Loaded first: the loader refuses a missing [stats] extra before the judge layer imports.
    judge = load_judge_file(path).judge
    from calibrated_agent_evaluation.judges import build_verdict_schema

    build_verdict_schema(judge)
    return StageOutcome(stage="judge", exit_code=EXIT_OK, summary="judge file ok")


def _targets_note(targets: list[Any], traces: list[Any], args: argparse.Namespace) -> str:
    """Where the verdicts landed: the traces, or for a span judge the spans across them."""
    from calibrated_agent_evaluation.cli.scope import scope_note

    scope = scope_note(args, len(traces))
    frame = getattr(args, "frame", None)
    if frame and not getattr(args, "trace_id", None):
        scope += f" in the {frame} frame"
    if all(t.span_id is None for t in targets):
        return scope
    if getattr(args, "trace_id", None):
        return f"{len(targets)} span(s) of {scope}"
    judged = len({t.trace_id for t in targets})
    return f"{len(targets)} span(s) in {judged} of {scope}"


def _with_rater(labels: Sequence[Label], judge: AgentCoreJudge) -> list[Label]:
    """AgentCore verdicts stamped with the judge that made them, as a prompt judge's verdicts are.

    ``calibrated-agent-evaluation run --judge`` corrects only the verdicts whose ``rater_id`` names
    the judge as it is now, and the evaluator's raw results carry none.
    """
    from calibrated_agent_evaluation.judges import verdict_rater_id

    rater = verdict_rater_id(judge)
    return [replace(label, rater_id=rater) for label in labels]


def _agentcore_run_record(
    judge: AgentCoreJudge, labels: list[Label], *, calib_id: str
) -> JudgeRunRecord:
    """The judge-run record of an AgentCore run, with the fields the prompt path's record has."""
    from calibrated_agent_evaluation.judges.run_record import JudgeRunRecord

    return JudgeRunRecord(
        run_id=_judge_run_id(),
        judge_id=judge.judge_id,
        model=judge.model,
        prompt_version=judge.prompt_version,
        calib_id=calib_id,
        level=judge.level,
        trace_ids=tuple(label.trace_id for label in labels),
        span_ids=tuple(label.span_id for label in labels),
        n_computed=len(labels),
        judge_fingerprint=judge.fingerprint,
    )


def _judge_run_id() -> str:
    import secrets
    from datetime import UTC, datetime

    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"judge-{stamp}-{secrets.token_hex(3)}"


def _plain() -> Any:
    from calibrated_agent_evaluation.cli import plain

    return plain


def _bias_command(args: argparse.Namespace, judge_path: str) -> str:
    return _plain().command(
        *("bias", "--judge", judge_path, "--probes", "DIR", "--thresholds", "FILE"),
        **_plain().carried_flags(args),
    )


def _calibrate_command(args: argparse.Namespace, judge_path: str) -> str:
    return _plain().command("calibrate", "--judge", judge_path, **_plain().carried_flags(args))
