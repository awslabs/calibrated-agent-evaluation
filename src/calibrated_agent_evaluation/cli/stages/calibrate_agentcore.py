# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AgentCore half of ``calibrate``, ``calibrate --check`` and ``judge``.

These commands take an ``agentcore`` judge file as well as a prompt one, and this module holds what
differs: opening the AWS clients, building the judge from its evaluator, and scoring through
AgentCore with the estimate confirmed by ``--yes``.

``calibrated-agent-evaluation calibrate`` runs its usual flow (the gold, the plan, the single-use
check, ``certify``, the record) for an AgentCore judge too; only the step that scores the locked
test differs. The judge is built from its evaluator (``GetEvaluator``), the locked test is scored
through AgentCore (one billed ``Evaluate`` call per trace, inside
``CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS`` when it is set, up to
``CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_CONCURRENCY`` at a time, 10 by default, with progress on
standard error), and without ``--yes`` the run shows its estimate and sends nothing. The session ids
the requests carried are stored as an ``agentcore-session`` record for the online read-back to skip,
and the gold's source is named, so rule-derived gold is never presented as human review.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError
from calibrated_agent_evaluation.cli.judgefile import JudgeFile

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.bias import AgentCoreBatteryRun
    from calibrated_agent_evaluation.agentcore.evaluate import AgentCoreJudgeRun
    from calibrated_agent_evaluation.agentcore.pricing import RunEstimate
    from calibrated_agent_evaluation.agentcore.session import AwsClients
    from calibrated_agent_evaluation.calibrate.record import CalibrationSet


@dataclass
class AgentCoreCalibrateRun:
    """An AgentCore judge a command scores with: its clients, its estimate and what it sent."""

    args: argparse.Namespace
    judge_file: (
        JudgeFile  # its judge is a calibrated_agent_evaluation.agentcore.judge.AgentCoreJudge
    )
    clients: AwsClients
    concurrency: int = 1
    estimates: list[RunEstimate] = field(default_factory=list)
    session_ids: tuple[str, ...] = ()
    gold_source: str | None = None

    def score(
        self, traces: Sequence[Any], pairs: Sequence[tuple[Any, Any]], *, set_id: str
    ) -> CalibrationSet | StageOutcome:
        """Score the locked test: its set, or the outcome when the estimate is declined."""
        from calibrated_agent_evaluation.agentcore.calibrate import (
            agentcore_locked_test_set,
            gold_source,
        )
        from calibrated_agent_evaluation.agentcore.errors import AgentCoreRunDeclined
        from calibrated_agent_evaluation.cli.agentcore_route import scoring_progress

        by_id = {trace.trace_id: trace for trace in traces}
        gold = [(by_id[label.trace_id], label) for _, label in pairs]
        self.gold_source = gold_source(label for _, label in pairs)
        try:
            run = agentcore_locked_test_set(
                self.judge_file.judge,
                gold,
                clients=self.clients,
                confirm=self._confirm,
                set_id=set_id,
                max_concurrency=self.concurrency,
                progress=scoring_progress(),
            )
        except AgentCoreRunDeclined:
            if self.args.yes or not self.estimates:
                raise
            verb = "Checking" if getattr(self.args, "check", False) else "Calibrating"
            return self._declined("calibrate", f"{verb} {self.judge_file.path}", len(gold), "gold ")
        self.session_ids = run.session_ids
        return run.calibration_set

    def judge_traces(self, traces: Sequence[Any]) -> AgentCoreJudgeRun | StageOutcome:
        """Score ``traces`` for ``judge``: the run, or the outcome when it is declined."""
        from calibrated_agent_evaluation.agentcore.errors import AgentCoreRunDeclined
        from calibrated_agent_evaluation.agentcore.evaluate import run_agentcore_judge_with_sessions
        from calibrated_agent_evaluation.cli.agentcore_route import scoring_progress

        try:
            run = run_agentcore_judge_with_sessions(
                self.judge_file.judge,
                list(traces),
                clients=self.clients,
                confirm=self._confirm,
                max_concurrency=self.concurrency,
                progress=scoring_progress(),
            )
        except AgentCoreRunDeclined:
            if self.args.yes or not self.estimates:
                raise
            return self._declined("judge", f"Judging with {self.judge_file.path}", len(traces), "")
        self.session_ids = run.session_ids
        return run

    def run_battery(
        self, calib: Any, thresholds: Any, probes: Sequence[Any]
    ) -> AgentCoreBatteryRun | StageOutcome:
        """Run the bias battery for ``bias``: the run, or the outcome when it is declined."""
        from calibrated_agent_evaluation.agentcore.bias import run_agentcore_battery
        from calibrated_agent_evaluation.agentcore.errors import AgentCoreRunDeclined
        from calibrated_agent_evaluation.cli.agentcore_route import scoring_progress

        try:
            run = run_agentcore_battery(
                self.judge_file.judge,
                calib,
                thresholds,
                probes,
                clients=self.clients,
                confirm=self._confirm,
                max_concurrency=self.concurrency,
                progress=scoring_progress(),
                battery_id=getattr(self.args, "battery_id", None),
            )
        except AgentCoreRunDeclined:
            if self.args.yes or not self.estimates:
                raise
            calls = self.estimates[0].calls
            return self._declined(
                "bias", f"The bias battery for {self.judge_file.path}", calls, "probe "
            )
        self.session_ids = run.session_ids
        return run

    def _confirm(self, estimate: RunEstimate) -> bool:
        self.estimates.append(estimate)
        return bool(self.args.yes)

    def _declined(self, stage: str, doing: str, count: int, kind: str) -> StageOutcome:
        """The outcome of a run declined at its estimate: what it would send, and how to send it."""
        return StageOutcome(
            stage=stage,
            exit_code=EXIT_OK,
            summary="\n".join(
                [
                    f"{doing} scores {count} {kind}trace(s) through AgentCore:",
                    f"  {self.estimates[0].describe()}",
                    "Nothing was sent. Run the same command with --yes to make the billed calls.",
                ]
            ),
            detail="",
        )

    def record_sessions(self, root: Path, record: Any) -> None:
        """Store the session ids the scoring sent, keyed by its calibration or battery record."""
        from calibrated_agent_evaluation.agentcore.online_record import AgentCoreSessionLog
        from calibrated_agent_evaluation.bias import BiasBatteryRecord
        from calibrated_agent_evaluation.cli.records import append_records

        judge = self.judge_file.judge
        battery = isinstance(record, BiasBatteryRecord)
        log = AgentCoreSessionLog(
            run_kind="bias" if battery else "calibration",
            run_id=record.battery_id if battery else record.calib_id,
            judge_id=judge.judge_id,
            evaluator_fingerprint=judge.fingerprint,
            session_ids=self.session_ids,
            # Recorded with the calibration, so read, publish and cost name it from any store.
            gold_source=None if battery else self.gold_source,
        )
        append_records(root, "agentcore-session", [log])

    def detail_lines(self, pairs: Sequence[tuple[Any, Any]]) -> list[str]:
        """The AgentCore lines of the detail: the evaluator, its pin, the gold source, the cost."""
        from calibrated_agent_evaluation.agentcore.calibrate import gold_source

        judge = self.judge_file.judge
        lines = [
            f"  AgentCore evaluator {judge.evaluator_id} ({judge.evaluator_type}), content version "
            f"{judge.prompt_version}, fingerprint {judge.fingerprint} (updated {judge.updated_at})",
            f"  gold source: {gold_source(label for _, label in pairs)}",
        ]
        if self.estimates:
            lines.append(f"  estimate approved: {self.estimates[0].describe()}")
        lines.append(f"  {len(self.session_ids)} session id(s) recorded for the online read-back")
        return lines

    def summary_lines(self, pairs: Sequence[tuple[Any, Any]]) -> list[str]:
        """The AgentCore line of the summary: the gold's source."""
        from calibrated_agent_evaluation.agentcore.calibrate import gold_source

        return [f"Gold source: {gold_source(label for _, label in pairs)}."]


def refuse_agentcore_flags(args: argparse.Namespace) -> None:
    """Refuse the AWS flags when the command's judge file is a prompt judge file.

    Raises:
        CliError: ``--yes``, ``--profile`` or ``--region`` was given; the message names it.
    """
    for flag, name in (("yes", "--yes"), ("profile", "--profile"), ("region", "--region")):
        if getattr(args, flag, None):
            raise UsageError(f"{name} applies to an AgentCore judge file only")


def open_agentcore_judge(
    args: argparse.Namespace, path: Path, held: contextlib.ExitStack
) -> AgentCoreCalibrateRun:
    """Read the ``agentcore`` judge file, open AWS clients on ``held`` and build its judge.

    Raises:
        CliError: ``--cassette`` was given (AgentCore scores the traces itself), the judge file
            cannot be read, or ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_CONCURRENCY`` is not from
            1 to 10.
        AgentCoreError: The evaluator cannot be read, or its level is not ``TRACE``; before any
            billed call.
    """
    from calibrated_agent_evaluation import agentcore
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
    from calibrated_agent_evaluation.agentcore.judge import (
        build_agentcore_judge,
        load_agentcore_judge_file,
    )
    from calibrated_agent_evaluation.cli.agentcore_route import (
        billed_call_budget,
        evaluate_concurrency,
    )

    if getattr(args, "cassette", None):
        raise UsageError(
            "--cassette replays a prompt judge's model calls; an AgentCore judge is scored by "
            "AgentCore, so it has none to replay"
        )
    try:
        loaded = load_agentcore_judge_file(path)
    except AgentCoreError as exc:
        raise CliError(str(exc)) from exc
    # Read before any client is built, so a bad value sends nothing.
    concurrency = evaluate_concurrency()
    clients = held.enter_context(
        agentcore.AwsClients.from_profile(
            region=args.region, profile=args.profile, handlers=[billed_call_budget()]
        )
    )
    judge = build_agentcore_judge(loaded, clients=clients)
    judge_file = JudgeFile(
        judge=judge, failure_mode=loaded.failure_mode, agent=loaded.agent, path=path
    )
    return AgentCoreCalibrateRun(
        args=args, judge_file=judge_file, clients=clients, concurrency=concurrency
    )


def open_agentcore_identity(
    args: argparse.Namespace, path: Path, held: contextlib.ExitStack
) -> JudgeFile:
    """The ``agentcore`` judge file's judge, built for its identity alone.

    Its clients, opened on ``held``, allow ``GetEvaluator`` and refuse every other AWS call, so a
    command that only looks up the judge's records can make no billed call.

    Raises:
        CliError: The judge file cannot be read.
        AgentCoreError: The evaluator (or a derived evaluator's base) cannot be read.
    """
    from calibrated_agent_evaluation import agentcore
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
    from calibrated_agent_evaluation.agentcore.judge import (
        build_agentcore_judge,
        load_agentcore_judge_file,
    )

    try:
        loaded = load_agentcore_judge_file(path)
    except AgentCoreError as exc:
        raise CliError(str(exc)) from exc
    guard = agentcore.ReadOnlyGuard({("bedrock-agentcore-control", "GetEvaluator")})
    clients = held.enter_context(
        agentcore.AwsClients.from_profile(
            region=args.region, profile=args.profile, handlers=[guard]
        )
    )
    judge = build_agentcore_judge(loaded, clients=clients)
    return JudgeFile(judge=judge, failure_mode=loaded.failure_mode, agent=loaded.agent, path=path)


def judge_file_for_identity(
    args: argparse.Namespace, path: str | Path, held: contextlib.ExitStack
) -> JudgeFile:
    """Load a judge file of either kind for a command that only reads the judge's records."""
    from calibrated_agent_evaluation.cli.judgefile import (
        declares_agentcore_backend,
        load_judge_file,
    )

    if declares_agentcore_backend(path):
        return open_agentcore_identity(args, Path(path), held)
    return load_judge_file(path)


def refuse_flags_without_agentcore(args: argparse.Namespace, paths: Sequence[str]) -> None:
    """Refuse the AWS flags unless one of ``paths`` is an ``agentcore`` judge file."""
    from calibrated_agent_evaluation.cli.judgefile import declares_agentcore_backend

    if not any(declares_agentcore_backend(path) for path in paths):
        refuse_agentcore_flags(args)
