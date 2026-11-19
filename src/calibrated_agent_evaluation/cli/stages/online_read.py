# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``online read``, ``publish`` and ``cost``: an attached online config after it ran.

``read`` reads the config's verdicts back into the store, checks each judge for staleness and,
with ``--drift-baseline STORE --drift-feature F``, the traffic for input drift, and stores a read
record naming each verdict it read. ``publish`` corrects each judge's online failure rate over
exactly those verdicts and publishes the pass rate and its interval to CloudWatch. ``cost`` counts
the evaluations the run made and prices them. Each needs the ``[agentcore]`` extra; ``read`` and
``publish`` need the statistics too.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.judge import AgentCoreJudgeFile
    from calibrated_agent_evaluation.agentcore.online_record import (
        OnlineEvaluationRecord,
        OnlineReadRecord,
    )
    from calibrated_agent_evaluation.agentcore.readback import ReadBackReport, ReadJudge
    from calibrated_agent_evaluation.agentcore.session import AwsClients
    from calibrated_agent_evaluation.backends.base import TraceStoreBackend
    from calibrated_agent_evaluation.calibrate.record import CalibrationRecord
    from calibrated_agent_evaluation.cli.drift import DriftOptions
    from calibrated_agent_evaluation.run import InputDriftReport
    from calibrated_agent_evaluation.schema.types import Label

_CONTROL = "bedrock-agentcore-control"
# The calls each action makes; its clients refuse any other.
_READ_OPERATIONS = frozenset(
    {
        (_CONTROL, "GetEvaluator"),
        ("logs", "StartQuery"),
        ("logs", "GetQueryResults"),
        ("logs", "FilterLogEvents"),
    }
)
# CloudWatch Omni's own service, which boto3 names ``cloudwatchomni`` (not ``logs``).
_OMNI_OPERATIONS = frozenset(
    {
        ("cloudwatchomni", "StartTelemetryQuerySession"),
        ("cloudwatchomni", "StartTelemetryQuery"),
        ("cloudwatchomni", "GetTelemetryQueryResults"),
        ("cloudwatchomni", "StopTelemetryQuerySession"),
    }
)
_PUBLISH_OPERATIONS = frozenset({(_CONTROL, "GetEvaluator"), ("cloudwatch", "PutMetricData")})
_COST_OPERATIONS = frozenset(
    {
        ("logs", "StartQuery"),
        ("logs", "GetQueryResults"),
        ("logs", "FilterLogEvents"),
        ("pricing", "GetProducts"),
    }
)


def online_read(args: argparse.Namespace) -> StageOutcome:
    """``calibrated-agent-evaluation online read``."""
    from calibrated_agent_evaluation import agentcore
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
    from calibrated_agent_evaluation.agentcore.online_record import OnlineReadRecord, new_read_id
    from calibrated_agent_evaluation.agentcore.readback import read_back
    from calibrated_agent_evaluation.cli.backends import local_root, open_backend
    from calibrated_agent_evaluation.cli.config import require_extras
    from calibrated_agent_evaluation.cli.drift import drift_options
    from calibrated_agent_evaluation.cli.records import append_records

    require_extras("scipy")
    root = local_root(args)
    record = _record(root, args.config)
    files = _matched_files(_judge_files(args.judge, "read"), record, root)
    since, until = _window(args, record)
    drift = drift_options(args)
    if args.seed is not None and drift.baseline is None:
        raise UsageError(
            "--seed sets the drift check's permutation test, so it needs --drift-baseline and "
            "--drift-feature; without them there is no drift check to seed"
        )
    if drift.baseline is not None and record.filtered:
        raise CliError(
            f"online config {record.config_id} has filters, so its sessions are not a random "
            "sample of the agent's traffic and drift is not measured on them; read it without "
            "--drift-baseline and --drift-feature"
        )
    if drift.baseline is not None:
        _check_baseline(drift.baseline)
    wait = 15 if args.wait_minutes is None else int(args.wait_minutes)
    if wait < 0:
        raise UsageError("--wait-minutes must be 0 or more")
    calibs = _calibrations(root)
    allowed = _READ_OPERATIONS | (_OMNI_OPERATIONS if args.omni else frozenset())
    backend_name = _backend_name(args)
    with contextlib.ExitStack() as held:
        clients = held.enter_context(
            agentcore.AwsClients.from_profile(
                region=args.region,
                profile=args.profile,
                handlers=[agentcore.ReadOnlyGuard(allowed)],
            )
        )
        judges = _read_judges(record, files, clients)
        backend = held.enter_context(open_backend(args))
        try:
            report = read_back(
                record,
                judges,
                clients=clients,
                backend=backend,
                since=since,
                until=until,
                omni=bool(args.omni),
                unscored_after_minutes=wait,
            )
        except AgentCoreError as exc:
            raise CliError(str(exc)) from exc
        gold_lines = _gold_lines(record, calibs, root, lambda: backend)
    read = OnlineReadRecord(
        read_id=new_read_id(record.config_id),
        config_id=record.config_id,
        backend=backend_name,
        since=since,
        until=until,
        verdicts=report.verdicts,
        errors=report.errors,
    )
    append_records(root, "agentcore-read", [read])

    summary = _read_summary(record, report, judges, since, until) + gold_lines
    detail: list[str] = [f"  read record {read.read_id}"]
    drift_plain, drift_detail, drift_report = _drift(args, drift, record, report)
    summary += drift_plain
    detail += drift_detail
    stale_plain, stale_detail = _staleness(record, judges, calibs, drift_report)
    summary += stale_plain
    detail += stale_detail
    summary.append(
        "Next: publish the corrected pass rate with `calibrated-agent-evaluation online publish "
        "--judge " + " --judge ".join(str(f.path) for f in files.values()) + "`."
    )
    return StageOutcome(
        stage="online", exit_code=EXIT_OK, summary="\n".join(summary), detail="\n".join(detail)
    )


def online_publish(args: argparse.Namespace) -> StageOutcome:
    """``calibrated-agent-evaluation online publish``."""
    from calibrated_agent_evaluation import agentcore
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
    from calibrated_agent_evaluation.agentcore.judge import build_agentcore_judge
    from calibrated_agent_evaluation.agentcore.metrics import publish_pass_rate
    from calibrated_agent_evaluation.cli.backends import local_root, open_backend
    from calibrated_agent_evaluation.cli.config import require_extras
    from calibrated_agent_evaluation.cli.records import load_records_of

    require_extras("scipy")
    root = local_root(args)
    record = _record(root, args.config)
    files = _matched_files(_judge_files(args.judge, "publish"), record, root)
    backend_name = _backend_name(args)
    reads = [
        r
        for r in load_records_of(root, "agentcore-read")
        if r.config_id == record.config_id and r.backend == backend_name
    ]
    if not reads:
        raise CliError(
            f"no read record of online config {record.config_id} for the {backend_name} store; "
            "run `calibrated-agent-evaluation online read` first, so there are online verdicts to "
            "correct"
        )
    calibs = _calibrations(root)
    summary: list[str] = []
    detail: list[str] = []
    refused = False
    with contextlib.ExitStack() as held:
        backend = held.enter_context(open_backend(args))
        labels = {
            e.evaluator_id: _named_verdicts(reads, e.evaluator_id, backend)
            for e in record.evaluators
        }
        missing = [e.calib_id for e in record.evaluators if e.calib_id not in calibs]
        if missing:
            raise CliError(
                f"calibration record(s) {', '.join(missing)}, which the online config was attached "
                "with, are not stored here; publish from the store root the judge was attached from"
            )
        golds = {
            e.evaluator_id: _gold_source(calibs[e.calib_id], root, lambda: backend)
            for e in record.evaluators
        }
        unknown = [
            e.evaluator_id for e in record.evaluators if golds[e.evaluator_id][0] == "unknown"
        ]
        if unknown:
            raise CliError(
                "the gold source of the calibration attached for evaluator "
                f"{', '.join(unknown)} is "
                "unknown: "
                "its session log records none and this store holds none of its gold, and "
                "publishing GoldSource=unknown would start a second CloudWatch series. Publish "
                "against the store `calibrated-agent-evaluation calibrate` graded the judge in, "
                "or calibrate the "
                "judge again with this version of `calibrated-agent-evaluation calibrate` so its "
                "session log records "
                "the source"
            )
        clients = held.enter_context(
            agentcore.AwsClients.from_profile(
                region=args.region,
                profile=args.profile,
                handlers=[agentcore.ReadOnlyGuard(_PUBLISH_OPERATIONS)],
            )
        )
        for attached in record.evaluators:
            try:
                judge = build_agentcore_judge(files[attached.evaluator_id], clients=clients)
            except AgentCoreError as exc:
                summary.append(f"Published nothing for evaluator {attached.evaluator_id}: {exc}.")
                refused = True
                continue
            calib = calibs[attached.calib_id]
            gold, gold_note = golds[attached.evaluator_id]
            try:
                published = publish_pass_rate(
                    record,
                    judge,
                    calib,
                    clients=clients,
                    rate_labels=labels[attached.evaluator_id],
                    gold_source=gold,
                    rate_cluster_ids=_sessions(labels[attached.evaluator_id], backend),
                )
            except AgentCoreError as exc:
                raise CliError(str(exc)) from exc
            count = len(labels[attached.evaluator_id])
            if published.refusal is not None:
                summary.append(
                    f"Published nothing for judge {judge.judge_id} ({count} online verdict(s)): "
                    f"{published.refusal}."
                )
                if published.detail:
                    detail.append(f"  {judge.judge_id}: {published.detail}")
                refused = True
                continue
            values = dict(published.published)
            summary.append(
                f"Published judge {judge.judge_id}'s corrected pass rate over {count} online "
                f"verdict(s) to CalibratedAgentEvaluation/Judges: CorrectedPassRate "
                f"{values['CorrectedPassRate']:.3f}, CorrectedPassRateLower95 "
                f"{values['CorrectedPassRateLower95']:.3f}, CorrectedPassRateUpper95 "
                f"{values['CorrectedPassRateUpper95']:.3f} (JudgeId {judge.judge_id}, "
                f"EvaluatorFingerprint {judge.fingerprint}, GoldSource {gold})."
            )
            if gold_note:
                summary.append(f"Gold source: {gold} {gold_note}.")
    if not refused:
        summary.append(
            "Set a CloudWatch alarm on CorrectedPassRateLower95 to hear when the agent's pass "
            "rate may have fallen."
        )
    return StageOutcome(
        stage="online",
        exit_code=EXIT_DOMAIN_ERROR if refused else EXIT_OK,
        summary="\n".join(summary),
        detail="\n".join(detail),
    )


def online_cost(args: argparse.Namespace) -> StageOutcome:
    """``calibrated-agent-evaluation online cost``."""
    from calibrated_agent_evaluation import agentcore
    from calibrated_agent_evaluation.agentcore.cost import online_run_cost
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
    from calibrated_agent_evaluation.agentcore.online import aws_error_code
    from calibrated_agent_evaluation.agentcore.pricing import lookup_evaluation_rates
    from calibrated_agent_evaluation.agentcore.readback import results_group
    from calibrated_agent_evaluation.agentcore.readers import run_logs_insights
    from calibrated_agent_evaluation.agentcore.sources import AgentCoreSourceConfig, TimeWindow
    from calibrated_agent_evaluation.cli.backends import local_root, open_backend
    from calibrated_agent_evaluation.cli.records import load_records_of

    root = local_root(args)
    record = _record(root, args.config)
    config = AgentCoreSourceConfig(
        runtime=record.runtime_service_name,
        window=TimeWindow(start=record.created_at, end=datetime.now(UTC).isoformat()),
    )
    with agentcore.AwsClients.from_profile(
        region=args.region,
        profile=args.profile,
        handlers=[agentcore.ReadOnlyGuard(_COST_OPERATIONS)],
    ) as clients:
        try:
            records = run_logs_insights(
                clients.client("logs"), results_group(record.config_id), config, cross_check=True
            )
        except AgentCoreError as exc:
            if aws_error_code(exc) != "ResourceNotFoundException":
                raise CliError(str(exc)) from exc
            records = []
        rates = None
        reason = None
        try:
            rates = lookup_evaluation_rates(clients)
        except AgentCoreError as exc:
            reason = str(exc)
    arns = {e.evaluator_arn for e in record.evaluators}
    scored = errors = other = 0
    for raw in records:
        attributes = raw.get("attributes") if isinstance(raw, dict) else None
        if not isinstance(attributes, dict) or raw.get("name") != "gen_ai.evaluation.result":
            continue
        # Read the evaluator as the read-back does: either attribute AgentCore may write.
        arn = attributes.get("aws.bedrock_agentcore.evaluator.arn") or attributes.get(
            "gen_ai.evaluation.evaluator.arn"
        )
        if arn is not None and arn not in arns:
            other += 1
        elif attributes.get("error.type") or attributes.get("error"):
            errors += 1
        else:
            scored += 1
    report = online_run_cost(
        record,
        load_records_of(root, "agentcore-session"),
        results=scored,
        error_results=errors,
        rates=rates,
        unpriced_reason=reason,
    )
    lines = report.describe()
    if other:
        lines.append(
            f"  {other} result(s) in the group name an evaluator this config did not attach and "
            "are not counted"
        )
    calibs = _calibrations(root)
    with contextlib.ExitStack() as held:
        opened: list[TraceStoreBackend] = []

        def store() -> TraceStoreBackend:
            # Opened once, and only if a calibration's session log records no gold source.
            if not opened:
                opened.append(held.enter_context(open_backend(args)))
            return opened[0]

        lines += _gold_lines(record, calibs, root, store)
    return StageOutcome(stage="online", exit_code=EXIT_OK, summary="\n".join(lines))


# --- shared --------------------------------------------------------------------------------------


def _record(root: Path, config_id: str | None) -> OnlineEvaluationRecord:
    """The online record ``config_id`` names, or the store's only one."""
    from calibrated_agent_evaluation.cli.records import load_records_of

    records = load_records_of(root, "agentcore-online")
    if config_id is not None:
        found = [r for r in records if config_id in (r.config_id, r.config_name)]
        if not found:
            raise CliError(
                f"no online config {config_id!r} is recorded under {root}; the recorded ones are "
                f"{', '.join(r.config_id for r in records) or 'none'}"
            )
        return found[0]
    if len(records) == 1:
        return records[0]
    if not records:
        raise CliError(
            f"no online config is recorded under {root}; attach one with "
            "`calibrated-agent-evaluation online attach`"
        )
    raise CliError(
        "more than one online config is recorded here; name one with --config "
        f"({', '.join(r.config_id for r in records)})"
    )


def _judge_files(paths: Sequence[str] | None, action: str) -> list[AgentCoreJudgeFile]:
    """The ``agentcore`` judge files named, read with no AWS call."""
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
    from calibrated_agent_evaluation.agentcore.judge import load_agentcore_judge_file
    from calibrated_agent_evaluation.cli.judgefile import declares_agentcore_backend

    files = []
    for path in paths or []:
        if not Path(path).is_file():
            raise CliError(f"cannot read judge file {path}: it does not exist")
        if not declares_agentcore_backend(path):
            raise CliError(
                f"{path} is not an agentcore judge file; {action} takes the judge files the "
                "config was attached with"
            )
        try:
            files.append(load_agentcore_judge_file(path))
        except AgentCoreError as exc:
            raise CliError(str(exc)) from exc
    return files


def _matched_files(
    files: Sequence[AgentCoreJudgeFile], record: OnlineEvaluationRecord, root: Path
) -> dict[str, AgentCoreJudgeFile]:
    """Each attached evaluator's judge file; refused unless they match one for one."""
    attached = {e.evaluator_id: e for e in record.evaluators}
    by_evaluator = {f.evaluator_id: f for f in files if f.evaluator_id in attached}
    problems = [
        f"judge file {f.path} names evaluator {f.evaluator_id}, which online config "
        f"{record.config_id} did not attach"
        for f in files
        if f.evaluator_id not in attached
    ]
    unnamed = [e for e in attached if e not in by_evaluator]
    if unnamed:
        problems.append(
            f"online config {record.config_id} attached evaluator(s) {', '.join(unnamed)}, and no "
            "--judge file names them"
        )
    if problems:
        raise CliError(
            "; ".join(problems) + "; name exactly the judge files the config was attached with"
        )
    calibs = _calibrations(root)
    for evaluator_id, judge_file in by_evaluator.items():
        calib = calibs.get(attached[evaluator_id].calib_id)
        if calib is None:
            raise CliError(
                f"the calibration record {attached[evaluator_id].calib_id} that evaluator "
                f"{evaluator_id} was attached with is not stored under {root}"
            )
        if calib.judge_id != judge_file.judge_id:
            raise CliError(
                f"judge file {judge_file.path} names evaluator {evaluator_id} as judge "
                f"{judge_file.judge_id!r}, but it was calibrated as {calib.judge_id!r}; use the "
                "judge file it was attached with"
            )
    return by_evaluator


def _calibrations(root: Path) -> dict[str, CalibrationRecord]:
    from calibrated_agent_evaluation.cli.records import load_records_of

    return {c.calib_id: c for c in load_records_of(root, "calibration")}


def _backend_name(args: argparse.Namespace) -> Literal["local", "langfuse"]:
    return "langfuse" if getattr(args, "backend", None) == "langfuse" else "local"


def _window(args: argparse.Namespace, record: OnlineEvaluationRecord) -> tuple[str, str]:
    """The read window: ``--since`` (default the config's creation) to ``--until`` (now)."""
    since = args.since or record.created_at
    until = args.until or datetime.now(UTC).isoformat()
    for flag, value in (("--since", since), ("--until", until)):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise UsageError(f"{flag} {value!r} is not an ISO-8601 time") from exc
        if parsed.utcoffset() is None:
            raise UsageError(f"{flag} {value!r} has no timezone; give one, for example +00:00")
    if datetime.fromisoformat(since) >= datetime.fromisoformat(until):
        raise UsageError(f"--since {since} is not before --until {until}")
    return since, until


def _read_summary(
    record: OnlineEvaluationRecord,
    report: ReadBackReport,
    judges: Mapping[str, ReadJudge],
    since: str,
    until: str,
) -> list[str]:
    fails = sum(1 for label in report.stored if label.outcome == "FAIL")
    lines = [
        f"Read {len(report.traces)} trace(s) of {record.runtime_service_name} from {since} to "
        f"{until} for online config {record.config_name} ({record.config_id}).",
        f"Stored {len(report.stored)} verdict(s) ({fails} FAIL, {len(report.stored) - fails} "
        f"PASS); {report.already_stored} already stored.",
    ]
    refused = [r for r in report.ingest.results if r.status != "ingested"]
    if refused:
        lines.append(
            f"The ingest did not store {len(refused)} trace(s); run with --explain for why."
        )
    if report.skipped_traces or report.skipped_sessions:
        lines.append(
            f"Left out {report.skipped_traces} trace(s) and {report.skipped_sessions} result(s) "
            "of the sessions the judges' calibration and bias runs sent."
        )
    for read_judge in judges.values():
        if not read_judge.status.certified:
            lines.append(f"{read_judge.status.reason}.")
    for evaluator_id, count in report.not_stored.items():
        lines.append(f"Stored none of evaluator {evaluator_id}'s {count} result(s).")
    if report.errors:
        kinds = ", ".join(f"{kind} {n}" for kind, n in sorted(report.errors.items()))
        lines.append(
            f"AgentCore could not score {sum(report.errors.values())} session(s): {kinds}."
        )
    else:
        lines.append("No error results.")
    if report.unreadable or report.outside_window:
        lines.append(
            f"{report.unreadable} record(s) could not be read as a verdict, and "
            f"{report.outside_window} result(s) were for traces outside this window."
        )
    if report.unscored_sessions:
        note = (
            ""
            if report.every_session_scored
            else " (the config does not score every session, so some were not sampled)"
        )
        lines.append(
            f"{len(report.unscored_sessions)} session(s) went idle long enough ago to have a "
            f"result and have none{note}."
        )
    if report.pending_sessions:
        lines.append(f"{len(report.pending_sessions)} session(s) may still be scored.")
    return lines


def _drift(
    args: argparse.Namespace,
    drift: DriftOptions,
    record: OnlineEvaluationRecord,
    report: ReadBackReport,
) -> tuple[list[str], list[str], InputDriftReport | None]:
    """The drift lines, plain and detailed, and the drift report (``None`` when not measured)."""
    from calibrated_agent_evaluation.cli.drift import measure_drift

    if record.filtered:
        return (
            [
                "Input drift is not measured on a filtered config's traffic: its sessions are "
                "not a random sample of the agent's traffic."
            ],
            [],
            None,
        )
    if drift.baseline is None:
        return (
            [
                "Input drift was not measured; to check the traffic against an earlier sample, "
                "pass --drift-baseline STORE --drift-feature input-chars|turns|tool-calls."
            ],
            [],
            None,
        )
    seed = 0 if args.seed is None else int(args.seed)
    feature = drift.feature or ""  # drift_options gives the feature with the baseline
    result = measure_drift(drift.baseline, feature, list(report.traces), seed=seed)
    return result.plain, result.detail, result.report


def _read_judges(
    record: OnlineEvaluationRecord,
    files: Mapping[str, AgentCoreJudgeFile],
    clients: AwsClients,
) -> dict[str, ReadJudge]:
    """Each attached evaluator as it is now: checked, then built as a judge when it can be."""
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
    from calibrated_agent_evaluation.agentcore.judge import build_agentcore_judge
    from calibrated_agent_evaluation.agentcore.online import aws_call_failed, check_attached
    from calibrated_agent_evaluation.agentcore.readback import ReadJudge

    try:
        statuses = check_attached(record, clients=clients)
    except AgentCoreError as exc:
        raise CliError(str(exc)) from exc
    judges: dict[str, ReadJudge] = {}
    for status in statuses:
        judge = None
        if status.state != "missing":
            try:
                judge = build_agentcore_judge(files[status.evaluator_id], clients=clients)
            except AgentCoreError as exc:
                if aws_call_failed(exc) or status.certified:
                    # A failed call (throttled, denied, timed out) says nothing about the
                    # evaluator; only one that can no longer be a judge is left unbuilt.
                    raise CliError(str(exc)) from exc
                status = replace(status, reason=f"{status.reason}; it cannot be built: {exc}")
        judges[status.evaluator_id] = ReadJudge(
            judge_file=files[status.evaluator_id], judge=judge, status=status
        )
    return judges


def _check_baseline(baseline: Path) -> None:
    """Refuse, before any AWS call, a drift baseline with no production-representative trace."""
    from calibrated_agent_evaluation.backends.local import LocalBackend

    with LocalBackend(root=baseline) as store:
        traces = store.query()
    if not any(t.frame == "production-representative" for t in traces):
        what = f"{len(traces)} trace(s), none of them" if traces else "no traces, so none"
        raise CliError(
            f"the drift baseline {baseline} holds {what} production-representative; drift "
            "compares two random samples, so the baseline needs an earlier random sample (for "
            "example the store an earlier `calibrated-agent-evaluation online read` wrote)"
        )


def _staleness(
    record: OnlineEvaluationRecord,
    judges: Mapping[str, ReadJudge],
    calibs: Mapping[str, CalibrationRecord],
    drift_report: InputDriftReport | None,
) -> tuple[list[str], list[str]]:
    from calibrated_agent_evaluation.cli.drift import staleness_lines
    from calibrated_agent_evaluation.run import check_staleness

    plain: list[str] = []
    decisions = []
    for attached in record.evaluators:
        read_judge = judges[attached.evaluator_id]
        if read_judge.judge is None:
            plain.append(
                f"Staleness of evaluator {attached.evaluator_id} is not checkable: "
                f"{read_judge.status.reason}."
            )
            continue
        decision = check_staleness(read_judge.judge, calibs[attached.calib_id], drift_report)
        if decision is not None:
            decisions.append(decision)
    detail, more = staleness_lines(decisions)
    return plain + more, detail


def _named_verdicts(
    reads: Sequence[OnlineReadRecord], evaluator_id: str, backend: TraceStoreBackend
) -> list[Label]:
    """The stored labels the read records name for ``evaluator_id``, one per trace."""
    from calibrated_agent_evaluation.agentcore.online_record import verdict_digest

    digests: dict[str, set[str]] = {}
    for read in reads:
        for verdict in read.verdicts:
            if verdict.evaluator_id == evaluator_id:
                digests.setdefault(verdict.trace_id, set()).add(verdict.verdict_digest)
    labels = []
    for trace_id, named in sorted(digests.items()):
        if len(named) > 1:
            raise CliError(
                f"the read records name {len(named)} different verdicts of evaluator "
                f"{evaluator_id} on trace {trace_id}; a trace counts once, so nothing was "
                "published"
            )
        (digest,) = named
        found = [label for label in backend.get_scores(trace_id) if verdict_digest(label) == digest]
        if not found:
            raise CliError(
                f"the verdict on trace {trace_id} a read record names is not in this store; "
                "publish from the store it was read into, or read it back again"
            )
        labels.append(found[0])
    return labels


def _sessions(labels: Sequence[Label], backend: TraceStoreBackend) -> list[str]:
    """Each verdict's session, so verdicts on traces of one session count as one cluster.

    One session can hold several traces, and their verdicts are not independent; a trace with
    no recorded session is its own cluster.
    """
    sessions = []
    for label in labels:
        trace = backend.get_trace(label.trace_id)
        sessions.append(trace.session_id or label.trace_id)
    return sessions


def _gold_source(
    calib: CalibrationRecord, root: Path, open_store: Callable[[], TraceStoreBackend]
) -> tuple[str, str]:
    """Where the calibration's gold came from, and a note when it was not recorded with it.

    The source ``calibrated-agent-evaluation calibrate`` recorded in the calibration's session log
    is named from any store. A log written before it was recorded has none, so the source is read
    from the gold this store holds for the calibration's traces and the note says so, or is
    ``unknown`` when the store holds none of it. ``open_store`` returns the store, so a command that
    needs no store opens one only for that fallback.
    """
    from calibrated_agent_evaluation.agentcore.calibrate import gold_source
    from calibrated_agent_evaluation.cli.records import load_records_of

    # The newest log for the calibration, as the newest calibration record is the one used.
    for log in reversed(load_records_of(root, "agentcore-session")):
        if log.run_kind == "calibration" and log.run_id == calib.calib_id and log.gold_source:
            return log.gold_source, ""
    store = open_store()
    gold = [
        label
        for trace_id in calib.trace_ids
        for label in store.get_scores(trace_id)
        if label.kind == "human"
    ]
    source = gold_source(gold)
    if source == "none":
        return "unknown", (
            "(this calibration's session log records none, and this store holds none of its gold)"
        )
    return source, "(read from this store's gold, not recorded at calibration)"


def _gold_lines(
    record: OnlineEvaluationRecord,
    calibs: Mapping[str, CalibrationRecord],
    root: Path,
    open_store: Callable[[], TraceStoreBackend],
) -> list[str]:
    """One ``Gold source:`` line per attached judge's calibration, naming where it came from."""
    lines = []
    for attached in record.evaluators:
        who = "" if len(record.evaluators) == 1 else f" for {attached.evaluator_id}"
        calib = calibs.get(attached.calib_id)
        if calib is None:
            lines.append(
                f"Gold source{who}: not checkable, since calibration record {attached.calib_id} "
                "is not stored here."
            )
            continue
        source, note = _gold_source(calib, root, open_store)
        lines.append(f"Gold source{who}: {source}" + (f" {note}." if note else "."))
    return lines
