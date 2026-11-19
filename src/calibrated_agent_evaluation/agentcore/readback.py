# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read an online evaluation config's verdicts back into a Calibrated Agent Evaluation store.

AgentCore scores a deployed agent's sessions on its own schedule and writes each result to the
config's own result log group. :func:`read_back` pulls the agent's traces for a window through the
same AgentCore ingest as ``calibrated-agent-evaluation ingest --source agentcore``, reads the
config's results, and stores each one as an advisory ``llm-judge`` verdict stamped with the judge
that made it, so the correction and the publish step can use it as they use any judge's verdicts. It
skips the sessions Calibrated Agent Evaluation itself sent while calibrating and bias-testing the
judge, stores nothing for a result AgentCore could not score or for an evaluator that is no longer
the one certified, and counts each of those, together with the sessions that went idle long enough
ago to have a result and have none.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreError,
    AgentCoreNoPassDirection,
    AgentCoreScoreIncomplete,
    AgentCoreScoreOffScale,
)
from calibrated_agent_evaluation.agentcore.judge import AgentCoreJudge, AgentCoreJudgeFile
from calibrated_agent_evaluation.agentcore.online import AttachedStatus, aws_error_code
from calibrated_agent_evaluation.agentcore.online_record import (
    OnlineEvaluationRecord,
    ReadVerdict,
    verdict_digest,
)
from calibrated_agent_evaluation.agentcore.settle import SettlePolicy
from calibrated_agent_evaluation.agentcore.sources import AgentCoreReader, RawTrace
from calibrated_agent_evaluation.backends.base import TraceStoreBackend
from calibrated_agent_evaluation.ingest.core import IngestReport
from calibrated_agent_evaluation.schema.types import Frame, Label, Ownership, Trace

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.session import AwsClients

__all__ = ["ReadBackReport", "ReadJudge", "read_back", "results_group"]


@dataclass(frozen=True)
class ReadJudge:
    """One evaluator the config attached, as the caller named it and as it is now.

    Attributes:
        judge_file: The ``agentcore`` judge file the caller named for it, which carries the judge
            id and the failure mode its verdicts are about.
        judge: The judge built from the evaluator, or ``None`` when it is missing or can no
            longer be built as a judge.
        status: Whether the evaluator is still the one certified.
    """

    judge_file: AgentCoreJudgeFile
    judge: AgentCoreJudge | None
    status: AttachedStatus


@dataclass(frozen=True)
class ReadBackReport:
    """What one read-back read and stored.

    Attributes:
        config_id: The online config read.
        ingest: The trace pull's own report.
        traces: The agent's traces in the window, as stored.
        stored: The verdicts newly stored.
        already_stored: Verdicts read that the store already held.
        verdicts: Every verdict stored or found already stored, by evaluator, trace and identity
            digest, for the read record that ``calibrated-agent-evaluation online publish`` corrects
            over.
        skipped_traces: Traces of sessions the judge's calibration or bias run sent, left out of
            the pull so they are neither stored nor counted as the agent's traffic.
        skipped_sessions: Results for those sessions, likewise not stored.
        not_stored: Per evaluator, results not stored because it is not the one certified.
        errors: Per AgentCore error type, results it could not score (``unknown`` when it named
            none).
        unreadable: Records that are not a result, or that named no trace, no attached
            evaluator, no verdict or a value off the evaluator's scale.
        outside_window: Results for traces this read did not pull (outside the window, or refused
            by the ingest).
        unscored_sessions: Sessions idle longer than the session timeout plus the wait, with no
            result.
        pending_sessions: Sessions with no result yet that are still inside that time.
        every_session_scored: Whether the config scores every session (100% sampling and no
            filters); when it does not, a session with no result may simply not have been
            sampled, so ``unscored_sessions`` is an upper bound on what went unscored.
    """

    config_id: str
    ingest: IngestReport
    traces: tuple[Trace, ...]
    stored: tuple[Label, ...]
    already_stored: int
    verdicts: tuple[ReadVerdict, ...]
    skipped_traces: int
    skipped_sessions: int
    not_stored: dict[str, int] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)
    unreadable: int = 0
    outside_window: int = 0
    unscored_sessions: tuple[str, ...] = ()
    pending_sessions: tuple[str, ...] = ()
    every_session_scored: bool = True


def results_group(config_id: str) -> str:
    """The log group AgentCore writes an online config's results to."""
    return f"/aws/bedrock-agentcore/evaluations/results/{config_id}"


class _SessionRecorder:
    """A reader that notes each trace's sessions and last span end.

    It holds back the traces of sessions Calibrated Agent Evaluation sent itself, which are not the
    agent's traffic.
    """

    def __init__(self, reader: AgentCoreReader, excluded: set[str]) -> None:
        self._reader = reader
        self._excluded = excluded
        self.sessions: dict[str, set[str]] = {}
        self.ended: dict[str, datetime | None] = {}
        self.held_back = 0

    def read_traces(self) -> Iterator[RawTrace]:
        for raw in self._reader.read_traces():
            sessions = {s for s in raw.session_ids if s.strip()}
            if sessions & self._excluded:
                self.held_back += 1
                continue
            self.sessions[raw.trace_id] = sessions
            self.ended[raw.trace_id] = _last_end(raw.span_records)
            yield raw

    def read_scores(self) -> Iterator[dict[str, Any]]:
        return iter(())


def read_back(
    record: OnlineEvaluationRecord,
    judges: Mapping[str, ReadJudge],
    *,
    clients: AwsClients,
    backend: TraceStoreBackend,
    since: str,
    until: str,
    omni: bool = False,
    unscored_after_minutes: int = 15,
    reader: AgentCoreReader | None = None,
    settle: SettlePolicy | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ReadBackReport:
    """Read ``record``'s results for the window ``[since, until]`` into ``backend``.

    The agent's traces are pulled for the window (Logs Insights, or CloudWatch Omni with ``omni``)
    and stamped ``production-representative`` when the config has no filters, since its sessions
    are then a random sample of the agent's traffic. The results come from the config's own
    result group only. Each result for an attached, certified evaluator, on a trace this read
    pulled, is stored as an advisory verdict read with its judge's scale and direction and
    stamped with ``<judge_id>@<fingerprint>`` and the judge file's failure mode; a verdict already
    stored is not stored twice.

    Args:
        record: The online config, as ``calibrated-agent-evaluation online attach`` recorded it.
        judges: Each attached evaluator id's :class:`ReadJudge`; every attached evaluator must be
            there.
        clients: The AWS clients.
        backend: The store the traces and verdicts go to.
        since: The window's start, timezone-aware ISO-8601.
        until: The window's end, likewise.
        omni: Read the traces through CloudWatch Omni instead of Logs Insights.
        unscored_after_minutes: How long after a session's timeout its result may still be
            coming before the session counts as unscored.
        reader: The trace reader (built from ``clients`` when not given).
        settle: How long to wait for the window's writes to settle (from the environment when not
            given).
        clock: The time now, which the results are never read past.

    Raises:
        AgentCoreError: An attached evaluator has no entry in ``judges``, or a read failed.
    """
    from calibrated_agent_evaluation.agentcore.ingest import ingest_agentcore
    from calibrated_agent_evaluation.agentcore.readers import (
        LogsInsightsReader,
        OmniReader,
        eval_record_to_result,
        run_logs_insights,
    )
    from calibrated_agent_evaluation.agentcore.sources import (
        AgentCoreSourceConfig,
        AgentCoreTraceSource,
        TimeWindow,
    )

    missing = [e.evaluator_id for e in record.evaluators if e.evaluator_id not in judges]
    if missing:
        raise AgentCoreError(
            f"no judge was given for evaluator(s) {', '.join(missing)}, which online config "
            f"{record.config_id} attached; name each attached judge's file"
        )
    config = AgentCoreSourceConfig(
        runtime=record.runtime_service_name,
        window=TimeWindow(start=since, end=until),
        omni=omni,
    )
    if reader is None:
        reader = OmniReader(clients, config) if omni else LogsInsightsReader(clients, config)
    excluded = set(record.excluded_session_ids)
    recorder = _SessionRecorder(reader, excluded)
    written = _WrittenTraces(backend)
    ingest = ingest_agentcore(
        AgentCoreTraceSource(recorder, config, settle=settle),
        backend=written,
        frame=None if record.filtered else "production-representative",
    )
    pulled = {r.trace_id for r in ingest.results if r.status == "ingested" and r.trace_id}
    # The traces as written, not read back: an eventually consistent store (Langfuse) cannot
    # return a trace for a few seconds after confirming its write.
    traces = tuple(written.traces[trace_id] for trace_id in sorted(pulled))

    # A session near the window's end is scored up to the timeout plus the wait later, so its
    # result is read up to then (never past now): its trace is in this window, not the next.
    after = timedelta(minutes=record.session_timeout_minutes + unscored_after_minutes)
    window_end = datetime.fromisoformat(until)
    results_end = min(window_end + after, clock())
    results_config = config.model_copy(
        update={"window": TimeWindow(start=since, end=max(results_end, window_end).isoformat())}
    )
    try:
        records = run_logs_insights(
            clients.client("logs"),
            results_group(record.config_id),
            results_config,
            cross_check=True,
        )
    except AgentCoreError as exc:
        # AgentCore creates the result group with the config's first result.
        if aws_error_code(exc) != "ResourceNotFoundException":
            raise
        records = []

    by_arn = {e.evaluator_arn: judges[e.evaluator_id] for e in record.evaluators}
    stored: list[Label] = []
    verdicts: list[ReadVerdict] = []
    already = skipped = unreadable = outside = 0
    not_stored: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    scored_sessions: set[str] = set()
    known: dict[str, set[Any]] = {}
    for raw_record in records:
        attributes = raw_record.get("attributes") if isinstance(raw_record, dict) else None
        if (
            not isinstance(raw_record, dict)
            or raw_record.get("name") != "gen_ai.evaluation.result"
            or not isinstance(attributes, dict)
        ):
            unreadable += 1
            continue
        session = attributes.get("session.id")
        # The library's own sessions are left out first, whatever their result, so an error on one
        # of them is not counted as an error in the agent's traffic.
        if session in excluded:
            skipped += 1
            continue
        if attributes.get("error.type") or attributes.get("error"):
            # Read from the raw record: the mapping below drops an error record without a count.
            # AgentCore did try to score the session, so it is not unscored.
            errors[str(attributes.get("error.type") or "unknown")] += 1
            if isinstance(session, str):
                scored_sessions.add(session)
            continue
        result = eval_record_to_result(raw_record)
        read_judge = by_arn.get(str((result or {}).get("evaluatorArn") or ""))
        if result is None or read_judge is None:
            unreadable += 1
            continue
        if isinstance(session, str):
            scored_sessions.add(session)
        if read_judge.judge is None or not read_judge.status.certified:
            not_stored[read_judge.status.evaluator_id] += 1
            continue
        label = _verdict(result, read_judge.judge, read_judge.judge_file)
        if label is None:
            unreadable += 1
            continue
        if label.trace_id not in pulled:
            outside += 1
            continue
        if not _store(label, backend, known, stored):
            already += 1
        verdicts.append(
            ReadVerdict(
                evaluator_id=read_judge.status.evaluator_id,
                trace_id=label.trace_id,
                verdict_digest=verdict_digest(label),
            )
        )

    unscored, pending = _unscored(
        recorder, pulled, excluded, scored_sessions, until=results_end, after=after
    )
    return ReadBackReport(
        config_id=record.config_id,
        ingest=ingest,
        traces=traces,
        stored=tuple(stored),
        already_stored=already,
        verdicts=tuple(verdicts),
        skipped_traces=recorder.held_back,
        skipped_sessions=skipped,
        not_stored=dict(not_stored),
        errors=dict(errors),
        unreadable=unreadable,
        outside_window=outside,
        unscored_sessions=unscored,
        pending_sessions=pending,
        every_session_scored=record.sampling_percentage >= 100.0 and not record.filtered,
    )


class _WrittenTraces:
    """``backend``, keeping each trace written through it, so they need not be read back."""

    def __init__(self, backend: TraceStoreBackend) -> None:
        self._backend = backend
        self.traces: dict[str, Trace] = {}

    def put_trace(self, trace: Trace) -> None:
        self._backend.put_trace(trace)
        self.traces[trace.trace_id] = trace

    def get_trace(self, trace_id: str) -> Trace:
        return self._backend.get_trace(trace_id)

    def query(
        self,
        frame: Frame | None = None,
        cell: dict[str, str] | None = None,
        ownership: Ownership | None = None,
    ) -> list[Trace]:
        return self._backend.query(frame, cell, ownership)

    def put_score(self, label: Label) -> None:
        self._backend.put_score(label)

    def get_scores(self, trace_id: str, span_id: str | None = None) -> list[Label]:
        return self._backend.get_scores(trace_id, span_id)

    def retract_score(self, label: Label) -> None:
        self._backend.retract_score(label)


def _verdict(
    result: dict[str, Any], judge: AgentCoreJudge, judge_file: AgentCoreJudgeFile
) -> Label | None:
    """The result as the judge's verdict, or ``None`` when it names no trace or carries none."""
    from calibrated_agent_evaluation.agentcore.scores import result_to_label
    from calibrated_agent_evaluation.judges.verdict import verdict_rater_id

    try:
        label = result_to_label(result, rating_scale=judge.rating_scale, inverted=judge.inverted)
    except (AgentCoreScoreIncomplete, AgentCoreNoPassDirection, AgentCoreScoreOffScale):
        return None
    return label.model_copy(
        update={"rater_id": verdict_rater_id(judge), "failure_mode": judge_file.failure_mode}
    )


def _store(
    label: Label, backend: TraceStoreBackend, known: dict[str, set[Any]], stored: list[Label]
) -> bool:
    """Store ``label`` unless the store already holds a verdict with its identity; say which."""
    from calibrated_agent_evaluation.schema.serde import label_identity

    seen = known.get(label.trace_id)
    if seen is None:
        seen = {label_identity(existing) for existing in backend.get_scores(label.trace_id)}
        known[label.trace_id] = seen
    identity = label_identity(label)
    if identity in seen:
        return False
    backend.put_score(label)
    seen.add(identity)
    stored.append(label)
    return True


def _unscored(
    recorder: _SessionRecorder,
    pulled: set[str],
    excluded: set[str],
    scored: set[str],
    *,
    until: datetime,
    after: timedelta,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The sessions with no result: those that should have one by ``until``, and the rest."""
    last: dict[str, datetime | None] = {}
    for trace_id in pulled:
        ended = recorder.ended.get(trace_id)
        for session in recorder.sessions.get(trace_id, ()):
            if session in excluded or session in scored:
                continue
            previous = last.get(session)
            # A session ends when its last trace does; an unknown end is taken as still open.
            if session not in last or (
                ended is not None and previous is not None and ended > previous
            ):
                last[session] = ended
            elif ended is None:
                last[session] = None
    unscored = sorted(
        s for s, ended in last.items() if ended is not None and ended + after <= until
    )
    pending = sorted(s for s in last if s not in unscored)
    return tuple(unscored), tuple(pending)


def _last_end(spans: Sequence[dict[str, Any]]) -> datetime | None:
    """When the last of ``spans`` ended, or ``None`` when none says."""
    ends = []
    for span in spans:
        raw = span.get("endTimeUnixNano")
        if isinstance(raw, (int, str)):
            try:
                ends.append(int(raw))
            except ValueError:
                continue
    if not ends:
        return None
    return datetime.fromtimestamp(max(ends) / 1_000_000_000, tz=UTC)
