# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentCore Gateway Policy decisions, read from the spans the Gateway writes to ``aws/spans``.

For each tool call an agent makes through a Gateway with a Policy engine attached, the Gateway
writes an ``AgentCore.Gateway.InvokeTool`` server span naming the tool (``tool.name``), and under
it an ``AgentCore.Policy.AuthorizeAction`` span holding the engine's decision, the gateway mode,
the determining policies and, for a policy in log-only mode inside an enforcing gateway, the
policies that would flip the decision. A ``tools/list`` call gets a
``PartiallyAuthorizeActions`` span instead, which records the tools the engine hid and is not a
decision on a call. With the whole gateway in log-only mode the engine evaluates asynchronously
and writes no ``AuthorizeAction`` span: the ``InvokeTool`` span says only ``LOG_ONLY_ASYNC``, so
those calls carry no readable decision.

:data:`AGENTCORE_POLICY_VOCABULARY` holds every attribute name and value spelling, pinned from a
recorded run against a real Gateway; :func:`extract_decisions` turns one trace's span records
into :class:`~calibrated_agent_evaluation.probe.policy.PolicyDecision` values, and refuses a span it
can't read whole. :class:`AgentCorePolicyDecisionSource` reads each trial's spans by its W3C trace
ID through the Logs Insights index, checked against the log store as the span reader checks it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreError,
    AgentCoreSettlingLag,
    AgentCoreTruncatedResult,
    PolicySpanMalformed,
)
from calibrated_agent_evaluation.agentcore.readers import (
    read_trace_events,
    require_otel_trace_id,
    run_logs_insights,
)
from calibrated_agent_evaluation.agentcore.runtime_errors import AgentReportedError
from calibrated_agent_evaluation.agentcore.sources import AgentCoreSourceConfig, TimeWindow
from calibrated_agent_evaluation.probe.errors import ProbeError
from calibrated_agent_evaluation.probe.policy import (
    AgentErrorTrial,
    Decision,
    GatewayMode,
    PolicyDecision,
    TrialDecisions,
    TrialRef,
)

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.run_manifest import RunRecord
    from calibrated_agent_evaluation.agentcore.session import AwsClients

__all__ = [
    "AGENTCORE_POLICY_VOCABULARY",
    "POLL_INTERVAL_SECONDS",
    "READ_WINDOW_PADDING_SECONDS",
    "SETTLE_BASE_SECONDS",
    "SETTLE_PER_TRIAL_SECONDS",
    "AgentCorePolicyDecisionSource",
    "PolicyVocabulary",
    "agent_reported_error_trial",
    "extract_decisions",
    "settle_deadline_for",
]

_LOGS_SERVICE = "logs"
_SPANS_LOG_GROUP = "aws/spans"
# Covers span timestamps that sit just outside the client's view of the call.
READ_WINDOW_PADDING_SECONDS = 60
# The wait for a battery's spans: a base, plus an allowance per trial, since one poll reads every
# trial still unsettled with two queries, and a large battery's first poll alone can take minutes.
SETTLE_BASE_SECONDS = 300.0
SETTLE_PER_TRIAL_SECONDS = 10.0
POLL_INTERVAL_SECONDS = 10.0


_SPAN_FORMAT_HINT = (
    "; look at that span in aws/spans, since AgentCore may have changed the span format this "
    "reader pins, and report it"
)


def settle_deadline_for(trials: int) -> float:
    """The default bound, in seconds, on the wait for ``trials`` trials' spans to settle."""
    return SETTLE_BASE_SECONDS + SETTLE_PER_TRIAL_SECONDS * trials


_RUNTIME_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


@dataclass(frozen=True)
class PolicyVocabulary:
    """The attribute names and value spellings of a Gateway's Policy decision spans."""

    selector_key: str
    per_call_value: str
    tool_list_value: str
    gateway_call_key: str
    gateway_call_value: str
    tool_key: str
    decision_key: str
    decisions: Mapping[str, Decision]
    asynchronous_decision: str
    mode_key: str
    modes: Mapping[str, GatewayMode]
    determining_key: str
    flipping_key: str
    incomplete_key: str
    engine_key: str
    reason_key: str


AGENTCORE_POLICY_VOCABULARY = PolicyVocabulary(
    selector_key="rpc.method",
    per_call_value="AuthorizeAction",
    tool_list_value="PartiallyAuthorizeActions",
    gateway_call_key="url.path",
    gateway_call_value="tools/call",
    tool_key="tool.name",
    decision_key="aws.agentcore.policy.authorization_decision",
    decisions={"ALLOW": "allow", "DENY": "deny"},
    asynchronous_decision="LOG_ONLY_ASYNC",
    mode_key="aws.agentcore.gateway.policy.mode",
    modes={"ENFORCE": "enforce", "LOG_ONLY": "log_only"},
    determining_key="aws.agentcore.policy.determining_policies",
    flipping_key="aws.agentcore.policy.log_only_decision_flipping_policies",
    incomplete_key="aws.agentcore.policy.log_only_eval_incomplete",
    engine_key="aws.agentcore.gateway.policy.arn",
    reason_key="aws.agentcore.policy.authorization_reason",
)


def _attributes(span: Mapping[str, Any]) -> Mapping[str, Any]:
    attrs = span.get("attributes")
    return attrs if isinstance(attrs, Mapping) else {}


def _names(span: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = _attributes(span).get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise PolicySpanMalformed(
            f"span {span.get('spanId')!r} of trace {span.get('traceId')!r} carries {key} as "
            f"{type(value).__name__}, not a list of strings; the decision is refused rather than "
            "guessed at" + _SPAN_FORMAT_HINT
        )
    return tuple(value)


def extract_decisions(
    spans: Iterable[Mapping[str, Any]],
    vocabulary: PolicyVocabulary = AGENTCORE_POLICY_VOCABULARY,
) -> tuple[PolicyDecision, ...]:
    """Every per-call Policy decision in one trace's span records, in the order the calls started.

    Spans without the decision attribute are skipped, and so are ``tools/list`` authorizations
    and the Gateway's own call spans, which repeat the decision their per-call child records; a
    call span whose child is missing, or whose value is unknown, is refused, and one that says a
    log-only gateway decided asynchronously is skipped. A span the store holds more than once is
    read once.

    Raises:
        PolicySpanMalformed: A span carries the decision attribute and is neither kind; or a
            per-call span's tool, decision or mode is missing or unknown, or a list attribute is
            in an unknown encoding. The message names the trace, the span and the attribute.
    """
    # The span store holds two or three records of one span (seen live), so each span counts once.
    by_id: dict[object, Mapping[str, Any]] = {}
    for span in spans:
        if isinstance(span, Mapping):
            by_id.setdefault((span.get("traceId"), span.get("spanId")), span)
    records = list(by_id.values())
    by_id = {span.get("spanId"): span for span in records}
    found: list[tuple[int, PolicyDecision]] = []
    for span in records:
        attrs = _attributes(span)
        if vocabulary.decision_key not in attrs:
            continue
        where = f"span {span.get('spanId')!r} of trace {span.get('traceId')!r}"
        selector = attrs.get(vocabulary.selector_key)
        if selector == vocabulary.tool_list_value:
            continue
        if selector != vocabulary.per_call_value:
            if attrs.get(vocabulary.gateway_call_key) == vocabulary.gateway_call_value:
                _check_gateway_call(span, attrs, records, vocabulary, where)
                continue
            raise PolicySpanMalformed(
                f"{where} carries {vocabulary.decision_key} but is neither a per-call "
                f"authorization ({vocabulary.selector_key}={vocabulary.per_call_value}) nor a "
                f"tools/list one ({vocabulary.tool_list_value}); it is refused rather than counted "
                "or dropped" + _SPAN_FORMAT_HINT
            )
        parent = by_id.get(span.get("parentSpanId"))
        tool = _attributes(parent).get(vocabulary.tool_key) if parent is not None else None
        if not isinstance(tool, str) or not tool:
            raise PolicySpanMalformed(
                f"{where} holds a Policy decision, but its parent Gateway span carries no "
                f"{vocabulary.tool_key}, so the decision can't be tied to a tool"
                + _SPAN_FORMAT_HINT
            )
        decision = vocabulary.decisions.get(str(attrs.get(vocabulary.decision_key)))
        mode = vocabulary.modes.get(str(attrs.get(vocabulary.mode_key)))
        if decision is None or mode is None:
            raise PolicySpanMalformed(
                f"{where} has {vocabulary.decision_key}={attrs.get(vocabulary.decision_key)!r} and "
                f"{vocabulary.mode_key}={attrs.get(vocabulary.mode_key)!r}; the decision must be "
                f"one of {sorted(vocabulary.decisions)} and the mode one of "
                f"{sorted(vocabulary.modes)}" + _SPAN_FORMAT_HINT
            )
        incomplete = attrs.get(vocabulary.incomplete_key, False)
        engine = attrs.get(vocabulary.engine_key)
        reason = attrs.get(vocabulary.reason_key)
        for key, value, wanted in (
            (vocabulary.incomplete_key, incomplete, bool),
            (vocabulary.engine_key, engine, (str, type(None))),
            (vocabulary.reason_key, reason, (str, type(None))),
        ):
            if not isinstance(value, wanted):
                raise PolicySpanMalformed(
                    f"{where} carries {key} as {type(value).__name__}; the decision is refused "
                    "rather than read with the value coerced" + _SPAN_FORMAT_HINT
                )
        try:
            record = PolicyDecision(
                trace_id=str(span.get("traceId") or "").lower(),
                span_id=str(span.get("spanId") or ""),
                tool=tool,
                decision=decision,
                mode=mode,
                determining_policies=_names(span, vocabulary.determining_key),
                log_only_flipping_policies=_names(span, vocabulary.flipping_key),
                log_only_incomplete=incomplete,
                engine=engine,
                reason=reason,
            )
        except ProbeError as exc:
            raise PolicySpanMalformed(f"{where} can't be read as a decision: {exc}") from exc
        started = span.get("startTimeUnixNano")
        found.append((started if isinstance(started, int) else 0, record))
    return tuple(record for _, record in sorted(found, key=lambda pair: pair[0]))


def _check_gateway_call(
    span: Mapping[str, Any],
    attrs: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    vocabulary: PolicyVocabulary,
    where: str,
) -> None:
    """Refuse a Gateway call span whose decision has no per-call record, or an unknown value.

    The Gateway's ``tools/call`` span repeats the decision its per-call child records, so it adds
    no decision of its own. It is skipped only when the engine decided asynchronously (a log-only
    gateway records no per-call decision); otherwise the child must be there, or the decision
    would be dropped.
    """
    value = attrs.get(vocabulary.decision_key)
    if value == vocabulary.asynchronous_decision:
        return
    if value not in vocabulary.decisions:
        raise PolicySpanMalformed(
            f"{where} carries {vocabulary.decision_key}={value!r}, which is neither "
            f"{', '.join(sorted(vocabulary.decisions))} nor {vocabulary.asynchronous_decision}; "
            "the call is refused rather than counted or dropped"
        )
    children = [
        s
        for s in records
        if s.get("parentSpanId") == span.get("spanId")
        and _attributes(s).get(vocabulary.selector_key) == vocabulary.per_call_value
    ]
    if not children:
        raise PolicySpanMalformed(
            f"{where} records the decision {value!r} but has no per-call "
            f"{vocabulary.per_call_value} span under it, so the decision can't be read whole; "
            "refused rather than dropped"
        )


def agent_reported_error_trial(exc: Exception) -> AgentErrorTrial | None:
    """The trial an agent-reported error stands for, or ``None`` for any other error."""
    if isinstance(exc, AgentReportedError) and exc.trace_id:
        return AgentErrorTrial(probe_trace_id=exc.trace_id, error_code=exc.error_code)
    return None


def _safe_runtime(runtime_arn: str) -> str:
    tail = runtime_arn.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    return "".join(c for c in tail if c in _RUNTIME_NAME_CHARS) or "runtime"


def _span_ids(spans: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    return frozenset(s["spanId"] for s in spans if isinstance(s.get("spanId"), str))


class AgentCorePolicyDecisionSource:
    """Reads each trial's Policy decisions from ``aws/spans`` by the trial's W3C trace ID.

    ``run_records`` is read when the decisions are read, never when the source is built: it maps
    each trial's probe trace ID to the run that sent it. Each poll reads every trial not yet
    settled through the Logs Insights index and straight from the log store. A trial settles when
    the two hold the same spans on two reads in a row, and is not read again. A trial with no span
    in either may simply not have reached the store yet, so it settles only at the bound, and then
    counts as a trial with no decision. The wait is bounded
    by ``settle_deadline_seconds``, by default :func:`settle_deadline_for` the number of trials;
    a trial whose last read before the bound matched still gets its confirming read. Every read is
    read-only, and none is billed.
    """

    def __init__(
        self,
        clients: AwsClients,
        *,
        run_records: Callable[[], Sequence[RunRecord]],
        settle_deadline_seconds: float | None = None,
        poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if poll_interval_seconds <= 0 or (
            settle_deadline_seconds is not None and settle_deadline_seconds < poll_interval_seconds
        ):
            raise AgentCoreError(
                f"poll_interval_seconds must be > 0 and settle_deadline_seconds at least that long "
                f"(got {poll_interval_seconds:g}s and {settle_deadline_seconds}s)"
            )
        self._clients = clients
        self._run_records = run_records
        self._deadline = settle_deadline_seconds
        self._interval = poll_interval_seconds
        self._clock = clock or (lambda: datetime.now(tz=UTC))
        # Looked up when called, so a test that replaces ``time.sleep`` still reaches it.
        self._sleep = sleep or (lambda seconds: time.sleep(seconds))
        self._monotonic = monotonic or (lambda: time.monotonic())

    def check_readable(self) -> None:
        """One read-only Logs Insights query on ``aws/spans`` over the last second.

        A truncated result counts as readable, since the query ran; any other failure, a missing
        ``logs:StartQuery`` permission among them, is raised before any trial is sent.
        """
        end = self._clock()
        window = TimeWindow(start=(end - timedelta(seconds=1)).isoformat(), end=end.isoformat())
        config = AgentCoreSourceConfig(runtime="runtime", window=window)
        try:
            run_logs_insights(self._clients.client(_LOGS_SERVICE), _SPANS_LOG_GROUP, config)
        except AgentCoreTruncatedResult:
            return

    def read_decisions(self, trials: Sequence[TrialRef]) -> Mapping[str, TrialDecisions]:
        """One answer per trial, keyed by probe trace ID, once every trial's spans have settled.

        Raises:
            ProbeError: A trial has no run record, or its record's W3C trace ID or times are
                unreadable.
            AgentCoreSettlingLag: Some trial's spans were still arriving at the deadline.
            PolicySpanMalformed: A decision span can't be read whole.
        """
        records = {r.trace_id: r for r in self._run_records()}
        runs: list[tuple[TrialRef, RunRecord, AgentCoreSourceConfig]] = []
        # Each trial's W3C trace ID as validated and normalized once, used for every read and for
        # its answer, so the reads and the extracted decisions name it the same way.
        w3c_ids: dict[str, str] = {}
        for trial in trials:
            record = records.get(trial.probe_trace_id)
            if record is None:
                raise ProbeError(
                    f"trial {trial.probe_trace_id} has no run record, so its W3C trace ID and "
                    "times are unknown and its spans can't be read"
                )
            try:
                w3c_ids[trial.probe_trace_id] = require_otel_trace_id(record.w3c_trace_id)
            except AgentCoreError as exc:
                raise ProbeError(
                    f"trial {trial.probe_trace_id}'s run record gives its W3C trace ID as "
                    f"{record.w3c_trace_id!r}, which is not 32 hex characters, so its spans "
                    "can't be read; run the battery again for fresh run records"
                ) from exc
            pad = timedelta(seconds=READ_WINDOW_PADDING_SECONDS)
            try:
                started = datetime.fromisoformat(record.started_at)
                ended = datetime.fromisoformat(record.ended_at)
            except ValueError as exc:
                raise ProbeError(
                    f"trial {trial.probe_trace_id}'s run record gives its start and end as "
                    f"{record.started_at!r} and {record.ended_at!r}, which are not ISO-8601 times, "
                    "so its spans can't be read; run the battery again for fresh run records"
                ) from exc
            window = TimeWindow(start=(started - pad).isoformat(), end=(ended + pad).isoformat())
            config = AgentCoreSourceConfig(runtime=_safe_runtime(record.runtime_arn), window=window)
            runs.append((trial, record, config))
        client = self._clients.client(_LOGS_SERVICE)
        bound = self._deadline if self._deadline is not None else settle_deadline_for(len(runs))
        deadline = self._monotonic() + bound
        settled: dict[str, list[dict[str, Any]]] = {}
        matched: dict[str, frozenset[str]] = {}
        pending = list(runs)
        confirming = False
        while pending:
            still: list[tuple[TrialRef, RunRecord, AgentCoreSourceConfig]] = []
            for run in pending:
                trial, record, config = run
                key = trial.probe_trace_id
                w3c = w3c_ids[key]
                index = _records(
                    run_logs_insights(client, _SPANS_LOG_GROUP, config, trace_id=w3c), w3c
                )
                store = _records(read_trace_events(client, _SPANS_LOG_GROUP, config, w3c), w3c)
                ids = _span_ids(index)
                if ids == _span_ids(store):
                    if not ids and self._monotonic() < deadline:
                        # Nothing in either yet: the spans may still be on their way to the store,
                        # so an empty trial is not taken as settled before the bound.
                        matched[key] = ids
                        still.append(run)
                        continue
                    if matched.get(key) == ids:
                        settled[key] = index
                        continue
                    matched[key] = ids
                else:
                    matched.pop(key, None)
                still.append(run)
            pending = still
            if not pending:
                break
            if confirming or self._monotonic() >= deadline:
                # Past the bound, a trial whose last read matched gets one confirming read; a trial
                # still behind, or one whose confirming read changed, is refused.
                unconfirmed = [
                    r[0].probe_trace_id for r in pending if r[0].probe_trace_id not in matched
                ]
                if confirming or unconfirmed:
                    lagging = unconfirmed or [r[0].probe_trace_id for r in pending]
                    raise AgentCoreSettlingLag(
                        f"the spans of {len(lagging)} trial(s) were still arriving at the "
                        f"{bound:g}s bound ({', '.join(lagging[:5])}); scoring them now could miss "
                        "decisions. Refused; run the battery again, since the index's lag "
                        "varies from run to run"
                    )
                confirming = True
            self._sleep(self._interval)
        return {
            trial.probe_trace_id: TrialDecisions(
                w3c_trace_id=w3c_ids[trial.probe_trace_id],
                decisions=extract_decisions(settled[trial.probe_trace_id]),
            )
            for trial, record, _ in runs
        }


def _records(rows: Sequence[Any], w3c: str) -> list[dict[str, Any]]:
    """The span records of one read, refusing one that is not a JSON object rather than guessing."""
    if any(not isinstance(row, dict) for row in rows):
        raise AgentCoreError(
            f"a span record read for trace {w3c!r} was not a JSON object; the read is refused "
            "rather than guessed at"
        )
    return list(rows)
