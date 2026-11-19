# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an AgentCore evaluator over a run of traces as an advisory judge.

The Evaluate invoker: it sends one call per trace (the service scores one session per call, and each
trace is its own session), up to a caller-given number at a time, and maps each result to an
advisory ``Label`` (reusing the shared evaluator-result mapping), addressed by the original trace
and span id; the calibration engine's clustered path groups span-level verdicts by that trace id.

Vendor boundary: the boto3 ``Evaluate`` call lives here, inside the AgentCore package;
the returned ``Label`` list is the only thing that crosses back into the methodology plane. The ids
on the request are conforming ids derived from our own (the Evaluate target requires an exact
length), so each result is mapped back to the original trace/span id before its label is stored.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreAmbiguousSessionResult,
    AgentCoreContentUnserializable,
    AgentCoreError,
    AgentCoreEvaluatorResultError,
    AgentCoreMixedSession,
    AgentCoreReferenceDropped,
    AgentCoreResultUnaddressable,
    AgentCoreRunDeclined,
    AgentCoreScoreIncomplete,
    BilledCallBudgetExceeded,
)
from calibrated_agent_evaluation.agentcore.evaluate_payload import (
    SYNTHETIC_TIMING_ATTRIBUTE,
    aws_span_id,
    aws_trace_id,
    conversation_session_id,
    conversation_to_evaluation_input,
    conversation_turn_trace_ids,
    trace_to_evaluation_input,
)
from calibrated_agent_evaluation.agentcore.judge import require_runnable_level, scores_inverted
from calibrated_agent_evaluation.agentcore.pricing import (
    EvaluationRates,
    RunEstimate,
    estimate_run,
    lookup_evaluation_rates,
    unpriced_reason,
    usage_cost,
)
from calibrated_agent_evaluation.agentcore.scores import result_to_label

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from typing import Any

    from calibrated_agent_evaluation.agentcore.judge import AgentCoreJudge
    from calibrated_agent_evaluation.agentcore.session import AwsClients
    from calibrated_agent_evaluation.backends.base import TraceStoreBackend
    from calibrated_agent_evaluation.schema.types import Label, Trace

    # A trace to score, its serialized span documents, whether any carries a synthetic time, the
    # session id those documents carry, and (for a session judge) the reference inputs that carry
    # its assertions.
    _Request = tuple[Trace, list[dict[str, Any]], bool, str, list[dict[str, Any]]]
logger = logging.getLogger(__name__)

_EVALUATE_SERVICE = "bedrock-agentcore"

MAX_CONCURRENCY = 10
"""The most ``Evaluate`` calls a run sends at once, and what the commands send by default.

It is the number of connections a boto3 client keeps pooled by default (``max_pool_connections``);
more concurrent calls on one client would open connections and throw them away afterwards."""

CONCURRENCY_ENV_VAR = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_CONCURRENCY"


def concurrency_from_env(environ: Mapping[str, str]) -> int:
    """The Evaluate concurrency the environment sets, else :data:`MAX_CONCURRENCY`.

    The variable is ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_CONCURRENCY``.

    Raises:
        AgentCoreError: The variable is set to anything but a whole number from 1 to
            :data:`MAX_CONCURRENCY`.
    """
    raw = environ.get(CONCURRENCY_ENV_VAR, "").strip()
    if not raw:
        return MAX_CONCURRENCY
    # Plain ASCII digits only: int() would also read "1_0", "+4" or other scripts' digits.
    value = int(raw) if raw.isascii() and raw.isdigit() else 0
    if not 1 <= value <= MAX_CONCURRENCY:
        raise AgentCoreError(
            f"{CONCURRENCY_ENV_VAR}={raw!r} is not a whole number from 1 to {MAX_CONCURRENCY}; "
            "set it in that range, or unset it to send up to "
            f"{MAX_CONCURRENCY} Evaluate calls at a time"
        )
    return value


@dataclass(frozen=True)
class AgentCoreJudgeRun:
    """The verdicts of one run and the session each of its ``Evaluate`` calls named.

    Attributes:
        labels: The advisory labels, in trace order.
        session_ids: The ``session.id`` each call's span documents carried, one per trace, in
            trace order, read from the requests as sent.
    """

    labels: tuple[Label, ...]
    session_ids: tuple[str, ...]


def run_agentcore_judge(
    judge: AgentCoreJudge,
    targets: Sequence[Trace],
    *,
    clients: AwsClients,
    confirm: Callable[[RunEstimate], bool],
    store: TraceStoreBackend | None = None,
    max_concurrency: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> list[Label]:
    """Score ``targets`` with ``judge`` and return one advisory ``Label`` per evaluator result.

    The labels of :func:`run_agentcore_judge_with_sessions`, which documents the run.
    """
    run = run_agentcore_judge_with_sessions(
        judge,
        targets,
        clients=clients,
        confirm=confirm,
        store=store,
        max_concurrency=max_concurrency,
        progress=progress,
    )
    return list(run.labels)


def run_agentcore_judge_with_sessions(
    judge: AgentCoreJudge,
    targets: Sequence[Trace],
    *,
    clients: AwsClients,
    confirm: Callable[[RunEstimate], bool],
    store: TraceStoreBackend | None = None,
    max_concurrency: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> AgentCoreJudgeRun:
    """Score ``targets`` with ``judge``: the advisory labels, and the session each call named.

    Sends one ``Evaluate`` call per trace, carrying only that trace's spans and that trace as the
    only target: the service refuses an input whose spans belong to more than one session, and each
    trace is its own session. Each result addresses a conforming id, mapped back to the original
    trace and span id before the advisory label is built; when ``store`` is given, the labels are
    persisted after every call has mapped. Every target is serialized before the first call, so a
    trace that cannot be sent refuses the run at no cost. A trace sent with any synthetic time (its
    tool calls, which never have stored times, or every span of a trace with no timing) has each of
    its verdicts marked ``timing=synthetic`` in its critique, and the run logs one warning with the
    count. Every verdict's critique also names the judge's fingerprint, ``[fingerprint=<16 hex>]``,
    since the evaluator ARN does not change when the evaluator is edited.

    Up to ``max_concurrency`` calls are in flight at once (1 to :data:`MAX_CONCURRENCY`; the
    default 1 sends them one after another), and the labels come back in trace order whatever order
    the calls end in. ``progress``, if given, is called from the calling thread after each call
    ends, as ``progress(scored, total)`` with ``scored`` strictly increasing. When a call fails, no
    further call is sent, the calls already in flight end and are counted in the usage log, and the
    first failure seen is raised; a ``ThrottlingException`` names the limit used and, above one call
    at a time, the concurrency to lower (at one, to wait and run again). The
    session id each call carried is read from its span documents, so the ids returned are the ones
    AWS saw, whatever the serializer puts there.

    Before the first billed call, the budget installed on ``clients`` must fit one call per trace,
    or the run is refused with :class:`BilledCallBudgetExceeded` before anything is asked. Then
    ``confirm`` receives the run's :class:`RunEstimate` (calls, approximate tokens, and the cost
    at the AWS Price List rates, or why there is none) and returns whether to go ahead; a declined
    run raises :class:`AgentCoreRunDeclined` and makes no call. A non-interactive approval is a
    ``confirm`` that returns true. Once a billed call has been made, the tokens AgentCore reported
    are logged when the run ends, whether it completed or was refused part-way.

    The estimate's cost is read from the AWS Price List API at the Tier1 rates. A built-in
    evaluator is priced per estimated token, model cost included; a custom evaluator is priced per
    evaluation, and the model usage (or Lambda) it is billed for on top is not included; a
    third-party or unknown evaluator type has no rate. If the lookup fails (no permission, a
    response that does not parse) or the region lists no rate, the run goes on: the estimate shows
    the calls and tokens with no cost and says why (see
    :func:`~calibrated_agent_evaluation.agentcore.pricing.estimate_run`).

    Each of these refuses the run, and no label from it is written:
    :class:`AgentCoreContentUnserializable` if a trace cannot be sent, or its documents carry no
    session id, and :class:`AgentCoreMixedSession` if they carry several (before any billed call);
    :class:`AgentCoreError` if ``max_concurrency`` is outside 1 to :data:`MAX_CONCURRENCY` (before
    anything is serialized);
    :class:`AgentCoreError` if the account is unreachable; :class:`AgentCoreEvaluatorResultError` if
    a result carries an ``errorCode`` (checked before anything else on it);
    :class:`AgentCoreReferenceDropped` if a result reports ignored reference inputs;
    :class:`AgentCoreResultUnaddressable` if a result names a trace its call did not target, a
    session that disagrees with its trace, or a span that is not a span of the trace it names;
    :class:`AgentCoreScoreIncomplete` if a result has no value and no label, or no address, or a
    call returns no result for its trace; and :class:`AgentCoreScoreOffScale` if a value is off the
    judge's published scale. :class:`AgentCoreNoPassDirection` (no pass direction) and
    :class:`AgentCoreEvaluatorLevelUnsupported` (a level other than ``TRACE``) are raised before any
    billed call. Each ``Evaluate`` call is sent once, with the SDK's retries off: a retry after a
    5xx or a read timeout could be billed again while the budget counts the call once, so such a
    failure refuses the run instead. On clients not already set to one attempt, each call waits up
    to :data:`EVALUATE_READ_TIMEOUT_SECONDS` for its response; clients set to one attempt keep the
    wait they set.

    A judge whose ``level`` is ``session`` runs a ``SESSION`` evaluator, such as
    ``Builtin.GoalSuccessRate``, over simulated-conversation traces, one call per conversation,
    under the same budget, estimate, confirmation, one-attempt and store-only-when-all-mapped
    rules. Each conversation is sent as one session with no ``evaluationTarget``, each turn under
    its own trace id, and its assertions as the session's reference input; its one result, which
    names only the session, becomes one label on the conversation's trace. A trace that is not a
    conversation is refused with :class:`AgentCoreNotAConversation`, one with a blank assertion
    with :class:`AgentCoreBlankAssertion`, and one past an ``Evaluate`` input limit with
    :class:`AgentCoreEvaluateLimitExceeded`, all before any billed call; a result naming another
    session is refused with :class:`AgentCoreResultUnaddressable`, and a call answered with more
    than one result with :class:`AgentCoreAmbiguousSessionResult`.
    """
    if not 1 <= max_concurrency <= MAX_CONCURRENCY:
        raise AgentCoreError(
            f"judge {judge.judge_id!r}: max_concurrency={max_concurrency} is not from 1 to "
            f"{MAX_CONCURRENCY}; refused before any call"
        )
    # Checked here as well as in the factory, so a judge built any other way is still refused
    # before a billed call when its evaluator has no pass direction or a level Calibrated Agent
    # Evaluation cannot send.
    scores_inverted({judge.evaluator_id})
    require_runnable_level(
        judge.evaluator_level, evaluator_id=judge.evaluator_id, level=judge.level
    )
    session = judge.level == "session"

    # Serialize every target before the first billed call, so a trace that cannot be sent (no
    # content, partial timing, a parent outside an untimed trace, not a conversation) refuses the
    # run at no cost.
    requests: list[_Request] = []
    for trace in targets:
        references: list[dict[str, Any]] = []
        if session:
            payload, references = conversation_to_evaluation_input(trace)
            documents = payload["sessionSpans"]
        else:
            documents = trace_to_evaluation_input(trace)["sessionSpans"]
        synthetic = any(doc["attributes"].get(SYNTHETIC_TIMING_ATTRIBUTE) for doc in documents)
        requests.append((trace, documents, synthetic, _session_of(trace, documents), references))
    synthetic_count = sum(synthetic for _, _, synthetic, _, _ in requests)
    if synthetic_count:
        logger.warning(
            "%d of %d traces had no step timestamps or carry tool calls, which never have them; "
            "Calibrated Agent Evaluation used synthetic times so AgentCore can score them, and "
            "each of their verdicts "
            "is marked timing=synthetic",
            synthetic_count,
            len(requests),
        )

    if not requests:
        return AgentCoreJudgeRun(labels=(), session_ids=())
    _require_budget(clients, calls=len(requests), judge_id=judge.judge_id)
    rates, no_rate_reason = _rates_for(judge.evaluator_type, clients)
    estimate = estimate_run(
        evaluator_id=judge.evaluator_id,
        evaluator_type=judge.evaluator_type,
        documents_per_trace=[documents for _, documents, _, _, _ in requests],
        rates=rates,
        no_rate_reason=no_rate_reason,
    )
    if not confirm(estimate):
        raise AgentCoreRunDeclined(
            f"judge {judge.judge_id!r}: the run was declined at its estimate, so no billed call "
            f"was made ({estimate.describe()})"
        )

    usage = _Usage()
    with _one_attempt(clients) as sender:
        try:
            # Fetched once here: AwsClients is not thread-safe, a botocore client is.
            labels = _score_all(
                judge,
                requests,
                client=sender.client(_EVALUATE_SERVICE),
                usage=usage,
                max_concurrency=max_concurrency,
                progress=progress,
            )
        finally:
            if usage.calls:
                _log_usage(judge, usage, rates)
    # Store only once every call has mapped cleanly, so a run that raises (an unreachable account,
    # or a flagged, incomplete or off-scale result on a later call) leaves the store as it found
    # it, matching the caller, who gets no labels back.
    if store is not None:
        for label in labels:
            store.put_score(label)
    logger.info(
        "ran AgentCore judge %s over %d target(s): %d advisory label(s)",
        judge.judge_id,
        len(requests),
        len(labels),
    )
    return AgentCoreJudgeRun(
        labels=tuple(labels), session_ids=tuple(sent for _, _, _, sent, _ in requests)
    )


def _session_of(trace: Trace, documents: list[dict[str, Any]]) -> str:
    """The one ``session.id`` a trace's span documents carry, which is the session AWS scores.

    Raises :class:`AgentCoreContentUnserializable` when no document carries one and
    :class:`AgentCoreMixedSession` when they carry several, before any billed call: either way the
    session recorded for the call would not be the one AWS scores.
    """
    sessions = {doc["attributes"].get("session.id") for doc in documents}
    sessions.discard(None)
    if not sessions:
        raise AgentCoreContentUnserializable(
            f"trace {trace.trace_id!r}: its span documents carry no session.id, so the session "
            "AWS would score cannot be recorded; refused before any billed call"
        )
    if len(sessions) > 1:
        raise AgentCoreMixedSession(
            f"trace {trace.trace_id!r}: its span documents carry {len(sessions)} session ids; "
            "AgentCore Evaluate scores one session per call, so it is refused before any billed "
            "call"
        )
    (session,) = sessions
    return str(session)


EVALUATE_READ_TIMEOUT_SECONDS = 120
"""How long an ``Evaluate`` call waits for its response on the clients the runner opens."""


@contextmanager
def _one_attempt(clients: AwsClients) -> Iterator[AwsClients]:
    """The clients to send ``Evaluate`` on, with the SDK's retries off.

    A retried ``Evaluate`` can be billed again after a 5xx or a read timeout, while the billed-call
    budget counts the call once, so each call is sent once. Clients already set to one attempt are
    used as they are, their wait included (a caller that sets them, a stubbed replay for one, owns
    that choice); otherwise a single-attempt view of them is opened and closed here, which also
    waits up to :data:`EVALUATE_READ_TIMEOUT_SECONDS` for each response, since the library's default
    30 seconds refused two live calibrations on one slow call each.
    """
    from botocore.config import Config

    if clients.client(_EVALUATE_SERVICE).meta.config.retries.get("total_max_attempts") == 1:
        yield clients
        return
    single = Config(
        read_timeout=EVALUATE_READ_TIMEOUT_SECONDS,
        retries={"mode": "standard", "total_max_attempts": 1},
    )
    with clients.with_config(single) as view:
        yield view


class _Usage:
    """The billed calls made so far and the tokens AgentCore reported for their results."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        # Calls run on several threads at once, and each updates these counts.
        self._lock = threading.Lock()

    def count_call(self) -> None:
        with self._lock:
            self.calls += 1

    def add(self, result: dict[str, Any]) -> None:
        # A count the service did not report as a whole number is left out, never guessed.
        reported = result.get("tokenUsage") or {}
        with self._lock:
            for field, attribute in (
                ("inputTokens", "input_tokens"),
                ("outputTokens", "output_tokens"),
            ):
                count = reported.get(field)
                if isinstance(count, int) and not isinstance(count, bool):
                    setattr(self, attribute, getattr(self, attribute) + count)


def _require_budget(clients: AwsClients, *, calls: int, judge_id: str) -> None:
    """Refuse a run the clients' billed-call budget cannot fit, before anything is asked."""
    budget = clients.billed_call_budget
    remaining = budget.remaining if budget is not None else None
    if remaining is not None and remaining < calls:
        raise BilledCallBudgetExceeded(
            f"judge {judge_id!r}: the run needs {calls} billed Evaluate call(s), one per trace, "
            f"but the billed-call budget has {remaining} left; refused before any call (raise "
            "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS or unset it to remove the "
            "limit)",
            operation=(_EVALUATE_SERVICE, "Evaluate"),
        )


def _rates_for(
    evaluator_type: str, clients: AwsClients
) -> tuple[EvaluationRates | None, str | None]:
    """The rates to price this run at, or the reason there are none; never refuses the run."""
    reason = unpriced_reason(evaluator_type)
    if reason is not None:
        return None, reason
    try:
        return lookup_evaluation_rates(clients), None
    except AgentCoreError as exc:
        return None, f"the AWS Price List lookup did not give a rate ({exc})"


def _log_usage(judge: AgentCoreJudge, usage: _Usage, rates: EvaluationRates | None) -> None:
    cost = None
    if rates is not None:
        cost, _ = usage_cost(
            judge.evaluator_type,
            rates,
            usage.input_tokens,
            usage.output_tokens,
            evaluations=usage.calls,
        )
    priced = f", ${cost:.4f} at the {rates.region} rates" if cost is not None and rates else ""
    logger.info(
        "AgentCore reported %d input and %d output tokens over %d billed call(s) for judge %s%s",
        usage.input_tokens,
        usage.output_tokens,
        usage.calls,
        judge.judge_id,
        priced,
    )


def _score_all(
    judge: AgentCoreJudge,
    requests: list[_Request],
    *,
    client: Any,
    usage: _Usage,
    max_concurrency: int,
    progress: Callable[[int, int], None] | None,
) -> list[Label]:
    """Send one Evaluate call per trace, up to ``max_concurrency`` at once; labels in trace order.

    Calls are sent as earlier ones end, so at most ``max_concurrency`` are in flight. Once a call
    fails, no further call is sent; the calls in flight end (they may be billed, and each counts in
    ``usage``) and the first failure seen is raised. ``progress`` runs on this thread only.
    """
    per_trace: list[list[Label]] = [[] for _ in requests]
    waiting = iter(enumerate(requests))
    in_flight: dict[Future[list[Label]], int] = {}
    failure: BaseException | None = None
    scored = 0
    with ThreadPoolExecutor(
        max_workers=max_concurrency, thread_name_prefix="calibrated-agent-evaluation-evaluate"
    ) as pool:

        def send_next() -> bool:
            item = next(waiting, None)
            if item is None:
                return False
            index, request = item
            future = pool.submit(
                _score_one, judge, request, client=client, usage=usage, limit=max_concurrency
            )
            in_flight[future] = index
            return True

        while len(in_flight) < max_concurrency and send_next():
            pass
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                index = in_flight.pop(future)
                error = future.exception()
                if error is not None:
                    failure = failure or error
                    continue
                per_trace[index] = future.result()
                if failure is None:
                    scored += 1
                    if progress is not None:
                        progress(scored, len(requests))
            while failure is None and len(in_flight) < max_concurrency and send_next():
                pass
    if failure is not None:
        raise failure
    return [label for labels in per_trace for label in labels]


def _score_one(
    judge: AgentCoreJudge, request: _Request, *, client: Any, usage: _Usage, limit: int
) -> list[Label]:
    """Send one trace's Evaluate call and map its results to labels, counting usage."""
    from botocore.exceptions import BotoCoreError, ClientError, ReadTimeoutError

    trace, documents, synthetic, session_id, references = request
    session = judge.level == "session"
    target_id = conversation_session_id(trace) if session else aws_trace_id(trace.trace_id)
    call: dict[str, Any] = {
        "evaluatorId": judge.evaluator_id,
        "evaluationInput": {"sessionSpans": documents},
    }
    if session:
        # A SESSION evaluator scores the whole input; the assertions ride on the session.
        if references:
            call["evaluationReferenceInputs"] = references
    else:
        call["evaluationTarget"] = {"traceIds": [target_id]}
    try:
        response = client.evaluate(**call)
        usage.count_call()
    except (ClientError, ReadTimeoutError) as exc:
        # The request reached the service (it answered with an error, or its response timed
        # out), so the call may have been billed; it counts in the usage log.
        usage.count_call()
        code = exc.response.get("Error", {}).get("Code") if isinstance(exc, ClientError) else None
        if code == "ThrottlingException":
            if limit == 1:
                advice = "the run sent one call at a time, so wait a minute and run it again"
            else:
                advice = (
                    f"the run sent up to {limit} at a time; set {CONCURRENCY_ENV_VAR} (or "
                    f"max_concurrency) below {limit} and run it again"
                )
            raise AgentCoreError(
                f"the AgentCore Evaluate call for judge {judge.judge_id!r} was throttled "
                f"(ThrottlingException): {advice}. {exc}"
            ) from exc
        raise AgentCoreError(
            f"the AgentCore Evaluate call for judge {judge.judge_id!r} failed before any "
            f"verdict was produced: {exc}"
        ) from exc
    except BotoCoreError as exc:
        raise AgentCoreError(
            f"the AgentCore Evaluate call for judge {judge.judge_id!r} failed before any "
            f"verdict was produced: {exc}"
        ) from exc
    # The request carries ids conformed to the Evaluate target's length. Each result is mapped
    # back against this call's own target and that trace's own spans, never run-wide.
    results = response.get("evaluationResults", [])
    if not results:
        raise AgentCoreScoreIncomplete(
            f"judge {judge.judge_id!r}: the Evaluate call for trace {trace.trace_id!r} "
            "returned no result, so that trace has no verdict; the run is refused and no label "
            "from it is stored"
        )
    for result in results:
        usage.add(result)
    if session and len(results) > 1:
        raise AgentCoreAmbiguousSessionResult(
            f"judge {judge.judge_id!r}: the Evaluate call for conversation "
            f"{trace.trace_id!r} returned {len(results)} results, and a session call scores "
            "one session, so it must return one; the run is refused and no label from it is "
            "stored"
        )
    labels: list[Label] = []
    for result in results:
        _refuse_flagged(result, judge_id=judge.judge_id)
        mapped = (
            _map_session_back(result, session_id=target_id, trace=trace, judge_id=judge.judge_id)
            if session
            else _map_ids_back(
                result,
                target_id=target_id,
                session_id=session_id,
                trace=trace,
                judge_id=judge.judge_id,
            )
        )
        label = result_to_label(
            mapped,
            rating_scale=judge.rating_scale,
            inverted=judge.inverted,
        )
        # The evaluator ARN stays the same when the evaluator is edited, so the verdict also
        # names the fingerprint of the judge version that produced it.
        critique = f"{label.critique} [fingerprint={judge.fingerprint}]"
        if synthetic:
            # The verdict was reached on synthetic times, so it says so beside its raw verdict.
            critique += " [timing=synthetic]"
        labels.append(label.model_copy(update={"critique": critique}))
    return labels


def _refuse_flagged(result: dict[str, Any], *, judge_id: str) -> None:
    """Refuse a result the service itself flagged, before anything else is read from it.

    The ``errorCode`` comes first: such a result usually has no value and may have no address, so
    any later check would report that instead and hide the service's own reason. A dropped
    reference input comes next, and is checked even though that result still carries a value.
    """
    code = result.get("errorCode")
    if code:
        raise AgentCoreEvaluatorResultError(
            f"judge {judge_id!r}: AgentCore could not score a result ({code}: "
            f"{result.get('errorMessage') or 'no message given'}); the run is refused and no label "
            "from it is stored"
        )
    dropped = result.get("ignoredReferenceInputFields")
    if dropped:
        raise AgentCoreReferenceDropped(
            f"judge {judge_id!r}: AgentCore scored a result without the reference input it was "
            f"given (ignored: {', '.join(map(str, dropped))}), so the verdict does not answer the "
            "question asked; the run is refused and no label from it is stored"
        )


def _map_ids_back(
    result: dict[str, Any], *, target_id: str, session_id: str, trace: Trace, judge_id: str
) -> dict[str, Any]:
    """Rewrite a result's conforming trace/span ids back to the original ids.

    ``target_id`` is the conforming trace id this call sent, for ``trace``, and ``session_id`` the
    session its span documents named. A result with no trace id but this call's session id maps to
    the call's one trace. A result naming neither is left for the mapping to refuse as incomplete.
    Raises :class:`AgentCoreResultUnaddressable` for a trace this call did not target, a session
    other than the one this call named, or a span that is not a span of the named trace.
    """
    span_context = dict((result.get("context") or {}).get("spanContext") or {})
    raw_trace = span_context.get("traceId")
    session = span_context.get("sessionId")
    if session and session != session_id:
        raise AgentCoreResultUnaddressable(
            f"judge {judge_id!r}: an Evaluate result names session {session!r}, but its call "
            f"named session {session_id!r}, so the verdict has no Calibrated Agent Evaluation "
            "trace to file under; "
            "the run is refused and no label from it is stored"
        )
    if not raw_trace and session:
        # Scored against this call's one session, which holds this call's one trace.
        raw_trace = target_id
    if not isinstance(raw_trace, str) or not raw_trace:
        return {**result, "context": {"spanContext": span_context}}
    if raw_trace != target_id:
        raise AgentCoreResultUnaddressable(
            f"judge {judge_id!r}: an Evaluate result names trace {raw_trace!r}, which its call did "
            "not target, so it has no Calibrated Agent Evaluation trace to file the verdict "
            "under; the run is refused "
            "and no label from it is stored"
        )
    span_context["traceId"] = trace.trace_id
    if session:
        # The shared result mapping reads the session id as the trace id when a result has no
        # trace id, so it is rewritten to the Calibrated Agent Evaluation id too, never left as the
        # request's session.
        span_context["sessionId"] = trace.trace_id
    raw_span = span_context.get("spanId")
    if raw_span:
        spans = {aws_span_id(span.span_id): span.span_id for span in trace.spans}
        if raw_span not in spans:
            raise AgentCoreResultUnaddressable(
                f"judge {judge_id!r}: an Evaluate result for trace {trace.trace_id!r} names span "
                f"{raw_span!r}, which is not a span of that trace; filing it there would put the "
                "verdict in the wrong trace's cluster, so the run is refused and no label from it "
                "is stored"
            )
        span_context["spanId"] = spans[raw_span]
    return {**result, "context": {"spanContext": span_context}}


def _map_session_back(
    result: dict[str, Any], *, session_id: str, trace: Trace, judge_id: str
) -> dict[str, Any]:
    """Rewrite a session-level result's ids back to the conversation's trace id.

    A session result names the session it scored (and, at most, one of that session's turn trace
    ids); it is filed under the conversation's one trace with no span. A result naming neither is
    left for the mapping to refuse as incomplete. Raises :class:`AgentCoreResultUnaddressable` for
    a session other than the one sent, a trace id that is not one of its turns, or a span.
    """
    span_context = dict((result.get("context") or {}).get("spanContext") or {})
    named_session = span_context.get("sessionId")
    named_trace = span_context.get("traceId")
    if not named_session and not named_trace:
        return {**result, "context": {"spanContext": span_context}}
    if named_session and named_session != session_id:
        raise AgentCoreResultUnaddressable(
            f"judge {judge_id!r}: an Evaluate result names session {named_session!r}, but its "
            f"call sent conversation {trace.trace_id!r} as session {session_id!r}, so the verdict "
            "has no conversation to file under; the run is refused and no label from it is stored"
        )
    turns = conversation_turn_trace_ids(trace)
    if named_trace and named_trace != session_id and named_trace not in turns:
        raise AgentCoreResultUnaddressable(
            f"judge {judge_id!r}: an Evaluate result for conversation {trace.trace_id!r} names "
            f"trace {named_trace!r}, which is not one of its turns; the run is refused and no "
            "label from it is stored"
        )
    if span_context.get("spanId"):
        raise AgentCoreResultUnaddressable(
            f"judge {judge_id!r}: a session-level result for conversation {trace.trace_id!r} names "
            "a span, and a session verdict scores the whole conversation; the run is refused and "
            "no label from it is stored"
        )
    return {
        **result,
        "context": {"spanContext": {"traceId": trace.trace_id, "sessionId": trace.trace_id}},
    }
