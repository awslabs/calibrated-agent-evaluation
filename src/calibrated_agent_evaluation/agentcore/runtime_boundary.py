# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The call path every caller of one AgentCore Runtime shares, kept in one place.

:class:`RuntimeCallBoundary` holds the parts of a run that the runtime executor and the runtime
probe adapter both need, so the rules for stopping sessions and keeping run records are written
once:

- the AWS clients: the caller's, or built on first use for the region in the runtime ARN, with the
  billed-call budget installed when ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS`` is
  set;
- the invocation view derived from those clients with the read timeout, closed on
  :meth:`~RuntimeCallBoundary.close`;
- the read-only preflight, run before the first invocation and kept once it succeeds;
- the invocation itself, retrying only a busy session or throttling;
- the sessions an invocation may have opened: each is stopped once a request may have been sent,
  and any whose stop did not succeed gets one more stop on close;
- the run record of every run, whatever its outcome, added to the run manifest of a log
  directory. A record that can't be written fails a run that succeeded, and is noted on the
  error of a run that had already failed.

Its callers add the rest: the executor its cassette, span read and owned ``Trace``, the probe
adapter its trace ID and probe reply. Each caller checks what it was given before it builds the
boundary, with :func:`check_limits` for the read timeout and the attempt limit, so each refusal
names the caller's own arguments. Both callers default those two settings to
:data:`DEFAULT_READ_TIMEOUT_SECONDS` and :data:`DEFAULT_ATTEMPT_LIMIT`, time their runs with
:func:`utc_now`, and leave their ``with`` block through :func:`close_on_exit`, as the boundary
does. Each refuses a prompt that is not text, or is empty, with :func:`check_prompt` before the
prompt's run starts, so it costs no call and leaves no run record. Cleanup goes through
:func:`run_all`, so every step runs even when one fails, and a cleanup failure never replaces
the error that ended a ``with`` block. This module is internal to
``calibrated_agent_evaluation.agentcore``; it is not one of the package's public names.

Every failure raises
:class:`~calibrated_agent_evaluation.agentcore.runtime_errors.RuntimeInvocationError`, chained from
its cause. Messages and log lines name IDs, operations, error codes and types, paths and settings;
they never carry a payload, a response body or an agent's error text.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Self, TypeVar

from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
from calibrated_agent_evaluation.agentcore.guards import BUDGET_ENV_VAR, BilledCallBudget
from calibrated_agent_evaluation.agentcore.run_manifest import (
    RunRecord,
    SessionStop,
    append_run_record,
)
from calibrated_agent_evaluation.agentcore.runtime import (
    CONTROL_SERVICE,
    DATA_SERVICE,
    InvocationProgress,
    RawResponse,
    RunIds,
    RuntimeInfo,
    RuntimeTarget,
    invoke,
    resolve_runtime,
    runtime_client_config,
    stop,
)
from calibrated_agent_evaluation.agentcore.runtime_errors import RuntimeInvocationError
from calibrated_agent_evaluation.agentcore.session import AwsClients

if TYPE_CHECKING:
    from botocore.client import BaseClient

__all__ = [
    "DEFAULT_ATTEMPT_LIMIT",
    "DEFAULT_READ_TIMEOUT_SECONDS",
    "Invocation",
    "RunState",
    "RuntimeCallBoundary",
    "check_limits",
    "check_prompt",
    "close_on_exit",
    "run_all",
    "utc_iso",
    "utc_now",
]

logger = logging.getLogger(__name__)

DEFAULT_READ_TIMEOUT_SECONDS = 300.0
"""How long an invocation waits for each read of its response when the caller names no time."""

DEFAULT_ATTEMPT_LIMIT = 3
"""The most attempts one invocation sends when the caller names no limit, the first included."""

_STOPPED: frozenset[SessionStop] = frozenset({"stopped", "already_gone"})
# What a caller's reading of an invocation's answer returns.
_T = TypeVar("_T")


def utc_now() -> datetime:
    """The current time in UTC, timezone-aware: the clock a run is timed by unless one is given."""
    return datetime.now(tz=UTC)


def utc_iso(when: datetime) -> str:
    """``when`` in UTC, as ISO 8601 with its offset, the form run records and traces store."""
    return when.astimezone(UTC).isoformat()


@dataclass
class RunState:
    """What one run has learned so far; it becomes the run's record in the run manifest.

    Make one with :meth:`RuntimeCallBoundary.start_run`. ``ids`` are what a live invocation
    carries. ``trace_id``, ``session_id``, ``w3c_trace_id`` and the timing are what the record
    names: the run's own, or a replayed entry's. ``ended`` stays ``None`` unless a replay names the
    recorded end; the record otherwise ends when it is built.
    """

    ids: RunIds
    trace_id: str
    started: datetime
    session_id: str
    w3c_trace_id: str
    ended: datetime | None = None
    attempts: int = 0
    status_code: int | None = None
    content_type: str | None = None
    response_traceparent: str | None = None
    runtime_info: RuntimeInfo | None = None
    replayed: bool = False
    session_stop: SessionStop = "not_started"


@dataclass(frozen=True)
class Invocation:
    """An invocation the runtime answered, as :meth:`RuntimeCallBoundary.call` hands it on.

    Attributes:
        response: The response, read whole and closed.
        started: When the invocation started, before its first attempt was sent.
        ended: When its response had been read.
        runtime_info: The runtime version the preflight resolved, which served the run.
    """

    response: RawResponse
    started: datetime
    ended: datetime
    runtime_info: RuntimeInfo


class RuntimeCallBoundary:
    """One AgentCore Runtime's call path: clients, preflight, invocation, stops and run records.

    It holds the AWS clients, the invocation view derived from them, the preflight's result once
    one succeeded, the sessions not recorded as stopped, and the record of every finished run.
    :meth:`call` sends one run's invocation and stops its session; :meth:`finish` writes the
    run's record, whatever the outcome; :meth:`close` stops the sessions left open and closes
    what it derived or built. Its caller builds the request body, names the run's trace ID and
    decides what the answer means. It builds no AWS client until the first :meth:`call`.

    Use it as a context manager, or have what owns it call :meth:`close` through
    :func:`close_on_exit` as its own ``with`` block is left. Either way, a failure on the way out,
    to stop the sessions left open or to close what was derived or built, is raised when the block
    succeeded, and logged and noted on the block's own error when it failed.

    Not safe to share across threads.

    The caller checks its arguments before it builds the boundary, so each refusal names the
    caller's own arguments: that the clients are for the runtime ARN's region, that ``clients``
    and ``profile`` are not both given, and the limits, with :func:`check_limits`. The boundary
    takes them as they are.

    Args:
        target: The runtime, endpoint qualifier, payload format and configuration bundle.
        clients: AWS clients to use instead of building them, for the region in the runtime ARN.
            They are never closed here, and no billed-call budget is installed on them. Give
            these or ``profile``, not both.
        profile: The AWS profile to build clients from on the first call, the default credential
            chain when ``None``. The clients built get the
            ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS`` budget when that is set.
        read_timeout_seconds: How long an invocation waits for each read of its response, a value
            :func:`check_limits` accepted.
        attempt_limit: The most attempts one invocation may send, the first one included, a
            value :func:`check_limits` accepted.
        sleep: Waits between retried attempts.
        clock: The current time, timezone-aware.
        setup_hint: The sentence that ends the refusal when the AWS clients could not be built,
            saying what to check. When ``None``, it names the AWS profile, or the default
            credential chain, that the clients come from.
        closed_on: What the refusal and the log lines of :meth:`stop_open_sessions` call the
            moment the sessions left open are stopped, as in "on exit".
        owner: What cleanup messages call the boundary's owner, as in "the AWS clients the
            executor built" and "closing the executor failed".
    """

    def __init__(
        self,
        target: RuntimeTarget,
        *,
        clients: AwsClients | None = None,
        profile: str | None = None,
        read_timeout_seconds: float,
        attempt_limit: int,
        sleep: Callable[[float], object] = time.sleep,
        clock: Callable[[], datetime] = utc_now,
        setup_hint: str | None = None,
        closed_on: str = "exit",
        owner: str = "the call boundary",
    ) -> None:
        self._target = target
        self._profile = profile
        self._read_timeout = read_timeout_seconds
        self._attempt_limit = attempt_limit
        self._sleep = sleep
        self._clock = clock
        self._setup_hint = setup_hint
        self._closed_on = closed_on
        self._owner = owner
        self._base = clients
        self._owns_base = False
        self._view: AwsClients | None = None
        self._invocation_client: BaseClient | None = None
        self._runtime_info: RuntimeInfo | None = None
        # Sessions an invocation may have opened and no stop has ended yet, oldest first.
        self._unstopped: list[str] = []
        self._records: list[RunRecord] = []
        # Why each run record that is in _records but not in the run manifest could not be added.
        self._manifest_errors: list[RuntimeInvocationError] = []

    @property
    def run_records(self) -> tuple[RunRecord, ...]:
        """The record of every run :meth:`finish` was given, whatever its outcome, in order.

        A record is listed even when the run manifest could not take it.
        """
        return tuple(self._records)

    @property
    def manifest_errors(self) -> tuple[RuntimeInvocationError, ...]:
        """The error of each failed write to the run manifest, in order.

        Empty when the manifest holds every record of :attr:`run_records`.
        """
        return tuple(self._manifest_errors)

    def start_run(self, ids: RunIds, trace_id: str) -> RunState:
        """A new run's state: ``ids``' session and W3C trace ID, named ``trace_id``, started now.

        Args:
            ids: The run's IDs, from
                :func:`~calibrated_agent_evaluation.agentcore.runtime.new_run_ids`.
            trace_id: The ID of the trace the run returns, or would have returned.

        Returns:
            The run's state, not replayed and with no session started.
        """
        return RunState(
            ids=ids,
            trace_id=trace_id,
            started=self._clock(),
            session_id=ids.session_id,
            w3c_trace_id=ids.w3c_trace_id,
        )

    def call(
        self,
        run: RunState,
        body: bytes,
        read: Callable[[Invocation], _T],
        *,
        stop_session: bool = True,
    ) -> tuple[Invocation, _T]:
        """Send ``run``'s invocation, read its answer with ``read``, and stop its session.

        On the first call, the clients are built (unless they were given) and the preflight runs;
        only a preflight that succeeded is kept, so one that failed is asked again on the next
        call. The run's session is then held as open and the invocation sent, retrying only a
        busy session or throttling. ``read`` runs on the answer inside the same guard, so a
        failure it raises ends the run as an invocation's failure does.

        Once a request may have been sent, or when the invocation was interrupted (whether a
        request went out is then unknown), the session is stopped before this returns or raises.
        ``run.session_stop`` reads ``failed`` while the stop runs, so an interrupted stop reads as
        not stopped, and then says how it ended. A session whose stop did not succeed stays held
        for :meth:`stop_open_sessions`. A run that sent nothing opened no session: it is let go
        and nothing is stopped.

        ``run`` keeps the runtime version, and the attempts, status code, content type and
        traceparent: the response's on success, what the invocation reached on a failure.

        With ``stop_session=False`` (a conversation's turn) the session is left open, held for
        :meth:`stop_session` or :meth:`stop_open_sessions`, whatever the call's outcome; a session
        already open from an earlier turn is held once.

        Args:
            run: The run, from :meth:`start_run`.
            body: The request body, built before this is called, so a body that can't be made
                costs no AWS call.
            read: Reads the answer; what it returns is returned with the invocation.
            stop_session: Whether to stop the session once the call ends; ``False`` keeps it open
                for the conversation's next turn.

        Returns:
            The invocation, and what ``read`` returned.

        Raises:
            RuntimeInvocationError: The clients could not be built, or the billed-call budget
                variable is invalid, naming the profile or the variable; the preflight refused
                the runtime or failed; or the invocation failed.
            Exception: What ``read`` raised, unchanged.
            BaseException: An interrupt, raised again once the session's stop was tried.
        """
        session = run.ids.session_id
        base, invocation_client = self._clients()
        if self._runtime_info is None:
            # Only a preflight that succeeded is kept; a refused one is asked again next run.
            self._runtime_info = resolve_runtime(base, self._target)
        info = self._runtime_info
        run.runtime_info = info
        already_open = session in self._unstopped
        if not already_open:
            self._unstopped.append(session)
        progress = InvocationProgress()
        sent = False
        try:
            invocation_started = self._clock()
            try:
                raw = invoke(
                    invocation_client,
                    self._target,
                    run.ids,
                    body,
                    attempt_limit=self._attempt_limit,
                    sleep=self._sleep,
                    progress=progress,
                )
            except Exception:
                _note_progress(run, progress)
                sent = progress.sent
                raise
            except BaseException:
                # Interrupted mid-call, so a request may have gone out: stop the session anyway,
                # since stopping one that never opened is harmless.
                _note_progress(run, progress)
                sent = True
                raise
            sent = True
            invocation = Invocation(
                response=raw, started=invocation_started, ended=self._clock(), runtime_info=info
            )
            run.attempts = raw.attempts
            run.status_code = raw.status_code
            run.content_type = raw.content_type
            run.response_traceparent = raw.response_traceparent
            answer = read(invocation)
        except Exception:
            if not sent and not already_open:
                # Nothing reached the service, so no session was opened.
                self._unstopped.remove(session)
            raise
        finally:
            if sent and stop_session:
                self._stop(run, base)
            elif sent:
                run.session_stop = "held_open"
        return invocation, answer

    def stop_session(self, session_id: str) -> SessionStop | None:
        """Stop a session a conversation kept open, once its last turn has run.

        Args:
            session_id: The conversation's session.

        Returns:
            How the stop ended, or ``None`` when the session is not held open (it was never
            opened, as in a replay, or it was already stopped). A session whose stop did not
            succeed stays held for :meth:`stop_open_sessions`.
        """
        base = self._base
        if session_id not in self._unstopped or base is None:
            return None
        try:
            client = base.client(DATA_SERVICE)
        except Exception as exc:
            # Any failure, one from a caller's own clients included: only its type is named.
            raise RuntimeInvocationError(
                f"could not build the AWS client to stop conversation session {session_id} "
                f"({type(exc).__name__}); it stays held for the stop on {self._closed_on}"
            ) from exc
        result = stop(client, self._target, session_id)
        if result in _STOPPED:
            self._unstopped.remove(session_id)
            logger.info("the stop of conversation session %s ended: %s", session_id, result)
        else:
            logger.warning(
                "the stop of conversation session %s ended: %s; it may stay billed until its idle "
                "timeout, and it is tried once more on %s",
                session_id,
                result,
                self._closed_on,
            )
        return result

    def record_for(self, run: RunState, failure: BaseException | None) -> RunRecord:
        """The run manifest's record of ``run``, ending now unless ``run`` names its end.

        Args:
            run: The run.
            failure: What ended the run, or ``None`` when it succeeded. A failed run's
                ``error_code`` is the AWS error code of a service error; else the type name of
                the cause underneath, or of the refusal itself when nothing underlies it.

        Returns:
            The record.
        """
        ended = run.ended or self._clock()
        info = run.runtime_info
        return RunRecord(
            trace_id=run.trace_id,
            w3c_trace_id=run.w3c_trace_id,
            session_id=run.session_id,
            runtime_arn=self._target.runtime_arn,
            qualifier=self._target.qualifier,
            runtime_version=None if info is None else info.runtime_version,
            runtime_version_source=None if info is None else info.version_source,
            config_bundle_arn=self._target.config_bundle_arn,
            config_bundle_version=self._target.config_bundle_version,
            payload_format=self._target.payload_format,
            started_at=utc_iso(run.started),
            ended_at=utc_iso(ended),
            duration_ms=(ended - run.started) // timedelta(milliseconds=1),
            attempts=run.attempts,
            status_code=run.status_code,
            content_type=run.content_type,
            response_traceparent=run.response_traceparent,
            replayed=run.replayed,
            session_stop=run.session_stop,
            error_code=None if failure is None else _error_code(failure),
        )

    def finish(
        self,
        run: RunState,
        manifest_dir: Path | Callable[[], Path],
        *,
        failure: BaseException | None,
    ) -> None:
        """Keep ``run``'s record in :attr:`run_records`, then add it to the run manifest.

        A record the manifest could not take stays in :attr:`run_records`, and why is kept in
        :attr:`manifest_errors`.

        Args:
            run: The run that ended.
            manifest_dir: The log directory whose run manifest gets the record, or a function
                that names it, called once the record is kept, so a directory it cannot name is
                handled as a write that failed.
            failure: What ended the run, or ``None`` when it succeeded. When the record can't be
                added to the manifest after a failed run, the manifest's error is noted on
                ``failure`` rather than raised, since the run's own error says what went wrong.

        Raises:
            RuntimeInvocationError: The run succeeded and its record could not be added to the
                manifest; the run manifest's own error, which names the manifest's path.
        """
        record = self.record_for(run, failure)
        self._records.append(record)
        try:
            directory = manifest_dir() if callable(manifest_dir) else manifest_dir
            append_run_record(directory, record)
        except RuntimeInvocationError as exc:
            self._manifest_errors.append(exc)
            if failure is None:
                raise
            # The run's own error says what went wrong; the manifest failure only adds to it.
            failure.add_note(str(exc))

    def stop_open_sessions(self) -> None:
        """Give every session not recorded as stopped one more ``StopRuntimeSession``.

        A session whose earlier stop failed or was interrupted is one of them. Each is let go
        whatever its stop's result, a session still open then being left to its idle timeout, and
        the result is logged at INFO with its session ID. Nothing is called when none is open.

        Raises:
            RuntimeInvocationError: The client to stop them could not be built, naming the
                sessions and the type of the error, never its text.
        """
        base = self._base
        if not self._unstopped or base is None:
            return
        try:
            client = base.client(DATA_SERVICE)
        except Exception as exc:
            # Any failure, one from a caller's own clients included: only its type is named.
            raise RuntimeInvocationError(
                f"could not build the AWS client to stop runtime sessions "
                f"{', '.join(self._unstopped)} on {self._closed_on} ({type(exc).__name__}); they "
                "may stay billed until their idle timeout ends"
            ) from exc
        for session in list(self._unstopped):
            result = stop(client, self._target, session)
            # One more stop each; one still open is left to its idle timeout, as stop() warns.
            self._unstopped.remove(session)
            logger.info(
                "on %s, the stop of runtime session %s ended: %s", self._closed_on, session, result
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        close_on_exit(self.close, exc, owner=self._owner)

    def close(self) -> None:
        """Stop the sessions left open, then close the invocation view and the clients built here.

        The caller's clients are never closed. Every step runs: the view and the clients built
        here are closed even when stopping the sessions, or closing one of them, fails. A later
        :meth:`call` builds them again.

        Raises:
            RuntimeInvocationError: The client to stop the open sessions could not be built, or
                the view or a client built here could not be closed. When more than one step
                fails, the first failure is raised, and each later one is logged as a warning
                and noted on it.
        """
        run_all((self.stop_open_sessions, self.close_clients), owner=self._owner)

    def close_clients(self) -> None:
        """Close the invocation view and the clients built here; the caller's stay open.

        Both are closed even when closing the first fails. A later :meth:`call` builds them
        again.

        Raises:
            RuntimeInvocationError: The view or a client built here could not be closed, naming
                which and the error's type, never its text. When both fail, the first is raised
                with the second noted on it.
        """
        # The caller's own clients are theirs to close.
        view, self._view, self._invocation_client = self._view, None, None
        built = self._base if self._owns_base else None
        if built is not None:
            self._base, self._owns_base = None, False
        steps: list[Callable[[], None]] = []
        if view is not None:
            steps.append(lambda: _close_clients(view, "the invocation clients"))
        if built is not None:
            steps.append(lambda: _close_clients(built, f"the AWS clients {self._owner} built"))
        run_all(steps, owner=self._owner)

    def _clients(self) -> tuple[AwsClients, BaseClient]:
        """The base clients and the invocation client, built on first use."""
        if self._base is not None and self._invocation_client is not None:
            return self._base, self._invocation_client
        if self._base is None:
            self._base = self._build_clients()
            self._owns_base = True
        base = self._base
        if self._view is not None:
            # Left by an earlier attempt that failed to build its client.
            self._view.close()
        try:
            view = base.with_config(runtime_client_config(self._read_timeout))
            self._view = view
            base.client(CONTROL_SERVICE)
            base.client(DATA_SERVICE)
            client = view.client(DATA_SERVICE)
        except Exception as exc:
            # A caller's own clients may raise anything here; only its type is named.
            hint = self._setup_hint or f"Check the AWS setup for {self._profile_named()}"
            raise RuntimeInvocationError(
                f"could not build the AWS clients for region {self._target.region} "
                f"({type(exc).__name__}), so nothing was invoked. {hint}"
            ) from exc
        self._invocation_client = client
        return base, client

    def _build_clients(self) -> AwsClients:
        try:
            budget = BilledCallBudget.from_env(os.environ)
        except AgentCoreError as exc:
            raise RuntimeInvocationError(
                f"{BUDGET_ENV_VAR}={os.environ.get(BUDGET_ENV_VAR, '')!r} is not a non-negative "
                "whole number, so no AWS client was built and nothing was invoked. Set it to the "
                "most billed calls to allow, or unset it to remove the limit"
            ) from exc
        handlers = () if budget.limit is None else (budget,)
        try:
            clients = AwsClients.from_profile(
                region=self._target.region, profile=self._profile, handlers=handlers
            )
        except AgentCoreError as exc:
            raise RuntimeInvocationError(
                f"could not open {self._profile_named()} for region {self._target.region} "
                f"({type(exc).__name__}), so nothing was invoked. Set AWS_PROFILE or pass "
                "profile= with a profile that exists"
            ) from exc
        return clients

    def _profile_named(self) -> str:
        if self._profile is not None:
            return f"the AWS profile {self._profile!r}"
        environment = os.environ.get("AWS_PROFILE")
        if environment:
            return f"the AWS profile {environment!r} (from AWS_PROFILE)"
        return "the default AWS credential chain"

    def _stop(self, run: RunState, base: AwsClients) -> None:
        session = run.ids.session_id
        # Recorded first, so a stop that is interrupted still reads as not stopped.
        run.session_stop = "failed"
        run.session_stop = stop(base.client(DATA_SERVICE), self._target, session)
        if run.session_stop in _STOPPED:
            self._unstopped.remove(session)


def check_limits(read_timeout_seconds: object, attempt_limit: int) -> None:
    """Refuse a read timeout or an attempt limit no invocation could run with.

    Args:
        read_timeout_seconds: How long an invocation waits for each read of its response: a
            positive, finite number of seconds of at most ``threading.TIMEOUT_MAX``, the longest a
            socket can wait. A bool is not one.
        attempt_limit: The most attempts one invocation may send, the first one included: 1 or
            more.

    Raises:
        RuntimeInvocationError: The read timeout, checked first, or the attempt limit is refused,
            naming the value.
    """
    _check_read_timeout(read_timeout_seconds)
    if attempt_limit < 1:
        raise RuntimeInvocationError(
            f"attempt_limit={attempt_limit!r} allows no invocation; pass 1 or more"
        )


def check_prompt(prompt: object, *, named: str, remedy: str) -> None:
    """Refuse a prompt that is not a non-empty string; call it before the prompt's run starts.

    The prompt becomes the billed request's body, so a bad one would cost a call. Checked before
    :meth:`RuntimeCallBoundary.start_run`, a refused prompt leaves no run and no run record.

    Args:
        prompt: What the caller was given to send.
        named: What the refusal calls the prompt, as in "the task's prompt".
        remedy: What to do instead, as a sentence that ends the refusal.

    Raises:
        RuntimeInvocationError: ``prompt`` is not a string, naming its type, or is empty. The
            message never holds the value itself.
    """
    if not isinstance(prompt, str) or not prompt:
        given = "an empty string" if isinstance(prompt, str) else f"of type {type(prompt).__name__}"
        raise RuntimeInvocationError(
            f"{named} is {given}, and a prompt must be a non-empty string, so nothing was run. "
            f"{remedy}"
        )


def close_on_exit(close: Callable[[], None], exc: BaseException | None, *, owner: str) -> None:
    """Run ``close`` as a ``with`` block is left, keeping the block's own error the one raised.

    Args:
        close: Stops the sessions left open and closes what was built.
        exc: What the block raised, or ``None`` when it succeeded.
        owner: What the note calls the object being closed, as in "the executor", when the
            cleanup failure is not one of this package's refusals and only its type is named.

    Raises:
        BaseException: The block succeeded and ``close`` failed: its error, unchanged. When the
            block failed, a failure of ``close``, an interrupt included, is logged as a warning
            and noted on ``exc`` with every note it carries, never raised over it. A refusal of
            this package is noted in its own words; any other error by its type alone.
    """
    if exc is None:
        close()
        return
    # The block's own error is the one the caller sees; a failure on the way out, an interrupt
    # included, is logged and noted on it.
    try:
        close()
    except BaseException as cleanup:  # noqa: BLE001 - logged and noted on the block's own error
        _note_cleanup_failure(exc, cleanup, owner)


def run_all(steps: Iterable[Callable[[], None]], *, owner: str) -> None:
    """Run every cleanup step, even after one fails or is interrupted, then raise the first failure.

    Each later failure is logged as a warning and noted on the first, with the notes it carries
    itself, so none is lost.

    Args:
        steps: The cleanup steps, in order.
        owner: What a note calls the object being closed, as in "the executor", when a later
            failure is not one of this package's refusals and only its type is named.

    Raises:
        BaseException: The first step's failure, unchanged, once every step ran.
    """
    first: BaseException | None = None
    for step in steps:
        try:
            step()
        except BaseException as exc:  # noqa: BLE001 - the first is raised below, once all ran
            if first is None:
                first = exc
                continue
            _note_cleanup_failure(first, exc, owner)
    if first is not None:
        raise first


# --- helpers ----------------------------------------------------------------------------------


def _check_read_timeout(seconds: object) -> None:
    # Refused here, as later layers refuse too late: urllib3 refuses a bool only once the client
    # is built, a non-number fails the comparison with a bare TypeError, and the socket layer
    # refuses an infinite or overlong timeout only after the billed request is sent; NaN fails both.
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, int | float)
        or not (0 < seconds <= threading.TIMEOUT_MAX)
    ):
        raise RuntimeInvocationError(
            f"read_timeout_seconds={seconds!r} is not a positive, finite number of seconds of at "
            f"most {threading.TIMEOUT_MAX:.0f}; pass how long an invocation may wait for each "
            "read of its response, such as 300"
        )


def _close_clients(clients: AwsClients, which: str) -> None:
    try:
        clients.close()
    except Exception as exc:
        # The text comes from the SDK's connection pool or a caller's clients: name the type only.
        raise RuntimeInvocationError(
            f"could not close {which} ({type(exc).__name__}); the runs' results stand, and the "
            "connections left open are released when the process exits"
        ) from exc


def _note_cleanup_failure(exc: BaseException, cleanup: BaseException, owner: str) -> None:
    # A cleanup failure that must not replace ``exc``: logged, then noted on it with its own notes,
    # since a step that ran several steps itself carries their later failures as notes.
    text = _cleanup_text(cleanup, owner)
    logger.warning("%s", text)
    exc.add_note(text)
    for note in getattr(cleanup, "__notes__", ()):
        exc.add_note(note)


def _cleanup_text(exc: BaseException, owner: str) -> str:
    # A refusal of this package names IDs, paths and types only. Any other error may carry text
    # from the SDK or the caller's code, so only its type is named.
    if isinstance(exc, RuntimeInvocationError):
        return str(exc)
    return (
        f"closing {owner} failed with {type(exc).__name__}; the runs' results stand, and "
        "anything left open is released when the process exits"
    )


def _note_progress(run: RunState, progress: InvocationProgress) -> None:
    # So the record of a failed or interrupted invocation keeps what it reached.
    run.attempts = progress.attempts
    run.status_code = progress.status_code
    run.content_type = progress.content_type
    run.response_traceparent = progress.response_traceparent


def _error_code(failure: BaseException) -> str:
    # The AWS error code, else the type of what ended the run (a refusal's cause, if it has one).
    cause = failure.__cause__ if isinstance(failure, RuntimeInvocationError) else failure
    if cause is None:
        return type(failure).__name__
    response = getattr(cause, "response", None)
    if isinstance(response, Mapping):
        code = (response.get("Error") or {}).get("Code")
        if isinstance(code, str) and code:
            return code
    return type(cause).__name__
