# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run test inputs through an agent hosted on Amazon Bedrock AgentCore Runtime.

:class:`AgentCoreRuntimeExecutor` is an
:class:`~calibrated_agent_evaluation.execute.base.ExecutorBackend`. Each
:class:`AgentCoreRuntimeTask` becomes one run: Calibrated Agent Evaluation makes the run's session
ID and W3C trace context, invokes the runtime once (retrying only a busy session or throttling),
reads the final answer, asks a
:class:`~calibrated_agent_evaluation.agentcore.run_spans.RunSpanReader` for the span tree, stops the
session, and returns one validated, owned ``Trace``. Every run, whatever its outcome, leaves one
:class:`~calibrated_agent_evaluation.agentcore.run_manifest.RunRecord` in
:attr:`AgentCoreRuntimeExecutor.run_records` and in the log directory's run manifest
(``agentcore_run_record.json`` and ``.csv``). :class:`AgentCoreTaskBuilder` turns synthesized inputs
into tasks, one per input.

No AWS client is built before the first live call, and then only for the runtime ARN's region,
so building an executor never touches a profile or credentials. Before that first call, read-only
calls check that the runtime can be invoked at all.

With a cassette, a run is recorded once and replayed offline from the entry's IDs and timing,
with no AWS client, credentials or connection. One executor serves each cassette entry to one
task only, so it never returns two traces carrying one run's IDs.

The executor's calls to the runtime go through
:class:`~calibrated_agent_evaluation.agentcore.runtime_boundary.RuntimeCallBoundary`, the internal
part of a run any caller of one runtime shares: the AWS clients, the preflight, the invocation, the
session stops and the run records. The executor adds the cassette, the span read and the owned
``Trace``.

Every failure raises
:class:`~calibrated_agent_evaluation.agentcore.runtime_errors.RuntimeInvocationError`, chained from
its cause. Messages and log lines name IDs, operations, error codes and types, paths and settings,
never a payload, a response body or an agent's error text. The one exception, the text of a span
reader's own :class:`~calibrated_agent_evaluation.agentcore.errors.AgentCoreError`, is described on
``RuntimeInvocationError``.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, get_args

from calibrated_agent_evaluation._conversation import ConversationTurn
from calibrated_agent_evaluation._finite import (
    MAX_JSON_DEPTH,
    NestedTooDeepError,
    NonFiniteNumberError,
    require_finite,
)
from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreError,
    BilledCallBudgetExceeded,
    ReadOnlyViolation,
)
from calibrated_agent_evaluation.agentcore.run_manifest import RunRecord, SessionStop
from calibrated_agent_evaluation.agentcore.run_spans import RunSpanQuery, RunSpanReader
from calibrated_agent_evaluation.agentcore.runtime import (
    CUSTOM_PAYLOAD_FORMAT,
    PayloadBuilder,
    RunIds,
    RuntimeInfo,
    RuntimeTarget,
    guard_hint,
    guard_refusal,
    new_run_ids,
    request_body,
    turn_run_ids,
)
from calibrated_agent_evaluation.agentcore.runtime_boundary import (
    DEFAULT_ATTEMPT_LIMIT,
    DEFAULT_READ_TIMEOUT_SECONDS,
    Invocation,
    RunState,
    RuntimeCallBoundary,
    check_limits,
    check_prompt,
    close_on_exit,
    run_all,
    utc_iso,
    utc_now,
)
from calibrated_agent_evaluation.agentcore.runtime_cassette import (
    RecordedRun,
    cassette_request,
    cassette_stamp,
    decode_entry,
    encode_entry,
    open_cassette,
    rekey_spans,
)
from calibrated_agent_evaluation.agentcore.runtime_errors import RuntimeInvocationError
from calibrated_agent_evaluation.agentcore.runtime_response import (
    DecodedValue,
    check_errors,
    decode,
    final_text,
)
from calibrated_agent_evaluation.agentcore.session import AwsClients
from calibrated_agent_evaluation.execute.cassette import (
    MODE_ENV_VAR,
    Cassette,
    CassetteError,
    CassetteMiss,
    CassetteModelMismatch,
    Mode,
    fingerprint,
    mode_from_env,
)
from calibrated_agent_evaluation.schema.errors import SchemaError
from calibrated_agent_evaluation.schema.types import Frame, Span, Trace
from calibrated_agent_evaluation.schema.validate import validate

if TYPE_CHECKING:
    # Imported for annotations only: calibrated_agent_evaluation.synth loads the sampling stack,
    # which the [agentcore] extra does not install.
    from calibrated_agent_evaluation.synth.types import SynthInput

__all__ = [
    "AgentCoreRuntimeExecutor",
    "AgentCoreRuntimeTask",
    "AgentCoreTaskBuilder",
    "PayloadBuilder",
]

# A guard's refusals. A span reader raises other AgentCoreError subclasses, which name no guard.
_GUARD_ERRORS = (BilledCallBudgetExceeded, ReadOnlyViolation)
# How deep a span reader's error chain is searched for a guard's refusal.
_CHAIN_DEPTH = 20
# What cleanup messages call the executor.
_OWNER = "the executor"


@dataclass(frozen=True)
class AgentCoreRuntimeTask:
    """One input to send to the AgentCore Runtime agent.

    Attributes:
        prompt: The test input, a non-empty string; :meth:`AgentCoreRuntimeExecutor.run` refuses
            any other before any AWS call. The returned ``Trace``'s ``input`` is
            ``{"prompt": prompt}``, whatever request body the payload format builds from it.
        trace_id: The ID the returned ``Trace`` carries. ``None`` (or an empty string) names it
            after the run's W3C trace ID, ``agentcore-<trace ID>``.
        taxonomy_cell: The taxonomy cell the input was written for, carried onto the ``Trace``.
        conversation: For one turn of a simulated conversation, which conversation and turn. Every
            turn of a conversation is sent on one runtime session, left open between turns and
            stopped by :meth:`AgentCoreRuntimeExecutor.end_conversation`; ``None`` is a
            single-turn run with its own session, stopped after it.
    """

    prompt: str
    trace_id: str | None = None
    taxonomy_cell: dict[str, str] = field(default_factory=dict)
    conversation: ConversationTurn | None = None


@dataclass(frozen=True)
class _UsedFingerprint:
    """What this executor did with a cassette fingerprint, as a later task's refusal names it.

    ``done`` says what happened to which run, ``outcome`` how a failed recording failed (empty
    otherwise), and ``why`` why the later task is refused and what to do instead.
    """

    done: str
    outcome: str
    why: str

    @classmethod
    def run(cls, done: str) -> _UsedFingerprint:
        """A run recorded or replayed: a second trace would carry its IDs."""
        return cls(
            done=done,
            outcome="",
            why="two traces would carry one run's IDs. Give each task its own prompt, or run it "
            "with another executor",
        )

    @classmethod
    def failed_recording(cls, session: str, failure: str) -> _UsedFingerprint:
        """A billed run whose recording failed: running the prompt again would bill it twice."""
        return cls(
            done=f"tried to record the run of session {session}",
            outcome=f", but {failure}",
            why="it already had its one billed run in this executor. Fix what stopped the "
            "recording, then run the prompt with a new executor, or without a cassette",
        )


class AgentCoreRuntimeExecutor:
    """An ``ExecutorBackend`` that runs each task against an AgentCore Runtime agent.

    Use it as a context manager. Leaving the context stops every session a run opened whose stop
    did not succeed, logging each result, closes the AWS clients the executor built (and the
    invocation clients it derived from yours), and removes its temporary log directory. A failure
    on the way out is raised when the block succeeded; when the block raised, the block's error
    is raised, with the failure logged as a warning and noted on it.

    Tasks run one after another; an executor is not safe to share across threads.

    Args:
        target: The runtime, endpoint qualifier, payload format and configuration bundle to use.
        span_reader: Reads each run's span tree: any object with the ``read_spans`` method of
            :class:`~calibrated_agent_evaluation.agentcore.run_spans.RunSpanReader`. Needed whenever
            the executor can call AWS: with no cassette, or a cassette in record or auto mode.
        clients: AWS clients to use instead of building them, for the runtime ARN's region. The
            executor installs no billed-call budget on them. Give these or ``profile``, not both.
        profile: The AWS profile to build clients from; the default credential chain when
            ``None``. Read on the first live call, never before.
        payload_builder: Builds the request body for the ``custom`` payload format; give none
            with a named format.
        builder_name: Names ``payload_builder``, so two builders are told apart. ``custom``
            needs one; a named format takes none. An empty name counts as none.
        cassette_path: A cassette to record runs into or replay them from.
        cassette_mode: Whether the cassette records, replays or does both; pass it only with
            ``cassette_path``. When ``None``, ``CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE`` decides
            (auto when unset); without a cassette, its record or auto mode runs every task live.
        log_dir: The directory whose run manifest gets each run's record; a temporary directory,
            removed on exit, when ``None``. A ``log_dir`` passed to :meth:`run` wins.
        frame: The sampling frame every returned ``Trace`` is stamped with.
        read_timeout_seconds: How long an invocation waits for each read of its response: a
            positive, finite number of seconds of at most ``threading.TIMEOUT_MAX``, the longest
            a socket can wait.
        attempt_limit: The most attempts one invocation may send, the first one included.
        sleep: Waits between retried attempts.
        clock: The current time, timezone-aware; it times every run.

    Raises:
        RuntimeInvocationError: ``clients`` and ``profile`` were both given, or ``clients`` are
            for another region; ``CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE`` or ``cassette_mode``
            names no mode; ``frame`` is not a sampling frame; the payload format and the builder or
            builder name don't pair up; the read timeout is out of the range above (a bool is not a
            number here), or the attempt limit is below 1; replay is asked for without a cassette,
            or ``cassette_mode`` without ``cassette_path``; or the executor can call AWS but has no
            span reader, or ``span_reader`` is not one.
    """

    def __init__(
        self,
        target: RuntimeTarget,
        *,
        span_reader: RunSpanReader | None = None,
        clients: AwsClients | None = None,
        profile: str | None = None,
        payload_builder: PayloadBuilder | None = None,
        builder_name: str | None = None,
        cassette_path: str | Path | None = None,
        cassette_mode: Mode | str | None = None,
        log_dir: str | Path | None = None,
        frame: Frame = "production-representative",
        read_timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS,
        attempt_limit: int = DEFAULT_ATTEMPT_LIMIT,
        sleep: Callable[[float], object] = time.sleep,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if clients is not None and profile is not None:
            # Refused rather than one of them silently ignored.
            raise RuntimeInvocationError(
                f"both clients= and profile={profile!r} were given, but clients passed in are "
                "used as they are and the profile would be ignored; pass clients=, or profile= "
                "and let the executor build the clients, not both"
            )
        self._target = target
        self._cassette_path = None if cassette_path is None else Path(cassette_path)
        if cassette_mode is not None:
            cassette_mode = _cassette_mode(cassette_mode)
        self._cassette_mode = cassette_mode or _mode_from_environment()
        _check_frame(frame)
        if clients is not None and clients.region != target.region:
            raise RuntimeInvocationError(
                f"the AWS clients passed in are for region {clients.region!r}, but runtime "
                f"{target.runtime_arn} is in {target.region!r}; pass clients for "
                f"{target.region!r}, or none and let the executor build them"
            )
        _check_builder(target, payload_builder, builder_name)
        check_limits(read_timeout_seconds, attempt_limit)
        replay = self._cassette_mode is Mode.REPLAY
        if replay and self._cassette_path is None:
            raise RuntimeInvocationError(
                "replay needs a cassette file: the cassette mode is replay, but no cassette was "
                "given. Pass the cassette to replay, or use record or auto mode to call AWS"
            )
        if cassette_mode is not None and self._cassette_path is None:
            # Refused, not ignored: every run would go live with nothing recorded.
            raise RuntimeInvocationError(
                f"cassette_mode={cassette_mode.value!r} was given without a cassette_path, so "
                "there is no cassette to record into and the mode would be ignored, with every "
                "run going live and billed. Pass cassette_path= too, or drop cassette_mode= to "
                "run every task live without a cassette"
            )
        if not replay and span_reader is None:
            raise RuntimeInvocationError(
                "this executor can call AWS, and live runs need the AgentCore span reader to "
                "read each run's spans, but none was given; offline runs need a cassette in "
                "replay mode. Pass span_reader=, or replay from a cassette"
            )
        if not replay and not isinstance(span_reader, RunSpanReader):
            # Refused here rather than after the first billed invocation, when its spans are read.
            raise RuntimeInvocationError(
                f"span_reader= is a {type(span_reader).__name__}, which has no read_spans method, "
                "so it is not a span reader, and live runs need the AgentCore span reader to read "
                "each run's spans; offline runs need a cassette in replay mode. Pass a span "
                "reader, or replay from a cassette"
            )
        self._span_reader = span_reader
        self._payload_builder = payload_builder
        self._builder_name = builder_name
        self._explicit_log_dir = None if log_dir is None else Path(log_dir)
        self._frame: Frame = frame
        # The clients, the preflight, the invocations, the session stops and the run records.
        self._boundary = RuntimeCallBoundary(
            target,
            clients=clients,
            profile=profile,
            read_timeout_seconds=read_timeout_seconds,
            attempt_limit=attempt_limit,
            sleep=sleep,
            clock=clock,
            owner=_OWNER,
        )
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._cassette: Cassette | None = None
        # Each fingerprint this executor recorded, tried to record or replayed. A later task with
        # one is refused, so no two traces carry one run's IDs and no failed recording is rebilled.
        self._used_fingerprints: dict[str, _UsedFingerprint] = {}
        # The session each conversation in progress runs on, by conversation ID.
        self._conversations: dict[str, str] = {}

    @property
    def run_records(self) -> tuple[RunRecord, ...]:
        """The record of every task run this executor finished, whatever its outcome, in order.

        A record is listed even when the run manifest could not take it; :attr:`manifest_errors`
        says which writes failed.
        """
        return self._boundary.run_records

    @property
    def manifest_errors(self) -> tuple[RuntimeInvocationError, ...]:
        """Why each run record that could not be added to the run manifest was not, in order.

        One error per failed write, each naming the manifest's path and what to do next. Empty
        when every record in :attr:`run_records` is in the manifest. A successful run whose record
        can't be written raises its error; a failed run keeps its own error, with this one noted.
        """
        return self._boundary.manifest_errors

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # The body's own error is the one the caller sees; a failure on the way out, an interrupt
        # included, is logged and noted on it.
        close_on_exit(self.close, exc, owner=_OWNER)

    def close(self) -> None:
        """Stop the sessions left open, close the clients built here, remove the temp log dir.

        Each session whose stop did not succeed, a failed or interrupted one included, gets one
        more ``StopRuntimeSession``, its result logged with its session ID. Every step runs even
        when an earlier one fails.

        Raises:
            RuntimeInvocationError: The client to stop the open sessions could not be built, an
                AWS client could not be closed, or the temporary log directory could not be
                removed. The first failure is raised; each later one is logged as a warning and
                noted on it.
        """
        # The boundary's two cleanup steps and the executor's own, all run even when one fails.
        run_all(
            (self._stop_open_sessions, self._boundary.close_clients, self._remove_tmpdir),
            owner=_OWNER,
        )

    def _stop_open_sessions(self) -> None:
        self._boundary.stop_open_sessions()

    def _remove_tmpdir(self) -> None:
        tmpdir, self._tmpdir = self._tmpdir, None
        if tmpdir is None:
            return
        try:
            tmpdir.cleanup()
        except Exception as exc:
            raise RuntimeInvocationError(
                f"could not remove the temporary log directory {tmpdir.name} "
                f"({type(exc).__name__}), which holds the run manifest; remove it yourself, or "
                "pass log_dir= to keep the run manifest in a directory you choose"
            ) from exc

    def run(self, task: Any, *, log_dir: str | Path | None = None) -> list[Trace]:
        """Run one task against the runtime and return its validated, owned ``Trace``.

        The run's record goes to the run manifest (``agentcore_run_record.json`` and ``.csv``)
        in ``log_dir``, else in the executor's log directory, whatever the outcome. With a
        cassette, a recorded run is replayed, and a live run is recorded once its ``Trace`` is
        valid.

        Args:
            task: An :class:`AgentCoreRuntimeTask`.
            log_dir: The directory whose run manifest gets this run's record, instead of the
                executor's log directory.

        Returns:
            One ``Trace``.

        Raises:
            RuntimeInvocationError: Before any AWS call and with no run record: the task is of
                another type, its prompt is not a non-empty string, its ``trace_id`` is not a
                string or None, or its ``taxonomy_cell`` is not a mapping of strings to strings.
                Otherwise the run was refused or failed, named by the failing step and the
                session ID. A run record that can't be written is refused naming the manifest's
                path, or that path is noted on the run's own error when the run already failed.
                With a cassette, refusals name its path or the fingerprint: the cassette is
                missing in replay mode, or can't be read, or can't be written after a live run; a
                replay miss; an entry recorded under another runtime, qualifier or configuration
                bundle, or with no record of which one; or a fingerprint this executor already
                recorded, tried to record or replayed, named with that earlier run's session.
        """
        if not isinstance(task, AgentCoreRuntimeTask):
            raise RuntimeInvocationError(
                f"AgentCoreRuntimeExecutor.run expects an AgentCoreRuntimeTask, got "
                f"{type(task).__name__}; build tasks with AgentCoreTaskBuilder"
            )
        # Checked first, as the probe adapter checks: the prompt becomes the billed request's body.
        check_prompt(
            task.prompt,
            named="the task's prompt",
            remedy="Give each AgentCoreRuntimeTask the text to send to the agent",
        )
        _check_task_fields(task)
        log_dir = None if log_dir is None else Path(log_dir)
        ids = self._ids_for(task.conversation)
        run = self._boundary.start_run(ids, task.trace_id or f"agentcore-{ids.w3c_trace_id}")
        try:
            trace = self._execute(task, run)
        except BaseException as exc:
            self._finish(run, log_dir, failure=exc)
            raise
        self._finish(run, log_dir, failure=None)
        return [trace]

    def end_conversation(self, conversation_id: str) -> SessionStop | None:
        """Stop the runtime session a conversation ran on, once its last turn has run.

        Call it when the conversation ends, fails or is interrupted. A conversation with no open
        session (unknown, already ended, or replayed from a cassette) is let go and nothing is
        stopped. A session whose stop did not succeed stays held, and :meth:`close` tries it once
        more.

        Each turn's run record says ``held_open``; the stop's own result is returned, and a stop
        that did not succeed is logged as a warning.

        Args:
            conversation_id: The conversation's ID, as its tasks' ``conversation`` named it.

        Returns:
            How the stop ended, or ``None`` when no session was open.

        Raises:
            RuntimeInvocationError: The client to stop the session could not be built.
        """
        session = self._conversations.pop(conversation_id, None)
        if session is None:
            return None
        return self._boundary.stop_session(session)

    # --- one run ----------------------------------------------------------------------------

    def _ids_for(self, conversation: ConversationTurn | None) -> RunIds:
        if conversation is None:
            return new_run_ids()
        session = self._conversations.get(conversation.conversation_id)
        if session is None:
            ids = new_run_ids()
            self._conversations[conversation.conversation_id] = ids.session_id
            return ids
        return turn_run_ids(session)

    def _execute(self, task: AgentCoreRuntimeTask, run: RunState) -> Trace:
        if self._cassette_path is None:
            # No cassette, no fingerprint: a repeated prompt is simply run again.
            return self._trace(task, self._live(task, run), run.trace_id)
        return self._through_cassette(task, run, self._cassette_path)

    def _through_cassette(self, task: AgentCoreRuntimeTask, run: RunState, path: Path) -> Trace:
        turn = task.conversation
        request = cassette_request(
            self._target,
            self._builder_name,
            task.prompt,
            conversation=None if turn is None else (turn.conversation_id, turn.turn),
        )
        fp = fingerprint(request)
        earlier = self._used_fingerprints.get(fp)
        if earlier is not None:
            raise RuntimeInvocationError(
                f"this executor already {earlier.done} for fingerprint {fp} (the same payload "
                f"format, builder and prompt) in cassette {path}{earlier.outcome}, so the prompt "
                f"is neither replayed nor recorded again for this task: {earlier.why}"
            )
        if self._cassette is None:
            self._cassette = open_cassette(path, self._cassette_mode)
        cassette = self._cassette
        live: list[Trace] = []
        session = run.ids.session_id

        def live_call() -> dict[str, Any]:
            recorded = self._live(task, run)
            try:
                entry = encode_entry(recorded, cassette_path=path)
            except RuntimeInvocationError:
                # Billed and not kept: a later task with this prompt is refused, not billed again.
                self._used_fingerprints[fp] = _UsedFingerprint.failed_recording(
                    session, "the run's entry is not standard JSON and was not kept"
                )
                raise
            # Built from the entry as a replay reads it, so an invalid run is never recorded and a
            # replay equals this run field for field.
            try:
                trace = self._trace(
                    task, decode_entry(entry, cassette_path=path, fingerprint=fp), run.trace_id
                )
            except RuntimeInvocationError:
                # Billed and not kept, like an entry that is not standard JSON.
                self._used_fingerprints[fp] = _UsedFingerprint.failed_recording(
                    session, "the run's trace is not valid and was not kept"
                )
                raise
            live.append(trace)
            # Marked before the cassette is written, so a run whose write fails still counts.
            self._used_fingerprints[fp] = _UsedFingerprint.run(
                f"recorded the run of session {session}"
            )
            return entry

        try:
            # An entry without the stamp every recording carries shows no runtime, so it is refused.
            response = cassette.play(
                request, live_call, model=cassette_stamp(self._target), require_model=True
            )
        except CassetteMiss as exc:
            raise RuntimeInvocationError(
                f"cassette miss: cassette {path} has no recorded run for fingerprint {fp}, "
                "and replay mode never calls AWS, so nothing was run. Record the prompt first, "
                "or use auto mode to run it live when it is missing"
            ) from exc
        except CassetteModelMismatch as exc:
            # Caught before its base, CassetteError. Its message names both stamps, or the run's
            # alone when the entry has none.
            raise RuntimeInvocationError(
                f"cassette {path} holds the run for fingerprint {fp}, but not under this run's "
                f"runtime, qualifier and configuration bundle, so it is not replayed: {exc}"
            ) from exc
        except CassetteError as exc:
            # Only a failed write raises it here, after the live run. The cassette drops the entry
            # it was writing, so no later write saves this run and no lookup finds it.
            if live:
                self._used_fingerprints[fp] = _UsedFingerprint.failed_recording(
                    session, "the cassette could not be written and does not hold the run"
                )
            raise RuntimeInvocationError(
                f"the run of session {run.ids.session_id} finished, but cassette {path} could "
                f"not be written ({type(exc.__cause__ or exc).__name__}), so the run is not kept "
                "in the cassette and its trace is not returned. Check that the cassette's "
                "directory is writable, then run again"
            ) from exc
        if live:
            return live[0]
        recorded = decode_entry(response, cassette_path=path, fingerprint=fp)
        self._replayed(task, run, recorded)
        trace = self._trace(task, recorded, run.trace_id)
        self._used_fingerprints[fp] = _UsedFingerprint.run(
            f"replayed the run of session {recorded.session_id}"
        )
        return trace

    def _replayed(self, task: AgentCoreRuntimeTask, run: RunState, recorded: RecordedRun) -> None:
        # The record names the recorded run, under the task's trace ID when it names one. A replay
        # opens no session, so none is stopped.
        run.trace_id = task.trace_id or recorded.trace_id
        run.session_id = recorded.session_id
        run.w3c_trace_id = recorded.w3c_trace_id
        run.started = datetime.fromisoformat(recorded.started_at)
        run.ended = datetime.fromisoformat(recorded.ended_at)
        run.attempts = recorded.attempts
        run.status_code = recorded.status_code
        run.content_type = recorded.content_type
        run.response_traceparent = recorded.response_traceparent
        run.runtime_info = RuntimeInfo(
            runtime_version=recorded.runtime_version,
            version_source=recorded.runtime_version_source,
        )
        run.replayed = True

    def _live(self, task: AgentCoreRuntimeTask, run: RunState) -> RecordedRun:
        session = run.ids.session_id
        # Built before any client, so a body that can't be made costs no AWS call.
        body = request_body(
            self._target, task.prompt, session, self._payload_builder, self._builder_name
        )

        def answer(invocation: Invocation) -> tuple[tuple[DecodedValue, ...], list[Span]]:
            content_type = invocation.response.content_type or ""
            values = decode(content_type, invocation.response.lines, session_id=session)
            check_errors(values, session_id=session)
            # Refused here, before the spans are read: a run with no answer did not succeed.
            final_text(values, content_type, session_id=session)
            return values, self._read_spans(run, invocation.started, invocation.ended)

        invocation, (values, spans) = self._boundary.call(
            run, body, answer, stop_session=task.conversation is None
        )
        raw, info = invocation.response, invocation.runtime_info
        return RecordedRun(
            content_type=raw.content_type or "",
            status_code=raw.status_code,
            values=list(values),
            spans=spans,
            runtime_version=info.runtime_version,
            runtime_version_source=info.version_source,
            session_id=session,
            w3c_trace_id=run.ids.w3c_trace_id,
            trace_id=run.trace_id,
            response_traceparent=raw.response_traceparent,
            started_at=utc_iso(invocation.started),
            ended_at=utc_iso(invocation.ended),
            attempts=raw.attempts,
        )

    def _read_spans(self, run: RunState, started: datetime, ended: datetime) -> list[Span]:
        session = run.ids.session_id
        reader = self._span_reader
        if reader is None:
            # The constructor refuses a live executor without a reader, so this is a defect.
            raise RuntimeInvocationError(
                f"no span reader to read trace {run.trace_id} (session {session}); pass "
                "span_reader="
            )
        query = RunSpanQuery(
            trace_id=run.trace_id,
            w3c_trace_id=run.ids.w3c_trace_id,
            session_id=session,
            runtime_arn=self._target.runtime_arn,
            qualifier=self._target.qualifier,
            started_at=utc_iso(started),
            ended_at=utc_iso(ended),
        )
        try:
            spans = list(reader.read_spans(query))
        except Exception as exc:
            guard = _guard_refusal_in(exc)
            if guard is not None:
                # The executor didn't make the call, so the guard's error names it.
                operation = guard.operation
                refused = "an AWS call" if operation is None else ":".join(operation)
                raise RuntimeInvocationError(
                    f"{guard_refusal(guard, refused)}, a call the span reader made for trace "
                    f"{run.trace_id} (session {session}), so the run's spans were not read and "
                    f"it returns no trace. {guard_hint(guard)}"
                ) from exc
            # Nothing re-reads a finished run's spans, so a retry is a new billed invocation.
            raise RuntimeInvocationError(
                f"the span reader could not read the spans of trace {run.trace_id} (session "
                f"{session}): it failed with {type(exc).__name__}, so the run returns no trace. "
                f"{_reader_message(exc)}The run's record keeps its trace and session IDs, but "
                "nothing reads a finished run's spans again yet, so a retry means running the "
                "task again, which is a new billed invocation of the runtime"
            ) from exc
        if not spans:
            raise RuntimeInvocationError(
                f"the span reader returned no spans for trace {run.trace_id} (session {session}), "
                "so the run returns no trace. Check that the runtime exports its spans"
            )
        if not all(isinstance(span, Span) for span in spans):
            raise RuntimeInvocationError(
                f"the span reader returned something other than normalized spans for trace "
                f"{run.trace_id} (session {session}), so the run returns no trace. Check the reader"
            )
        if any(span.trace_id != run.trace_id for span in spans):
            raise RuntimeInvocationError(
                f"the span reader returned a span of another trace for trace {run.trace_id} "
                f"(session {session}), so the run returns no trace. Check the reader"
            )
        for span in spans:
            _require_json_span(span, run.trace_id, session)
        return spans

    def _trace(self, task: AgentCoreRuntimeTask, recorded: RecordedRun, trace_id: str) -> Trace:
        session = recorded.session_id
        # A replayed entry is checked like a live answer, so an error value never becomes a trace.
        check_errors(recorded.values, session_id=session)
        text = final_text(recorded.values, recorded.content_type, session_id=session)
        try:
            trace = Trace(
                trace_id=trace_id,
                frame=self._frame,
                ownership="owned",
                input={"prompt": task.prompt},
                output={"text": text},
                taxonomy_cell=dict(task.taxonomy_cell),
                spans=rekey_spans(recorded.spans, recorded.trace_id, trace_id),
                captured_at=recorded.started_at,
                # The runtime session the agent ran under, so turns of one session group together.
                session_id=session,
            )
            validate(trace)
        except SchemaError as exc:
            raise RuntimeInvocationError(
                f"the trace {trace_id} of session {session} is not valid "
                f"({type(exc).__name__}), so it is not returned. Check that the span reader "
                "returns the run's whole span tree, or re-record the cassette entry"
            ) from exc
        return trace

    def _finish(
        self, run: RunState, log_dir: Path | None, *, failure: BaseException | None
    ) -> None:
        self._boundary.finish(run, lambda: self._manifest_path(log_dir), failure=failure)

    def _manifest_path(self, log_dir: Path | None) -> Path:
        """The log directory whose run manifest gets this run's record."""
        directory = log_dir or self._explicit_log_dir
        if directory is None:
            if self._tmpdir is None:
                try:
                    self._tmpdir = tempfile.TemporaryDirectory(
                        prefix="calibrated_agent_evaluation_agentcore_"
                    )
                except OSError as exc:
                    raise RuntimeInvocationError(
                        f"could not create a temporary log directory for the run manifest "
                        f"({type(exc).__name__}); pass log_dir="
                    ) from exc
            directory = Path(self._tmpdir.name)
        return directory


class AgentCoreTaskBuilder:
    """Turns synthesized inputs into :class:`AgentCoreRuntimeTask`s, one per input.

    Each task's prompt is the input's text and its ``taxonomy_cell`` the input's cell, so every
    returned trace can be traced to the cell it was written for.
    """

    def build_all(self, inputs: list[SynthInput]) -> list[AgentCoreRuntimeTask]:
        """One task per input, in order.

        Args:
            inputs: The synthesized inputs.

        Returns:
            The tasks.

        Raises:
            SynthError: ``inputs`` is empty.
        """
        # Imported here: calibrated_agent_evaluation.synth loads the sampling stack, which
        # [agentcore] does not install.
        from calibrated_agent_evaluation.synth.errors import SynthError

        if not inputs:
            raise SynthError(
                "AgentCoreTaskBuilder.build_all received no inputs; refusing an empty run"
            )
        return [self.build([item]) for item in inputs]

    def build(self, inputs: list[SynthInput]) -> AgentCoreRuntimeTask:
        """The task for a single input. Use :meth:`build_all` for a list.

        Args:
            inputs: Exactly one synthesized input.

        Returns:
            The task.

        Raises:
            SynthError: ``inputs`` is empty, or holds more than one input, which only
                :meth:`build_all` can run without dropping the rest.
        """
        from calibrated_agent_evaluation.synth.errors import SynthError

        if len(inputs) > 1:
            raise SynthError(
                f"AgentCoreTaskBuilder.build received {len(inputs)} inputs but a task carries "
                "exactly one; call build_all(inputs) to run them all"
            )
        if not inputs:
            raise SynthError(
                "AgentCoreTaskBuilder.build received no inputs; refusing an empty task"
            )
        item = inputs[0]
        return AgentCoreRuntimeTask(prompt=item.text, taxonomy_cell=dict(item.cell))

    def build_conversation_turn(
        self, item: SynthInput, history: list[dict[str, Any]], turn: ConversationTurn
    ) -> AgentCoreRuntimeTask:
        """The task for one agent turn of a simulated conversation.

        A runtime keeps the conversation's memory in its session, so the turn sends only the
        latest customer message; ``turn`` keeps every turn of the conversation on one session.

        Args:
            item: The synthesized input the conversation started from.
            history: The conversation so far, ending with the customer's message for this turn.
            turn: Which conversation and turn this is.

        Returns:
            The turn's task.

        Raises:
            SynthError: ``history`` is empty or does not end with a customer message.
        """
        from calibrated_agent_evaluation.synth.conversation import message_text
        from calibrated_agent_evaluation.synth.errors import SynthError

        if not history or history[-1].get("role") != "user":
            raise SynthError(
                "a conversation turn needs the history up to the customer's latest message"
            )
        return AgentCoreRuntimeTask(
            prompt=message_text(history[-1]),
            taxonomy_cell=dict(item.cell),
            conversation=turn,
        )


# --- helpers ----------------------------------------------------------------------------------


def _mode_from_environment() -> Mode:
    try:
        mode = mode_from_env(default=Mode.AUTO)
    except ValueError as exc:
        raise RuntimeInvocationError(
            f"{MODE_ENV_VAR} is set to {os.environ.get(MODE_ENV_VAR, '')!r}, which is not a "
            f"cassette mode; set it to one of {', '.join(item.value for item in Mode)}, or "
            "unset it"
        ) from exc
    return mode


def _check_builder(
    target: RuntimeTarget, builder: PayloadBuilder | None, builder_name: str | None
) -> None:
    # An empty builder name counts as not given, as an empty bundle part does.
    named = bool(builder_name)
    if target.payload_format == CUSTOM_PAYLOAD_FORMAT:
        missing = [
            name
            for name, given in (("payload_builder", builder is not None), ("builder_name", named))
            if not given
        ]
        if missing:
            raise RuntimeInvocationError(
                f"payload_format={CUSTOM_PAYLOAD_FORMAT!r} needs both a payload_builder and a "
                f"builder_name, but {' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} "
                "missing (an empty builder_name counts as missing); the name tells one builder's "
                "recorded runs from another's"
            )
        return
    stray = [
        name
        for name, given in (("payload_builder", builder is not None), ("builder_name", named))
        if given
    ]
    if stray:
        raise RuntimeInvocationError(
            f"{' and '.join(stray)} {'was' if len(stray) == 1 else 'were'} given with "
            f"payload_format={target.payload_format!r}, which builds its own body and takes "
            f"neither; use payload_format={CUSTOM_PAYLOAD_FORMAT!r} with a builder and its name, "
            f"or drop {'it' if len(stray) == 1 else 'them'}"
        )


def _cassette_mode(mode: Mode | str) -> Mode:
    # A Mode is a StrEnum, so a caller may pass its text; the executor compares by identity.
    try:
        return Mode(mode)
    except ValueError:
        raise RuntimeInvocationError(
            f"cassette_mode={mode!r} is not a cassette mode; pass one of "
            f"{', '.join(item.value for item in Mode)}"
        ) from None


def _check_frame(frame: object) -> None:
    frames = get_args(Frame)
    if frame not in frames:
        raise RuntimeInvocationError(
            f"frame={frame!r} is not a sampling frame; pass one of {', '.join(map(repr, frames))}"
        )


def _check_task_fields(task: AgentCoreRuntimeTask) -> None:
    # Checked first, like the prompt: both are used only after the billed call, where a bad value
    # would cost the call and lose the run.
    if task.trace_id is not None and not isinstance(task.trace_id, str):
        raise RuntimeInvocationError(
            f"the task's trace_id is of type {type(task.trace_id).__name__}, and it must be a "
            "string or None, so nothing was run"
        )
    turn = task.conversation
    if turn is not None and not (
        isinstance(turn, ConversationTurn)
        and isinstance(turn.conversation_id, str)
        and turn.conversation_id
        and isinstance(turn.turn, int)
        and not isinstance(turn.turn, bool)
        and turn.turn >= 1
    ):
        raise RuntimeInvocationError(
            "the task's conversation must be a ConversationTurn with a non-empty conversation_id "
            "and a turn of 1 or more, so nothing was run"
        )
    cell = task.taxonomy_cell
    if not isinstance(cell, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in cell.items()
    ):
        raise RuntimeInvocationError(
            "the task's taxonomy_cell must be a mapping of strings to strings, so nothing was run"
        )


def _require_json_span(span: Span, trace_id: str, session: str) -> None:
    # On the Python dump: pydantic's JSON dump would write a NaN or an infinity as null silently.
    # The depth is counted from the span itself.
    refused = (
        f"span {span.span_id} that the span reader returned for trace {trace_id} (session "
        f"{session})"
    )
    outcome = "so the run returns no trace and nothing of it is recorded. Check the reader"
    try:
        require_finite(span.model_dump(mode="python"), where=f"span {span.span_id}")
    except NonFiniteNumberError as exc:
        raise RuntimeInvocationError(
            f"{refused} holds a number JSON cannot represent (NaN or an infinity), {outcome}"
        ) from exc
    except (NestedTooDeepError, RecursionError) as exc:
        raise RuntimeInvocationError(
            f"{refused} nests more than {MAX_JSON_DEPTH} levels deep, {outcome}"
        ) from exc


def _guard_refusal_in(exc: BaseException) -> BilledCallBudgetExceeded | ReadOnlyViolation | None:
    # The error itself, or one it was raised from or while handling.
    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(_CHAIN_DEPTH):
        if current is None or id(current) in seen:
            return None
        if isinstance(current, _GUARD_ERRORS):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def _reader_message(exc: Exception) -> str:
    # The reader contract keeps an AgentCoreError's text to IDs, log groups, counts and times, so
    # it is repeated; any other error's text may quote span content.
    if not isinstance(exc, AgentCoreError):
        return ""
    try:
        text = str(exc).strip()
    except Exception as unreadable:  # noqa: BLE001 - named in the refusal, which is raised
        # A message that fails to render must not replace the executor's refusal.
        return f"The reader's message could not be read ({type(unreadable).__name__}). "
    if not text:
        return ""
    end = "" if text.endswith((".", "!", "?")) else "."
    return f"The reader's message: {text}{end} "
