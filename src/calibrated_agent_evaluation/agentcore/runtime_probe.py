# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe an agent hosted on Amazon Bedrock AgentCore Runtime, one prompt per run.

:class:`RuntimeProbeAdapter` is a :class:`~calibrated_agent_evaluation.probe.adapter.ProbeAdapter`,
so :func:`~calibrated_agent_evaluation.probe.harness.probe_with` can send the probe's prompts
through it and build the probe's report from the replies. It reads only each reply, never the
agent's spans, so the probe stores outcome-only ``probed`` traces.

Each prompt is one run of its own, as one task is for the executor: a new runtime session ID and
W3C trace context, a request body in the named payload format, a read-only preflight before the
first invocation, one invocation (retrying only a busy session or throttling, with the same IDs
and body), the reply decoded and checked, the session stopped, and one
:class:`~calibrated_agent_evaluation.agentcore.run_manifest.RunRecord` added to the log directory's
run manifest whatever the outcome. The reply's trace is named
``probe-<fingerprint>-<W3C trace ID>``, where the fingerprint is the endpoint fingerprint of the
adapter's :attr:`~RuntimeProbeAdapter.target_label`, ``<runtime ARN>@<qualifier>``, so each run
record points at exactly one stored trace. Pass that label to
:func:`~calibrated_agent_evaluation.probe.harness.probe_with` as its ``target``, so each trace's
``endpoint_fingerprint`` is the one its trace ID carries.

The adapter builds no client of its own and picks no directory: the caller passes the AWS
clients, with any billed-call budget already installed, and the log directory. It derives one
invocation view from those clients, and closing the adapter closes only that view.

Its calls go through
:class:`~calibrated_agent_evaluation.agentcore.runtime_boundary.RuntimeCallBoundary`, the call path
the executor uses too, so a prompt's preflight, session stop and run record follow the executor's
rules. A prompt that is not a non-empty string is refused by the executor's own check before its run
starts. The adapter adds the probe trace ID and the probe reply.

Every failure raises
:class:`~calibrated_agent_evaluation.agentcore.runtime_errors.RuntimeInvocationError`, chained from
its cause. Messages and log lines name IDs, operations, error codes and types, paths and settings;
they never carry a prompt, a request body, a reply or an agent's error text.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Self

from calibrated_agent_evaluation.agentcore.endpoint_label import endpoint_label
from calibrated_agent_evaluation.agentcore.run_manifest import RunRecord
from calibrated_agent_evaluation.agentcore.runtime import RuntimeTarget, new_run_ids, request_body
from calibrated_agent_evaluation.agentcore.runtime_boundary import (
    DEFAULT_ATTEMPT_LIMIT,
    DEFAULT_READ_TIMEOUT_SECONDS,
    Invocation,
    RunState,
    RuntimeCallBoundary,
    check_limits,
    check_prompt,
    close_on_exit,
    utc_now,
)
from calibrated_agent_evaluation.agentcore.runtime_errors import RuntimeInvocationError
from calibrated_agent_evaluation.agentcore.runtime_response import (
    DecodedValue,
    check_errors,
    decode,
    final_text,
    is_event_stream,
)
from calibrated_agent_evaluation.probe.adapter import PAYLOAD_FORMATS, ProbeResponse, tool_names
from calibrated_agent_evaluation.probe.fingerprint import endpoint_fingerprint

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.session import AwsClients

__all__ = ["RuntimeProbeAdapter"]

# The request shapes a probe can send; a custom builder's body is not one of them.
_PROBE_FORMATS: tuple[str, ...] = tuple(payload_format.name for payload_format in PAYLOAD_FORMATS)
# What a refusal for clients that could not be built tells the caller to check: the adapter only
# ever uses the clients passed in.
_SETUP_HINT = "Check the AWS setup of the clients passed in"
# What cleanup messages call the adapter.
_OWNER = "the probe adapter"


class RuntimeProbeAdapter:
    """A probe adapter that sends each prompt to an AgentCore Runtime agent as one run.

    Pass it to :func:`~calibrated_agent_evaluation.probe.harness.probe_with`, with
    :attr:`target_label` as the probe's ``target``. Use it as a context manager: leaving the context
    stops every runtime session a prompt opened whose stop did not succeed, logging each result, and
    closes the invocation view it derived from your clients. Your clients stay open; they are yours
    to close. A failure on the way out is raised when the block succeeded. When the block raised,
    its own error is the one raised, and the failure is logged as a warning and noted on it.

    Prompts run one after another; an adapter is not safe to share across threads.

    Args:
        target: The runtime, endpoint qualifier, payload format and configuration bundle to
            probe. The format is one of the probe's request shapes: ``generic_message``,
            ``openai``, ``agentcore_sse`` or ``nested_agentcore``.
        clients: The AWS clients every call goes through, built for the region in the runtime
            ARN, with any billed-call budget already installed. The adapter adds no guard.
        log_dir: The directory whose run manifest (``agentcore_run_record.json`` and ``.csv``)
            gets each prompt's run record.
        read_timeout_seconds: How long an invocation waits for each read of its reply: a
            positive, finite number of seconds of at most ``threading.TIMEOUT_MAX``, the longest
            a socket can wait.
        attempt_limit: The most attempts one prompt's invocation may send, the first included.
        sleep: Waits between retried attempts.
        clock: The current time, timezone-aware; it times every prompt's run.

    Raises:
        RuntimeInvocationError: Raised before any client is built: the payload format is
            ``custom`` or not one of the probe's four, which the message lists; ``clients`` are
            for another region than the runtime ARN's, both named; the read timeout is not a
            positive, finite number of seconds of at most ``threading.TIMEOUT_MAX`` (a bool is
            not one); or the attempt limit is below 1. Each message names the value.
    """

    def __init__(
        self,
        target: RuntimeTarget,
        *,
        clients: AwsClients,
        log_dir: Path,
        read_timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS,
        attempt_limit: int = DEFAULT_ATTEMPT_LIMIT,
        sleep: Callable[[float], object] = time.sleep,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        _check_format(target)
        if clients.region != target.region:
            raise RuntimeInvocationError(
                f"the AWS clients passed in are for region {clients.region!r}, but runtime "
                f"{target.runtime_arn} is in {target.region!r}; pass clients built for "
                f"{target.region!r}"
            )
        check_limits(read_timeout_seconds, attempt_limit)
        self._target = target
        self._log_dir = Path(log_dir)
        self._label = endpoint_label(target.runtime_arn, target.qualifier)
        # The endpoint fingerprint every probed trace is stamped with, so a run record's trace_id
        # names exactly one stored trace.
        self._fingerprint = endpoint_fingerprint(self._label)
        # The clients, the invocation view, the preflight, the stops and the run records.
        self._boundary = RuntimeCallBoundary(
            target,
            clients=clients,
            read_timeout_seconds=read_timeout_seconds,
            attempt_limit=attempt_limit,
            sleep=sleep,
            clock=clock,
            setup_hint=_SETUP_HINT,
            closed_on="close",
            owner=_OWNER,
        )

    @property
    def target_label(self) -> str:
        """What this adapter probes, ``<runtime ARN>@<qualifier>``, to pass as ``target``.

        Every trace ID this adapter names carries the endpoint fingerprint of this label. Passed
        to :func:`~calibrated_agent_evaluation.probe.harness.probe_with` as ``target``, it names the
        target in the authorization refusal, and every trace and the finding carry that same
        fingerprint.
        """
        return self._label

    @property
    def run_records(self) -> tuple[RunRecord, ...]:
        """The record of every prompt this adapter sent, whatever its outcome, in order.

        A record is listed even when the run manifest could not take it; :attr:`manifest_errors`
        says which writes failed.
        """
        return self._boundary.run_records

    @property
    def manifest_errors(self) -> tuple[RuntimeInvocationError, ...]:
        """Why each run record that could not be added to the run manifest was not, in order.

        One error per failed write, each naming the manifest's path and what to do next. Empty
        when every record in :attr:`run_records` is in the manifest. A prompt that succeeded but
        whose record can't be written raises its error; a failed prompt keeps its own error, with
        this one noted.
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
        # The body's own error is the one the caller sees. Any failure on the way out, to stop
        # the open sessions or to close the view, is logged and noted on that error, not raised
        # over it.
        close_on_exit(self.close, exc, owner=_OWNER)

    def close(self) -> None:
        """Stop the sessions left open, then close the invocation view derived from your clients.

        Each session whose stop did not succeed, a failed or interrupted one included, gets one
        more ``StopRuntimeSession``, and its result is logged at INFO with its session ID. The
        view is closed even when stopping fails. Your clients are left open.

        Raises:
            RuntimeInvocationError: The client to stop the open sessions could not be built, or
                the view could not be closed, naming the error's type and never its text. When
                both fail, the first is raised, and the second is logged as a warning and noted
                on it.
        """
        self._boundary.close()

    def send(self, prompt: str) -> ProbeResponse:
        """Send one probe prompt as a run of its own, and return the agent's reply.

        The prompt gets a new runtime session and W3C trace context. Its body is built in the
        target's payload format before any AWS call. Before the first invocation the runtime is
        checked with read-only calls, and only a check that succeeded is kept for later prompts.
        A busy session or throttling is retried with the same IDs and body. Once a request may
        have reached the service, the session is stopped before this returns or raises. The run
        record goes to the run manifest of the log directory, whatever the outcome.

        Args:
            prompt: The probe prompt, a non-empty string.

        Returns:
            The reply: status 200, its final text, the tools its stream events name (each
            string ``tool_name``, in first-seen order, once; a single JSON document names none),
            its decoded objects as the events, and
            its trace ID, ``probe-<fingerprint>-<W3C trace ID>``.

        Raises:
            RuntimeInvocationError: The prompt is not a string or is empty, refused before any
                AWS call and with no run record; or the prompt was refused or failed, named by
                the step and the session ID: the body could not be built; the clients could not
                be built; the
                preflight failed, or found a runtime that only accepts OAuth or does not serve
                HTTP; the invocation was refused by a guard, never sent, refused by the service,
                still busy or throttled on its last attempt, or failed after it was sent; or the
                reply could not be decoded, reported an agent error, or held no answer. A prompt
                whose run record could not be written is refused naming the manifest's path;
                when the prompt had already failed, its own error is raised with that path noted.
        """
        # Before the run starts, as the executor checks a task's prompt: a refused prompt costs no
        # call and leaves no run record.
        check_prompt(
            prompt,
            named="the probe prompt",
            remedy="Give each probe prompt the text to send to the agent",
        )
        ids = new_run_ids()
        run = self._boundary.start_run(ids, f"probe-{self._fingerprint}-{ids.w3c_trace_id}")
        try:
            response = self._send(prompt, run)
        except BaseException as exc:
            self._boundary.finish(run, self._log_dir, failure=exc)
            raise
        self._boundary.finish(run, self._log_dir, failure=None)
        return response

    # --- one prompt -------------------------------------------------------------------------

    def _send(self, prompt: str, run: RunState) -> ProbeResponse:
        session = run.ids.session_id
        # Built before any AWS call, so a body that can't be made costs nothing.
        body = request_body(self._target, prompt, session)

        def reply(invocation: Invocation) -> tuple[tuple[DecodedValue, ...], str, bool]:
            # Read inside the boundary's guard, so a reply that fails here still has its session
            # stopped and its run recorded as failed.
            content_type = invocation.response.content_type or ""
            values = decode(content_type, invocation.response.lines, session_id=session)
            check_errors(values, session_id=session, trace_id=run.trace_id)
            text = final_text(values, content_type, session_id=session)
            return values, text, is_event_stream(content_type)

        _, (values, text, streamed) = self._boundary.call(run, body, reply)
        objects = tuple(value for value in values if isinstance(value, dict))
        return ProbeResponse(
            status=200,
            text=text,
            # The URL probe's rule: only stream events name tools, and a single JSON document is
            # read for its text alone, so one reply body reports the same tools on either route.
            observed_tools=tool_names(objects) if streamed else (),
            events=objects,
            trace_id=run.trace_id,
        )


# --- helpers ----------------------------------------------------------------------------------


def _check_format(target: RuntimeTarget) -> None:
    if target.payload_format not in _PROBE_FORMATS:
        raise RuntimeInvocationError(
            f"payload_format={target.payload_format!r} is not one of the probe's request shapes, "
            f"so it can't be probed; name one of {', '.join(_PROBE_FORMATS)}. The probe builds "
            "each request body itself and takes no payload builder"
        )
