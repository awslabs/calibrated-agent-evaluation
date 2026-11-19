# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runtime call boundary is a context manager. Leaving its block stops every session a call
left open and closes the invocation view it derived, leaving the caller's clients open. Every
cleanup step runs even when one fails. A failure on the way out is raised when the block
succeeded, and noted on the block's own error, not raised over it, when the block failed."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("boto3")

from botocore.client import BaseClient

from _agentcore_runtime_fakes import (
    RUNTIME_ARN,
    StubbedAwsClients,
    add_invocation_response,
    add_preflight_responses,
    add_stop_response,
    stop_params,
)
from calibrated_agent_evaluation.agentcore import (
    AgentCoreError,
    RuntimeInvocationError,
    RuntimeTarget,
)
from calibrated_agent_evaluation.agentcore.runtime import new_run_ids, request_body
from calibrated_agent_evaluation.agentcore.runtime_boundary import (
    Invocation,
    RunState,
    RuntimeCallBoundary,
    close_on_exit,
    run_all,
)

_TARGET = RuntimeTarget(runtime_arn=RUNTIME_ARN, payload_format="agentcore_sse")
_EXIT_STOP_REFUSED = "could not build the AWS client to stop runtime sessions"
_SENTINEL = "SENTINEL-b07d"


@pytest.fixture
def clients() -> Iterator[StubbedAwsClients]:
    """Stubbed clients for one test, closed when the test ends."""
    with StubbedAwsClients() as stubbed:
        yield stubbed


@pytest.fixture(autouse=True)
def _no_ambient_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hide the machine's AWS setup (profile, credentials, config files) and the budget."""
    for name in list(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", raising=False)
    empty = tmp_path / "empty-aws-config"
    empty.write_text("")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(empty))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(empty))


class _BlockFailed(Exception):
    """What the code inside a boundary's block raised."""


def _boundary(clients: StubbedAwsClients) -> RuntimeCallBoundary:
    return RuntimeCallBoundary(
        _TARGET,
        clients=clients,
        read_timeout_seconds=300.0,
        attempt_limit=1,
        sleep=lambda _seconds: None,
    )


def _status(invocation: Invocation) -> int:
    return invocation.response.status_code


def _call(boundary: RuntimeCallBoundary) -> RunState:
    """Send one run through ``boundary`` and return its state."""
    ids = new_run_ids()
    run = boundary.start_run(ids, f"boundary-{ids.w3c_trace_id}")
    boundary.call(run, request_body(_TARGET, "Is claim CLM-4471 covered?", ids.session_id), _status)
    return run


def _answer_then_fail_the_stop(clients: StubbedAwsClients) -> None:
    """Queue the preflight and one answer, then a stop that is refused, leaving the session open."""
    add_preflight_responses(clients)
    add_invocation_response(clients)
    clients.stub("bedrock-agentcore").add_client_error(
        "stop_runtime_session",
        service_error_code="ThrottlingException",
        service_message="stop refused",
        http_status_code=429,
        expected_params=stop_params(),
    )


def _fail_the_exit_stop_client(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """Make the caller's clients refuse to build a client once ``switch["on"]`` is set."""
    switch = {"on": False}
    real_client = StubbedAwsClients.client

    def client(self: StubbedAwsClients, service_name: str) -> BaseClient:
        if switch["on"] and not self.is_invocation_view:
            raise AgentCoreError(f"could not build an AWS client for {service_name!r}")
        return real_client(self, service_name)

    monkeypatch.setattr(StubbedAwsClients, "client", client)
    return switch


def _call_then_break_the_exit_stop(
    boundary: RuntimeCallBoundary, switch: dict[str, bool], sessions: list[str], *, fail: bool
) -> None:
    """Send one run, keep its session in ``sessions``, and make the exit's stop fail.

    With ``fail``, the block then raises its own error.
    """
    sessions.append(_call(boundary).session_id)
    switch["on"] = True
    if fail:
        raise _BlockFailed("the block's own failure")


def _stopped_sessions(clients: StubbedAwsClients) -> list[str]:
    return [call.params["runtimeSessionId"] for call in clients.calls_to("StopRuntimeSession")]


def test_leaving_a_call_boundarys_block_stops_a_session_left_open_and_closes_its_view(
    clients: StubbedAwsClients,
) -> None:
    _answer_then_fail_the_stop(clients)
    # The one more stop the session gets as the block is left.
    add_stop_response(clients)

    with _boundary(clients) as boundary:
        run = _call(boundary)
        assert run.session_stop == "failed"

    assert _stopped_sessions(clients) == [run.session_id, run.session_id]
    views = [entry for entry in clients.built if entry.view]
    assert views
    assert all(entry.closed for entry in views)
    # The caller's own clients are the caller's to close.
    assert not any(entry.closed for entry in clients.built if not entry.view)
    clients.assert_no_pending_responses()


def test_a_failure_to_stop_sessions_on_leaving_the_block_is_noted_on_the_blocks_own_error(
    monkeypatch: pytest.MonkeyPatch, clients: StubbedAwsClients, caplog: pytest.LogCaptureFixture
) -> None:
    switch = _fail_the_exit_stop_client(monkeypatch)
    _answer_then_fail_the_stop(clients)
    sessions: list[str] = []

    with caplog.at_level(logging.WARNING, logger="calibrated_agent_evaluation"):
        with pytest.raises(_BlockFailed) as raised, _boundary(clients) as boundary:
            _call_then_break_the_exit_stop(boundary, switch, sessions, fail=True)

    [session] = sessions
    assert str(raised.value) == "the block's own failure"
    notes = getattr(raised.value, "__notes__", [])
    assert any(_EXIT_STOP_REFUSED in note and session in note for note in notes)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(_EXIT_STOP_REFUSED in line and session in line for line in warnings)
    assert all(entry.closed for entry in clients.built if entry.view)


def test_a_failure_to_stop_sessions_on_leaving_a_block_that_succeeded_is_raised(
    monkeypatch: pytest.MonkeyPatch, clients: StubbedAwsClients
) -> None:
    switch = _fail_the_exit_stop_client(monkeypatch)
    _answer_then_fail_the_stop(clients)
    sessions: list[str] = []

    boundary = _boundary(clients)
    with pytest.raises(RuntimeInvocationError, match=_EXIT_STOP_REFUSED) as raised, boundary:
        _call_then_break_the_exit_stop(boundary, switch, sessions, fail=False)

    [session] = sessions
    assert session in str(raised.value)
    assert isinstance(raised.value.__cause__, AgentCoreError)
    assert all(entry.closed for entry in clients.built if entry.view)


# --- the cleanup steps ------------------------------------------------------------------------


def test_every_cleanup_step_runs_and_each_later_failure_is_noted_on_the_first(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ran: list[str] = []

    def stop() -> None:
        ran.append("stop")
        raise RuntimeInvocationError("could not stop the open sessions")

    def close_view() -> None:
        ran.append("view")

    def close_built() -> None:
        ran.append("built")
        # A step that ran steps of its own carries their later failures as notes.
        nested = RuntimeError(f"pool text {_SENTINEL}")
        nested.add_note("could not close the AWS clients the executor built (OSError)")
        raise nested

    with (
        caplog.at_level(logging.WARNING, logger="calibrated_agent_evaluation"),
        pytest.raises(RuntimeInvocationError, match="could not stop the open sessions") as raised,
    ):
        run_all((stop, close_view, close_built), owner="the probe adapter")

    assert ran == ["stop", "view", "built"]
    notes = getattr(raised.value, "__notes__", [])
    assert any("closing the probe adapter failed with RuntimeError" in note for note in notes)
    assert "could not close the AWS clients the executor built (OSError)" in notes
    assert not any(_SENTINEL in note for note in notes)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("closing the probe adapter failed with RuntimeError" in line for line in warnings)


def test_an_interrupted_cleanup_step_still_lets_the_later_steps_run_and_is_raised() -> None:
    ran: list[str] = []

    def interrupted() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_all((interrupted, lambda: ran.append("view")), owner="the executor")

    assert ran == ["view"]


def test_a_cleanup_failure_after_a_block_that_succeeded_is_raised() -> None:
    def close() -> None:
        raise RuntimeInvocationError("could not stop the open sessions")

    with pytest.raises(RuntimeInvocationError, match="could not stop the open sessions"):
        close_on_exit(close, None, owner="the probe adapter")


def test_an_interrupted_cleanup_after_a_block_that_failed_is_noted_on_the_blocks_error() -> None:
    block = _BlockFailed("the block's own failure")

    def close() -> None:
        raise KeyboardInterrupt

    close_on_exit(close, block, owner="the executor")

    notes = getattr(block, "__notes__", [])
    assert any("closing the executor failed with KeyboardInterrupt" in note for note in notes)


def test_a_cleanup_failure_of_another_type_is_noted_on_the_blocks_error_by_its_type_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    block = _BlockFailed("the block's own failure")
    cleanup = RuntimeError(f"pool text {_SENTINEL}")
    cleanup.add_note("could not close the invocation clients (OSError)")

    def close() -> None:
        raise cleanup

    with caplog.at_level(logging.WARNING, logger="calibrated_agent_evaluation"):
        close_on_exit(close, block, owner="the probe adapter")

    notes = getattr(block, "__notes__", [])
    assert any("closing the probe adapter failed with RuntimeError" in note for note in notes)
    # The notes the cleanup failure carried reach the block's error too.
    assert "could not close the invocation clients (OSError)" in notes
    assert not any(_SENTINEL in note for note in notes)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("closing the probe adapter failed with RuntimeError" in line for line in warnings)
