# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AgentCore probe adapter sends each probe prompt as one run of its own: a new session and
trace context, the named request shape, the reply read and checked, the session stopped, and one
run record kept whatever the outcome. It refuses a setup it can't probe before building any
client, and leaves the caller's clients open."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")

from botocore.client import BaseClient
from botocore.config import Config

from _agentcore_runtime_fakes import (
    ANSWER,
    REGION,
    RUNTIME_ARN,
    StubbedAwsClients,
    add_invocation_response,
    add_preflight_responses,
    add_stop_response,
    json_body,
    sse_body,
    stop_params,
    streaming_body,
)
from calibrated_agent_evaluation.agentcore import (
    AgentCoreError,
    AgentReportedError,
    BilledCallBudget,
    RunRecord,
    RuntimeInvocationError,
    RuntimeProbeAdapter,
    RuntimeTarget,
)
from calibrated_agent_evaluation.agentcore.run_manifest import RUN_RECORD_STORE, manifest_path
from calibrated_agent_evaluation.probe import endpoint_fingerprint, probe_with
from calibrated_agent_evaluation.probe.adapter import PAYLOAD_FORMATS, ProbeResponse

_WHEN = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
_PROMPT = "Is the vacancy exclusion triggered for claim CLM-4471?"
_FOUR_FORMATS = ("generic_message", "openai", "agentcore_sse", "nested_agentcore")
_FINGERPRINT = endpoint_fingerprint(f"{RUNTIME_ARN}@DEFAULT")
_SENTINEL = "SENTINEL-7f3a"


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


class _Clock:
    """A UTC clock that moves on one second each time it is read."""

    def __init__(self) -> None:
        self.now = _WHEN

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def _no_sleep(_seconds: float) -> None:
    return None


def _target(**overrides: object) -> RuntimeTarget:
    fields: dict[str, object] = {"runtime_arn": RUNTIME_ARN, "payload_format": "agentcore_sse"}
    fields.update(overrides)
    return RuntimeTarget(**fields)  # pyright: ignore[reportArgumentType]


def _adapter(clients: StubbedAwsClients, log_dir: Path, **overrides: Any) -> RuntimeProbeAdapter:
    target = overrides.pop("target", None) or _target()
    kwargs: dict[str, Any] = {
        "clients": clients,
        "log_dir": log_dir,
        "sleep": _no_sleep,
        "clock": _Clock(),
    }
    kwargs.update(overrides)
    return RuntimeProbeAdapter(target, **kwargs)


def _fail(clients: StubbedAwsClients, code: str, status: int) -> None:
    clients.stub("bedrock-agentcore", view=True).add_client_error(
        "invoke_agent_runtime",
        service_error_code=code,
        service_message=f"agent text that must stay out of messages {_SENTINEL}",
        http_status_code=status,
    )


def _stop_fails(clients: StubbedAwsClients) -> None:
    clients.stub("bedrock-agentcore").add_client_error(
        "stop_runtime_session",
        service_error_code="ThrottlingException",
        service_message="stop refused",
        http_status_code=429,
        expected_params=stop_params(),
    )


def _one_prompt(clients: StubbedAwsClients, *, preflight: bool = True) -> None:
    """Queue a whole successful prompt: the preflight when asked for, one answer and one stop."""
    if preflight:
        add_preflight_responses(clients)
    add_invocation_response(clients)
    add_stop_response(clients)


def _manifest(log_dir: Path) -> list[RunRecord]:
    return RUN_RECORD_STORE.load(log_dir, fmt="json")


def _w3c_trace_id(traceparent: str) -> str:
    return traceparent.split("-")[1]


def _sessions(clients: StubbedAwsClients, operation: str) -> list[str]:
    return [call.params["runtimeSessionId"] for call in clients.calls_to(operation)]


def _unwritable(tmp_path: Path) -> Path:
    """A log directory that can't be created, because a file stands where its parent should be."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    return blocker / "logs"


# --- what a prompt answers --------------------------------------------------------------------


def test_a_probe_prompt_returns_its_final_text_tool_names_events_and_probe_trace_id(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    objects = [
        {"type": "tool_use", "tool_name": "lookup_policy"},
        {"type": "tool_use", "tool_name": "score_fraud"},
        {"type": "tool_use", "tool_name": "lookup_policy"},
        {"response": "the vacancy exclusion applies."},
    ]
    add_preflight_responses(clients)
    # A plain-text event is part of the answer but is not an object, so it is no event.
    add_invocation_response(clients, sse_body(objects[0], "Denied: ", *objects[1:]))
    add_stop_response(clients)

    with _adapter(clients, tmp_path) as adapter:
        response = adapter.send(_PROMPT)

    [sent] = clients.calls_to("InvokeAgentRuntime")
    w3c_trace_id = _w3c_trace_id(sent.params["traceParent"])
    assert response == ProbeResponse(
        status=200,
        text="Denied: the vacancy exclusion applies.",
        observed_tools=("lookup_policy", "score_fraud"),
        events=tuple(objects),
        trace_id=f"probe-{_FINGERPRINT}-{w3c_trace_id}",
    )
    [record] = adapter.run_records
    assert record.trace_id == response.trace_id
    assert record.w3c_trace_id == w3c_trace_id


def test_a_tool_named_only_in_a_single_json_reply_is_not_an_observed_tool(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    # The URL probe's rule: a single JSON document is read for its text alone, so the same reply
    # body reports the same tools on both probe routes.
    add_preflight_responses(clients)
    add_invocation_response(
        clients,
        json_body({"response": ANSWER, "tool_name": "score_fraud"}),
        content_type="application/json",
    )
    add_stop_response(clients)

    with _adapter(clients, tmp_path) as adapter:
        response = adapter.send(_PROMPT)

    assert response.text == ANSWER
    assert response.observed_tools == ()


def test_a_tool_name_that_is_not_a_string_is_not_an_observed_tool(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    objects = [
        {"type": "tool_use", "tool_name": 7},
        {"type": "tool_use", "tool_name": None},
        {"type": "tool_use", "tool_name": ["lookup_policy"]},
        {"type": "tool_use", "tool_name": {"name": "lookup_policy"}},
        {"type": "tool_use", "tool_name": "score_fraud"},
        {"response": ANSWER},
    ]
    add_preflight_responses(clients)
    add_invocation_response(clients, sse_body(*objects))
    add_stop_response(clients)

    with _adapter(clients, tmp_path) as adapter:
        response = adapter.send(_PROMPT)

    assert response.observed_tools == ("score_fraud",)
    assert response.events == tuple(objects)
    assert response.text == ANSWER


def test_two_probe_prompts_get_different_trace_ids(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _one_prompt(clients)
    _one_prompt(clients, preflight=False)

    with _adapter(clients, tmp_path) as adapter:
        first = adapter.send(_PROMPT)
        second = adapter.send(_PROMPT)

    w3c_trace_ids = [
        _w3c_trace_id(call.params["traceParent"]) for call in clients.calls_to("InvokeAgentRuntime")
    ]
    assert [first.trace_id, second.trace_id] == [
        f"probe-{_FINGERPRINT}-{w3c_trace_id}" for w3c_trace_id in w3c_trace_ids
    ]
    assert first.trace_id != second.trace_id
    assert [record.trace_id for record in adapter.run_records] == [
        first.trace_id,
        second.trace_id,
    ]


def test_a_probe_by_the_adapters_target_label_stamps_each_trace_with_the_fingerprint_in_its_id(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _one_prompt(clients)
    _one_prompt(clients, preflight=False)

    with _adapter(clients, tmp_path) as adapter:
        report = probe_with(
            adapter,
            target=adapter.target_label,
            authorized=True,
            seeds=("first prompt", "second prompt"),
            payload_format="agentcore_sse",
        )

    assert adapter.target_label == f"{RUNTIME_ARN}@DEFAULT"
    assert len(report.traces) == 2
    for trace, record in zip(report.traces, adapter.run_records, strict=True):
        assert trace.endpoint_fingerprint == _FINGERPRINT
        assert trace.trace_id == f"probe-{_FINGERPRINT}-{record.w3c_trace_id}"
        assert record.trace_id == trace.trace_id
    [finding] = report.findings
    assert finding.endpoint_fingerprint == _FINGERPRINT


# --- one run per prompt -----------------------------------------------------------------------


def test_each_probe_prompt_gets_its_own_session_and_one_stop(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _one_prompt(clients)
    _fail(clients, "RuntimeClientError", 424)
    add_stop_response(clients)

    with _adapter(clients, tmp_path) as adapter:
        adapter.send(_PROMPT)
        with pytest.raises(RuntimeInvocationError, match="RuntimeClientError"):
            adapter.send(_PROMPT)

    sessions = _sessions(clients, "InvokeAgentRuntime")
    assert len(set(sessions)) == 2
    assert _sessions(clients, "StopRuntimeSession") == sessions
    assert [record.session_id for record in adapter.run_records] == sessions
    assert [record.session_stop for record in adapter.run_records] == ["stopped", "stopped"]
    clients.assert_no_pending_responses()


def test_a_busy_probe_session_is_retried_with_the_same_session_and_payload(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    add_preflight_responses(clients)
    _fail(clients, "RetryableConflictException", 409)
    _fail(clients, "ThrottlingException", 429)
    add_invocation_response(clients)
    add_stop_response(clients)
    waits: list[float] = []

    with _adapter(clients, tmp_path, sleep=waits.append) as adapter:
        response = adapter.send(_PROMPT)

    attempts = clients.calls_to("InvokeAgentRuntime")
    assert len(attempts) == 3
    for name in ("runtimeSessionId", "payload", "traceParent", "baggage", "qualifier"):
        assert len({str(attempt.params[name]) for attempt in attempts}) == 1, name
    assert response.text == ANSWER
    assert waits == [1, 2]
    [record] = adapter.run_records
    assert record.attempts == 3
    assert record.error_code is None
    assert _sessions(clients, "StopRuntimeSession") == [record.session_id]


def test_each_probe_prompt_adds_one_run_record_whatever_its_outcome(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _one_prompt(clients)
    _fail(clients, "RuntimeClientError", 424)
    add_stop_response(clients)
    add_invocation_response(
        clients, sse_body({"error": "boom", "error_type": "ValueError", "message": "failed"})
    )
    add_stop_response(clients)

    with _adapter(clients, tmp_path) as adapter:
        response = adapter.send(_PROMPT)
        for _ in range(2):
            with pytest.raises(RuntimeInvocationError):
                adapter.send(_PROMPT)

    records = adapter.run_records
    assert _manifest(tmp_path) == list(records)
    assert records[0].trace_id == response.trace_id
    assert [record.error_code for record in records] == [
        None,
        "RuntimeClientError",
        "AgentReportedError",
    ]
    assert [record.status_code for record in records] == [200, 424, 200]
    sessions = _sessions(clients, "InvokeAgentRuntime")
    assert [record.session_id for record in records] == sessions
    for record in records:
        assert record.trace_id == f"probe-{_FINGERPRINT}-{record.w3c_trace_id}"
        assert record.replayed is False
        assert record.session_stop == "stopped"
        assert record.attempts == 1
        assert record.runtime_arn == RUNTIME_ARN
        assert record.qualifier == "DEFAULT"
        assert record.payload_format == "agentcore_sse"
        assert record.runtime_version == "24"
        assert record.runtime_version_source == "endpoint"
        assert record.config_bundle_arn is None
        started = datetime.fromisoformat(record.started_at)
        ended = datetime.fromisoformat(record.ended_at)
        assert started.tzinfo is not None
        assert record.duration_ms == (ended - started) // timedelta(milliseconds=1) > 0
    assert adapter.manifest_errors == ()


def test_a_probe_prompt_refused_before_invoking_stops_nothing_and_records_not_started(
    tmp_path: Path,
) -> None:
    with StubbedAwsClients(handlers=[BilledCallBudget(0)]) as clients:
        add_preflight_responses(clients)
        # Queued so the stub accepts the call; the budget refuses it before it is answered.
        add_invocation_response(clients)
        # Queued so that a stop, if one were asked for, would be answered and seen.
        add_stop_response(clients)
        adapter = _adapter(clients, tmp_path)

        with adapter, pytest.raises(RuntimeInvocationError) as refused:
            adapter.send(_PROMPT)

    message = str(refused.value)
    assert "BilledCallBudgetExceeded" in message
    assert "InvokeAgentRuntime" in message
    assert clients.calls_to("InvokeAgentRuntime") == []
    # Nothing was sent, so no session was opened: none is stopped, now or on close.
    assert clients.calls_to("StopRuntimeSession") == []
    [record] = adapter.run_records
    assert record.session_stop == "not_started"
    assert record.attempts == 0
    assert record.error_code == "BilledCallBudgetExceeded"
    assert _manifest(tmp_path) == [record]


@pytest.mark.parametrize(
    ("prompt", "given"),
    [
        ("", "an empty string"),
        (None, "of type NoneType"),
        (123, "of type int"),
        (b"bytes", "of type bytes"),
    ],
    ids=["empty", "none", "int", "bytes"],
)
def test_an_empty_or_non_text_probe_prompt_is_refused_before_any_call(
    tmp_path: Path, clients: StubbedAwsClients, prompt: object, given: str
) -> None:
    # Queued so that a prompt sent anyway would be answered and seen.
    _one_prompt(clients)

    with _adapter(clients, tmp_path) as adapter, pytest.raises(RuntimeInvocationError) as refused:
        adapter.send(prompt)  # pyright: ignore[reportArgumentType]

    assert f"the probe prompt is {given}, and a prompt must be a non-empty string" in str(
        refused.value
    )
    assert clients.built == []
    assert clients.calls == []
    # Refused before the run started, so no run and no record exist.
    assert adapter.run_records == ()
    assert not manifest_path(tmp_path).exists()


def test_a_probe_with_an_empty_seed_is_refused_with_no_call_and_no_report(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _one_prompt(clients)
    reports: list[object] = []

    with _adapter(clients, tmp_path) as adapter, pytest.raises(RuntimeInvocationError) as refused:
        reports.append(
            probe_with(
                adapter,
                target=adapter.target_label,
                authorized=True,
                seeds=("",),
                payload_format="agentcore_sse",
            )
        )

    assert "the probe prompt is an empty string" in str(refused.value)
    assert reports == []
    assert clients.calls == []
    assert adapter.run_records == ()
    assert not manifest_path(tmp_path).exists()


def test_a_probe_prompt_with_an_unwritable_manifest_is_refused_naming_its_path(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    log_dir = _unwritable(tmp_path)
    _one_prompt(clients)
    returned: list[ProbeResponse] = []

    with _adapter(clients, log_dir) as adapter, pytest.raises(RuntimeInvocationError) as refused:
        returned.append(adapter.send(_PROMPT))

    assert str(manifest_path(log_dir)) in str(refused.value)
    assert returned == []
    # The prompt itself went through, and its session was stopped.
    assert len(clients.calls_to("InvokeAgentRuntime")) == 1
    [record] = adapter.run_records
    assert record.session_stop == "stopped"
    assert adapter.manifest_errors == (refused.value,)


def test_a_failed_probe_prompt_with_an_unwritable_manifest_keeps_its_own_error_with_the_path_noted(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    log_dir = _unwritable(tmp_path)
    add_preflight_responses(clients)
    _fail(clients, "RuntimeClientError", 424)
    add_stop_response(clients)

    with _adapter(clients, log_dir) as adapter, pytest.raises(RuntimeInvocationError) as refused:
        adapter.send(_PROMPT)

    assert "RuntimeClientError" in str(refused.value)
    assert str(manifest_path(log_dir)) not in str(refused.value)
    assert str(manifest_path(log_dir)) in "\n".join(getattr(refused.value, "__notes__", []))
    [record] = adapter.run_records
    assert record.error_code == "RuntimeClientError"
    [unwritten] = adapter.manifest_errors
    assert str(manifest_path(log_dir)) in str(unwritten)


# --- the preflight ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("from_endpoint", "source"), [(True, "endpoint"), (False, "latest")], ids=["live", "latest"]
)
def test_the_preflight_runs_once_before_the_first_probe_invocation(
    tmp_path: Path, clients: StubbedAwsClients, from_endpoint: bool, source: str
) -> None:
    add_preflight_responses(clients, from_endpoint=from_endpoint)
    _one_prompt(clients, preflight=False)
    _one_prompt(clients, preflight=False)

    with _adapter(clients, tmp_path) as adapter:
        adapter.send(_PROMPT)
        adapter.send(_PROMPT)

    assert [call.operation for call in clients.calls] == [
        "GetAgentRuntimeEndpoint",
        "GetAgentRuntime",
        "InvokeAgentRuntime",
        "StopRuntimeSession",
        "InvokeAgentRuntime",
        "StopRuntimeSession",
    ]
    assert [(r.runtime_version, r.runtime_version_source) for r in adapter.run_records] == [
        ("24", source),
        ("24", source),
    ]
    clients.assert_no_pending_responses()


def test_an_oauth_only_runtime_is_refused_before_any_probe_invocation(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    add_preflight_responses(clients, oauth_only=True)
    # Queued so that an invocation or a stop, if one were asked for, would be answered and seen.
    add_invocation_response(clients)
    add_stop_response(clients)

    with _adapter(clients, tmp_path) as adapter, pytest.raises(RuntimeInvocationError) as refused:
        adapter.send(_PROMPT)

    assert "OAuth" in str(refused.value)
    assert RUNTIME_ARN in str(refused.value)
    assert clients.calls_to("InvokeAgentRuntime") == []
    assert clients.calls_to("StopRuntimeSession") == []
    [record] = adapter.run_records
    assert record.session_stop == "not_started"
    assert record.runtime_version is None
    assert record.error_code == "RuntimeInvocationError"


def test_a_refused_probe_preflight_is_asked_again_on_the_next_prompt(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    clients.stub("bedrock-agentcore-control").add_client_error(
        "get_agent_runtime_endpoint",
        service_error_code="AccessDeniedException",
        service_message="denied",
        http_status_code=403,
    )
    _one_prompt(clients)

    with _adapter(clients, tmp_path) as adapter:
        with pytest.raises(RuntimeInvocationError, match="AccessDeniedException"):
            adapter.send(_PROMPT)
        response = adapter.send(_PROMPT)

    assert response.text == ANSWER
    assert len(clients.calls_to("GetAgentRuntimeEndpoint")) == 2
    assert len(clients.calls_to("InvokeAgentRuntime")) == 1
    assert [record.error_code for record in adapter.run_records] == [
        "AccessDeniedException",
        None,
    ]


# --- closing ----------------------------------------------------------------------------------


def test_closing_the_probe_adapter_stops_a_session_left_open_and_keeps_caller_clients_open(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, clients: StubbedAwsClients
) -> None:
    add_preflight_responses(clients)
    add_invocation_response(clients)
    _stop_fails(clients)
    add_stop_response(clients)

    with caplog.at_level(logging.INFO, logger="calibrated_agent_evaluation"):
        with _adapter(clients, tmp_path) as adapter:
            adapter.send(_PROMPT)
            [record] = adapter.run_records
            assert record.session_stop == "failed"
            assert not any(entry.closed for entry in clients.built)
            caplog.clear()

    assert _sessions(clients, "StopRuntimeSession") == [record.session_id, record.session_id]
    closing_logs = [r for r in caplog.records if r.name.startswith("calibrated_agent_evaluation")]
    assert any(
        r.levelno == logging.INFO
        and record.session_id in r.getMessage()
        and r.getMessage().endswith("stopped")
        for r in closing_logs
    )
    views = [entry for entry in clients.built if entry.view]
    assert views
    assert all(entry.closed for entry in views)
    # The caller's own clients are the caller's to close.
    assert not any(entry.closed for entry in clients.built if not entry.view)
    clients.assert_no_pending_responses()


@pytest.mark.parametrize("used", [False, True], ids=["never-sent", "session-already-stopped"])
def test_closing_a_probe_adapter_with_no_open_session_makes_no_call(
    tmp_path: Path, clients: StubbedAwsClients, used: bool
) -> None:
    if used:
        _one_prompt(clients)
    # Queued so that a stop on close, if one were asked for, would be answered and seen.
    add_stop_response(clients)
    adapter = _adapter(clients, tmp_path)
    if used:
        adapter.send(_PROMPT)
    calls_before = list(clients.calls)

    adapter.close()

    assert clients.calls == calls_before
    if not used:
        assert clients.built == []


def _view_close_raises(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    """Make closing an invocation view raise ``error``; the caller's own clients close as usual."""
    real_close = StubbedAwsClients.close

    def close(self: StubbedAwsClients) -> None:
        if self.is_invocation_view:
            raise error
        real_close(self)

    monkeypatch.setattr(StubbedAwsClients, "close", close)


def _fail_the_close_stop_client(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception] = AgentCoreError
) -> dict[str, bool]:
    """Make the caller's clients fail to build a client with ``error`` once ``switch["on"]``."""
    switch = {"on": False}
    real_client = StubbedAwsClients.client

    def client(self: StubbedAwsClients, service_name: str) -> BaseClient:
        if switch["on"] and not self.is_invocation_view:
            raise error(f"could not build an AWS client for {service_name!r} {_SENTINEL}")
        return real_client(self, service_name)

    monkeypatch.setattr(StubbedAwsClients, "client", client)
    return switch


def _send_and_leave_the_session_open(
    clients: StubbedAwsClients, adapter: RuntimeProbeAdapter
) -> str:
    """Send one prompt whose stop is refused, so its session is left for close; its session ID."""
    add_preflight_responses(clients)
    add_invocation_response(clients)
    _stop_fails(clients)
    adapter.send(_PROMPT)
    [record] = adapter.run_records
    return record.session_id


class _BlockFailed(Exception):
    """What the code inside the adapter's block raised."""


@pytest.mark.parametrize(
    "error",
    [OSError(f"the view could not be closed {_SENTINEL}"), RuntimeError(_SENTINEL)],
    ids=["os-error", "other-error"],
)
def test_a_probe_view_that_fails_to_close_is_refused_naming_its_type_and_not_its_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clients: StubbedAwsClients, error: Exception
) -> None:
    _one_prompt(clients)
    adapter = _adapter(clients, tmp_path)
    adapter.send(_PROMPT)
    _view_close_raises(monkeypatch, error)

    with pytest.raises(RuntimeInvocationError, match="could not close the invocation") as refused:
        adapter.close()

    assert refused.value.__cause__ is error
    assert f"({type(error).__name__})" in str(refused.value)
    # The error's own text is the SDK's or the caller's, so it stays out of the message.
    assert _SENTINEL not in str(refused.value)
    assert not any(entry.closed for entry in clients.built if not entry.view)


@pytest.mark.parametrize(
    "error",
    # Any error the clients raise, such as one from a caller's own AwsClients subclass.
    [AgentCoreError, RuntimeError],
    ids=["agentcore-error", "other-error"],
)
def test_a_probe_stop_client_that_fails_to_build_on_close_is_refused_naming_the_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clients: StubbedAwsClients,
    error: type[Exception],
) -> None:
    switch = _fail_the_close_stop_client(monkeypatch, error)
    adapter = _adapter(clients, tmp_path)
    session = _send_and_leave_the_session_open(clients, adapter)
    switch["on"] = True

    with pytest.raises(RuntimeInvocationError, match="stop runtime sessions") as refused:
        adapter.close()

    message = str(refused.value)
    assert session in message
    assert "on close" in message
    assert f"({error.__name__})" in message
    assert _SENTINEL not in message
    assert type(refused.value.__cause__) is error
    # The view is closed all the same.
    assert all(entry.closed for entry in clients.built if entry.view)


def test_closing_the_probe_adapter_notes_a_later_view_close_failure_on_the_stop_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clients: StubbedAwsClients
) -> None:
    switch = _fail_the_close_stop_client(monkeypatch)
    adapter = _adapter(clients, tmp_path)
    _send_and_leave_the_session_open(clients, adapter)
    switch["on"] = True
    _view_close_raises(monkeypatch, OSError("the view could not be closed"))

    with pytest.raises(RuntimeInvocationError, match="stop runtime sessions") as raised:
        adapter.close()

    # The stop's failure comes first and is the one raised; the view's is noted on it.
    notes = getattr(raised.value, "__notes__", [])
    assert any("could not close the invocation" in note and "OSError" in note for note in notes)


def test_every_cleanup_failure_on_leaving_the_probe_adapter_is_noted_on_the_blocks_own_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clients: StubbedAwsClients,
    caplog: pytest.LogCaptureFixture,
) -> None:
    switch = _fail_the_close_stop_client(monkeypatch)
    adapter = _adapter(clients, tmp_path)
    sessions: list[str] = []

    def send_then_break_the_close() -> None:
        sessions.append(_send_and_leave_the_session_open(clients, adapter))
        switch["on"] = True
        _view_close_raises(monkeypatch, RuntimeError(_SENTINEL))
        raise _BlockFailed("the block's own failure")

    with caplog.at_level(logging.WARNING, logger="calibrated_agent_evaluation"):
        with pytest.raises(_BlockFailed) as raised, adapter:
            send_then_break_the_close()

    [session] = sessions
    assert str(raised.value) == "the block's own failure"
    notes = getattr(raised.value, "__notes__", [])
    # The stop's failure, and the view's that was noted on it, both reach the block's error.
    assert any("stop runtime sessions" in note and session in note for note in notes)
    assert any(
        "could not close the invocation" in note and "RuntimeError" in note for note in notes
    )
    assert not any(_SENTINEL in note for note in notes)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("stop runtime sessions" in line and session in line for line in warnings)
    assert any("could not close the invocation" in line for line in warnings)


def test_an_interrupt_while_leaving_the_probe_adapter_is_noted_on_the_blocks_own_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _one_prompt(clients)
    adapter = _adapter(clients, tmp_path)

    def send_then_interrupt_the_close() -> None:
        adapter.send(_PROMPT)
        _view_close_raises(monkeypatch, KeyboardInterrupt())
        raise _BlockFailed("the block's own failure")

    with pytest.raises(_BlockFailed) as raised, adapter:
        send_then_interrupt_the_close()

    notes = getattr(raised.value, "__notes__", [])
    assert any("closing the probe adapter failed with KeyboardInterrupt" in n for n in notes)


class _ConfigCannotBeApplied(StubbedAwsClients):
    """Clients whose ``with_config`` fails with an error that is not an AgentCoreError."""

    def with_config(self, config: Config) -> StubbedAwsClients:
        raise RuntimeError(f"bad config {_SENTINEL}")


def test_clients_that_fail_to_apply_the_invocation_config_are_refused_before_any_probe_call(
    tmp_path: Path,
) -> None:
    with _ConfigCannotBeApplied() as clients:
        adapter = _adapter(clients, tmp_path)

        with adapter, pytest.raises(RuntimeInvocationError) as refused:
            adapter.send(_PROMPT)

    message = str(refused.value)
    assert "could not build the AWS clients" in message
    assert "(RuntimeError)" in message
    assert "Check the AWS setup of the clients passed in" in message
    assert _SENTINEL not in message
    assert clients.calls == []
    [record] = adapter.run_records
    assert record.session_stop == "not_started"


class _InterruptsTheInvocation:
    """A call handler that interrupts each ``InvokeAgentRuntime`` as it is sent, as Ctrl-C would."""

    def __call__(self, **kwargs: Any) -> None:
        if kwargs["model"].name == "InvokeAgentRuntime":
            raise KeyboardInterrupt


def test_an_interrupted_probe_invocation_still_stops_its_session(tmp_path: Path) -> None:
    with StubbedAwsClients(handlers=[_InterruptsTheInvocation()]) as clients:
        add_preflight_responses(clients)
        # Queued so the stub accepts the call; the interrupt comes before it answers.
        add_invocation_response(clients)
        add_stop_response(clients)
        adapter = _adapter(clients, tmp_path)

        with adapter:
            with pytest.raises(KeyboardInterrupt):
                adapter.send(_PROMPT)
            # Stopped before send() raised, not left for the adapter's close.
            stopped_by_the_prompt = _sessions(clients, "StopRuntimeSession")

    [record] = adapter.run_records
    # Whether the request left is unknown, so the session is stopped anyway.
    assert stopped_by_the_prompt == [record.session_id]
    assert len(clients.calls_to("StopRuntimeSession")) == 1
    assert record.session_stop == "stopped"
    assert record.error_code == "KeyboardInterrupt"


# --- replies that are refused -----------------------------------------------------------------


def test_a_probe_reply_with_an_agent_error_is_refused_naming_the_session(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, clients: StubbedAwsClients
) -> None:
    add_preflight_responses(clients)
    add_invocation_response(
        clients,
        sse_body(
            {"response": f"partial {_SENTINEL}"},
            {"error": f"boom {_SENTINEL}", "error_type": "ValueError", "message": "failed"},
        ),
    )
    add_stop_response(clients)

    adapter = _adapter(clients, tmp_path)
    with caplog.at_level(logging.DEBUG, logger="calibrated_agent_evaluation"):
        with adapter, pytest.raises(RuntimeInvocationError) as refused:
            adapter.send(f"{_PROMPT} {_SENTINEL}")

    [record] = adapter.run_records
    message = str(refused.value)
    assert record.session_id in message
    assert "ValueError" in message
    # Neither the prompt, the reply nor the agent's error text reaches a message or a log line.
    assert _SENTINEL not in message
    assert not any(_SENTINEL in r.getMessage() for r in caplog.records)
    assert record.session_stop == "stopped"
    # Typed so a caller can treat such a run as a result: the error code and the probe trace ID
    # ride on the error, never the agent's text.
    assert isinstance(refused.value, AgentReportedError)
    assert refused.value.error_code == "ValueError"
    assert refused.value.trace_id == record.trace_id
    assert _SENTINEL not in repr(vars(refused.value))
    assert record.error_code == "AgentReportedError"


@pytest.mark.parametrize(
    "body",
    [b"", b'data: {"type": "tool_use", "tool_name": "lookup_policy"}\n\n'],
    ids=["no-body", "no-text"],
)
def test_an_empty_probe_reply_is_refused(
    tmp_path: Path, clients: StubbedAwsClients, body: bytes
) -> None:
    add_preflight_responses(clients)
    add_invocation_response(clients, streaming_body(body))
    add_stop_response(clients)

    with _adapter(clients, tmp_path) as adapter, pytest.raises(RuntimeInvocationError) as refused:
        adapter.send(_PROMPT)

    [record] = adapter.run_records
    assert record.session_id in str(refused.value)
    assert record.session_stop == "stopped"
    assert _manifest(tmp_path) == [record]


def test_a_failed_probe_prompt_stops_the_probe_at_that_prompt(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    add_preflight_responses(clients)
    _fail(clients, "RuntimeClientError", 424)
    add_stop_response(clients)

    with _adapter(clients, tmp_path) as adapter, pytest.raises(RuntimeInvocationError) as refused:
        probe_with(
            adapter,
            target=f"{RUNTIME_ARN}@DEFAULT",
            authorized=True,
            seeds=("first prompt", "second prompt"),
            payload_format="agentcore_sse",
        )

    # The adapter's own error, not wrapped as a probe error.
    assert type(refused.value) is RuntimeInvocationError
    assert "RuntimeClientError" in str(refused.value)
    [sent] = clients.calls_to("InvokeAgentRuntime")
    assert json.loads(sent.params["payload"])["prompt"] == "first prompt"
    assert len(adapter.run_records) == 1


# The object is one level, and each of the 256 lists inside it one more.
_DEEP_REPLY = b'{"response": "x", "score": ' + b"[" * 256 + b"]" * 256 + b"}"


def _assert_json_reply_refused(clients: StubbedAwsClients, log_dir: Path, body: bytes) -> None:
    """Send one prompt whose ``application/json`` reply is ``body``, and check it was refused.

    The refusal names the session and the content type, the session is stopped, and the run
    manifest holds the prompt's one record.
    """
    add_preflight_responses(clients)
    add_invocation_response(clients, streaming_body(body), content_type="application/json")
    add_stop_response(clients)

    with _adapter(clients, log_dir) as adapter, pytest.raises(RuntimeInvocationError) as refused:
        adapter.send(_PROMPT)

    [record] = adapter.run_records
    assert record.session_id in str(refused.value)
    assert "application/json" in str(refused.value)
    assert _sessions(clients, "StopRuntimeSession") == [record.session_id]
    assert _manifest(log_dir) == [record]


def test_a_probe_reply_holding_a_number_too_large_for_a_float_is_refused(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _assert_json_reply_refused(clients, tmp_path, b'{"response": "x", "score": 1e999}')


def test_a_probe_reply_nested_deeper_than_256_levels_is_refused(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _assert_json_reply_refused(clients, tmp_path, _DEEP_REPLY)


# --- what the adapter refuses to be built with ------------------------------------------------


@pytest.mark.parametrize("payload_format", ["custom", "mystery"])
def test_the_probe_adapter_refuses_the_custom_and_unknown_formats_listing_the_four(
    tmp_path: Path, clients: StubbedAwsClients, payload_format: str
) -> None:
    if payload_format == "custom":
        target = _target(payload_format="custom")
    else:
        # RuntimeTarget refuses an unknown name itself, so this one is built without validation.
        target = RuntimeTarget.model_construct(
            runtime_arn=RUNTIME_ARN,
            qualifier="DEFAULT",
            payload_format=payload_format,
            config_bundle_arn=None,
            config_bundle_version=None,
        )

    with pytest.raises(RuntimeInvocationError) as refused:
        _adapter(clients, tmp_path, target=target)

    message = str(refused.value)
    assert repr(payload_format) in message
    for name in _FOUR_FORMATS:
        assert name in message
    assert clients.built == []


def test_the_probe_adapter_refuses_clients_for_another_region(tmp_path: Path) -> None:
    with StubbedAwsClients(region="us-east-1") as clients:
        with pytest.raises(RuntimeInvocationError) as refused:
            _adapter(clients, tmp_path)

    message = str(refused.value)
    assert "'us-east-1'" in message
    assert f"'{REGION}'" in message
    assert clients.built == []


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("read_timeout_seconds", 0),
        ("read_timeout_seconds", -1),
        ("read_timeout_seconds", -0.0),
        ("read_timeout_seconds", math.nan),
        ("read_timeout_seconds", -math.inf),
        ("attempt_limit", 0),
    ],
    ids=["zero", "negative", "negative-zero", "nan", "negative-infinity", "no-attempt"],
)
def test_the_probe_adapter_refuses_a_non_positive_read_timeout_or_an_attempt_limit_below_one(
    tmp_path: Path, clients: StubbedAwsClients, setting: str, value: float
) -> None:
    with pytest.raises(RuntimeInvocationError) as refused:
        _adapter(clients, tmp_path, **{setting: value})

    assert f"{setting}={value!r}" in str(refused.value)
    assert clients.built == []


@pytest.mark.parametrize(
    "seconds",
    [math.inf, math.nextafter(threading.TIMEOUT_MAX, math.inf), 1e10],
    ids=["infinite", "just-over-the-longest", "longer-than-a-socket-can-wait"],
)
def test_the_probe_adapter_refuses_a_read_timeout_no_socket_can_wait_for_naming_it(
    tmp_path: Path, clients: StubbedAwsClients, seconds: float
) -> None:
    with pytest.raises(RuntimeInvocationError) as refused:
        _adapter(clients, tmp_path, read_timeout_seconds=seconds)

    message = str(refused.value)
    assert f"read_timeout_seconds={seconds!r}" in message
    assert "finite" in message
    assert f"{threading.TIMEOUT_MAX:.0f}" in message
    assert clients.built == []


@pytest.mark.parametrize("seconds", [True, "300", None], ids=["bool", "string", "none"])
def test_the_probe_adapter_refuses_a_read_timeout_that_is_not_a_number_of_seconds_naming_it(
    tmp_path: Path, clients: StubbedAwsClients, seconds: object
) -> None:
    with pytest.raises(RuntimeInvocationError) as refused:
        _adapter(clients, tmp_path, read_timeout_seconds=seconds)

    message = str(refused.value)
    assert f"read_timeout_seconds={seconds!r}" in message
    assert "finite number of seconds" in message
    assert clients.built == []


def test_the_probe_adapter_accepts_the_longest_read_timeout_a_socket_can_wait_for(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _one_prompt(clients)

    with _adapter(clients, tmp_path, read_timeout_seconds=threading.TIMEOUT_MAX) as adapter:
        response = adapter.send(_PROMPT)

    assert response.text == ANSWER
    [invoker] = [entry.client for entry in clients.built if entry.view]
    assert invoker.meta.config.read_timeout == threading.TIMEOUT_MAX


@pytest.mark.parametrize("payload_format", _FOUR_FORMATS)
def test_each_named_format_is_accepted_by_the_probe_adapter(
    tmp_path: Path, clients: StubbedAwsClients, payload_format: str
) -> None:
    _one_prompt(clients)

    with _adapter(clients, tmp_path, target=_target(payload_format=payload_format)) as adapter:
        response = adapter.send(_PROMPT)

    assert response.text == ANSWER
    [sent] = clients.calls_to("InvokeAgentRuntime")
    [shape] = [shape for shape in PAYLOAD_FORMATS if shape.name == payload_format]
    assert json.loads(sent.params["payload"]) == shape.build_body(
        _PROMPT, sent.params["runtimeSessionId"]
    )
    # The qualifier is the call's own parameter, whatever URL suffix the shape has.
    assert sent.params["qualifier"] == "DEFAULT"
    assert sent.params["agentRuntimeArn"] == RUNTIME_ARN
    [record] = adapter.run_records
    assert record.payload_format == payload_format
