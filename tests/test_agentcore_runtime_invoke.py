# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The preflight refuses runtimes that can't be invoked, a request body is a JSON object or is
refused, and every invocation is sent once, retried only while the session is busy or the runtime
throttles, and refused by name otherwise, with what its attempts reached kept for the caller. A
session stop that doesn't return in time counts as failed."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

pytest.importorskip("boto3")

import urllib3
from botocore import xform_name
from botocore.client import BaseClient
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    CredentialRetrievalError,
    EndpointConnectionError,
    IncompleteReadError,
    NoCredentialsError,
    ParamValidationError,
    PartialCredentialsError,
    ProxyConnectionError,
    ReadTimeoutError,
    RefreshWithMFAUnsupportedError,
    ResponseStreamingError,
    SSLError,
    SSOTokenLoadError,
    TokenRetrievalError,
    UnauthorizedSSOTokenError,
)
from botocore.response import StreamingBody

from _agentcore_runtime_fakes import (
    RUNTIME_ARN,
    CountingAwsClients,
    Scripted,
    ScriptedSends,
    SendCounter,
    StubbedAwsClients,
    clm_4471_spans,
    json_answer,
    preflight_sends,
    sse_body,
    stop_params,
    streaming_body,
    zero_retry_backoff,
)
from calibrated_agent_evaluation.agentcore import (
    AgentCoreRuntimeExecutor,
    AgentCoreRuntimeTask,
    AwsClients,
    BilledCallBudget,
    RuntimeInvocationError,
    RuntimeTarget,
)
from calibrated_agent_evaluation.agentcore.run_manifest import VersionSource
from calibrated_agent_evaluation.agentcore.run_spans import RunSpanQuery
from calibrated_agent_evaluation.agentcore.runtime import (
    InvocationProgress,
    RawResponse,
    RunIds,
    RuntimeInfo,
    invoke,
    new_run_ids,
    request_body,
    resolve_runtime,
    runtime_client_config,
    stop,
)
from calibrated_agent_evaluation.backends._bounded import BoundedOutcome
from calibrated_agent_evaluation.probe.adapter import PAYLOAD_FORMATS
from calibrated_agent_evaluation.schema.types import Span

_RUNTIME_ID = "calibrated_agent_evaluation_fixture-AbCdEf1234"
_WHEN = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
_PROMPT = "Is the vacancy exclusion triggered?"
_ANSWER = "Yes, it is."
_BODY = json.dumps({"prompt": _PROMPT, "session_id": "s"}).encode()
_ENDPOINT = "https://bedrock-agentcore.us-west-2.amazonaws.com"


@pytest.fixture
def clients() -> Iterator[StubbedAwsClients]:
    """Stubbed clients for one test, closed when the test ends."""
    with StubbedAwsClients() as stubbed:
        yield stubbed


def _says(error: BaseException, *parts: str, never: Sequence[str] = ()) -> None:
    """Assert that the message of ``error`` holds every one of ``parts`` and none of ``never``."""
    message = str(error)
    for part in parts:
        assert part in message
    for part in never:
        assert part not in message


def _target(**overrides: object) -> RuntimeTarget:
    fields: dict[str, object] = {"runtime_arn": RUNTIME_ARN, "payload_format": "agentcore_sse"}
    fields.update(overrides)
    return RuntimeTarget(**fields)  # pyright: ignore[reportArgumentType]


@contextmanager
def _invocation_client(clients: AwsClients, read_timeout: float = 300.0) -> Iterator[BaseClient]:
    """The invocation client over ``clients``; its view of them is closed when the block ends."""
    with clients.with_config(runtime_client_config(read_timeout)) as view:
        yield view.client("bedrock-agentcore")


def _invoke(
    clients: AwsClients,
    *,
    target: RuntimeTarget | None = None,
    ids: RunIds | None = None,
    attempt_limit: int = 3,
    sleeps: list[float] | None = None,
    progress: InvocationProgress | None = None,
) -> RawResponse:
    waits = [] if sleeps is None else sleeps
    with _invocation_client(clients) as client:
        return invoke(
            client,
            target or _target(),
            ids or new_run_ids(),
            _BODY,
            attempt_limit=attempt_limit,
            sleep=waits.append,
            progress=progress,
        )


def _answer(
    clients: StubbedAwsClients, *, status: int = 200, body: StreamingBody | None = None
) -> None:
    clients.stub("bedrock-agentcore", view=True).add_response(
        "invoke_agent_runtime",
        {
            "response": body if body is not None else sse_body({"response": _ANSWER}),
            "contentType": "text/event-stream",
            "statusCode": status,
            "traceParent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        },
    )


def _fail(clients: StubbedAwsClients, code: str, status: int) -> None:
    clients.stub("bedrock-agentcore", view=True).add_client_error(
        "invoke_agent_runtime",
        service_error_code=code,
        service_message="agent text that must stay out of messages",
        http_status_code=status,
    )


def _endpoint(
    clients: StubbedAwsClients, *, live_version: str | None = "24", qualifier: str = "DEFAULT"
) -> None:
    response: dict[str, Any] = {
        "agentRuntimeEndpointArn": f"{RUNTIME_ARN}/runtime-endpoint/{qualifier}",
        "agentRuntimeArn": RUNTIME_ARN,
        "status": "READY",
        "createdAt": _WHEN,
        "lastUpdatedAt": _WHEN,
        "name": qualifier,
        "id": qualifier,
    }
    if live_version is not None:
        response["liveVersion"] = live_version
    clients.stub("bedrock-agentcore-control").add_response(
        "get_agent_runtime_endpoint",
        response,
        {"agentRuntimeId": _RUNTIME_ID, "endpointName": qualifier},
    )


def _runtime(
    clients: StubbedAwsClients,
    *,
    version: str = "24",
    requested: str | None = "24",
    protocol: str | None = "HTTP",
    authorizer: dict[str, Any] | None = None,
) -> None:
    response: dict[str, Any] = {
        "agentRuntimeArn": RUNTIME_ARN,
        "agentRuntimeName": "calibrated_agent_evaluation_fixture",
        "agentRuntimeId": _RUNTIME_ID,
        "agentRuntimeVersion": version,
        "createdAt": _WHEN,
        "lastUpdatedAt": _WHEN,
        "roleArn": "arn:aws:iam::111122223333:role/calibrated-agent-evaluation-fixture-runtime",
        "status": "READY",
        "lifecycleConfiguration": {},
    }
    if protocol is not None:
        response["protocolConfiguration"] = {"serverProtocol": protocol}
    if authorizer is not None:
        response["authorizerConfiguration"] = authorizer
    expected: dict[str, Any] = {"agentRuntimeId": _RUNTIME_ID}
    if requested is not None:
        expected["agentRuntimeVersion"] = requested
    clients.stub("bedrock-agentcore-control").add_response("get_agent_runtime", response, expected)


def _deny(clients: StubbedAwsClients, method: str) -> None:
    clients.stub("bedrock-agentcore-control").add_client_error(
        method,
        service_error_code="AccessDeniedException",
        service_message="User: arn:aws:sts::111122223333:assumed-role/x is not authorized",
        http_status_code=403,
    )


# --- running a task through the executor ----------------------------------------------------


class _Clock:
    """A UTC clock that moves on one second each time it is read."""

    def __init__(self) -> None:
        self.now = _WHEN

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


class _TreeReader:
    """A span reader that answers every query with the CLM-4471 tree re-keyed to its trace ID."""

    def read_spans(self, query: RunSpanQuery) -> list[Span]:
        return clm_4471_spans(query.trace_id)


def _executor(clients: AwsClients, log_dir: Path, **overrides: Any) -> AgentCoreRuntimeExecutor:
    """An executor over ``clients`` that reads the CLM-4471 spans and never waits."""
    target = overrides.pop("target", None) or _target()
    kwargs: dict[str, Any] = {
        "span_reader": _TreeReader(),
        "clients": clients,
        "log_dir": log_dir,
        "sleep": lambda _seconds: None,
        "clock": _Clock(),
    }
    kwargs.update(overrides)
    return AgentCoreRuntimeExecutor(target, **kwargs)


def _custom_executor(
    clients: AwsClients, log_dir: Path, builder: Callable[[str, str], Mapping[str, object]]
) -> AgentCoreRuntimeExecutor:
    """An executor whose request body ``builder``, named ``claims-body``, makes."""
    return _executor(
        clients,
        log_dir,
        target=_target(payload_format="custom"),
        payload_builder=builder,
        builder_name="claims-body",
    )


def _task(prompt: str = _PROMPT) -> AgentCoreRuntimeTask:
    return AgentCoreRuntimeTask(prompt=prompt)


def _preflight(clients: StubbedAwsClients, *, qualifier: str = "DEFAULT") -> None:
    """Queue the two read-only preflight answers: the endpoint serves version 24 over HTTP."""
    _endpoint(clients, qualifier=qualifier)
    _runtime(clients)


def _stopped(clients: StubbedAwsClients, *, qualifier: str = "DEFAULT") -> None:
    clients.stub("bedrock-agentcore").add_response(
        "stop_runtime_session", {"statusCode": 200}, stop_params(qualifier=qualifier)
    )


def _one_run(clients: StubbedAwsClients) -> None:
    """Queue a whole successful first run: preflight, one answer and one stop."""
    _preflight(clients)
    _answer(clients)
    _stopped(clients)


# --- preflight ------------------------------------------------------------------------------


def test_preflight_resolves_version_authorizer_and_protocol_before_the_first_invocation(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _one_run(clients)

    with _executor(clients, tmp_path) as executor:
        [trace] = executor.run(_task())

    assert trace.output == {"text": _ANSWER}
    assert [call.operation for call in clients.calls] == [
        "GetAgentRuntimeEndpoint",
        "GetAgentRuntime",
        "InvokeAgentRuntime",
        "StopRuntimeSession",
    ]
    [record] = executor.run_records
    assert (record.runtime_version, record.runtime_version_source) == ("24", "endpoint")


def test_several_tasks_share_one_preflight_and_each_run_gets_a_new_session(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _preflight(clients)
    for _ in range(3):
        _answer(clients)
        _stopped(clients)

    # The same prompt twice, so a new session can't come from a new prompt.
    with _executor(clients, tmp_path) as executor:
        for prompt in (_PROMPT, _PROMPT, "Is water damage covered?"):
            executor.run(_task(prompt))

    assert len(clients.calls_to("GetAgentRuntimeEndpoint")) == 1
    assert len(clients.calls_to("GetAgentRuntime")) == 1
    sent = [call.params for call in clients.calls_to("InvokeAgentRuntime")]
    sessions = [params["runtimeSessionId"] for params in sent]
    assert len(sent) == len(set(sessions)) == len({params["traceParent"] for params in sent}) == 3
    records = [(record.session_id, record.runtime_version) for record in executor.run_records]
    assert records == [(session, "24") for session in sessions]
    clients.assert_no_pending_responses()


def test_a_refused_preflight_is_checked_again_on_the_next_task(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _deny(clients, "get_agent_runtime_endpoint")
    _one_run(clients)
    executor = _executor(clients, tmp_path)

    with executor:
        with pytest.raises(RuntimeInvocationError, match="GetAgentRuntimeEndpoint"):
            executor.run(_task())
        [trace] = executor.run(_task())

    assert trace.output == {"text": _ANSWER}
    assert len(clients.calls_to("GetAgentRuntimeEndpoint")) == 2
    assert len(clients.calls_to("InvokeAgentRuntime")) == 1
    refused, succeeded = executor.run_records
    assert (refused.error_code, refused.session_stop) == ("AccessDeniedException", "not_started")
    assert (succeeded.error_code, succeeded.runtime_version) == (None, "24")


@pytest.mark.parametrize(
    ("live_version", "version", "source", "protocol"),
    [
        (None, "7", "latest", "HTTP"),
        ("24", "24", "endpoint", "HTTP"),
        ("24", "24", "endpoint", None),
    ],
    ids=["no-live-version-reads-the-latest", "live-version", "no-protocol-configuration-is-http"],
)
def test_an_http_runtime_resolves_to_the_endpoints_live_version_or_else_its_latest(
    live_version: str | None,
    version: str,
    source: VersionSource,
    protocol: str | None,
    clients: StubbedAwsClients,
) -> None:
    _endpoint(clients, live_version=live_version)
    _runtime(clients, version=version, requested=live_version, protocol=protocol)

    info = resolve_runtime(clients, _target())

    assert info == RuntimeInfo(runtime_version=version, version_source=source)
    read: dict[str, str] = {"agentRuntimeId": _RUNTIME_ID}
    if live_version is not None:
        read["agentRuntimeVersion"] = live_version
    assert [call.params for call in clients.calls_to("GetAgentRuntime")] == [read]
    clients.assert_no_pending_responses()


_OAUTH_ONLY = {
    "customJWTAuthorizer": {
        "discoveryUrl": "https://idp.example.com/.well-known/openid-configuration"
    }
}


@pytest.mark.parametrize(
    ("runtime", "why"),
    [
        pytest.param(
            {"authorizer": _OAUTH_ONLY},
            "OAuth-only runtimes are not supported yet",
            id="oauth-only",
        ),
        *(pytest.param({"protocol": p}, p, id=p) for p in ["MCP", "A2A", "AGUI", "WEBSOCKET_NEXT"]),
    ],
)
def test_an_oauth_only_or_non_http_runtime_is_refused_naming_why_and_the_runtime(
    runtime: dict[str, Any], why: str, clients: StubbedAwsClients
) -> None:
    _endpoint(clients)
    _runtime(clients, **runtime)

    with pytest.raises(RuntimeInvocationError) as refused:
        resolve_runtime(clients, _target())

    _says(refused.value, why, RUNTIME_ARN)


@pytest.mark.parametrize(
    "operation",
    ["GetAgentRuntimeEndpoint", "GetAgentRuntime"],
    ids=["endpoint-read", "runtime-read"],
)
def test_a_denied_preflight_call_is_refused_naming_the_operation_and_code(
    operation: str, clients: StubbedAwsClients
) -> None:
    if operation == "GetAgentRuntime":
        _endpoint(clients)
    _deny(clients, xform_name(operation))

    with pytest.raises(RuntimeInvocationError) as refused:
        resolve_runtime(clients, _target())

    _says(refused.value, operation, "AccessDeniedException", never=["not authorized"])
    assert isinstance(refused.value.__cause__, ClientError)


def _without(answer: Scripted, key: str) -> Scripted:
    """The scripted JSON ``answer`` with ``key`` taken out of its document."""
    assert not isinstance(answer, Exception)
    document = json.loads(answer[2])
    del document[key]
    return json_answer(document)


def test_a_runtime_read_without_its_version_is_refused_naming_the_operation() -> None:
    # Answered on the wire: a Stubber won't queue an answer that lacks a required member, and
    # botocore doesn't check the answers it parses.
    preflight = preflight_sends()
    sends = ScriptedSends(
        {
            "GetAgentRuntimeEndpoint": [
                _without(preflight["GetAgentRuntimeEndpoint"][0], "liveVersion")
            ],
            "GetAgentRuntime": [_without(preflight["GetAgentRuntime"][0], "agentRuntimeVersion")],
        }
    )

    with CountingAwsClients(sends) as clients, pytest.raises(RuntimeInvocationError) as refused:
        resolve_runtime(clients, _target())

    _says(
        refused.value,
        "bedrock-agentcore-control:GetAgentRuntime",
        RUNTIME_ARN,
        "nothing was invoked",
        never=["RuntimeInfo"],
    )


def test_an_unreachable_control_plane_is_refused_before_any_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zero_retry_backoff(monkeypatch)

    def unreachable(**_kwargs: Any) -> None:
        raise EndpointConnectionError(endpoint_url=_ENDPOINT)

    with CountingAwsClients(SendCounter(200)) as clients:
        control = clients.client("bedrock-agentcore-control")
        control.meta.events.register_first("before-send.*.*", unreachable)

        with pytest.raises(RuntimeInvocationError) as refused:
            resolve_runtime(clients, _target())

    _says(refused.value, "GetAgentRuntimeEndpoint", "EndpointConnectionError")
    assert isinstance(refused.value.__cause__, EndpointConnectionError)


@pytest.mark.parametrize(
    "error",
    [
        urllib3.exceptions.ProtocolError("Connection broken: IncompleteRead"),
        urllib3.exceptions.ReadTimeoutError(None, "/runtimes", "Read timed out."),  # pyright: ignore[reportArgumentType]
        urllib3.exceptions.SSLError("record layer failure"),
        # Such as a caller's own handler on the clients raising while the call is made.
        ValueError("SENTINEL-unclassified-preflight"),
    ],
    ids=["protocol-error", "read-timeout", "ssl-error", "unclassified"],
)
def test_a_preflight_call_failing_with_a_non_aws_error_is_refused_naming_the_operation(
    error: Exception,
) -> None:
    # botocore reads the whole answer of a call that does not stream outside its own error
    # handling, so an HTTP-layer error there reaches the caller as urllib3 raised it.
    sends = ScriptedSends({"GetAgentRuntimeEndpoint": [error]})

    with CountingAwsClients(sends) as clients, pytest.raises(RuntimeInvocationError) as refused:
        resolve_runtime(clients, _target())

    _says(
        refused.value,
        "bedrock-agentcore-control:GetAgentRuntimeEndpoint",
        type(error).__name__,
        "nothing was invoked",
        never=["SENTINEL"],
    )
    assert refused.value.__cause__ is error


# --- invocation: the request ----------------------------------------------------------------


@pytest.mark.parametrize("payload_format", [fmt.name for fmt in PAYLOAD_FORMATS])
def test_each_named_format_builds_the_probe_body_for_the_prompt_and_session(
    payload_format: str,
) -> None:
    [named] = [fmt for fmt in PAYLOAD_FORMATS if fmt.name == payload_format]
    session = new_run_ids().session_id

    body = request_body(_target(payload_format=payload_format), _PROMPT, session)

    expected = named.build_body(_PROMPT, session)
    assert body == json.dumps(expected).encode("utf-8")
    # Built with the prompt first and the session second, never the other way round.
    assert expected != named.build_body(session, _PROMPT)


@pytest.mark.parametrize("mapping", [dict, MappingProxyType], ids=["dict", "read-only-mapping"])
def test_a_custom_builder_body_is_built_as_returned(
    mapping: Callable[[dict[str, str]], Any],
) -> None:
    asked: list[tuple[str, str]] = []

    def builder(prompt: str, session_id: str) -> Mapping[str, object]:
        asked.append((prompt, session_id))
        return mapping({"question": prompt, "conversation": session_id, "locale": "en-GB"})

    session = new_run_ids().session_id

    body = request_body(_target(payload_format="custom"), _PROMPT, session, builder, "claims-body")

    assert asked == [(_PROMPT, session)]
    assert body == json.dumps(
        {"question": _PROMPT, "conversation": session, "locale": "en-GB"}
    ).encode("utf-8")


@pytest.mark.parametrize(
    "result",
    [["SENTINEL-item"], "SENTINEL-text", 271828],
    ids=["list", "string", "number"],
)
def test_a_builder_result_that_is_not_a_json_object_is_refused_naming_its_type(
    result: object,
) -> None:
    def builder(_prompt: str, _session_id: str) -> Any:
        return result

    session = new_run_ids().session_id

    with pytest.raises(RuntimeInvocationError) as refused:
        request_body(_target(payload_format="custom"), _PROMPT, session, builder, "claims-body")

    _says(
        refused.value,
        type(result).__name__,
        "claims-body",
        "nothing was invoked",
        never=["SENTINEL", "271828"],
    )


def _builder_never_called(_prompt: str, _session_id: str) -> Mapping[str, object]:
    raise AssertionError("a named payload format builds its own body, so no builder is called")


@pytest.mark.parametrize(
    ("payload_format", "builder", "builder_name"),
    [
        ("custom", None, "claims-body"),
        ("agentcore_sse", _builder_never_called, None),
        ("agentcore_sse", None, "claims-body"),
    ],
    ids=["custom-without-a-builder", "named-format-with-a-builder", "named-format-with-a-name"],
)
def test_request_body_refuses_a_builder_pairing_that_does_not_match_its_format(
    payload_format: str,
    builder: Callable[[str, str], Mapping[str, object]] | None,
    builder_name: str | None,
) -> None:
    target = _target(payload_format=payload_format)

    with pytest.raises(RuntimeInvocationError) as refused:
        request_body(target, _PROMPT, new_run_ids().session_id, builder, builder_name)

    _says(
        refused.value,
        "nothing was invoked",
        f"payload_format={payload_format!r}",
        never=[_PROMPT],
    )


def test_request_body_counts_an_empty_builder_name_as_none_given() -> None:
    target = _target(payload_format="agentcore_sse")
    session = new_run_ids().session_id

    body = request_body(target, _PROMPT, session, None, "")

    assert body == request_body(target, _PROMPT, session)


def _nested_lists(levels: int) -> list[Any]:
    """Lists ``levels`` deep, built with a loop."""
    value: list[Any] = []
    for _ in range(levels - 1):
        value = [value]
    return value


@pytest.mark.parametrize(
    ("extra", "why"),
    [
        ({"opts": {"t": float("inf")}}, "a number JSON cannot represent"),
        ({"xs": [float("nan")]}, "a number JSON cannot represent"),
        ({"deep": [{"t": -float("inf")}]}, "a number JSON cannot represent"),
        # The body is the first level, so this value adds 256 more.
        ({"v": _nested_lists(256)}, "nests more than 256 levels"),
    ],
    ids=["infinity-in-an-object", "nan-in-a-list", "-infinity-deeper", "257-levels"],
)
def test_a_body_holding_nan_infinity_or_more_than_256_levels_is_refused_before_any_invocation(
    tmp_path: Path, clients: StubbedAwsClients, extra: dict[str, object], why: str
) -> None:
    def builder(prompt: str, _session_id: str) -> Mapping[str, object]:
        return {"prompt": prompt, **extra}

    executor = _custom_executor(clients, tmp_path, builder)

    with executor, pytest.raises(RuntimeInvocationError) as refused:
        executor.run(_task())

    _says(refused.value, "claims-body", "nothing was invoked", why, never=[_PROMPT])
    assert clients.calls_to("InvokeAgentRuntime") == []
    assert clients.calls_to("StopRuntimeSession") == []
    [record] = executor.run_records
    assert record.session_stop == "not_started"


def test_a_body_nested_256_levels_is_sent(tmp_path: Path, clients: StubbedAwsClients) -> None:
    def builder(prompt: str, _session_id: str) -> Mapping[str, object]:
        return {"prompt": prompt, "v": _nested_lists(255)}

    _one_run(clients)

    with _custom_executor(clients, tmp_path, builder) as executor:
        executor.run(_task())

    [sent] = clients.calls_to("InvokeAgentRuntime")
    assert json.loads(sent.params["payload"]) == {"prompt": _PROMPT, "v": _nested_lists(255)}


@pytest.mark.parametrize("payload_format", [fmt.name for fmt in PAYLOAD_FORMATS])
def test_a_rejected_format_is_not_followed_by_another_format(
    tmp_path: Path, payload_format: str, clients: StubbedAwsClients
) -> None:
    _preflight(clients)
    _fail(clients, "ValidationException", 400)
    _stopped(clients)
    target = _target(payload_format=payload_format)
    executor = _executor(clients, tmp_path, target=target)

    with executor, pytest.raises(RuntimeInvocationError, match="ValidationException"):
        executor.run(_task())

    # One request, in the format named, and no other format tried after the service refused it.
    [sent] = clients.calls_to("InvokeAgentRuntime")
    session = sent.params["runtimeSessionId"]
    assert sent.params["payload"] == request_body(target, _PROMPT, session)
    [record] = executor.run_records
    assert (record.attempts, record.error_code) == (1, "ValidationException")


@pytest.mark.parametrize(
    ("payload_format", "qualifier"),
    [("openai", None), ("agentcore_sse", "live"), ("nested_agentcore", "live")],
    ids=["default", "named", "nested-format-keeps-the-named-qualifier"],
)
def test_the_invocation_and_the_stop_go_to_the_named_qualifier_or_default(
    payload_format: str, qualifier: str | None, tmp_path: Path, clients: StubbedAwsClients
) -> None:
    # The probe sends the nested format to the URL with ?qualifier=DEFAULT; here the qualifier is
    # the caller's, for the invocation and for the stop alike.
    sent_to = qualifier or "DEFAULT"
    _preflight(clients, qualifier=sent_to)
    _answer(clients)
    _stopped(clients, qualifier=sent_to)
    named = {} if qualifier is None else {"qualifier": qualifier}
    target = _target(payload_format=payload_format, **named)

    with _executor(clients, tmp_path, target=target) as executor:
        executor.run(_task())

    [sent] = clients.calls_to("InvokeAgentRuntime")
    assert (sent.params["agentRuntimeArn"], sent.params["qualifier"]) == (RUNTIME_ARN, sent_to)
    assert sent.params["payload"] == request_body(target, _PROMPT, sent.params["runtimeSessionId"])
    [stopped] = clients.calls_to("StopRuntimeSession")
    assert stopped.params["qualifier"] == sent_to


# --- invocation: SDK attempts and timeouts --------------------------------------------------


def test_an_invocation_is_sent_once_on_a_500_while_stops_and_control_plane_calls_retry_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zero_retry_backoff(monkeypatch)
    counter = SendCounter.error("InternalServerException", 500)

    with CountingAwsClients(counter) as clients:
        with pytest.raises(RuntimeInvocationError, match="InternalServerException"):
            _invoke(clients)
        stopped = stop(clients.client("bedrock-agentcore"), _target(), new_run_ids().session_id)
        with pytest.raises(RuntimeInvocationError, match="GetAgentRuntimeEndpoint"):
            resolve_runtime(clients, _target())

    assert counter.count("InvokeAgentRuntime") == 1
    assert stopped == "failed"
    assert counter.count("StopRuntimeSession") > 1
    assert counter.count("GetAgentRuntimeEndpoint") > 1


@pytest.mark.parametrize(
    ("overrides", "read_timeout"),
    [({}, 300.0), ({"read_timeout_seconds": 45.5}, 45.5)],
    ids=["executor-default", "caller-set"],
)
def test_only_invocations_take_the_read_timeout_and_they_connect_within_10_seconds(
    overrides: dict[str, float], read_timeout: float, tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _one_run(clients)

    with _executor(clients, tmp_path, **overrides) as executor:
        executor.run(_task())

    [invoker] = [built for built in clients.built if built.view]
    control = next(b for b in clients.built if b.service == "bedrock-agentcore-control")
    stopper = next(b for b in clients.built if b.service == "bedrock-agentcore" and not b.view)
    assert invoker.client.meta.config.connect_timeout == 10
    assert invoker.client.meta.config.read_timeout == read_timeout
    # The control plane and the stop keep the default 30 seconds.
    assert control.client.meta.config.read_timeout == 30
    assert stopper.client.meta.config.read_timeout == 30
    assert len(clients.calls_to("InvokeAgentRuntime")) == 1


# --- invocation: failures -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("ValidationException", 400),
        ("ServiceQuotaExceededException", 402),
        ("AccessDeniedException", 403),
        ("ResourceNotFoundException", 404),
        ("RuntimeClientError", 424),
        ("InternalServerException", 500),
    ],
)
def test_other_aws_errors_are_refused_after_one_attempt(
    code: str, status: int, clients: StubbedAwsClients
) -> None:
    _fail(clients, code, status)
    ids = new_run_ids()
    sleeps: list[float] = []

    with pytest.raises(RuntimeInvocationError) as refused:
        _invoke(clients, ids=ids, sleeps=sleeps)

    _says(refused.value, code, ids.session_id, never=["agent text"])
    assert len(clients.calls_to("InvokeAgentRuntime")) == 1
    assert sleeps == []


def _raise_on(client: BaseClient, hook: str, error: Exception) -> None:
    """Make ``client`` raise ``error`` at ``hook``, before any other handler runs."""

    def handler(**_kwargs: Any) -> None:
        raise error

    client.meta.events.register_first(f"{hook}.*.*", handler)


# botocore refuses a runtime session ID shorter than 33 characters before signing anything. Built
# without validation, since RunIds itself refuses it, so the refusal tested is botocore's own.
_SHORT_SESSION = RunIds.model_construct(
    session_id="short-session",
    w3c_trace_id="0af7651916cd43dd8448eb211c80319c",
    parent_span_id="b7ad6b7169203331",
)
_FAILS_BEFORE_SIGNING = {
    "no-credentials": NoCredentialsError(),
    "partial-credentials": PartialCredentialsError(
        provider="env", cred_var="AWS_SECRET_ACCESS_KEY"
    ),
    "token-retrieval": TokenRetrievalError(provider="sso", error_msg="token expired"),
    "credential-refresh": CredentialRetrievalError(provider="process", error_msg="refresh failed"),
    "sso-token": SSOTokenLoadError(error_msg="token expired"),
    "sso-session-expired": UnauthorizedSSOTokenError(),
    "refresh-needs-mfa": RefreshWithMFAUnsupportedError(),
}
_FAILS_BEFORE_SENDING = {
    "endpoint-connection": EndpointConnectionError(endpoint_url=_ENDPOINT),
    "connect-timeout": ConnectTimeoutError(endpoint_url=_ENDPOINT),
    "ssl": SSLError(endpoint_url=_ENDPOINT, error="verify failed"),
    "proxy-connection": ProxyConnectionError(
        proxy_url="http://proxy.example.com:3128", error="refused"
    ),
}


@pytest.mark.parametrize(
    ("hook", "error"),
    [
        pytest.param(None, None, id="invalid-parameter"),
        *(pytest.param("before-sign", e, id=name) for name, e in _FAILS_BEFORE_SIGNING.items()),
        *(pytest.param("before-send", e, id=name) for name, e in _FAILS_BEFORE_SENDING.items()),
    ],
)
def test_a_request_that_never_left_is_refused_as_not_sent(
    hook: str | None, error: Exception | None
) -> None:
    counter = SendCounter(200)
    progress = InvocationProgress()
    ids = _SHORT_SESSION if error is None else new_run_ids()
    cause = ParamValidationError if error is None else type(error)

    with CountingAwsClients(counter) as clients, _invocation_client(clients) as client:
        if hook and error:
            _raise_on(client, hook, error)
        with pytest.raises(RuntimeInvocationError) as refused:
            invoke(
                client, _target(), ids, _BODY, attempt_limit=3, sleep=[].append, progress=progress
            )

    _says(refused.value, "not sent", ids.session_id, cause.__name__, never=["may still have run"])
    assert type(refused.value.__cause__) is cause
    assert counter.sends == []
    assert progress == InvocationProgress()
    assert not progress.sent
    # A connect timeout or a refused proxy is an OSError, like a read timeout, and still counts as
    # never sent.
    if cause in (ConnectTimeoutError, ProxyConnectionError):
        assert isinstance(refused.value.__cause__, OSError)


class _FailsMidStream(io.RawIOBase):
    """A response stream that yields the start of an event, then fails with ``error``."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.reads = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        self.reads += 1
        if self.reads == 1:
            return b'data: {"response": "SENTINEL-partial-answer'
        raise self.error


def _send_fails(error: Exception) -> Callable[[], AwsClients]:
    """New clients, for the caller to close, whose invocation fails on the wire with ``error``."""

    def make() -> AwsClients:
        return CountingAwsClients(
            ScriptedSends(
                {
                    **preflight_sends(),
                    "InvokeAgentRuntime": [error],
                    "StopRuntimeSession": [json_answer({})],
                }
            )
        )

    return make


def _answered_with(body: Callable[[], StreamingBody]) -> Callable[[], AwsClients]:
    """New clients, for the caller to close, whose invocation answers with ``body``."""

    def make() -> AwsClients:
        clients = StubbedAwsClients()
        _preflight(clients)
        _answer(clients, body=body())
        _stopped(clients)
        return clients

    return make


def _breaks_mid_stream(error: Exception) -> Callable[[], AwsClients]:
    return _answered_with(lambda: StreamingBody(_FailsMidStream(error), None))


def _stopped_sessions(clients: AwsClients) -> list[str | None]:
    if isinstance(clients, StubbedAwsClients):
        return [call.params["runtimeSessionId"] for call in clients.calls_to("StopRuntimeSession")]
    assert isinstance(clients, CountingAwsClients)
    assert isinstance(clients.counter, ScriptedSends)
    return clients.counter.sessions("StopRuntimeSession")


_SENT = "after the request was sent"


@pytest.mark.parametrize(
    ("make_clients", "cause", "when"),
    [
        pytest.param(
            _send_fails(ReadTimeoutError(endpoint_url=_ENDPOINT)),
            ReadTimeoutError,
            _SENT,
            id="read-timeout",
        ),
        pytest.param(
            _breaks_mid_stream(
                urllib3.exceptions.ProtocolError("Connection broken: IncompleteRead")
            ),
            ResponseStreamingError,
            _SENT,
            id="broken-mid-stream",
        ),
        pytest.param(
            _answered_with(
                lambda: StreamingBody(io.BytesIO(b'data: {"response": "SENTINEL"}\n'), 4096)
            ),
            IncompleteReadError,
            _SENT,
            id="incomplete-body",
        ),
        pytest.param(
            _send_fails(ConnectionClosedError(endpoint_url=_ENDPOINT)),
            ConnectionClosedError,
            _SENT,
            id="connection-closed",
        ),
        # StreamingBody wraps urllib3's read timeouts and protocol errors, and lets every other
        # urllib3 error through as it is.
        pytest.param(
            _breaks_mid_stream(urllib3.exceptions.SSLError("record layer failure")),
            urllib3.exceptions.SSLError,
            _SENT,
            id="ssl-while-reading",
        ),
        pytest.param(
            _breaks_mid_stream(urllib3.exceptions.DecodeError("bad gzip stream")),
            urllib3.exceptions.DecodeError,
            _SENT,
            id="decode-while-reading",
        ),
        pytest.param(
            _send_fails(urllib3.exceptions.HTTPError("connection pool failure")),
            urllib3.exceptions.HTTPError,
            _SENT,
            id="http-error-while-sending",
        ),
        # Not an error known to come before sending, so the request may have gone out.
        pytest.param(
            _send_fails(ValueError("SENTINEL-unclassified-send")),
            ValueError,
            "when the request may already have been sent",
            id="unclassified-while-sending",
        ),
        pytest.param(
            _breaks_mid_stream(ValueError("SENTINEL-unclassified-read")),
            ValueError,
            _SENT,
            id="unclassified-while-reading",
        ),
    ],
)
def test_a_failure_once_the_request_may_have_left_is_refused_saying_the_agent_may_have_run(
    tmp_path: Path, make_clients: Callable[[], AwsClients], cause: type[Exception], when: str
) -> None:
    with make_clients() as clients:
        executor = _executor(clients, tmp_path)

        with executor, pytest.raises(RuntimeInvocationError) as refused:
            executor.run(_task())

    [record] = executor.run_records
    _says(
        refused.value,
        "may still have run",
        record.session_id,
        f"failed with {cause.__name__} {when}",
        never=["not sent", "SENTINEL"],
    )
    assert type(refused.value.__cause__) is cause
    # The request may have opened the session, so it is stopped, once, before run() raises.
    assert _stopped_sessions(clients) == [record.session_id]
    assert (record.attempts, record.session_stop) == (1, "stopped")


def test_an_unclassified_invocation_failure_counts_as_one_attempt_that_may_have_run() -> None:
    progress = InvocationProgress()

    with (
        _send_fails(ValueError("SENTINEL-unclassified-send"))() as clients,
        pytest.raises(RuntimeInvocationError, match="may still have run") as refused,
    ):
        _invoke(clients, progress=progress)

    # Nothing shows whether it was sent, so the message doesn't say it was.
    _says(refused.value, "failed with ValueError when the request may already have been sent")
    assert isinstance(refused.value.__cause__, ValueError)
    assert progress.attempts == 1
    assert progress.sent


@pytest.mark.parametrize("status", [202, 204, None], ids=["202", "204", "no-status"])
def test_a_success_status_other_than_200_is_refused_naming_it(
    status: int | None, clients: StubbedAwsClients
) -> None:
    if status is None:
        # The answer a client gets when the response carries no status code at all.
        clients.stub("bedrock-agentcore", view=True).add_response(
            "invoke_agent_runtime",
            {"response": sse_body({"response": _ANSWER}), "contentType": "text/event-stream"},
        )
    else:
        _answer(clients, status=status, body=streaming_body(b""))
    ids = new_run_ids()

    with pytest.raises(RuntimeInvocationError) as refused:
        _invoke(clients, ids=ids)

    answered = "no HTTP status" if status is None else f"HTTP status {status}"
    _says(refused.value, answered, "not 200", ids.session_id)


# --- invocation: retrying a busy session or throttling --------------------------------------


def test_a_conflict_then_success_returns_a_trace_after_two_attempts(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _preflight(clients)
    _fail(clients, "RetryableConflictException", 409)
    _answer(clients)
    _stopped(clients)
    sleeps: list[float] = []

    with _executor(clients, tmp_path, sleep=sleeps.append) as executor:
        [trace] = executor.run(_task())

    assert trace.output == {"text": _ANSWER}
    first, second = clients.calls_to("InvokeAgentRuntime")
    assert first.params["runtimeSessionId"] == second.params["runtimeSessionId"]
    [record] = executor.run_records
    assert record.attempts == 2
    assert record.session_id == first.params["runtimeSessionId"]
    assert sleeps == [1]
    assert len(clients.calls_to("StopRuntimeSession")) == 1


def test_an_attempt_limit_of_one_does_not_retry(clients: StubbedAwsClients) -> None:
    _fail(clients, "RetryableConflictException", 409)
    _answer(clients)
    sleeps: list[float] = []

    with pytest.raises(RuntimeInvocationError) as refused:
        _invoke(clients, attempt_limit=1, sleeps=sleeps)

    _says(refused.value, "RetryableConflictException", "1 attempt")
    assert sleeps == []
    assert len(clients.calls_to("InvokeAgentRuntime")) == 1


@pytest.mark.parametrize(
    ("second_code", "second_status"),
    [("ThrottlingException", 429), ("RetryableConflictException", 409)],
    ids=["throttled-twice", "throttled-then-busy"],
)
def test_a_throttled_or_busy_attempt_is_resent_unchanged_after_one_then_two_seconds(
    clients: StubbedAwsClients, second_code: str, second_status: int
) -> None:
    _fail(clients, "ThrottlingException", 429)
    _fail(clients, second_code, second_status)
    _answer(clients)
    ids = new_run_ids()
    sleeps: list[float] = []

    response = _invoke(clients, ids=ids, sleeps=sleeps)

    assert sleeps == [1, 2]
    first, second, third = clients.calls_to("InvokeAgentRuntime")
    assert first.params == second.params == third.params
    assert first.params["runtimeSessionId"] == ids.session_id
    assert first.params["traceParent"] == (f"00-{ids.w3c_trace_id}-{ids.parent_span_id}-01")
    assert first.params["baggage"] == f"session.id={ids.session_id}"
    assert first.params["payload"] == _BODY
    assert (response.attempts, response.status_code) == (3, 200)
    assert response.lines == (b'data: {"response": "Yes, it is."}', b"")


def test_conflicts_on_every_attempt_are_refused_naming_code_session_and_attempts(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _preflight(clients)
    # One more conflict than the default limit allows, so a higher limit would send it.
    for _ in range(4):
        _fail(clients, "RetryableConflictException", 409)
    _stopped(clients)
    sleeps: list[float] = []
    # Built without an attempt limit, so the default one applies.
    executor = _executor(clients, tmp_path, sleep=sleeps.append)

    with executor, pytest.raises(RuntimeInvocationError) as refused:
        executor.run(_task())

    [record] = executor.run_records
    _says(refused.value, "RetryableConflictException", record.session_id, "3 attempts")
    assert sleeps == [1, 2]
    assert len(clients.calls_to("InvokeAgentRuntime")) == 3
    assert record.attempts == 3
    assert isinstance(refused.value.__cause__, ClientError)


# --- invocation: what the attempts reached --------------------------------------------------


def test_invoke_reports_attempts_and_the_last_answer_when_it_fails(
    clients: StubbedAwsClients,
) -> None:
    _fail(clients, "RetryableConflictException", 409)
    _fail(clients, "RuntimeClientError", 424)
    progress = InvocationProgress()

    with pytest.raises(RuntimeInvocationError, match="RuntimeClientError"):
        _invoke(clients, progress=progress)

    assert (progress.attempts, progress.status_code) == (2, 424)
    assert progress.sent


def test_a_guard_refusal_leaves_progress_at_no_attempts() -> None:
    progress = InvocationProgress()

    with StubbedAwsClients(handlers=[BilledCallBudget(0)]) as clients:
        # Queued so the stub accepts the call; the guard refuses it before the stub answers.
        _answer(clients)
        with pytest.raises(RuntimeInvocationError, match="BilledCallBudgetExceeded"):
            _invoke(clients, progress=progress)

    assert clients.calls_to("InvokeAgentRuntime") == []
    assert progress == InvocationProgress()
    assert not progress.sent


# --- stopping a session ---------------------------------------------------------------------


class _BoundedCallOnAFakeClock:
    """A bounded call that runs the call in line, timed on a clock the test moves, so no test
    waits on a real clock. The call finished in time if the clock moved no further than the
    deadline while it ran."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(
        self, fn: Callable[[], object], *, timeout_seconds: float, name: str
    ) -> BoundedOutcome:
        started = self.now
        error: BaseException | None = None
        try:
            fn()
        except BaseException as exc:  # handed back, as the real bounded call does
            error = exc
        if self.now - started > timeout_seconds:
            return BoundedOutcome(finished=False)
        return BoundedOutcome(finished=True, error=error)


@pytest.mark.parametrize(
    ("seconds", "result"), [(5.1, "failed"), (4.9, "stopped")], ids=["too-late", "in-time"]
)
def test_a_stop_counts_as_failed_and_warns_only_when_it_does_not_return_in_time(
    seconds: float, result: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = _BoundedCallOnAFakeClock()
    monkeypatch.setattr("calibrated_agent_evaluation.agentcore.runtime.call_bounded", clock)
    session = new_run_ids().session_id

    def takes_its_time(**kwargs: Any) -> None:
        if kwargs["model"].name == "StopRuntimeSession":
            clock.now += seconds

    with StubbedAwsClients(handlers=[takes_its_time]) as clients:
        _stopped(clients)
        with caplog.at_level(logging.WARNING, logger="calibrated_agent_evaluation"):
            stopped = stop(
                clients.client("bedrock-agentcore"), _target(), session, timeout_seconds=5
            )

    assert stopped == result
    warned = any(
        record.levelno == logging.WARNING and session in record.getMessage()
        for record in caplog.records
    )
    assert warned == (result == "failed")
    [call] = clients.calls_to("StopRuntimeSession")
    assert call.params["runtimeSessionId"] == session
    clients.assert_no_pending_responses()


class _FailsTheStop:
    """A call handler that fails each ``StopRuntimeSession`` with ``error``, as the HTTP layer would
    while botocore reads the answer of a call that does not stream, or as Ctrl-C would."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __call__(self, **kwargs: Any) -> None:
        if kwargs["model"].name == "StopRuntimeSession":
            raise self.error


@pytest.mark.parametrize(
    "error",
    [
        urllib3.exceptions.ProtocolError("Connection broken: IncompleteRead"),
        urllib3.exceptions.ReadTimeoutError(None, "/sessions/stop", "Read timed out."),  # pyright: ignore[reportArgumentType]
        urllib3.exceptions.SSLError("record layer failure"),
        OSError("connection reset by peer"),
    ],
    ids=["protocol-error", "read-timeout", "ssl-error", "os-error"],
)
def test_a_stop_that_fails_below_the_sdk_is_recorded_as_failed_and_warned(
    error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    session = new_run_ids().session_id

    with StubbedAwsClients(handlers=[_FailsTheStop(error)]) as clients:
        # Queued so the stub accepts the call; the failure comes before the stub answers.
        _stopped(clients)
        with caplog.at_level(logging.WARNING, logger="calibrated_agent_evaluation"):
            result = stop(
                clients.client("bedrock-agentcore"), _target(), session, timeout_seconds=5
            )

    assert result == "failed"
    [warning] = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert session in warning.getMessage()
    assert type(error).__name__ in warning.getMessage()


def test_an_interrupt_during_a_bounded_stop_reaches_the_caller() -> None:
    with StubbedAwsClients(handlers=[_FailsTheStop(KeyboardInterrupt())]) as clients:
        _stopped(clients)

        with pytest.raises(KeyboardInterrupt):
            stop(
                clients.client("bedrock-agentcore"),
                _target(),
                new_run_ids().session_id,
                timeout_seconds=5,
            )
