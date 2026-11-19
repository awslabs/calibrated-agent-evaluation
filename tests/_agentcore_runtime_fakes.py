# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test doubles for the AgentCore Runtime executor's tests. Each one mirrors a real surface.

- :class:`StubbedAwsClients` is a real :class:`~calibrated_agent_evaluation.agentcore.AwsClients`
  over fake credentials. Every client it builds gets a botocore ``Stubber``, and so does every
  client of a set derived from it with ``with_config`` (the invocation view). Stubs are keyed by
  service and by whether they answer the invocation view, and can be queued before the client
  exists. A misspelled operation, member or response field fails the test. Each call that reaches a
  stub is recorded with its parameters, because ``Stubber`` keeps no record of them.
- :class:`CountingAwsClients` builds unstubbed clients whose HTTP sends a :class:`SendCounter`
  answers and counts. A ``Stubber`` answers at ``before-call``, ahead of botocore's retry
  handler, so retry behavior can only be seen on clients without one; :func:`zero_retry_backoff`
  takes the waiting out of those retries. :class:`ScriptedSends` answers each operation's sends
  in turn, or fails them as the HTTP layer would, and :func:`preflight_sends` scripts the two
  read-only preflight answers.
- :func:`streaming_body`, :func:`sse_body` and :func:`json_body` build a real
  ``botocore.response.StreamingBody``.
- :func:`add_preflight_responses`, :func:`add_invocation_response` and :func:`add_stop_response`
  queue the stubbed answers of one run on :class:`StubbedAwsClients`: the two read-only preflight
  reads (an OAuth-only runtime, or an endpoint with no live version, on request), one invocation
  answer, and one session stop.
- :class:`FakeRunSpanReader` returns the CLM-4471 span tree re-keyed to each query's trace ID, and
  :class:`FakeAgentCoreSpanReader` stands in for the AgentCore span reader in CLI tests.

The sessions read no profile, AWS config file or credentials file, so a test sees the same thing
whatever the machine's AWS setup is.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Self, TypeAlias

import boto3
import botocore.session
import pytest
import urllib3
from botocore.awsrequest import AWSResponse
from botocore.client import BaseClient
from botocore.config import Config
from botocore.response import StreamingBody
from botocore.retries import standard
from botocore.stub import ANY, Stubber

import calibrated_agent_evaluation.agentcore
from _helpers import load_corpus_trace
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.agentcore.session import (
    DEFAULT_CLIENT_CONFIG,
    AwsClients,
    BeforeCallHandler,
)
from calibrated_agent_evaluation.schema.types import Span

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.run_spans import RunSpanQuery

__all__ = [
    "ANSWER",
    "BUNDLE_ARN",
    "BUNDLE_VERSION",
    "REGION",
    "RETURNED_TRACEPARENT",
    "RUNTIME_ARN",
    "RUNTIME_ID",
    "BuiltClient",
    "CountingAwsClients",
    "FakeAgentCoreSpanReader",
    "FakeRunSpanReader",
    "Scripted",
    "ScriptedSends",
    "SendCounter",
    "StubbedAwsClients",
    "StubbedCall",
    "add_invocation_response",
    "add_preflight_responses",
    "add_reader_response",
    "add_stop_response",
    "clm_4471_spans",
    "fake_session",
    "json_answer",
    "json_body",
    "preflight_sends",
    "register_fake_agentcore_span_reader",
    "sse_body",
    "stop_params",
    "streaming_body",
    "zero_retry_backoff",
]

REGION = "us-west-2"
# AWS's documentation account: a synthetic ARN that can't name a real runtime.
_ACCOUNT = "111122223333"
RUNTIME_ARN = (
    f"arn:aws:bedrock-agentcore:{REGION}:{_ACCOUNT}:runtime/"
    "calibrated_agent_evaluation_fixture-AbCdEf1234"
)
BUNDLE_ARN = (
    f"arn:aws:bedrock-agentcore:{REGION}:{_ACCOUNT}:configuration-bundle/"
    "calibrated_agent_evaluation_bundle-AbCdEf1234"
)
BUNDLE_VERSION = "0f8b6c1e-3a2d-4c5b-9e7f-1a2b3c4d5e6f"
# The runtime's ID, the ARN's resource after ``runtime/``, which the preflight reads it by.
RUNTIME_ID = RUNTIME_ARN.rpartition("/")[2]
# The traceparent a stubbed invocation answer carries back.
RETURNED_TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
# The final text of the default stubbed invocation answer.
ANSWER = "Denied: the vacancy exclusion applies."

_CLM_4471 = "CLM-4471"
# The fake reader's telemetry query session; any value the service's pattern accepts.
_READER_QUERY_SESSION = "calibrated-agent-evaluation-fake-span-reader"
_STASHED_PARAMS = "calibrated_agent_evaluation_test_api_params"
# The header both runtime calls carry their session ID in, per the installed service model.
_SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"


def fake_session(region: str = REGION) -> boto3.Session:
    """A boto3 session with fake static credentials that reads nothing from the environment."""
    core = botocore.session.Session(
        session_vars={
            "profile": (None, None, None, None),
            "config_file": (None, None, os.devnull, None),
            "credentials_file": (None, None, os.devnull, None),
        }
    )
    return boto3.Session(
        botocore_session=core,
        region_name=region,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )


@dataclass
class BuiltClient:
    """One client a fake set of clients built, and whether that set has closed it since."""

    service: str
    view: bool
    client: BaseClient
    owner: AwsClients
    closed: bool = False


class _RecordingAwsClients(AwsClients):
    """Marks the set ``with_config`` derives as the invocation view, and records built clients.

    ``with_config`` copies the set, so the view shares this set's records.
    """

    def __init__(
        self,
        *,
        region: str,
        handlers: Sequence[BeforeCallHandler],
        config: Config,
    ) -> None:
        self.session = fake_session(region)
        super().__init__(self.session, handlers=handlers, config=config)
        self.is_invocation_view = False
        self.built: list[BuiltClient] = []

    def with_config(self, config: Config) -> Self:
        view = super().with_config(config)
        view.is_invocation_view = True
        return view

    def client(self, service_name: str, *, region: str | None = None) -> BaseClient:
        client = super().client(service_name, region=region)
        if not any(entry.client is client for entry in self.built):
            self.built.append(BuiltClient(service_name, self.is_invocation_view, client, self))
            self._on_build(service_name, client)
        return client

    def close(self) -> None:
        for entry in self.built:
            if entry.owner is self:
                entry.closed = True
        super().close()

    def _on_build(self, service_name: str, client: BaseClient) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class StubbedCall:
    """One call that reached a stub: its service, operation and the parameters it was given."""

    service: str
    operation: str
    params: dict[str, Any]
    view: bool


class _StubSlot:
    """The ``Stubber`` for one (service, view) key, which may exist before its client does."""

    def __init__(self, stubber: Stubber, stand_in: BaseClient | None = None) -> None:
        self.stubber = stubber
        self.stand_in = stand_in
        self.bound: BaseClient | None = None

    def bind(self, client: BaseClient) -> None:
        if self.bound is not None:
            self.stubber.deactivate()
        # A stub queued before its client existed was made over a stand-in client of the same
        # service, which only checked the queued responses against the model; from here on it
        # answers the client the code under test built.
        self.stubber.client = client
        self.stubber.activate()
        self.bound = client
        self.close_stand_in()

    def close_stand_in(self) -> None:
        stand_in, self.stand_in = self.stand_in, None
        if stand_in is not None:
            stand_in.close()


def _stash_params(params: dict[str, Any], context: dict[str, Any], **_kwargs: Any) -> None:
    # The parameters as the caller passed them, before botocore fills in idempotency tokens.
    context[_STASHED_PARAMS] = dict(params)


class StubbedAwsClients(_RecordingAwsClients):
    """Clients over fake credentials whose every call a ``Stubber`` answers.

    Guards in ``handlers`` run before each stub answers, as they would before a request is sent.
    Queue answers with :meth:`stub`; read what was called from :attr:`calls`. Use it as a context
    manager, like the real set; :attr:`calls` and :attr:`built` can still be read once it is closed.
    """

    def __init__(
        self,
        *,
        region: str = REGION,
        handlers: Sequence[BeforeCallHandler] = (),
        config: Config = DEFAULT_CLIENT_CONFIG,
    ) -> None:
        super().__init__(region=region, handlers=handlers, config=config)
        self.calls: list[StubbedCall] = []
        self._slots: dict[tuple[str, bool], _StubSlot] = {}

    def stub(self, service_name: str, *, view: bool = False) -> Stubber:
        """The stub that answers ``service_name`` on these clients, or on the invocation view."""
        return self._slot(service_name, view).stubber

    def calls_to(self, operation: str) -> list[StubbedCall]:
        """Every recorded call to ``operation`` (e.g. ``"InvokeAgentRuntime"``), in order."""
        return [call for call in self.calls if call.operation == operation]

    def assert_no_pending_responses(self) -> None:
        """Fail if any queued answer, on these clients or the view, was never asked for."""
        for slot in self._slots.values():
            slot.stubber.assert_no_pending_responses()

    def close(self) -> None:
        """Close the clients built here, and the stand-ins of stubs no client has taken over."""
        try:
            super().close()
        finally:
            # The view shares the stubs, so whichever set closes first closes the stand-ins. A
            # closed stand-in keeps its service model, so a stub queued later is still checked.
            for slot in self._slots.values():
                slot.close_stand_in()

    def _slot(self, service_name: str, view: bool, client: BaseClient | None = None) -> _StubSlot:
        key = (service_name, view)
        slot = self._slots.get(key)
        if slot is None:
            if client is not None:
                slot = _StubSlot(Stubber(client))
            else:
                stand_in = self.session.client(service_name, config=DEFAULT_CLIENT_CONFIG)  # pyright: ignore[reportUnknownMemberType]
                slot = _StubSlot(Stubber(stand_in), stand_in)
            self._slots[key] = slot
        return slot

    def _on_build(self, service_name: str, client: BaseClient) -> None:
        view = self.is_invocation_view
        calls = self.calls

        def record(model: Any, context: dict[str, Any], **_kwargs: Any) -> None:
            calls.append(StubbedCall(service_name, model.name, context[_STASHED_PARAMS], view))

        client.meta.events.register("before-parameter-build.*.*", _stash_params)
        # Registered ahead of the stub's own handler, and after the guards (which run first), so
        # a call a guard refuses is never recorded as one the service received.
        client.meta.events.register("before-call.*.*", record)
        self._slot(service_name, view, client).bind(client)


def stop_params(
    session_id: Any = ANY, *, runtime_arn: str = RUNTIME_ARN, qualifier: str = "DEFAULT"
) -> dict[str, Any]:
    """The parameters a ``StopRuntimeSession`` stub expects, ``clientToken`` left out.

    ``Stubber`` checks parameters before botocore fills in that idempotency token, and the caller
    never passes it. ``ANY`` matches any session ID.
    """
    return {"agentRuntimeArn": runtime_arn, "qualifier": qualifier, "runtimeSessionId": session_id}


def streaming_body(data: bytes) -> StreamingBody:
    """A real ``StreamingBody`` over ``data``."""
    return StreamingBody(io.BytesIO(data), len(data))


def sse_body(*values: object) -> StreamingBody:
    """An event stream with one event per value: a string as its text, anything else as JSON.

    A string with newlines becomes one ``data:`` line per line, as an SSE server sends it.
    """
    events = []
    for value in values:
        text = value if isinstance(value, str) else json.dumps(value)
        events.append("".join(f"data: {line}\n" for line in text.split("\n")) + "\n")
    return streaming_body("".join(events).encode("utf-8"))


def json_body(value: object) -> StreamingBody:
    """A JSON document as a ``StreamingBody``."""
    return streaming_body(json.dumps(value).encode("utf-8"))


class SendCounter:
    """A ``before-send`` handler that answers every send with one response and counts the sends.

    It runs once per attempt that would go on the wire, after botocore's retry handler decided to
    send it, so the count is the number of HTTP attempts.
    """

    def __init__(
        self, status: int, *, body: bytes = b"", headers: Mapping[str, str] | None = None
    ) -> None:
        self.status = status
        self.body = body
        self.headers = dict(headers or {})
        self.sends: list[str] = []

    @classmethod
    def error(cls, code: str, status: int) -> SendCounter:
        """A counter whose every answer is the AWS error ``code`` at HTTP ``status``."""
        return cls(
            status,
            body=json.dumps({"message": "stubbed error"}).encode("utf-8"),
            headers={"x-amzn-errortype": code, "content-type": "application/json"},
        )

    def count(self, operation: str) -> int:
        """How many attempts of ``operation`` (e.g. ``"StopRuntimeSession"``) were sent."""
        return self.sends.count(operation)

    def __call__(self, request: Any, event_name: str = "", **_kwargs: Any) -> AWSResponse:
        self.sends.append(event_name.rpartition(".")[2])
        raw = urllib3.HTTPResponse(
            body=io.BytesIO(self.body),
            headers=self.headers,
            status=self.status,
            preload_content=False,
        )
        return AWSResponse(request.url, self.status, self.headers, raw)


class CountingAwsClients(_RecordingAwsClients):
    """Unstubbed clients over fake credentials whose every HTTP send ``counter`` answers."""

    def __init__(
        self,
        counter: SendCounter,
        *,
        region: str = REGION,
        handlers: Sequence[BeforeCallHandler] = (),
        config: Config = DEFAULT_CLIENT_CONFIG,
    ) -> None:
        super().__init__(region=region, handlers=handlers, config=config)
        self.counter = counter

    def _on_build(self, service_name: str, client: BaseClient) -> None:
        client.meta.events.register("before-send.*.*", self.counter)


Scripted: TypeAlias = tuple[int, dict[str, str], bytes] | Exception
"""One scripted answer to a send: ``(status, headers, body)``, or the exception the send raises."""


class ScriptedSends(SendCounter):
    """Answers each HTTP send of an operation with that operation's next scripted answer.

    An exception is raised as the HTTP layer would raise it, before or while sending. Each send's
    runtime session header is kept, so a test can see which session a call was for.
    """

    def __init__(self, script: Mapping[str, Sequence[Scripted]]) -> None:
        super().__init__(200)
        self.script = {operation: list(answers) for operation, answers in script.items()}
        self.sent_sessions: list[tuple[str, str | None]] = []

    def sessions(self, operation: str) -> list[str | None]:
        """The runtime session each send of ``operation`` carried, in order."""
        return [session for name, session in self.sent_sessions if name == operation]

    def __call__(self, request: Any, event_name: str = "", **_kwargs: Any) -> AWSResponse:
        operation = event_name.rpartition(".")[2]
        self.sends.append(operation)
        session = request.headers.get(_SESSION_HEADER)
        self.sent_sessions.append(
            (operation, session.decode() if isinstance(session, bytes) else session)
        )
        answer = self.script[operation].pop(0)
        if isinstance(answer, Exception):
            raise answer
        status, headers, body = answer
        raw = urllib3.HTTPResponse(
            body=io.BytesIO(body), headers=headers, status=status, preload_content=False
        )
        return AWSResponse(request.url, status, headers, raw)


def json_answer(document: Mapping[str, Any]) -> Scripted:
    """A 200 answer carrying ``document`` as JSON."""
    return 200, {"content-type": "application/json"}, json.dumps(document).encode("utf-8")


# When the fixture runtime and its endpoint were created, as the preflight answers say.
_CREATED = datetime(2026, 10, 26, 12, 0, tzinfo=UTC).timestamp()


def preflight_sends(version: str = "24") -> dict[str, list[Scripted]]:
    """The two preflight answers on the wire: endpoint DEFAULT serves ``version`` over HTTP."""
    return {
        "GetAgentRuntimeEndpoint": [
            json_answer(
                {
                    "agentRuntimeEndpointArn": f"{RUNTIME_ARN}/runtime-endpoint/DEFAULT",
                    "agentRuntimeArn": RUNTIME_ARN,
                    "status": "READY",
                    "createdAt": _CREATED,
                    "lastUpdatedAt": _CREATED,
                    "name": "DEFAULT",
                    "id": "DEFAULT",
                    "liveVersion": version,
                }
            )
        ],
        "GetAgentRuntime": [
            json_answer(
                {
                    "agentRuntimeArn": RUNTIME_ARN,
                    "agentRuntimeName": "calibrated_agent_evaluation_fixture",
                    "agentRuntimeId": RUNTIME_ARN.rpartition("/")[2],
                    "agentRuntimeVersion": version,
                    "createdAt": _CREATED,
                    "lastUpdatedAt": _CREATED,
                    "roleArn": (
                        f"arn:aws:iam::{_ACCOUNT}:role/calibrated-agent-evaluation-fixture-runtime"
                    ),
                    "status": "READY",
                    "lifecycleConfiguration": {},
                    "protocolConfiguration": {"serverProtocol": "HTTP"},
                }
            )
        ],
    }


def add_preflight_responses(
    clients: StubbedAwsClients,
    *,
    version: str = "24",
    from_endpoint: bool = True,
    oauth_only: bool = False,
    protocol: str | None = "HTTP",
) -> None:
    """Queue the two read-only preflight answers for endpoint DEFAULT of the fixture runtime.

    Args:
        clients: The stubbed clients the preflight reads through.
        version: The runtime version that serves the endpoint.
        from_endpoint: Whether the endpoint names ``version`` as its live version. When it names
            none, the runtime is read without a version and answers with ``version`` as its
            latest.
        oauth_only: Give the runtime a custom JWT authorizer, so it only accepts OAuth calls.
        protocol: The runtime's server protocol, or ``None`` for no protocol configuration.
    """
    control = clients.stub("bedrock-agentcore-control")
    endpoint: dict[str, Any] = {
        "agentRuntimeEndpointArn": f"{RUNTIME_ARN}/runtime-endpoint/DEFAULT",
        "agentRuntimeArn": RUNTIME_ARN,
        "status": "READY",
        "createdAt": _CREATED,
        "lastUpdatedAt": _CREATED,
        "name": "DEFAULT",
        "id": "DEFAULT",
    }
    runtime_params: dict[str, Any] = {"agentRuntimeId": RUNTIME_ID}
    if from_endpoint:
        endpoint["liveVersion"] = version
        runtime_params["agentRuntimeVersion"] = version
    control.add_response(
        "get_agent_runtime_endpoint",
        endpoint,
        {"agentRuntimeId": RUNTIME_ID, "endpointName": "DEFAULT"},
    )
    runtime: dict[str, Any] = {
        "agentRuntimeArn": RUNTIME_ARN,
        "agentRuntimeName": "calibrated_agent_evaluation_fixture",
        "agentRuntimeId": RUNTIME_ID,
        "agentRuntimeVersion": version,
        "createdAt": _CREATED,
        "lastUpdatedAt": _CREATED,
        "roleArn": f"arn:aws:iam::{_ACCOUNT}:role/calibrated-agent-evaluation-fixture-runtime",
        "status": "READY",
        "lifecycleConfiguration": {},
    }
    if protocol is not None:
        runtime["protocolConfiguration"] = {"serverProtocol": protocol}
    if oauth_only:
        runtime["authorizerConfiguration"] = {
            "customJWTAuthorizer": {
                "discoveryUrl": "https://idp.example.com/.well-known/openid-configuration"
            }
        }
    control.add_response("get_agent_runtime", runtime, runtime_params)


def add_invocation_response(
    clients: StubbedAwsClients,
    body: StreamingBody | None = None,
    *,
    content_type: str = "text/event-stream",
    status: int = 200,
    traceparent: str | None = RETURNED_TRACEPARENT,
) -> None:
    """Queue one ``InvokeAgentRuntime`` answer on the invocation view.

    Args:
        clients: The stubbed clients whose invocation view is asked.
        body: The response body; by default an event stream whose two events make :data:`ANSWER`.
        content_type: The response's content type.
        status: The response's status code.
        traceparent: The ``traceparent`` the answer carries back, or ``None`` for none.
    """
    head, _, tail = ANSWER.partition(": ")
    answer: dict[str, Any] = {
        "response": body
        if body is not None
        else sse_body({"response": f"{head}: "}, {"response": tail}),
        "contentType": content_type,
        "statusCode": status,
    }
    if traceparent is not None:
        answer["traceParent"] = traceparent
    clients.stub("bedrock-agentcore", view=True).add_response("invoke_agent_runtime", answer)


def add_stop_response(clients: StubbedAwsClients, session_id: Any = ANY) -> None:
    """Queue one successful ``StopRuntimeSession`` answer, for ``session_id`` or any session."""
    clients.stub("bedrock-agentcore").add_response(
        "stop_runtime_session", {"statusCode": 200}, stop_params(session_id)
    )


def zero_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make botocore's standard-mode retries wait no time between attempts."""

    def no_delay(_backoff: Any, _context: Any) -> float:
        return 0.0

    monkeypatch.setattr(standard.ExponentialBackoff, "delay_amount", no_delay)


def clm_4471_spans(trace_id: str) -> list[Span]:
    """The committed CLM-4471 span tree (five agents), re-keyed to ``trace_id``.

    Each span's ``trace_id``, and the trace-ID prefix of its ``span_id`` and ``parent``, become
    ``trace_id``, so two runs' spans never share an ID in a store.
    """
    source = load_corpus_trace(_CLM_4471)

    def rekey(span_id: str) -> str:
        head, sep, tail = span_id.partition("/")
        return f"{trace_id}/{tail}" if sep and head == source.trace_id else span_id

    return [
        replace(
            span,
            trace_id=trace_id,
            span_id=rekey(span.span_id),
            parent=None if span.parent is None else rekey(span.parent),
        )
        for span in source.spans
    ]


class FakeRunSpanReader:
    """A span reader that answers every query with the CLM-4471 tree re-keyed to its trace ID.

    It records each query, and whether ``clients`` had already been asked to stop the query's
    session when the read happened.
    """

    def __init__(self, clients: StubbedAwsClients) -> None:
        self.clients = clients
        self.queries: list[RunSpanQuery] = []
        self.stopped_when_read: list[bool] = []

    def read_spans(self, query: RunSpanQuery) -> list[Span]:
        """The re-keyed CLM-4471 spans."""
        self.queries.append(query)
        stops = self.clients.calls_to("StopRuntimeSession")
        self.stopped_when_read.append(
            any(call.params.get("runtimeSessionId") == query.session_id for call in stops)
        )
        return clm_4471_spans(query.trace_id)


class FakeAgentCoreSpanReader:
    """Stands in for the AgentCore span reader in CLI tests.

    Built like the real one, from the clients the CLI builds. Each read makes one billed call
    (CloudWatch's ``StartTelemetryQuery``) through those clients, so it counts against their
    billed-call budget, then returns the re-keyed CLM-4471 tree. Queue that call's answer with
    :func:`add_reader_response`.
    """

    def __init__(self, clients: AwsClients) -> None:
        self.clients = clients
        self.queries: list[RunSpanQuery] = []

    def read_spans(self, query: RunSpanQuery) -> list[Span]:
        """The re-keyed CLM-4471 spans, after one billed call."""
        self.queries.append(query)
        self.clients.client("cloudwatchomni").start_telemetry_query(  # pyright: ignore[reportAttributeAccessIssue]
            queryString=f"filter attributes.session.id = '{query.session_id}'",
            sessionId=_READER_QUERY_SESSION,
        )
        return clm_4471_spans(query.trace_id)


def add_reader_response(clients: StubbedAwsClients) -> None:
    """Queue the answer to one :class:`FakeAgentCoreSpanReader` read on ``clients``."""
    clients.stub("cloudwatchomni").add_response(
        "start_telemetry_query",
        {"queryId": "calibrated-agent-evaluation-fake-query", "sessionId": _READER_QUERY_SESSION},
        {"queryString": ANY, "sessionId": _READER_QUERY_SESSION},
    )


def register_fake_agentcore_span_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``calibrated_agent_evaluation.agentcore.AgentCoreSpanReader`` the fake for the rest of
    the test."""
    # Not raising: until the real reader is registered there is no attribute to replace.
    monkeypatch.setattr(
        calibrated_agent_evaluation.agentcore,
        "AgentCoreSpanReader",
        FakeAgentCoreSpanReader,
        raising=False,
    )
