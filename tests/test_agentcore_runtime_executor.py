# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AgentCore Runtime executor turns one task into one owned trace, refuses what it can't run,
stops every session it opens, and leaves one run record per run in the run manifest."""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote

import pytest

pytest.importorskip("boto3")

import urllib3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import CredentialRetrievalError, EndpointConnectionError
from botocore.response import StreamingBody

from _agentcore_runtime_fakes import (
    BUNDLE_ARN,
    BUNDLE_VERSION,
    REGION,
    RUNTIME_ARN,
    CountingAwsClients,
    FakeAgentCoreSpanReader,
    FakeRunSpanReader,
    ScriptedSends,
    StubbedAwsClients,
    add_reader_response,
    clm_4471_spans,
    json_answer,
    preflight_sends,
    sse_body,
    stop_params,
    streaming_body,
    zero_retry_backoff,
)
from _helpers import load_corpus_trace
from calibrated_agent_evaluation._finite import NonFiniteNumberError
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.agentcore import (
    AgentCoreError,
    AgentCoreRuntimeExecutor,
    AgentCoreRuntimeTask,
    AwsClients,
    BilledCallBudget,
    ReadOnlyGuard,
    RunRecord,
    RuntimeInvocationError,
    RuntimeTarget,
)
from calibrated_agent_evaluation.agentcore.run_manifest import (
    RUN_RECORD_STEM,
    RUN_RECORD_STORE,
    manifest_path,
)
from calibrated_agent_evaluation.agentcore.run_spans import RunSpanQuery
from calibrated_agent_evaluation.agentcore.runtime import STOP_TIMEOUT_SECONDS
from calibrated_agent_evaluation.agentcore.session import DEFAULT_CLIENT_CONFIG, BeforeCallHandler
from calibrated_agent_evaluation.execute import ExecutorBackend, Mode
from calibrated_agent_evaluation.execute.strands_executor import StrandsTask
from calibrated_agent_evaluation.schema.errors import SchemaError
from calibrated_agent_evaluation.schema.types import Span, Trace
from calibrated_agent_evaluation.schema.validate import validate

_RUNTIME_ID = "calibrated_agent_evaluation_fixture-AbCdEf1234"
_WHEN = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
_CLM_4471 = load_corpus_trace("CLM-4471")
_NARRATIVE = str(_CLM_4471.input["narrative"])
_CELL = dict(_CLM_4471.taxonomy_cell)
_FIVE_AGENTS = {"supervisor", "intake", "policy-checker", "fraud-scorer", "payout"}
_ANSWER_PARTS = ("Denied: ", "the vacancy exclusion applies.")
_ANSWER = "".join(_ANSWER_PARTS)
_RETURNED_TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
_BUDGET_VARIABLE = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS"
_MODE_VARIABLE = "CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE"
_ENDPOINT = "https://bedrock-agentcore.us-west-2.amazonaws.com"
_SENTINEL = "SENTINEL-7f3a"


@pytest.fixture
def clients() -> Iterator[StubbedAwsClients]:
    """Stubbed clients for one test, closed when the test ends."""
    with StubbedAwsClients() as stubbed:
        yield stubbed


@pytest.fixture(autouse=True)
def _no_ambient_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hide the machine's AWS setup (profile, credentials, config files) and cassette mode."""
    for name in list(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv(_BUDGET_VARIABLE, raising=False)
    monkeypatch.delenv(_MODE_VARIABLE, raising=False)
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


def _task(**overrides: Any) -> AgentCoreRuntimeTask:
    fields: dict[str, Any] = {"prompt": _NARRATIVE, "taxonomy_cell": dict(_CELL)}
    fields.update(overrides)
    return AgentCoreRuntimeTask(**fields)


def _executor(
    clients: StubbedAwsClients, log_dir: Path | None, **overrides: Any
) -> AgentCoreRuntimeExecutor:
    target = overrides.pop("target", None) or _target()
    kwargs: dict[str, Any] = {
        "span_reader": FakeRunSpanReader(clients),
        "clients": clients,
        "log_dir": log_dir,
        "sleep": _no_sleep,
        "clock": _Clock(),
    }
    kwargs.update(overrides)
    return AgentCoreRuntimeExecutor(target, **kwargs)


def _preflight(
    clients: StubbedAwsClients,
    *,
    protocol: str | None = "HTTP",
    authorizer: dict[str, Any] | None = None,
) -> None:
    """Queue the two read-only preflight answers: endpoint DEFAULT serves version 24."""
    control = clients.stub("bedrock-agentcore-control")
    control.add_response(
        "get_agent_runtime_endpoint",
        {
            "agentRuntimeEndpointArn": f"{RUNTIME_ARN}/runtime-endpoint/DEFAULT",
            "agentRuntimeArn": RUNTIME_ARN,
            "status": "READY",
            "createdAt": _WHEN,
            "lastUpdatedAt": _WHEN,
            "name": "DEFAULT",
            "id": "DEFAULT",
            "liveVersion": "24",
        },
        {"agentRuntimeId": _RUNTIME_ID, "endpointName": "DEFAULT"},
    )
    runtime: dict[str, Any] = {
        "agentRuntimeArn": RUNTIME_ARN,
        "agentRuntimeName": "calibrated_agent_evaluation_fixture",
        "agentRuntimeId": _RUNTIME_ID,
        "agentRuntimeVersion": "24",
        "createdAt": _WHEN,
        "lastUpdatedAt": _WHEN,
        "roleArn": "arn:aws:iam::111122223333:role/calibrated-agent-evaluation-fixture-runtime",
        "status": "READY",
        "lifecycleConfiguration": {},
    }

    if protocol is not None:
        runtime["protocolConfiguration"] = {"serverProtocol": protocol}
    if authorizer is not None:
        runtime["authorizerConfiguration"] = authorizer
    control.add_response(
        "get_agent_runtime",
        runtime,
        {"agentRuntimeId": _RUNTIME_ID, "agentRuntimeVersion": "24"},
    )


def _oauth_only(clients: StubbedAwsClients) -> None:
    _preflight(
        clients,
        authorizer={
            "customJWTAuthorizer": {
                "discoveryUrl": "https://idp.example.com/.well-known/openid-configuration"
            }
        },
    )


def _answer(
    clients: StubbedAwsClients,
    body: StreamingBody | None = None,
    *,
    content_type: str = "text/event-stream",
    status: int = 200,
) -> None:
    """Queue one invocation answer; by default an event stream with the answer in two events."""
    clients.stub("bedrock-agentcore", view=True).add_response(
        "invoke_agent_runtime",
        {
            "response": body
            if body is not None
            else sse_body(*({"response": part} for part in _ANSWER_PARTS)),
            "contentType": content_type,
            "statusCode": status,
            "traceParent": _RETURNED_TRACEPARENT,
        },
    )


def _fail(
    clients: StubbedAwsClients,
    code: str,
    status: int,
    message: str = "agent text that must stay out of messages",
) -> None:
    clients.stub("bedrock-agentcore", view=True).add_client_error(
        "invoke_agent_runtime",
        service_error_code=code,
        service_message=message,
        http_status_code=status,
    )


def _stopped(clients: StubbedAwsClients) -> None:
    clients.stub("bedrock-agentcore").add_response(
        "stop_runtime_session", {"statusCode": 200}, stop_params()
    )


def _stop_fails(clients: StubbedAwsClients, code: str = "ThrottlingException") -> None:
    status = {"ThrottlingException": 429, "ResourceNotFoundException": 404}.get(code, 500)
    clients.stub("bedrock-agentcore").add_client_error(
        "stop_runtime_session",
        service_error_code=code,
        service_message="stop refused",
        http_status_code=status,
        expected_params=stop_params(),
    )


def _one_run(clients: StubbedAwsClients) -> None:
    """Queue a whole successful first run: preflight, one answer and one stop."""
    _preflight(clients)
    _answer(clients)
    _stopped(clients)


def _refused(executor: AgentCoreRuntimeExecutor, task: object = None) -> RuntimeInvocationError:
    """What ``executor`` refuses running ``task`` (the CLM-4471 one by default) in its context."""
    with executor, pytest.raises(RuntimeInvocationError) as refused:
        executor.run(_task() if task is None else task)
    return refused.value


class _Raises:
    """A call handler that raises ``error`` as each ``operation`` call is sent (only the first
    with ``once``), and counts the calls it raised on."""

    def __init__(self, operation: str, error: BaseException, *, once: bool = False) -> None:
        self.operation, self.error, self.once = operation, error, once
        self.raised = 0

    def __call__(self, **kwargs: Any) -> None:
        if kwargs["model"].name == self.operation and not (self.once and self.raised):
            self.raised += 1
            raise self.error


def _manifest(log_dir: Path) -> list[dict[str, Any]]:
    """The run records in ``log_dir``'s run manifest, as the JSON the manifest stores."""
    return [record.model_dump(mode="json") for record in RUN_RECORD_STORE.load(log_dir, fmt="json")]


def _no_manifest_in(log_dir: Path) -> bool:
    """Whether neither file of the run manifest pair exists in ``log_dir``."""
    return not any((log_dir / f"{RUN_RECORD_STEM}.{suffix}").exists() for suffix in ("json", "csv"))


def _baggage(header: str) -> dict[str, str]:
    """The W3C baggage members of ``header``, values percent-decoded."""
    entries: dict[str, str] = {}
    for member in header.split(","):
        key, _, value = member.strip().partition("=")
        entries[key.strip()] = unquote(value.split(";")[0].strip())
    return entries


def _w3c_trace_id(traceparent: str) -> str:
    return traceparent.split("-")[1]


def _when(text: str) -> datetime:
    return datetime.fromisoformat(text)


class _OwnClients:
    """Stands in for ``AwsClients.from_profile``: like it, it takes the environment's region when
    given none and runs its handlers before every call. ``prepare`` queues each set's answers."""

    def __init__(self, prepare: Callable[[StubbedAwsClients], None]) -> None:
        self.prepare = prepare
        self.built: list[StubbedAwsClients] = []

    def __call__(
        self,
        *,
        region: str | None = None,
        profile: str | None = None,
        handlers: tuple[BeforeCallHandler, ...] | list[BeforeCallHandler] = (),
        config: Config = DEFAULT_CLIENT_CONFIG,
    ) -> StubbedAwsClients:
        clients = StubbedAwsClients(
            region=region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            handlers=handlers,
            config=config,
        )
        self.prepare(clients)
        self.built.append(clients)
        return clients


def _own_clients(
    monkeypatch: pytest.MonkeyPatch, prepare: Callable[[StubbedAwsClients], None]
) -> _OwnClients:
    factory = _OwnClients(prepare)
    monkeypatch.setattr(AwsClients, "from_profile", factory)
    return factory


def _own_executor(log_dir: Path) -> AgentCoreRuntimeExecutor:
    """An executor that builds its own clients."""
    return AgentCoreRuntimeExecutor(
        _target(), span_reader=_Reader(), log_dir=log_dir, sleep=_no_sleep, clock=_Clock()
    )


def _tree(query: RunSpanQuery) -> list[Span]:
    return clm_4471_spans(query.trace_id)


class _Reader:
    """A span reader that raises ``answer`` if it is an error, else returns ``answer(query)``."""

    def __init__(self, answer: Exception | Callable[[RunSpanQuery], list[Span]] = _tree) -> None:
        self.answer = answer

    def read_spans(self, query: RunSpanQuery) -> list[Span]:
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer(query)


class _SpanReadFailed(AgentCoreError):
    """The reader's own typed refusal, as a real reader raises when its wait runs out."""


def _with_one_span_changed(change: Callable[[Span], Span]) -> Callable[[RunSpanQuery], list[Span]]:
    """The CLM-4471 tree with ``change`` applied to its first span that made a tool call."""

    def spans(query: RunSpanQuery) -> list[Span]:
        tree = clm_4471_spans(query.trace_id)
        index = next(i for i, span in enumerate(tree) if span.tool_calls)
        return [*tree[:index], change(tree[index]), *tree[index + 1 :]]

    return spans


# --- building the executor ------------------------------------------------------------------


def test_building_the_executor_touches_no_profile_or_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # boto3 refuses a named profile that doesn't exist as soon as a session is opened.
    monkeypatch.setenv("AWS_PROFILE", "calibrated-agent-evaluation-no-such-profile")

    with AgentCoreRuntimeExecutor(_target(), span_reader=_Reader(), log_dir=tmp_path) as executor:
        assert executor.run_records == ()

    assert _no_manifest_in(tmp_path)


def test_live_clients_use_the_arns_region_not_the_environments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    factory = _own_clients(monkeypatch, _one_run)

    with _own_executor(tmp_path) as executor:
        [trace] = executor.run(_task())

    [clients] = factory.built
    assert trace.output == {"text": _ANSWER}
    assert {entry.client.meta.region_name for entry in clients.built} == {REGION}


class _NotesClientsAtTheFirstPreflight:
    """A call handler that notes which clients exist when the first preflight call is made."""

    def __init__(self) -> None:
        self.clients: StubbedAwsClients | None = None
        self.seen: set[tuple[str, bool]] | None = None

    def __call__(self, **kwargs: Any) -> None:
        if kwargs["model"].name == "GetAgentRuntimeEndpoint" and self.seen is None:
            assert self.clients is not None
            self.seen = {(entry.service, entry.view) for entry in self.clients.built}


def test_the_invocation_view_is_built_before_the_preflight_and_closed_on_exit(
    tmp_path: Path,
) -> None:
    notes = _NotesClientsAtTheFirstPreflight()
    with StubbedAwsClients(handlers=[notes]) as clients:
        notes.clients = clients
        _one_run(clients)

        with _executor(clients, tmp_path) as executor:
            executor.run(_task())
            assert not any(entry.closed for entry in clients.built)
        views = [entry for entry in clients.built if entry.view]
        assert views
        assert all(entry.closed for entry in views)
        # The caller's own clients are the caller's to close.
        assert not any(entry.closed for entry in clients.built if not entry.view)

    # The control plane, the data plane and the single-attempt invocation view, all three.
    assert notes.seen == {
        ("bedrock-agentcore-control", False),
        ("bedrock-agentcore", False),
        ("bedrock-agentcore", True),
    }


class _InvocationViewCannotBeBuilt(StubbedAwsClients):
    """Clients whose invocation view can't build its client, as a bad client setting makes it."""

    def client(self, service_name: str, *, region: str | None = None) -> BaseClient:
        if self.is_invocation_view:
            raise AgentCoreError(f"could not build an AWS client for {service_name!r}")
        return super().client(service_name, region=region)


class _ConfigCannotBeApplied(StubbedAwsClients):
    """Clients whose ``with_config`` fails with an error that is not an AgentCoreError."""

    def with_config(self, config: Config) -> StubbedAwsClients:
        raise RuntimeError(f"bad config {_SENTINEL}")


@pytest.mark.parametrize(
    ("broken", "error"),
    [(_InvocationViewCannotBeBuilt, "AgentCoreError"), (_ConfigCannotBeApplied, "RuntimeError")],
    ids=["view-client-cannot-be-built", "config-cannot-be-applied"],
)
def test_invocation_clients_that_cannot_be_built_are_refused_before_the_preflight(
    tmp_path: Path, broken: type[StubbedAwsClients], error: str
) -> None:
    with broken() as clients:
        # Queued, so a preflight that went ahead would be answered and show up in the calls.
        _preflight(clients)
        executor = _executor(clients, tmp_path)
        message = str(_refused(executor))

    assert "could not build the AWS clients" in message
    assert f"({error}), so nothing was invoked" in message
    assert _SENTINEL not in message
    assert clients.calls == []
    [record] = executor.run_records
    assert record.session_stop == "not_started"


def _builder(prompt: str, session_id: str) -> Mapping[str, object]:
    return {"question": prompt, "conversation": session_id}


_CASSETTE = Path("runtime.cassette.json")  # never opened: each executor below is refused first
_RECORD = {"cassette_path": _CASSETTE, "cassette_mode": Mode.RECORD}
_AUTO = {"cassette_path": _CASSETTE, "cassette_mode": Mode.AUTO}
_CUSTOM = _target(payload_format="custom")
_GENERIC = _target(payload_format="generic_message")
_NO_READER = (
    "live runs need the AgentCore span reader",
    "offline runs need a cassette in replay mode",
)
_NOT_A_READER = ("span_reader= is a function", "read_spans", _NO_READER[0])
_REPLAY = "replay needs a cassette file"
_MODES = ("record", "replay", "auto")
_NEEDS_BOTH = ("custom", "payload_builder", "builder_name")
_BOTH = {"payload_builder": _builder, "builder_name": "claims-body"}


def _case(name: str, *named: str, variable: str | None = None, **settings: Any) -> Any:
    """An executor setup to refuse: its ``settings``, the mode variable, and what to name."""
    return pytest.param(settings, variable, named, id=name)


@pytest.mark.parametrize(
    ("settings", "variable", "named"),
    [
        _case("no-reader-no-cassette", *_NO_READER, span_reader=None),
        _case("no-reader-record", *_NO_READER, span_reader=None, **_RECORD),
        _case("no-reader-auto", *_NO_READER, span_reader=None, **_AUTO),
        _case("no-reader-auto-by-default", *_NO_READER, span_reader=None, cassette_path=_CASSETTE),
        _case("reader-is-a-function", *_NOT_A_READER, span_reader=clm_4471_spans),
        _case("replay-without-a-cassette", _REPLAY, cassette_mode=Mode.REPLAY),
        _case("replay-as-text-without-a-cassette", _REPLAY, cassette_mode="replay"),
        _case("replay-from-the-variable-without-a-cassette", _REPLAY, variable="replay"),
        _case(
            "record-without-a-cassette",
            *("cassette_mode='record'", "cassette_path"),
            cassette_mode=Mode.RECORD,
        ),
        _case(
            "auto-without-a-cassette",
            *("cassette_mode='auto'", "cassette_path"),
            cassette_mode=Mode.AUTO,
        ),
        _case("not-a-mode", "'sometimes'", *_MODES, cassette_mode="sometimes"),
        _case("variable-not-a-mode", _MODE_VARIABLE, *_MODES, variable="sometimes"),
        _case("custom-without-a-builder", *_NEEDS_BOTH, target=_CUSTOM, builder_name="claims-body"),
        _case("custom-without-a-name", *_NEEDS_BOTH, target=_CUSTOM, payload_builder=_builder),
        _case("custom-without-either", *_NEEDS_BOTH, target=_CUSTOM),
        _case(
            "custom-with-an-empty-name",
            "builder_name is missing",
            target=_CUSTOM,
            payload_builder=_builder,
            builder_name="",
        ),
        _case(
            "builder-with-generic-message", "generic_message", "custom", target=_GENERIC, **_BOTH
        ),
        _case("builder-with-agentcore-sse", "agentcore_sse", "custom", **_BOTH),
        _case(
            "name-with-generic-message",
            *("builder_name", "generic_message", "custom"),
            target=_GENERIC,
            builder_name="claims-body",
        ),
        _case(
            "name-with-agentcore-sse",
            *("builder_name", "agentcore_sse", "custom"),
            builder_name="claims-body",
        ),
        _case("unknown-frame", "'discovry'", "production-representative", frame="discovry"),
        _case("attempt-limit-zero", "attempt_limit=0", attempt_limit=0),
        _case("attempt-limit-negative", "attempt_limit=-2", attempt_limit=-2),
        _case("clients-for-another-region", "eu-west-1", REGION, region="eu-west-1"),
        _case(
            "clients-and-a-profile",
            *("clients=", "profile=", "calibrated-agent-evaluation-claims"),
            profile="calibrated-agent-evaluation-claims",
        ),
    ],
)
def test_a_setting_the_executor_cannot_use_is_refused_naming_it_before_any_client_is_built(
    monkeypatch: pytest.MonkeyPatch,
    settings: dict[str, Any],
    variable: str | None,
    named: tuple[str, ...],
) -> None:
    if variable is not None:
        monkeypatch.setenv(_MODE_VARIABLE, variable)
    settings = dict(settings)
    target = settings.pop("target", _target())
    factory = _own_clients(monkeypatch, _one_run)

    with StubbedAwsClients(region=settings.pop("region", REGION)) as clients:
        with pytest.raises(RuntimeInvocationError) as refused:
            AgentCoreRuntimeExecutor(
                target, **{"span_reader": _Reader(), "clients": clients, **settings}
            )

    for fragment in named:
        assert fragment in str(refused.value)
    assert factory.built == []
    assert clients.built == []
    assert clients.calls == []


@pytest.mark.parametrize("from_variable", [False, True], ids=["passed", "from-variable"])
def test_replay_from_a_cassette_needs_no_span_reader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, from_variable: bool
) -> None:
    cassette = tmp_path / "runtime.cassette.json"
    cassette.write_text(json.dumps({"version": 1, "entries": []}))
    mode: Mode | None = Mode.REPLAY
    if from_variable:
        # Case and surrounding spaces don't change the mode the variable names.
        monkeypatch.setenv(_MODE_VARIABLE, " Replay ")
        mode = None

    with AgentCoreRuntimeExecutor(
        _target(), cassette_path=cassette, cassette_mode=mode, log_dir=tmp_path
    ) as executor:
        assert executor.run_records == ()


def test_custom_with_a_builder_and_name_is_accepted(tmp_path: Path) -> None:
    with AgentCoreRuntimeExecutor(
        _CUSTOM,
        span_reader=_Reader(),
        payload_builder=_builder,
        builder_name="claims-body",
        log_dir=tmp_path,
    ) as executor:
        assert executor.run_records == ()


@pytest.mark.parametrize(
    ("variable", "value", "mode"),
    [
        (_MODE_VARIABLE, "record", None),
        (_MODE_VARIABLE, "sometimes", Mode.AUTO),
        (_MODE_VARIABLE, "replay", Mode.AUTO),
        (_BUDGET_VARIABLE, "0", None),
    ],
    ids=["record-without-a-cassette", "auto-over-an-invalid-mode", "auto-over-replay", "budget"],
)
def test_a_mode_or_budget_variable_that_does_not_apply_leaves_the_run_live(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clients: StubbedAwsClients,
    variable: str,
    value: str,
    mode: Mode | None,
) -> None:
    # Only a mode the caller passes is refused without a cassette; a mode passed means the mode
    # variable is neither read nor obeyed (auto over a cassette with no entry runs live); and the
    # budget variable caps only clients the executor builds, not the caller's.
    monkeypatch.setenv(variable, value)
    _one_run(clients)
    cassette = None if mode is None else tmp_path / "runtime.cassette.json"

    with _executor(
        clients, tmp_path / "logs", cassette_path=cassette, cassette_mode=mode
    ) as executor:
        [trace] = executor.run(_task())

    assert trace.output == {"text": _ANSWER}
    assert len(clients.calls_to("InvokeAgentRuntime")) == 1


@pytest.mark.parametrize(
    "seconds",
    [
        *(0.0, -0.0, 0, -1.5, -math.inf, math.nan),
        *(True, "300", None),
        *(math.inf, math.nextafter(threading.TIMEOUT_MAX, math.inf), 1e10),
    ],
    ids=[
        *("zero", "negative-zero", "integer-zero", "negative", "-infinity", "nan"),
        *("bool", "string", "none"),
        *("infinity", "just-over-the-longest", "longer-than-a-socket-can-wait"),
    ],
)
def test_a_read_timeout_that_is_not_seconds_a_socket_can_wait_for_is_refused_naming_it(
    seconds: object, clients: StubbedAwsClients
) -> None:
    # A bool passes a numeric comparison as 1 second, and a string fails it with a bare TypeError.
    # A socket refuses an infinite or overlong timeout only once the billed request was sent.
    with pytest.raises(RuntimeInvocationError) as refused:
        AgentCoreRuntimeExecutor(
            _target(),
            span_reader=_Reader(),
            clients=clients,
            read_timeout_seconds=seconds,  # pyright: ignore[reportArgumentType]
        )

    message = str(refused.value)
    assert f"read_timeout_seconds={seconds!r}" in message
    assert "finite number of seconds" in message
    assert f"{threading.TIMEOUT_MAX:.0f}" in message
    assert clients.built == []


@pytest.mark.parametrize(
    "seconds",
    [5e-324, 1, threading.TIMEOUT_MAX],
    ids=["the-smallest-float", "a-whole-number", "the-longest-a-socket-can-wait-for"],
)
def test_a_read_timeout_a_socket_can_wait_for_is_accepted(
    tmp_path: Path, clients: StubbedAwsClients, seconds: float
) -> None:
    _one_run(clients)

    with _executor(clients, tmp_path, read_timeout_seconds=seconds) as executor:
        [trace] = executor.run(_task())

    assert trace.output == {"text": _ANSWER}
    [invoker] = [entry.client for entry in clients.built if entry.view]
    assert invoker.meta.config.read_timeout == seconds


def test_an_attempt_limit_of_one_is_accepted(tmp_path: Path, clients: StubbedAwsClients) -> None:
    _preflight(clients)
    _fail(clients, "RetryableConflictException", 409)
    _answer(clients)
    _stopped(clients)
    executor = _executor(clients, tmp_path, attempt_limit=1)

    with executor, pytest.raises(RuntimeInvocationError, match="RetryableConflictException"):
        executor.run(_task())

    assert len(clients.calls_to("InvokeAgentRuntime")) == 1
    assert executor.run_records[0].attempts == 1


_PROMPT = "the task's prompt is {}, and a prompt must be a non-empty string"
_NOT_RUN = "nothing was run"


@pytest.mark.parametrize(
    ("task", "named"),
    [
        pytest.param(
            StrandsTask(build=lambda model: model, input=_NARRATIVE),
            ("StrandsTask",),
            id="another-type",
        ),
        pytest.param(_task(prompt=None), (_PROMPT.format("of type NoneType"),), id="prompt-none"),
        pytest.param(_task(prompt=123), (_PROMPT.format("of type int"),), id="prompt-int"),
        pytest.param(_task(prompt=""), (_PROMPT.format("an empty string"),), id="prompt-empty"),
        pytest.param(_task(prompt=b"bytes"), (_PROMPT.format("of type bytes"),), id="prompt-bytes"),
        pytest.param(_task(trace_id=123), ("trace_id", _NOT_RUN), id="trace-id-int"),
        pytest.param(_task(taxonomy_cell=None), ("taxonomy_cell", _NOT_RUN), id="cell-none"),
        pytest.param(
            _task(taxonomy_cell={"domain": 1}), ("taxonomy_cell", _NOT_RUN), id="cell-value-int"
        ),
    ],
)
def test_a_task_the_executor_cannot_run_is_refused_with_no_aws_call(
    tmp_path: Path, task: object, named: tuple[str, ...], clients: StubbedAwsClients
) -> None:
    executor = _executor(clients, tmp_path)

    refused = _refused(executor, task)

    for fragment in named:
        assert fragment in str(refused)
    assert clients.built == []
    assert clients.calls == []
    assert executor.run_records == ()
    assert _no_manifest_in(tmp_path)


def test_a_replayed_task_whose_prompt_is_empty_is_refused_before_the_cassette_is_read(
    tmp_path: Path,
) -> None:
    cassette = tmp_path / "missing.cassette.json"
    executor = AgentCoreRuntimeExecutor(
        _target(), cassette_path=cassette, cassette_mode=Mode.REPLAY, log_dir=tmp_path
    )

    refused = str(_refused(executor, _task(prompt="")))

    assert "prompt" in refused
    assert "cassette" not in refused
    assert executor.run_records == ()


def test_a_missing_profile_on_the_first_live_call_is_refused_naming_it(tmp_path: Path) -> None:
    executor = AgentCoreRuntimeExecutor(
        _target(),
        span_reader=_Reader(),
        profile="calibrated-agent-evaluation-no-such-profile",
        log_dir=tmp_path,
    )

    assert "calibrated-agent-evaluation-no-such-profile" in str(_refused(executor))
    [record] = executor.run_records
    assert record.session_stop == "not_started"
    assert record.attempts == 0
    assert record.error_code == "AwsCredentialsMissing"


@pytest.mark.parametrize("value", ["abc", "-1", "1.5"])
def test_an_invalid_budget_variable_is_refused_naming_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str
) -> None:
    monkeypatch.setenv(_BUDGET_VARIABLE, value)
    factory = _own_clients(monkeypatch, _one_run)
    executor = _own_executor(tmp_path)

    assert _BUDGET_VARIABLE in str(_refused(executor))
    assert all(clients.calls == [] for clients in factory.built)
    assert executor.run_records[0].session_stop == "not_started"


def _raising_builder(prompt: str, session_id: str) -> Mapping[str, object]:
    raise ValueError(f"cannot shape {prompt} ({_SENTINEL})")


def _builder_of(value: object) -> Callable[[str, str], Mapping[str, object]]:
    return lambda prompt, session_id: {"question": prompt, "detail": value}


@pytest.mark.parametrize(
    ("builder", "said", "cause"),
    [
        pytest.param(_raising_builder, "failed with ValueError", ValueError, id="raises"),
        pytest.param(_builder_of(_WHEN), "JSON", TypeError, id="datetime"),
        pytest.param(_builder_of(math.nan), "JSON", NonFiniteNumberError, id="nan"),
        pytest.param(_builder_of(math.inf), "JSON", NonFiniteNumberError, id="infinity"),
        pytest.param(_builder_of(-math.inf), "JSON", NonFiniteNumberError, id="-infinity"),
    ],
)
def test_a_custom_builder_that_raises_or_builds_a_body_that_is_not_json_is_refused_naming_it(
    tmp_path: Path,
    clients: StubbedAwsClients,
    builder: Callable[[str, str], Mapping[str, object]],
    said: str,
    cause: type[Exception],
) -> None:
    executor = _executor(
        clients, tmp_path, target=_CUSTOM, payload_builder=builder, builder_name="claims-body"
    )

    refused = _refused(executor)

    for fragment in ("claims-body", said, "nothing was invoked"):
        assert fragment in str(refused)
    assert _SENTINEL not in str(refused)
    assert isinstance(refused.__cause__, cause)

    assert clients.calls_to("InvokeAgentRuntime") == []
    assert executor.run_records[0].session_stop == "not_started"


# --- one run --------------------------------------------------------------------------------


def test_a_clm_4471_run_returns_one_owned_trace_with_prompt_text_cell_and_five_agents(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _one_run(clients)

    # Caller clients with no profile are accepted.
    with _executor(clients, tmp_path, profile=None) as executor:
        assert isinstance(executor, ExecutorBackend)
        traces = executor.run(_task())

    [trace] = traces
    assert type(trace) is Trace
    assert trace.ownership == "owned"
    assert trace.frame == "production-representative"
    assert trace.input == {"prompt": _NARRATIVE}
    assert trace.output == {"text": _ANSWER}
    assert trace.taxonomy_cell == _CELL
    assert {span.agent for span in trace.spans} == _FIVE_AGENTS
    validate(trace)
    [sent] = clients.calls_to("InvokeAgentRuntime")
    session = sent.params["runtimeSessionId"]
    assert len(session) == 60
    assert session.startswith("calibrated-agent-evaluation-")
    assert _baggage(sent.params["baggage"])["session.id"] == session
    assert sent.params["traceParent"].startswith("00-")


@pytest.mark.parametrize(
    "given", [None, "", "CLM-4471-agentcore"], ids=["no-trace-id", "empty-trace-id", "trace-id"]
)
def test_a_task_trace_id_is_kept_and_without_one_the_trace_is_named_after_its_w3c_trace_id(
    tmp_path: Path, clients: StubbedAwsClients, given: str | None
) -> None:
    _one_run(clients)

    with _executor(clients, tmp_path) as executor:
        [trace] = executor.run(_task(trace_id=given))

    [sent] = clients.calls_to("InvokeAgentRuntime")
    named = given or f"agentcore-{_w3c_trace_id(sent.params['traceParent'])}"
    assert trace.trace_id == named
    assert {span.trace_id for span in trace.spans} == {named}
    assert [line["trace_id"] for line in _manifest(tmp_path)] == [named]


def test_the_trace_input_is_the_prompt_not_the_formats_body(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _one_run(clients)

    with _executor(clients, tmp_path, target=_target(payload_format="openai")) as executor:
        [trace] = executor.run(_task())

    [sent] = clients.calls_to("InvokeAgentRuntime")
    assert json.loads(sent.params["payload"]) == {
        "messages": [{"role": "user", "content": _NARRATIVE}]
    }
    assert trace.input == {"prompt": _NARRATIVE}


# --- the span reader ------------------------------------------------------------------------


def test_the_reader_is_asked_for_the_runs_ids_and_times_before_the_session_is_stopped(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _one_run(clients)
    reader = FakeRunSpanReader(clients)

    with _executor(clients, tmp_path, span_reader=reader) as executor:
        [trace] = executor.run(_task())

    [query] = reader.queries
    [sent] = clients.calls_to("InvokeAgentRuntime")
    [record] = executor.run_records
    assert query.trace_id == trace.trace_id
    assert query.w3c_trace_id == _w3c_trace_id(sent.params["traceParent"])
    assert query.session_id == sent.params["runtimeSessionId"]
    assert query.runtime_arn == RUNTIME_ARN
    assert query.qualifier == "DEFAULT"
    assert query.started_at == trace.captured_at
    assert _when(record.started_at) <= _when(query.started_at) < _when(query.ended_at)
    assert _when(query.ended_at) <= _when(record.ended_at)
    assert reader.stopped_when_read == [False]
    stops = [stop.params["runtimeSessionId"] for stop in clients.calls_to("StopRuntimeSession")]
    assert stops == [query.session_id]


@pytest.mark.parametrize(
    "failure",
    [
        lambda clients: _fail(clients, "RuntimeClientError", 424),
        lambda clients: _answer(clients, sse_body({"error": "boom", "error_type": "ValueError"})),
        lambda clients: _answer(clients, sse_body({"tokens": 12})),
    ],
    ids=["service-error", "error-event", "no-text"],
)
def test_the_reader_is_not_asked_when_the_invocation_fails(
    tmp_path: Path, failure: Callable[[StubbedAwsClients], None], clients: StubbedAwsClients
) -> None:
    _preflight(clients)
    failure(clients)
    _stopped(clients)
    reader = FakeRunSpanReader(clients)
    executor = _executor(clients, tmp_path, span_reader=reader)

    with executor, pytest.raises(RuntimeInvocationError):
        executor.run(_task())

    assert reader.queries == []
    assert executor.run_records[0].session_stop == "stopped"


class _UnrenderableRefusal(AgentCoreError):
    """A typed refusal whose message itself fails to render."""

    def __str__(self) -> str:
        raise ValueError(_SENTINEL)


_LAGGING = (
    "Logs Insights returned no records for '/aws/bedrock-agentcore/runtimes/"
    "calibrated_agent_evaluation_fixture-AbCdEf1234-DEFAULT' in the window, but the log group "
    "already holds events "
    "there; the Insights index is lagging. Retry shortly."
)
_UNQUOTED = "so the run returns no trace. The run's record"
_RERUN = "a retry means running the task again, which is a new billed invocation of the runtime"


@pytest.mark.parametrize(
    ("error", "said"),
    [
        pytest.param(
            _SpanReadFailed("the settling wait ran out"),
            "The reader's message: the settling wait ran out. The run's record keeps",
            id="typed-refusal",
        ),
        pytest.param(AgentCoreError(_LAGGING), f"The reader's message: {_LAGGING} ", id="lagging"),
        pytest.param(_SpanReadFailed(""), f"failed with _SpanReadFailed, {_UNQUOTED}", id="empty"),
        pytest.param(
            _UnrenderableRefusal(),
            "The reader's message could not be read (ValueError).",
            id="message-fails-to-render",
        ),
        # Only typed refusals are quoted: a stray error's text may carry span data.
        pytest.param(RuntimeError(f"bad span {_SENTINEL}"), _UNQUOTED, id="runtime-error"),
        pytest.param(KeyError(_SENTINEL), _UNQUOTED, id="key-error"),
    ],
)
def test_a_reader_error_is_refused_naming_trace_and_session_and_a_rerun_is_called_billed(
    tmp_path: Path, clients: StubbedAwsClients, error: Exception, said: str
) -> None:
    _one_run(clients)
    executor = _executor(clients, tmp_path, span_reader=_Reader(error))
    refused = _refused(executor)

    [record] = executor.run_records
    message = str(refused)
    assert record.trace_id in message
    assert record.session_id in message
    assert f"it failed with {type(error).__name__}, so the run returns no trace." in message
    assert said in message
    assert "nothing reads a finished run's spans again yet" in message
    assert _RERUN in message
    assert _SENTINEL not in message
    assert refused.__cause__ is error
    assert (record.attempts, record.session_stop) == (1, "stopped")
    assert record.error_code == type(error).__name__


def _span_dicts(query: RunSpanQuery) -> list[Span]:
    # What a reader that skipped normalizing would hand back: each span as a plain dict.
    return cast("list[Span]", [span.model_dump() for span in clm_4471_spans(query.trace_id)])


def _spans_and_a_string(query: RunSpanQuery) -> list[Span]:
    return [*clm_4471_spans(query.trace_id), cast(Span, "not a span")]


def _orphaned(span: Span) -> Span:
    return replace(span, parent=f"{span.trace_id}/no-such-span")


@pytest.mark.parametrize(
    ("returned", "said", "cause"),
    [
        pytest.param(
            _with_one_span_changed(_orphaned), "is not valid", SchemaError, id="not-a-tree"
        ),
        pytest.param(lambda _query: [], "returned no spans", None, id="no-spans"),
        pytest.param(
            _with_one_span_changed(lambda span: replace(span, trace_id="some-other-trace")),
            "a span of another trace",
            None,
            id="a-span-of-another-trace",
        ),
        pytest.param(_span_dicts, "other than normalized spans", None, id="dicts"),
        pytest.param(
            _spans_and_a_string, "other than normalized spans", None, id="spans-and-a-string"
        ),
    ],
)
def test_a_reader_answer_that_is_not_the_runs_span_tree_is_refused_naming_trace_and_session(
    tmp_path: Path,
    returned: Callable[[RunSpanQuery], list[Span]],
    said: str,
    cause: type[Exception] | None,
    clients: StubbedAwsClients,
) -> None:
    _one_run(clients)
    executor = _executor(clients, tmp_path, span_reader=_Reader(returned))
    refused = _refused(executor)

    [record] = executor.run_records
    for fragment in (said, record.trace_id, record.session_id):
        assert fragment in str(refused)
    assert isinstance(refused.__cause__, cause or type(None))
    assert record.session_stop == "stopped"
    assert record.error_code == (cause or RuntimeInvocationError).__name__


def _nested_lists(levels: int) -> list[Any]:
    """Lists ``levels`` deep, built with a loop."""
    value: list[Any] = []
    for _ in range(levels - 1):
        value = [value]
    return value


def _output_with(value: object) -> Callable[[Span], Span]:
    return lambda span: replace(span, output={**span.output, "note": _SENTINEL, "score": value})


def _input_with(value: object) -> Callable[[Span], Span]:
    return lambda span: replace(span, input={**span.input, "note": _SENTINEL, "limit": value})


def _tool_arguments_with(value: object) -> Callable[[Span], Span]:
    def change(span: Span) -> Span:
        first, *rest = span.tool_calls
        args = {**first.args, "note": _SENTINEL, "threshold": value}
        return replace(span, tool_calls=[replace(first, args=args), *rest])

    return change


_NOT_JSON = ("a number JSON cannot represent", "NonFiniteNumberError")


@pytest.mark.parametrize(
    ("change", "said", "code"),
    [
        pytest.param(_output_with(math.nan), *_NOT_JSON, id="nan-in-output"),
        pytest.param(_input_with(math.inf), *_NOT_JSON, id="infinity-in-input"),
        pytest.param(
            _tool_arguments_with(-math.inf), *_NOT_JSON, id="-infinity-in-tool-call-arguments"
        ),
        # The span's dump is the first level and its output the second, so 255 more make 257.
        pytest.param(
            _output_with(_nested_lists(255)),
            "nests more than 256 levels",
            "NestedTooDeepError",
            id="nested-257-levels",
        ),
    ],
)
def test_a_reader_span_holding_nan_infinity_or_over_256_levels_is_refused_naming_the_span(
    tmp_path: Path, clients: StubbedAwsClients, change: Callable[[Span], Span], said: str, code: str
) -> None:
    changed: list[str] = []

    def reading(span: Span) -> Span:
        changed.append(span.span_id)
        return change(span)

    _one_run(clients)
    cassette = tmp_path / "runtime.cassette.json"
    executor = _executor(
        clients,
        tmp_path / "logs",
        span_reader=_Reader(_with_one_span_changed(reading)),
        cassette_path=cassette,
        cassette_mode=Mode.RECORD,
    )
    message = str(_refused(executor))

    # Nothing was recorded or returned, and the session was stopped.
    assert not cassette.exists()
    assert len(clients.calls_to("StopRuntimeSession")) == 1
    [record] = executor.run_records
    assert _manifest(tmp_path / "logs") == [record.model_dump(mode="json")]
    [span_id] = changed

    for fragment in (record.trace_id, record.session_id, span_id, said):
        assert fragment in message
    assert _SENTINEL not in message
    assert (record.session_stop, record.error_code) == ("stopped", code)


@pytest.mark.parametrize(
    "value",
    [[0.25, -0.0, 1.7976931348623157e308], _nested_lists(254)],
    ids=["finite-floats", "nested-256-levels"],
)
def test_a_reader_span_with_finite_floats_or_nested_256_levels_is_accepted(
    tmp_path: Path, clients: StubbedAwsClients, value: object
) -> None:
    _one_run(clients)
    reader = _Reader(_with_one_span_changed(_output_with(value)))

    with _executor(clients, tmp_path, span_reader=reader) as executor:
        [trace] = executor.run(_task())

    [scored] = [span for span in trace.spans if "score" in span.output]
    assert scored.output["score"] == value
    validate(trace)


@pytest.mark.parametrize("chained", [False, True], ids=["as-is", "chained"])
def test_a_guard_refusal_inside_the_reader_names_the_guard(tmp_path: Path, chained: bool) -> None:
    # One billed call is allowed: the invocation spends it, so the reader's billed read is refused.
    with StubbedAwsClients(handlers=[BilledCallBudget(1)]) as clients:
        _one_run(clients)
        # Queued so the stub accepts the call's parameters; the guard refuses it before it is sent.
        add_reader_response(clients)
        reader: Any = FakeAgentCoreSpanReader(clients)
        if chained:
            inner = reader

            class _Wrapping:
                def read_spans(self, query: RunSpanQuery) -> list[Span]:
                    try:
                        return inner.read_spans(query)
                    except AgentCoreError as exc:
                        raise _SpanReadFailed("could not read the run's spans") from exc

            reader = _Wrapping()
        executor = _executor(clients, tmp_path, span_reader=reader)
        message = str(_refused(executor))

    [record] = executor.run_records
    assert "BilledCallBudgetExceeded" in message
    assert "cloudwatchomni:StartTelemetryQuery" in message
    assert record.trace_id in message
    assert record.session_id in message
    assert clients.calls_to("StartTelemetryQuery") == []
    assert record.session_stop == "stopped"


# --- stopping sessions ----------------------------------------------------------------------


def test_a_successful_and_a_failed_run_each_stop_their_session_once(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _one_run(clients)
    _fail(clients, "RuntimeClientError", 424)
    _stopped(clients)
    executor = _executor(clients, tmp_path)

    with executor:
        executor.run(_task())
        with pytest.raises(RuntimeInvocationError, match="RuntimeClientError"):
            executor.run(_task())

    sessions = [call.params["runtimeSessionId"] for call in clients.calls_to("InvokeAgentRuntime")]
    stops = [call.params["runtimeSessionId"] for call in clients.calls_to("StopRuntimeSession")]
    assert len(set(sessions)) == 2
    assert stops == sessions
    assert [record.session_stop for record in executor.run_records] == ["stopped", "stopped"]
    clients.assert_no_pending_responses()


def _interrupted_wait(_seconds: float) -> None:

    raise KeyboardInterrupt


@pytest.mark.parametrize("moment", ["during-the-send", "during-the-retry-wait"])
def test_an_interrupt_while_an_invocation_is_in_flight_still_stops_its_session(
    tmp_path: Path, moment: str
) -> None:
    sending = moment == "during-the-send"
    interrupt = _Raises("InvokeAgentRuntime", KeyboardInterrupt())
    with StubbedAwsClients(handlers=[interrupt] if sending else []) as clients:
        _preflight(clients)
        if sending:
            # Queued so the stub accepts the call; the interrupt comes before it answers.
            _answer(clients)
        else:
            # The session is busy, and the wait before the retry is interrupted.
            _fail(clients, "RetryableConflictException", 409)
        _stopped(clients)
        executor = _executor(clients, tmp_path, sleep=_no_sleep if sending else _interrupted_wait)

        with executor:
            with pytest.raises(KeyboardInterrupt):
                executor.run(_task())
            # Whether a request went out is unknown, so the session was stopped before run()
            # raised, not left for the context's exit.
            stopped_by_the_run = [
                call.params["runtimeSessionId"] for call in clients.calls_to("StopRuntimeSession")
            ]

    [record] = executor.run_records
    assert stopped_by_the_run == [record.session_id]
    assert record.session_stop == "stopped"
    assert record.error_code == "KeyboardInterrupt"
    assert len(clients.calls_to("StopRuntimeSession")) == 1


class _CredentialsExpireOnInvoke(CountingAwsClients):
    """Clients whose credentials can't be refreshed when an invocation is about to be signed."""

    def _on_build(self, service_name: str, client: BaseClient) -> None:
        super()._on_build(service_name, client)

        def expired(**_kwargs: Any) -> None:
            raise CredentialRetrievalError(provider="custom-process", error_msg="refresh failed")

        client.meta.events.register("before-sign.bedrock-agentcore.InvokeAgentRuntime", expired)


@pytest.mark.parametrize(
    "refusal", ["oauth-only-runtime", "request-never-left", "credentials-not-refreshed"]
)
def test_a_run_refused_before_invoking_stops_nothing_and_records_not_started(
    tmp_path: Path, refusal: str
) -> None:
    sends = ScriptedSends(
        {
            **preflight_sends(),
            "InvokeAgentRuntime": [EndpointConnectionError(endpoint_url=_ENDPOINT)],
            "StopRuntimeSession": [json_answer({})],
        }
    )
    clients: AwsClients
    if refusal == "oauth-only-runtime":
        clients = StubbedAwsClients()
        _oauth_only(clients)
    elif refusal == "request-never-left":
        # The connection is refused before the request leaves, so no session was opened.
        clients = CountingAwsClients(sends)
    else:
        # The credentials fail to refresh while the request is signed, before any send.
        clients = _CredentialsExpireOnInvoke(sends)

    with clients:
        executor = AgentCoreRuntimeExecutor(
            _target(), span_reader=_Reader(), clients=clients, log_dir=tmp_path, clock=_Clock()
        )
        expected = "OAuth" if isinstance(clients, StubbedAwsClients) else "was not sent"

        with executor, pytest.raises(RuntimeInvocationError, match=expected):
            executor.run(_task())

    [record] = executor.run_records
    assert (record.session_stop, record.attempts) == ("not_started", 0)
    if isinstance(clients, StubbedAwsClients):
        # Nothing was invoked, and leaving the context found no open session to stop.
        assert clients.calls_to("InvokeAgentRuntime") == []
        assert clients.calls_to("StopRuntimeSession") == []
        assert record.error_code == "RuntimeInvocationError"
        assert (record.runtime_version, record.runtime_version_source) == (None, None)
        assert record.status_code is None
    else:
        assert sends.count("InvokeAgentRuntime") == (1 if refusal == "request-never-left" else 0)
        assert sends.count("StopRuntimeSession") == 0


def test_a_session_that_is_already_gone_counts_as_stopped(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    _preflight(clients)
    _answer(clients)
    _stop_fails(clients, "ResourceNotFoundException")

    with _executor(clients, tmp_path) as executor:
        [trace] = executor.run(_task())

    assert trace.output == {"text": _ANSWER}
    assert executor.run_records[0].session_stop == "already_gone"
    # Already gone counts as stopped, so leaving the context stops nothing more.
    assert len(clients.calls_to("StopRuntimeSession")) == 1
    clients.assert_no_pending_responses()


def test_a_stop_that_succeeds_on_an_sdk_retry_is_recorded_as_stopped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    zero_retry_backoff(monkeypatch)
    sends = ScriptedSends(
        {
            **preflight_sends(),
            "InvokeAgentRuntime": [
                (
                    200,
                    {"content-type": "text/event-stream", "traceparent": _RETURNED_TRACEPARENT},
                    f"data: {json.dumps({'response': _ANSWER})}\n\n".encode(),
                )
            ],
            "StopRuntimeSession": [
                (
                    500,
                    {
                        "x-amzn-errortype": "InternalServerException",
                        "content-type": "application/json",
                    },
                    b'{"message": "try again"}',
                ),
                json_answer({}),
            ],
        }
    )
    with CountingAwsClients(sends) as clients:
        executor = AgentCoreRuntimeExecutor(
            _target(),
            span_reader=_Reader(),
            clients=clients,
            log_dir=tmp_path,
            sleep=_no_sleep,
            clock=_Clock(),
        )

        with executor:
            [trace] = executor.run(_task())

    assert trace.output == {"text": _ANSWER}
    assert sends.count("StopRuntimeSession") == 2
    [record] = executor.run_records
    assert record.session_stop == "stopped"
    assert record.response_traceparent == _RETURNED_TRACEPARENT


def test_a_failed_stop_warns_keeps_the_trace_and_is_retried_once_on_exit_within_30_seconds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    clients: StubbedAwsClients,
) -> None:
    from calibrated_agent_evaluation.backends._bounded import call_bounded

    waits: list[float] = []

    def timed(fn: Callable[[], object], *, timeout_seconds: float, name: str) -> Any:
        waits.append(timeout_seconds)
        return call_bounded(fn, timeout_seconds=timeout_seconds, name=name)

    # Every session stop goes through the bounded call; this notes how long each may wait.
    monkeypatch.setattr("calibrated_agent_evaluation.agentcore.runtime.call_bounded", timed)
    _preflight(clients)
    _answer(clients)
    _stop_fails(clients)
    _stopped(clients)

    with caplog.at_level(logging.INFO, logger="calibrated_agent_evaluation"):
        with _executor(clients, tmp_path) as executor:
            [trace] = executor.run(_task())
            [record] = executor.run_records
            warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
            caplog.clear()

    assert trace.output == {"text": _ANSWER}
    assert record.session_stop == "failed"
    assert any(record.session_id in warning for warning in warnings)
    stops = [call.params["runtimeSessionId"] for call in clients.calls_to("StopRuntimeSession")]
    assert stops == [record.session_id, record.session_id]
    exit_logs = [
        r.getMessage() for r in caplog.records if r.name.startswith("calibrated_agent_evaluation")
    ]
    assert any(record.session_id in line and line.endswith("stopped") for line in exit_logs)
    clients.assert_no_pending_responses()
    # The run's own stop, which fails, and the one more stop it gets on exit.
    assert STOP_TIMEOUT_SECONDS == 30
    assert waits == [30, 30]


def _stops_fail_below_the_sdk() -> _Raises:
    """Fails each stop with an HTTP-layer error that botocore lets through unwrapped."""
    error = urllib3.exceptions.ProtocolError("Connection broken: IncompleteRead")
    return _Raises("StopRuntimeSession", error)


@pytest.mark.parametrize("below_the_sdk", [False, True], ids=["service-error", "below-the-sdk"])
def test_a_failed_stop_does_not_replace_the_runs_own_error(
    tmp_path: Path, below_the_sdk: bool
) -> None:
    handlers = [_stops_fail_below_the_sdk()] if below_the_sdk else []
    with StubbedAwsClients(handlers=handlers) as clients:
        _preflight(clients)
        _fail(clients, "RuntimeClientError", 424)
        if not below_the_sdk:
            _stop_fails(clients)
        _stopped(clients)
        executor = _executor(clients, tmp_path)

        with executor, pytest.raises(RuntimeInvocationError, match="RuntimeClientError"):
            executor.run(_task())

    [record] = executor.run_records
    assert (record.session_stop, record.error_code) == ("failed", "RuntimeClientError")


def test_a_stop_that_fails_below_the_sdk_keeps_the_trace_and_is_retried_on_exit(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    failing = _stops_fail_below_the_sdk()
    with StubbedAwsClients(handlers=[failing]) as clients:
        _preflight(clients)
        for _ in range(2):
            _answer(clients)
        # Queued so the stub accepts each stop; the failure comes before it answers.
        _stopped(clients)

        with caplog.at_level(logging.WARNING, logger="calibrated_agent_evaluation"):
            with _executor(clients, tmp_path) as executor:
                traces = [executor.run(_task())[0], executor.run(_task())[0]]

    assert [trace.output for trace in traces] == [{"text": _ANSWER}] * 2
    assert [record.session_stop for record in executor.run_records] == ["failed", "failed"]
    # Each run's own stop, then one more each on exit: one failing stop did not end the others.
    assert failing.raised == 4
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    for record in executor.run_records:
        assert any(record.session_id in line and "ProtocolError" in line for line in warnings)


def _first_stop_interrupted() -> _Raises:
    """Ctrl-C during the first ``StopRuntimeSession``."""
    return _Raises("StopRuntimeSession", KeyboardInterrupt(), once=True)


def test_context_exit_stops_a_session_whose_stop_was_interrupted(tmp_path: Path) -> None:
    with StubbedAwsClients(handlers=[_first_stop_interrupted()]) as clients:
        _preflight(clients)
        _answer(clients)
        _stopped(clients)
        executor = _executor(clients, tmp_path)

        with pytest.raises(KeyboardInterrupt), executor:
            executor.run(_task())

    [sent] = clients.calls_to("InvokeAgentRuntime")
    [stop] = clients.calls_to("StopRuntimeSession")
    assert stop.params["runtimeSessionId"] == sent.params["runtimeSessionId"]
    clients.assert_no_pending_responses()


def _view_close_raises(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """Make closing an invocation view raise ``error``; other client sets close as usual."""
    real_close = StubbedAwsClients.close

    def close(self: StubbedAwsClients) -> None:
        if self.is_invocation_view:
            raise error
        real_close(self)

    monkeypatch.setattr(StubbedAwsClients, "close", close)


@pytest.mark.parametrize(
    "error",
    [OSError(f"the view could not be closed {_SENTINEL}"), RuntimeError(_SENTINEL)],
    ids=["os-error", "other-error"],
)
def test_a_view_that_fails_to_close_still_closes_the_clients_the_executor_built(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: Exception
) -> None:
    factory = _own_clients(monkeypatch, _one_run)
    _view_close_raises(monkeypatch, error)
    executor = _own_executor(tmp_path)
    executor.run(_task())

    with pytest.raises(RuntimeInvocationError, match="could not close") as refused:
        executor.close()

    assert refused.value.__cause__ is error
    assert type(error).__name__ in str(refused.value)
    # The error's own text is the SDK's or the caller's, so it stays out of the message.
    assert _SENTINEL not in str(refused.value)
    [clients] = factory.built
    own = [entry for entry in clients.built if not entry.view]
    assert own
    assert all(entry.closed for entry in own)


def test_every_later_cleanup_failure_is_noted_on_the_one_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    factory = _own_clients(monkeypatch, _one_run)
    real_close = StubbedAwsClients.close

    def close_fails(self: StubbedAwsClients) -> None:
        real_close(self)
        raise OSError("the pool would not close")

    executor = _own_executor(tmp_path)
    executor.run(_task())
    monkeypatch.setattr(StubbedAwsClients, "close", close_fails)

    def stop_fails() -> None:
        raise RuntimeInvocationError("could not stop the open sessions")

    monkeypatch.setattr(executor, "_stop_open_sessions", stop_fails)

    with pytest.raises(RuntimeInvocationError, match="could not stop the open sessions") as refused:
        executor.close()

    notes = " ".join(getattr(refused.value, "__notes__", ()))
    assert "could not close the invocation clients" in notes
    assert "could not close the AWS clients the executor built" in notes
    assert factory.built


def _permission_denied(name: str) -> Exception:
    return PermissionError(13, "Permission denied", name)


def _temp_dir_cleanup_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: Callable[[str], Exception] = _permission_denied,
) -> Path:
    """Put temporary directories under ``tmp_path``, and make removing one raise ``error(name)``."""
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))

    def cleanup(self: tempfile.TemporaryDirectory[str]) -> None:
        raise error(self.name)

    monkeypatch.setattr(tempfile.TemporaryDirectory, "cleanup", cleanup)
    return temp_root


def test_a_temp_dir_that_cannot_be_removed_on_exit_does_not_replace_the_run_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _temp_dir_cleanup_raises(monkeypatch, tmp_path)
    cassette = tmp_path / "runtime.cassette.json"
    cassette.write_text(json.dumps({"version": 1, "entries": []}))
    executor = AgentCoreRuntimeExecutor(
        _target(), cassette_path=cassette, cassette_mode=Mode.REPLAY, clock=_Clock()
    )

    with caplog.at_level(logging.WARNING, logger="calibrated_agent_evaluation"):
        with pytest.raises(RuntimeInvocationError, match="cassette miss") as raised, executor:
            executor.run(_task())

    notes = getattr(raised.value, "__notes__", [])
    assert any("temporary log directory" in note and "PermissionError" in note for note in notes)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("temporary log directory" in line for line in warnings)
    assert [record.error_code for record in executor.run_records] == ["CassetteMiss"]


@pytest.mark.parametrize(
    ("error", "kind"),
    [(_permission_denied, PermissionError), (lambda _name: RuntimeError(_SENTINEL), RuntimeError)],
    ids=["permission-error", "other-error"],
)
def test_a_temp_dir_that_cannot_be_removed_is_refused_naming_it_after_a_run_that_succeeded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clients: StubbedAwsClients,
    error: Callable[[str], Exception],
    kind: type[Exception],
) -> None:
    temp_root = _temp_dir_cleanup_raises(monkeypatch, tmp_path, error)
    _one_run(clients)
    executor = _executor(clients, None)

    # A run that succeeded: leaving the context is what raises.
    with pytest.raises(RuntimeInvocationError) as refused, executor:
        executor.run(_task())

    [temp_dir] = list(temp_root.iterdir())
    assert "could not remove the temporary log directory" in str(refused.value)
    assert str(temp_dir) in str(refused.value)
    assert f"({kind.__name__})" in str(refused.value)
    assert _SENTINEL not in str(refused.value)
    assert type(refused.value.__cause__) is kind
    # The clients are closed all the same.
    assert all(entry.closed for entry in clients.built if entry.view)


@pytest.mark.parametrize(
    "error",
    [OSError("the view could not be closed"), RuntimeError("a caller's own close failed")],
    ids=["os-error", "other-error"],
)
def test_a_client_that_fails_to_close_on_exit_does_not_replace_the_run_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: Exception,
    caplog: pytest.LogCaptureFixture,
    clients: StubbedAwsClients,
) -> None:
    _view_close_raises(monkeypatch, error)
    _oauth_only(clients)
    # The invocation view is built before the preflight refuses the runtime.
    executor = _executor(clients, tmp_path)

    with caplog.at_level(logging.WARNING, logger="calibrated_agent_evaluation"):
        with pytest.raises(RuntimeInvocationError, match="only accepts OAuth") as raised, executor:
            executor.run(_task())

    notes = getattr(raised.value, "__notes__", [])
    assert any(type(error).__name__ in note for note in notes)
    assert not any(str(error) in note for note in notes)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(type(error).__name__ in line for line in warnings)


def _fail_the_exit_stop_client(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception] = AgentCoreError
) -> dict[str, bool]:
    """Make the base clients fail to build a client with ``error`` once ``switch["on"]`` is set."""
    switch = {"on": False}
    real_client = StubbedAwsClients.client

    def client(
        self: StubbedAwsClients, service_name: str, *, region: str | None = None
    ) -> BaseClient:
        if switch["on"] and not self.is_invocation_view:
            raise error(f"could not build an AWS client for {service_name!r} {_SENTINEL}")
        return real_client(self, service_name, region=region)

    monkeypatch.setattr(StubbedAwsClients, "client", client)
    return switch


def _run_then_break_the_exit_stop(
    executor: AgentCoreRuntimeExecutor, switch: dict[str, bool]
) -> None:
    try:
        executor.run(_task())
    finally:
        switch["on"] = True


def test_an_interrupt_while_cleaning_up_on_exit_does_not_replace_the_run_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clients: StubbedAwsClients
) -> None:
    executor = _executor(clients, tmp_path)

    def interrupted() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(executor, "_stop_open_sessions", interrupted)

    with pytest.raises(RuntimeInvocationError, match="the run failed") as raised, executor:
        raise RuntimeInvocationError("the run failed")

    assert any("KeyboardInterrupt" in note for note in getattr(raised.value, "__notes__", []))


def test_a_failure_to_stop_sessions_on_exit_does_not_replace_the_run_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    switch = _fail_the_exit_stop_client(monkeypatch)
    with StubbedAwsClients(handlers=[_first_stop_interrupted()]) as clients:
        _preflight(clients)
        _answer(clients)
        # Queued so the stub lets the run's own stop reach the handler that interrupts it.
        _stopped(clients)
        executor = _executor(clients, tmp_path)

        with caplog.at_level(logging.WARNING, logger="calibrated_agent_evaluation"):
            with pytest.raises(KeyboardInterrupt) as raised, executor:
                _run_then_break_the_exit_stop(executor, switch)

    session = executor.run_records[0].session_id
    notes = getattr(raised.value, "__notes__", [])
    assert any("could not build the AWS client to stop runtime sessions" in n for n in notes)
    assert any(session in n for n in notes)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(session in line and "on exit" in line for line in warnings)


@pytest.mark.parametrize(
    ("error", "view_close_fails"),
    # Any error the clients raise, such as one from a caller's own AwsClients subclass.
    [(AgentCoreError, False), (RuntimeError, False), (AgentCoreError, True)],
    ids=["agentcore-error", "other-error", "with-a-later-close-failure"],
)
def test_a_failure_to_stop_sessions_on_exit_is_raised_when_the_run_succeeded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clients: StubbedAwsClients,
    error: type[Exception],
    view_close_fails: bool,
) -> None:
    switch = _fail_the_exit_stop_client(monkeypatch, error)
    if view_close_fails:
        _view_close_raises(monkeypatch, OSError("the view could not be closed"))
    _preflight(clients)
    _answer(clients)
    _stop_fails(clients)
    executor = _executor(clients, tmp_path)

    with pytest.raises(RuntimeInvocationError, match="stop runtime sessions") as raised, executor:
        _run_then_break_the_exit_stop(executor, switch)

    assert executor.run_records[0].session_id in str(raised.value)
    assert f"({error.__name__})" in str(raised.value)
    assert _SENTINEL not in str(raised.value)
    assert type(raised.value.__cause__) is error
    if view_close_fails:
        # The stop's failure comes first and is the one raised; the view's is noted on it.
        notes = getattr(raised.value, "__notes__", [])
        assert any("could not close" in note and "OSError" in note for note in notes)
    else:
        assert all(entry.closed for entry in clients.built if entry.view)


# --- the billed-call budget -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("guard", "named", "calls"),
    [
        pytest.param(
            lambda: BilledCallBudget(0),
            ("BilledCallBudgetExceeded", "InvokeAgentRuntime"),
            ["GetAgentRuntimeEndpoint", "GetAgentRuntime"],
            id="zero-budget-at-the-first-invocation",
        ),
        # A read-only guard that allows nothing refuses the first preflight read.
        pytest.param(
            lambda: ReadOnlyGuard(allowed=set()),
            (
                "ReadOnlyViolation",
                "bedrock-agentcore-control:GetAgentRuntimeEndpoint",
                "nothing was invoked",
            ),
            [],
            id="read-only-guard-at-the-preflight",
        ),
    ],
)
def test_a_guard_refusing_a_call_before_the_invocation_names_itself_and_the_operation(
    tmp_path: Path,
    guard: Callable[[], BeforeCallHandler],
    named: tuple[str, ...],
    calls: list[str],
) -> None:
    with StubbedAwsClients(handlers=[guard()]) as clients:
        _one_run(clients)
        executor = _executor(clients, tmp_path)
        refused = str(_refused(executor))

    for fragment in named:
        assert fragment in refused
    # Nothing was invoked, so no session was stopped.
    assert [call.operation for call in clients.calls] == calls
    [record] = executor.run_records
    assert (record.session_stop, record.attempts) == ("not_started", 0)
    assert record.error_code == named[0]


def _two_runs_queued(clients: StubbedAwsClients) -> None:
    _one_run(clients)
    _answer(clients)


@pytest.mark.parametrize("from_variable", [False, True], ids=["caller-clients", "the-variable"])
def test_a_budget_of_one_refuses_the_second_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, from_variable: bool
) -> None:
    if from_variable:
        # The variable caps the clients the executor builds.
        monkeypatch.setenv(_BUDGET_VARIABLE, "1")
    factory = _own_clients(monkeypatch, _two_runs_queued)
    with StubbedAwsClients(handlers=[BilledCallBudget(1)]) as caller_clients:
        _two_runs_queued(caller_clients)
        executor = _own_executor(tmp_path) if from_variable else _executor(caller_clients, tmp_path)

        with executor:
            [first] = executor.run(_task())
            with pytest.raises(RuntimeInvocationError, match="BilledCallBudgetExceeded"):
                executor.run(_task())

    [clients] = factory.built if from_variable else [caller_clients]
    assert first.output == {"text": _ANSWER}
    assert len(clients.calls_to("InvokeAgentRuntime")) == 1
    assert len(clients.calls_to("StopRuntimeSession")) == 1


def test_retries_spend_the_budget(tmp_path: Path) -> None:
    with StubbedAwsClients(handlers=[BilledCallBudget(2)]) as clients:
        _preflight(clients)
        _fail(clients, "RetryableConflictException", 409)
        _fail(clients, "RetryableConflictException", 409)
        _answer(clients)
        _stopped(clients)
        executor = _executor(clients, tmp_path)

        with executor, pytest.raises(RuntimeInvocationError, match="BilledCallBudgetExceeded"):
            executor.run(_task())

    assert len(clients.calls_to("InvokeAgentRuntime")) == 2
    [record] = executor.run_records
    assert record.attempts == 2
    # Two attempts reached the session, so it is stopped.
    assert record.session_stop == "stopped"


@pytest.mark.parametrize("value", [None, "", "  "], ids=["unset", "empty", "blank"])
def test_no_budget_variable_means_no_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str | None
) -> None:
    if value is not None:
        monkeypatch.setenv(_BUDGET_VARIABLE, value)

    def prepare(clients: StubbedAwsClients) -> None:
        _preflight(clients)
        for _ in range(3):
            _answer(clients)
            _stopped(clients)

    factory = _own_clients(monkeypatch, prepare)

    with _own_executor(tmp_path) as executor:
        traces = [executor.run(_task())[0] for _ in range(3)]

    [clients] = factory.built
    assert len(traces) == 3
    assert len(clients.calls_to("InvokeAgentRuntime")) == 3


# --- run records and the manifest -----------------------------------------------------------


def test_run_records_lists_a_success_a_failure_and_a_refusal_in_call_order(
    tmp_path: Path,
) -> None:
    with StubbedAwsClients(handlers=[BilledCallBudget(2)]) as clients:
        _one_run(clients)
        _fail(clients, "RuntimeClientError", 424)
        _stopped(clients)
        _answer(clients)
        executor = _executor(clients, tmp_path)

        with executor:
            [trace] = executor.run(_task())
            for _ in range(2):
                with pytest.raises(RuntimeInvocationError):
                    executor.run(_task())

    success, failure, refusal = executor.run_records
    assert success.trace_id == trace.trace_id
    assert [(r.error_code, r.session_stop, r.attempts) for r in executor.run_records] == [
        (None, "stopped", 1),
        ("RuntimeClientError", "stopped", 1),
        ("BilledCallBudgetExceeded", "not_started", 0),
    ]
    assert (failure.status_code, failure.runtime_version) == (424, "24")
    sent = [call.params["runtimeSessionId"] for call in clients.calls_to("InvokeAgentRuntime")]
    assert sent == [success.session_id, failure.session_id]
    assert len({success.session_id, failure.session_id, refusal.session_id}) == 3

    # A failed or refused run still writes its run record.
    assert [RunRecord.model_validate(line) for line in _manifest(tmp_path)] == list(
        executor.run_records
    )


def test_two_runs_write_two_run_records_keyed_by_trace_id_and_stamp_the_bundle(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    cassette = tmp_path / "runtime.cassette.json"
    _one_run(clients)
    _answer(clients)
    _stopped(clients)
    target = _target(config_bundle_arn=BUNDLE_ARN, config_bundle_version=BUNDLE_VERSION)

    with _executor(
        clients, tmp_path, target=target, cassette_path=cassette, cassette_mode=Mode.RECORD
    ) as executor:
        traces = [executor.run(_task())[0], executor.run(_task(prompt="Is it covered?"))[0]]

    lines = _manifest(tmp_path)
    assert [line["trace_id"] for line in lines] == [trace.trace_id for trace in traces]
    assert executor.manifest_errors == ()
    sent = clients.calls_to("InvokeAgentRuntime")
    for line, call in zip(lines, sent, strict=True):
        assert line["session_id"] == call.params["runtimeSessionId"]
        assert line["w3c_trace_id"] == _w3c_trace_id(call.params["traceParent"])
        assert line["runtime_arn"] == RUNTIME_ARN
        assert line["qualifier"] == "DEFAULT"
        assert (line["runtime_version"], line["runtime_version_source"]) == ("24", "endpoint")
        assert line["config_bundle_arn"] == BUNDLE_ARN
        assert line["config_bundle_version"] == BUNDLE_VERSION
        assert line["payload_format"] == "agentcore_sse"
        started, ended = _when(line["started_at"]), _when(line["ended_at"])
        assert started < ended
        assert line["duration_ms"] == (ended - started) // timedelta(milliseconds=1)
        assert (line["attempts"], line["status_code"]) == (1, 200)
        assert line["content_type"] == "text/event-stream"
        assert line["response_traceparent"] == _RETURNED_TRACEPARENT
        assert line["replayed"] is False
        assert line["session_stop"] == "stopped"
        assert line["error_code"] is None
        baggage = _baggage(call.params["baggage"])
        assert baggage["aws.agentcore.configbundle_arn"] == BUNDLE_ARN
        assert baggage["aws.agentcore.configbundle_version"] == BUNDLE_VERSION
    entries = json.loads(cassette.read_text(encoding="utf-8"))["entries"]
    assert [entry["model"] for entry in entries] == 2 * [
        {
            "id": f"{RUNTIME_ARN}@DEFAULT",
            "config": {"config_bundle_arn": BUNDLE_ARN, "config_bundle_version": BUNDLE_VERSION},
        }
    ]


def test_a_replay_miss_writes_a_run_record_with_fresh_ids(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    cassette = tmp_path / "runtime.cassette.json"
    _one_run(clients)
    with _executor(
        clients, tmp_path / "record", cassette_path=cassette, cassette_mode=Mode.RECORD
    ) as recorder:
        recorder.run(_task())
    replayer = AgentCoreRuntimeExecutor(
        _target(),
        cassette_path=cassette,
        cassette_mode=Mode.REPLAY,
        log_dir=tmp_path / "replay",
        clock=_Clock(),
    )

    with replayer, pytest.raises(RuntimeInvocationError, match="cassette miss"):
        replayer.run(_task(prompt="Is water damage from a burst pipe covered?"))

    [line] = _manifest(tmp_path / "replay")
    recorded = cassette.read_text(encoding="utf-8")
    assert re.fullmatch(r"calibrated-agent-evaluation-[0-9a-f]{32}", line["session_id"])
    assert re.fullmatch(r"[0-9a-f]{32}", line["w3c_trace_id"])
    assert line["w3c_trace_id"].strip("0")
    assert line["trace_id"] == f"agentcore-{line['w3c_trace_id']}"
    # Made for this run, not taken from the cassette's recorded run.
    assert line["session_id"] not in recorded
    assert line["w3c_trace_id"] not in recorded
    assert (line["replayed"], line["error_code"], line["session_stop"]) == (
        False,
        "CassetteMiss",
        "not_started",
    )
    assert (line["attempts"], line["runtime_version"]) == (0, None)


@pytest.mark.parametrize("to_run", [False, True], ids=["the-executors-given-as-text", "run-wins"])
def test_the_run_record_goes_to_the_log_dir_passed_to_run_or_else_the_executors(
    tmp_path: Path, clients: StubbedAwsClients, to_run: bool
) -> None:
    executor_dir, run_dir = tmp_path / "executor", tmp_path / "run"
    _one_run(clients)

    with _executor(clients, cast(Any, str(executor_dir))) as executor:
        [trace] = executor.run(_task(), log_dir=run_dir if to_run else None)

    [line] = _manifest(run_dir if to_run else executor_dir)
    assert (line["trace_id"], line["session_stop"]) == (trace.trace_id, "stopped")
    assert _no_manifest_in(executor_dir) is to_run


def test_without_a_log_dir_the_manifest_goes_to_a_temp_dir_removed_on_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clients: StubbedAwsClients
) -> None:
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))
    _one_run(clients)

    with _executor(clients, None) as executor:
        [trace] = executor.run(_task())
        [manifest] = list(temp_root.glob(f"*/{RUN_RECORD_STEM}.json"))
        assert [line["trace_id"] for line in _manifest(manifest.parent)] == [trace.trace_id]

    assert list(temp_root.iterdir()) == []
    assert [record.trace_id for record in executor.run_records] == [trace.trace_id]


def _unwritable_log_dir(tmp_path: Path) -> Path:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    return blocker / "logs"


def test_a_successful_run_with_an_unwritable_manifest_is_refused_naming_its_path(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    log_dir = _unwritable_log_dir(tmp_path)
    _one_run(clients)
    executor = _executor(clients, log_dir)
    returned: list[Trace] = []

    with executor, pytest.raises(RuntimeInvocationError) as refused:
        returned.extend(executor.run(_task()))

    assert str(manifest_path(log_dir)) in str(refused.value)
    assert returned == []
    # The run itself went through, and its session was stopped.
    assert len(clients.calls_to("InvokeAgentRuntime")) == 1
    assert executor.run_records[0].session_stop == "stopped"
    # Listed, and said to be missing from the manifest.
    assert executor.manifest_errors == (refused.value,)


def test_a_failed_run_with_an_unwritable_manifest_raises_the_runs_own_error_with_the_path_noted(
    tmp_path: Path, clients: StubbedAwsClients
) -> None:
    log_dir = _unwritable_log_dir(tmp_path)
    _preflight(clients)
    _fail(clients, "RuntimeClientError", 424)
    _stopped(clients)
    executor = _executor(clients, log_dir)
    refused = _refused(executor)

    assert "RuntimeClientError" in str(refused)
    assert str(manifest_path(log_dir)) in "\n".join(getattr(refused, "__notes__", []))
    assert executor.run_records[0].error_code == "RuntimeClientError"
    [unwritten] = executor.manifest_errors
    assert str(manifest_path(log_dir)) in str(unwritten)


# --- what stays out of messages and logs ----------------------------------------------------


class _TimesOutMidStream(io.RawIOBase):
    """A response stream that yields the start of an event, then times out reading the rest."""

    def __init__(self, head: bytes) -> None:
        self.head = head
        self.reads = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        self.reads += 1
        if self.reads == 1:
            return self.head
        raise urllib3.exceptions.ReadTimeoutError(cast("Any", None), _ENDPOINT, "Read timed out.")


_STARTED = f'{{"response": "{_SENTINEL}'.encode()  # the start of an answer, cut off
_STREAMING = "An error occurred during streaming"
_ERROR_EVENT = {"error": f"boom {_SENTINEL}", "error_type": "ValueError", "message": _STREAMING}


@pytest.mark.parametrize(
    "failure",
    [
        lambda clients: _fail(clients, "ValidationException", 400, f"bad payload: {_SENTINEL}"),
        lambda clients: _answer(
            clients, StreamingBody(_TimesOutMidStream(b"data: " + _STARTED), None)
        ),
        lambda clients: _answer(clients, streaming_body(_STARTED), content_type="application/json"),
        lambda clients: _answer(
            clients, sse_body({"response": f"partial {_SENTINEL}"}, _ERROR_EVENT)
        ),
    ],
    ids=["service-error", "read-timeout", "undecodable-body", "error-event"],
)
def test_payloads_responses_and_error_text_never_reach_calibrated_agent_evaluation_logs_or_errors(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    failure: Callable[[StubbedAwsClients], None],
    clients: StubbedAwsClients,
) -> None:
    _preflight(clients)
    failure(clients)
    _stop_fails(clients)
    _stopped(clients)
    executor = _executor(clients, tmp_path)

    with caplog.at_level(logging.DEBUG, logger="calibrated_agent_evaluation"):
        prompt = f"Is the vacancy exclusion triggered? {_SENTINEL}"
        refused = _refused(executor, _task(prompt=prompt))

    [sent] = clients.calls_to("InvokeAgentRuntime")
    assert _SENTINEL in sent.params["payload"].decode()
    assert _SENTINEL not in str(refused)
    assert _SENTINEL not in "\n".join(getattr(refused, "__notes__", []))
    calibrated_agent_evaluation_logs = [
        r.getMessage() for r in caplog.records if r.name.startswith("calibrated_agent_evaluation")
    ]
    assert calibrated_agent_evaluation_logs
    assert not any(_SENTINEL in line for line in calibrated_agent_evaluation_logs)
    for suffix in ("json", "csv"):
        stored = (tmp_path / f"{RUN_RECORD_STEM}.{suffix}").read_text(encoding="utf-8")
        assert _SENTINEL not in stored


# --- the session a trace belongs to ----------------------------------------------------------


@pytest.mark.parametrize("given", [None, "CLM-4471-agentcore"], ids=["no-trace-id", "trace-id"])
def test_a_live_trace_carries_the_runtime_session_it_was_invoked_under(
    tmp_path: Path, clients: StubbedAwsClients, given: str | None
) -> None:
    _one_run(clients)

    with _executor(clients, tmp_path) as executor:
        [trace] = executor.run(_task(trace_id=given))

    [sent] = clients.calls_to("InvokeAgentRuntime")
    assert trace.session_id == sent.params["runtimeSessionId"]
