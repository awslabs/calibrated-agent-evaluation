# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation probe --executor agentcore`: the probe's prompts go to an agent on
AgentCore Runtime.

The route sends each `--prompt`, or the probe's two built-in prompts, to the agent a runtime ARN
names, over the SigV4 call path `calibrated-agent-evaluation generate --executor agentcore` uses. It
reads no spans, so each reply is stored as a `probed`, outcome-only trace. Before any AWS request it
refuses what it can't run: the URL probe's own flags, a missing runtime ARN or payload format, a
missing `--authorized`, a cassette mode it has no use for, a missing `[agentcore]` extra and a trace
store that can't take the traces. It then prints what it will call and goes ahead only with `--yes`.
Each prompt's run record goes to the run manifest under `<root>/records/agentcore`, or under
`--log-dir`.

Without `--executor agentcore` the probe is the URL probe it has always been, and each flag only
the AgentCore route reads is refused before anything is sent.

A URL probe's requests go to a stub transport, and the AgentCore runs answer through stubbed AWS
clients, so no test opens a socket or touches AWS.
"""

from __future__ import annotations

import os
import socket
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote

import pytest

pytest.importorskip("boto3")

import boto3.session
import botocore.session
from botocore.config import Config

from _agentcore_runtime_fakes import (
    ANSWER,
    BUNDLE_ARN,
    BUNDLE_VERSION,
    REGION,
    RUNTIME_ARN,
    StubbedAwsClients,
    add_invocation_response,
    add_preflight_responses,
    add_stop_response,
    sse_body,
)
from calibrated_agent_evaluation.agentcore import AwsClients
from calibrated_agent_evaluation.agentcore.run_manifest import RUN_RECORD_STORE, manifest_path
from calibrated_agent_evaluation.agentcore.session import DEFAULT_CLIENT_CONFIG, BeforeCallHandler
from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend, LangfuseLike
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.args import AGENTCORE_PAYLOAD_FORMATS
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.probe import DEFAULT_SEEDS, endpoint_fingerprint
from calibrated_agent_evaluation.schema.types import Trace
from test_backends_langfuse import StubLangfuseClient

_URL = "https://stub.local/invoke"
_BUNDLE_ARN = (
    "arn:aws:bedrock-agentcore:us-west-2:111122223333:configuration-bundle/"
    "calibrated_agent_evaluation_fixture-bundle"
)
_PROMPT = "What can you help me with?"
_MODE_VARIABLE = "CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE"
_BUDGET_VARIABLE = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS"
# The opening words of the plan the route prints before any AWS request.
_PLAN = "this run can call AWS"

# Each flag only the AgentCore route reads, with a value it would accept.
_AGENTCORE_ONLY_FLAGS: list[tuple[str, ...]] = [
    ("--runtime-arn", RUNTIME_ARN),
    ("--payload-format", "agentcore_sse"),
    ("--qualifier", "live"),
    ("--config-bundle-arn", _BUNDLE_ARN),
    ("--config-bundle-version", "1"),
    ("--read-timeout", "60"),
    ("--profile", "calibrated-agent-evaluation-test"),
    ("--log-dir", "agentcore-logs"),
    ("--yes",),
]


class _SendSpy:
    """A URL probe transport that records each request and answers none of them."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.sent: list[str] = []
        monkeypatch.setattr("calibrated_agent_evaluation.probe.adapter._urllib_transport", self)

    def __call__(
        self, url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, list[bytes]]:
        self.sent.append(url)
        return 200, []


def _probed(root: Path) -> list[str]:
    return [t.trace_id for t in LocalBackend(root=root).query() if t.ownership == "probed"]


@pytest.fixture(autouse=True)
def _no_ambient_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hide the machine's AWS setup, cassette mode, budget and local root."""
    for name in list(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    for name in (_MODE_VARIABLE, _BUDGET_VARIABLE, "CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT"):
        monkeypatch.delenv(name, raising=False)
    empty = tmp_path / "empty-aws-config"
    empty.write_text("")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(empty))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(empty))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "store"


def _agentcore_probe(
    root: Path,
    *flags: str,
    runtime_arn: str | None = RUNTIME_ARN,
    payload_format: str | None = "agentcore_sse",
    authorized: bool = True,
    yes: bool = True,
    backend: str | None = None,
) -> int:
    """``calibrated-agent-evaluation probe --executor agentcore`` against the fixture runtime, plus
    ``flags``."""
    argv = [] if backend is None else ["--backend", backend]
    argv += ["--root", str(root), "probe", "--executor", "agentcore"]
    if runtime_arn is not None:
        argv += ["--runtime-arn", runtime_arn]
    if payload_format is not None:
        argv += ["--payload-format", payload_format]
    if authorized:
        argv.append("--authorized")
    if yes:
        argv.append("--yes")
    return main([*argv, *flags])


def _stored_probed(root: Path) -> list[Trace]:
    if not root.exists():
        return []
    with LocalBackend(root=root) as backend:
        return [trace for trace in backend.query() if trace.ownership == "probed"]


def _manifest(path: Path) -> list[dict[str, Any]]:
    """The run records in the run manifest whose JSON file is ``path``, as the JSON it stores."""
    records = RUN_RECORD_STORE.load(path.parent, fmt="json")
    return [record.model_dump(mode="json") for record in records]


def _default_manifest(root: Path) -> Path:
    return manifest_path(root / "records" / "agentcore")


# A tool the stubbed agent's reply names, so the probe sees one.
_TOOL = "lookup_policy"


class _StubbedRuntime:
    """Stands in for ``AwsClients.from_profile``: stubbed clients that answer like a runtime.

    Each set it builds, for the region, profile and handlers it is given, answers the two
    read-only preflight calls, then ``replies`` invocations, each followed by the stop of its
    session. Each reply names :data:`_TOOL` and says :data:`ANSWER`; the one at index ``fail_at``
    is a 424 instead. No span reader is offered: a probe reads none. ``on_build`` runs as each
    set is built, before any call goes through it.
    """

    def __init__(
        self,
        replies: int = 1,
        *,
        fail_at: int | None = None,
        on_build: Callable[[], None] | None = None,
    ) -> None:
        self.replies = replies
        self.fail_at = fail_at
        self.on_build = on_build
        self.built: list[StubbedAwsClients] = []
        self.requested: list[tuple[str | None, str | None]] = []

    def __call__(
        self,
        *,
        region: str | None = None,
        profile: str | None = None,
        handlers: tuple[BeforeCallHandler, ...] | list[BeforeCallHandler] = (),
        config: Config = DEFAULT_CLIENT_CONFIG,
    ) -> StubbedAwsClients:
        if self.on_build is not None:
            self.on_build()
        self.requested.append((region, profile))
        clients = StubbedAwsClients(region=region or REGION, handlers=handlers, config=config)
        add_preflight_responses(clients)
        for index in range(self.replies):
            if index == self.fail_at:
                clients.stub("bedrock-agentcore", view=True).add_client_error(
                    "invoke_agent_runtime",
                    service_error_code="RuntimeClientError",
                    service_message="the agent failed",
                    http_status_code=424,
                )
            else:
                reply = sse_body({"type": "tool_use", "tool_name": _TOOL}, {"response": ANSWER})
                add_invocation_response(clients, reply)
            add_stop_response(clients)
        self.built.append(clients)
        return clients

    @property
    def invocations(self) -> int:
        return sum(len(clients.calls_to("InvokeAgentRuntime")) for clients in self.built)


def _stubbed_runtime(monkeypatch: pytest.MonkeyPatch, runtime: _StubbedRuntime) -> _StubbedRuntime:
    """Route the CLI's AWS clients through ``runtime``."""
    monkeypatch.setattr(AwsClients, "from_profile", runtime)
    return runtime


class _CheckedLangfuse(StubLangfuseClient):
    """A Langfuse client whose ``auth_check`` answers ``accepted``, or raises ``unreachable``.

    Each check calls ``on_check`` first, so a test can see what had happened by then.
    """

    def __init__(
        self,
        *,
        accepted: bool = True,
        unreachable: Exception | None = None,
        on_check: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.accepted = accepted
        self.unreachable = unreachable
        self.on_check = on_check

    def auth_check(self) -> bool:
        if self.on_check is not None:
            self.on_check()
        if self.unreachable is not None:
            raise self.unreachable
        return self.accepted

    def stored_traces(self) -> int:
        return len(self.api.trace.list(page=1, limit=50).data)


def _langfuse_store(monkeypatch: pytest.MonkeyPatch, client: _CheckedLangfuse) -> None:
    """Make ``--backend langfuse`` a Langfuse backend over ``client``."""
    import calibrated_agent_evaluation.cli.backends as cli_backends

    real = cli_backends.backend_from_flag

    def backend(flag: str, **kwargs: Any) -> Any:
        if flag == "langfuse":
            return LangfuseBackend(cast("LangfuseLike", client))
        return real(flag, **kwargs)

    monkeypatch.setattr(cli_backends, "backend_from_flag", backend)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-test")


class _AwsSpy:
    """Records, and refuses, every AWS session, AWS client and socket connection a command opens."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.opened: list[str] = []

        def refuse(kind: str) -> Callable[..., Any]:
            def handler(*_args: Any, **_kwargs: Any) -> Any:
                self.opened.append(kind)
                raise OSError(f"this test allows no {kind}")

            return handler

        monkeypatch.setattr(socket.socket, "connect", refuse("connection"))
        monkeypatch.setattr(socket.socket, "connect_ex", refuse("connection"))
        monkeypatch.setattr(boto3.session.Session, "__init__", refuse("AWS session"))
        monkeypatch.setattr(botocore.session.Session, "create_client", refuse("AWS client"))


# --- the URL probe ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag", _AGENTCORE_ONLY_FLAGS, ids=[flag[0] for flag in _AGENTCORE_ONLY_FLAGS]
)
def test_agentcore_only_flags_on_a_url_probe_are_refused_naming_the_executor(
    seeded_local_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport = _SendSpy(monkeypatch)

    rc = main(["probe", "--endpoint", _URL, "--authorized", *flag])

    err = capsys.readouterr().err
    assert rc == EXIT_USAGE
    assert flag[0] in err
    assert "pass --executor agentcore to send the prompts to an AgentCore Runtime agent" in err
    assert "Traceback" not in err
    assert "Probing " not in err
    assert transport.sent == []
    assert _probed(seeded_local_root) == []


def test_an_agentcore_only_flag_is_refused_before_the_interactive_setup_or_the_battery(
    seeded_local_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    transport = _SendSpy(monkeypatch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input", lambda *_a, **_k: pytest.fail("the refusal comes before any prompt")
    )

    interactive = main(["probe", "--yes"])
    interactive_err = capsys.readouterr().err
    battery = main(["probe", "--endpoint", _URL, "--authorized", "--battery", "--profile", "p"])
    battery_err = capsys.readouterr().err

    assert interactive == EXIT_USAGE
    assert "--yes" in interactive_err
    assert battery == EXIT_USAGE
    assert "--profile" in battery_err
    assert "--trials" not in battery_err
    assert transport.sent == []
    assert _probed(seeded_local_root) == []


# --- refused before any AWS request ------------------------------------------------------------

# Each URL probe flag the AgentCore route has no use for, with a value the URL probe would accept.
_URL_PROBE_FLAGS: list[tuple[str, ...]] = [
    ("--endpoint", _URL),
    ("--auth-bearer-env", "AGENT_API_TOKEN"),
    ("--auth-api-key-env", "AGENT_API_KEY"),
    ("--auth-api-key-header", "x-api-key"),
    ("--auth-header-env", "AGENT_HEADERS"),
    ("--auth-subprotocol-env", "AGENT_WS_TOKEN"),
    ("--timeout", "60"),
]


@pytest.mark.parametrize("flag", _URL_PROBE_FLAGS, ids=[flag[0] for flag in _URL_PROBE_FLAGS])
def test_an_agentcore_probe_refuses_each_url_auth_and_timeout_flag_naming_it(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    flag: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    aws = _AwsSpy(monkeypatch)

    rc = _agentcore_probe(root, *flag)

    err = capsys.readouterr().err
    assert rc == EXIT_USAGE
    assert f"{flag[0]} doesn't apply to --executor agentcore:" in err
    assert "the runtime ARN names the target" in err
    assert "signed with SigV4 from --profile" in err
    assert "OAuth-only runtimes are not supported yet" in err
    assert ("--read-timeout sets the invocation timeout" in err) == (flag[0] == "--timeout")
    assert "Traceback" not in err
    assert _PLAN not in err
    assert aws.opened == []
    assert not (root / "records" / "agentcore").exists()


_BATTERY_FLAGS: list[tuple[str, ...]] = [
    ("--battery",),
    ("--battery", "xstest"),
    ("--trials", "20"),
    ("--allow-use-restricted",),
    ("--grader-model", "bedrock/global.anthropic.claude-sonnet-4-6"),
]


@pytest.mark.parametrize(
    "flag", _BATTERY_FLAGS, ids=["--battery", "--battery-named", "--trials", "--allow", "--grader"]
)
def test_an_agentcore_probe_refuses_the_battery_flags_naming_them(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    flag: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    aws = _AwsSpy(monkeypatch)

    rc = _agentcore_probe(root, *flag)

    err = capsys.readouterr().err
    assert rc == EXIT_USAGE
    assert f"{flag[0]} doesn't apply to --executor agentcore:" in err
    assert "the benchmark battery runs only against a URL probe's --endpoint" in err
    assert "Traceback" not in err
    assert _PLAN not in err
    assert aws.opened == []
    assert not (root / "records" / "agentcore").exists()


@pytest.mark.parametrize(
    ("runtime_arn", "payload_format", "named"),
    [
        (None, "agentcore_sse", ("--runtime-arn",)),
        (RUNTIME_ARN, None, ("--payload-format", *AGENTCORE_PAYLOAD_FORMATS)),
        (None, None, ("--runtime-arn", "--payload-format", *AGENTCORE_PAYLOAD_FORMATS)),
    ],
    ids=["no-runtime-arn", "no-payload-format", "neither"],
)
def test_an_agentcore_probe_without_runtime_arn_or_payload_format_names_them(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    runtime_arn: str | None,
    payload_format: str | None,
    named: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    aws = _AwsSpy(monkeypatch)

    rc = _agentcore_probe(root, runtime_arn=runtime_arn, payload_format=payload_format)

    err = capsys.readouterr().err
    assert rc == EXIT_USAGE
    for name in named:
        assert name in err
    assert "Traceback" not in err
    assert _PLAN not in err
    assert aws.opened == []


def test_an_agentcore_probe_on_a_terminal_never_asks_for_its_options(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    aws = _AwsSpy(monkeypatch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input", lambda *_a, **_k: pytest.fail("the AgentCore probe never asks")
    )

    rc = _agentcore_probe(root, runtime_arn=None, payload_format=None)

    err = capsys.readouterr().err
    assert rc == EXIT_USAGE
    assert "UsageError" in err
    assert "--runtime-arn" in err
    assert "--payload-format" in err
    assert aws.opened == []


@pytest.mark.parametrize(
    ("flags", "label"),
    [((), f"{RUNTIME_ARN}@DEFAULT"), (("--qualifier", "live"), f"{RUNTIME_ARN}@live")],
    ids=["default-qualifier", "named-qualifier"],
)
def test_an_agentcore_probe_without_authorized_is_refused_before_the_plan(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    flags: tuple[str, ...],
    label: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    aws = _AwsSpy(monkeypatch)

    rc = _agentcore_probe(root, *flags, authorized=False)

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert "AuthorizationError" in err
    assert f"probing {label!r} requires confirmed authorization" in err
    assert "Traceback" not in err
    assert _PLAN not in err
    assert aws.opened == []
    assert not (root / "records" / "agentcore").exists()


@pytest.mark.parametrize("value", ["replay", " Replay "])
def test_an_agentcore_probe_refuses_replay_mode_saying_nothing_was_sent(
    monkeypatch: pytest.MonkeyPatch, root: Path, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(_MODE_VARIABLE, value)
    aws = _AwsSpy(monkeypatch)

    rc = _agentcore_probe(root, "--prompt", _PROMPT)

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert (
        "CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE is replay, but probe has no replay; nothing "
        "was sent" in err
    )
    assert "Traceback" not in err
    assert _PLAN not in err
    assert aws.opened == []
    assert not (root / "records" / "agentcore").exists()


def test_an_agentcore_probe_with_an_invalid_cassette_mode_variable_exits_1_naming_it(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(_MODE_VARIABLE, "rewind")
    aws = _AwsSpy(monkeypatch)

    rc = _agentcore_probe(root, "--prompt", _PROMPT)

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert _MODE_VARIABLE in err
    assert "rewind" in err
    assert "Traceback" not in err
    assert _PLAN not in err
    assert aws.opened == []


# --- sending the prompts ----------------------------------------------------------------------


def test_an_agentcore_probe_stores_one_probed_outcome_only_trace_per_prompt(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=1))
    logs = tmp_path / "logs"

    rc = _agentcore_probe(root, "--prompt", _PROMPT, "--log-dir", str(logs))

    captured = capsys.readouterr()
    assert rc == EXIT_OK, captured.err
    [trace] = _stored_probed(root)
    assert trace.spans == []
    assert trace.input == {"prompt": _PROMPT}
    assert trace.output == {"text": ANSWER}
    assert trace.frame == "discovery"
    assert trace.taxonomy_cell == {}
    assert trace.endpoint_fingerprint == endpoint_fingerprint(f"{RUNTIME_ARN}@DEFAULT")
    [clients] = runtime.built
    assert [call.operation for call in clients.calls] == [
        "GetAgentRuntimeEndpoint",
        "GetAgentRuntime",
        "InvokeAgentRuntime",
        "StopRuntimeSession",
    ]
    clients.assert_no_pending_responses()
    [record] = _manifest(manifest_path(logs))
    assert record["trace_id"] == trace.trace_id
    assert record["replayed"] is False
    assert record["session_stop"] == "stopped"
    summary = captured.out
    assert f"{RUNTIME_ARN}@DEFAULT" in summary
    assert "1 probed trace(s) stored" in summary
    assert _TOOL in summary
    assert "agentcore_sse" in summary
    assert str(manifest_path(logs)) in summary
    assert "a refusal is not proof the agent cannot do something" in summary


def test_an_agentcore_probe_builds_its_clients_for_the_arns_region_from_the_profile(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=1))

    rc = _agentcore_probe(
        root, "--prompt", _PROMPT, "--profile", "calibrated-agent-evaluation-test"
    )

    assert rc == EXIT_OK, capsys.readouterr().err
    assert runtime.requested == [(REGION, "calibrated-agent-evaluation-test")]
    assert runtime.invocations == 1


def test_a_second_agentcore_probe_adds_traces_instead_of_replacing_them(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=1))

    first = _agentcore_probe(root, "--prompt", _PROMPT)
    second = _agentcore_probe(root, "--prompt", _PROMPT)

    assert (first, second) == (EXIT_OK, EXIT_OK), capsys.readouterr().err
    stored = _stored_probed(root)
    assert len(stored) == 2
    assert {trace.input["prompt"] for trace in stored} == {_PROMPT}
    records = _manifest(_default_manifest(root))
    assert sorted(record["trace_id"] for record in records) == sorted(t.trace_id for t in stored)


def test_an_agentcore_probe_with_no_prompt_sends_the_two_default_prompts(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=2))

    rc = _agentcore_probe(root)

    captured = capsys.readouterr()
    assert rc == EXIT_OK, captured.err
    assert "up to 2 prompts" in captured.err
    assert runtime.invocations == 2
    assert sorted(t.input["prompt"] for t in _stored_probed(root)) == sorted(DEFAULT_SEEDS)


@pytest.mark.parametrize(
    ("prompts", "named"),
    [(("",), "--prompt is empty"), ((_PROMPT, ""), "1 of the 2 --prompt values is empty")],
    ids=["the-only-prompt", "after-another-prompt"],
)
def test_an_agentcore_probe_with_an_empty_prompt_is_refused_before_the_plan_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    capsys: pytest.CaptureFixture[str],
    prompts: tuple[str, ...],
    named: str,
) -> None:
    aws = _AwsSpy(monkeypatch)
    flags = [arg for prompt in prompts for arg in ("--prompt", prompt)]

    rc = _agentcore_probe(root, *flags)

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert f"{named}, so nothing was sent" in err
    assert _PLAN not in err
    assert "Traceback" not in err
    assert aws.opened == []
    assert _stored_probed(root) == []
    assert not (root / "records" / "agentcore").exists()


def test_a_url_probe_calls_no_agentcore_runtime_and_writes_no_run_record(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=1))

    def reply(url: str, headers: dict[str, str], body: bytes, timeout: float) -> Any:
        return 200, [b'data: {"response": "I help with orders."}']

    monkeypatch.setattr("calibrated_agent_evaluation.probe.adapter._urllib_transport", reply)

    rc = main(["--root", str(root), "probe", "--endpoint", _URL, "--authorized"])

    assert rc == EXIT_OK, capsys.readouterr().err
    assert _stored_probed(root)
    # No AWS client was built, so no runtime was called.
    assert runtime.built == []
    assert not (root / "records" / "agentcore").exists()


def test_an_agentcore_probe_sends_the_named_configuration_bundle_and_records_it(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=1))
    bundle = ("--config-bundle-arn", BUNDLE_ARN, "--config-bundle-version", BUNDLE_VERSION)

    rc = _agentcore_probe(root, "--prompt", _PROMPT, *bundle)

    assert rc == EXIT_OK, capsys.readouterr().err
    [clients] = runtime.built
    [sent] = clients.calls_to("InvokeAgentRuntime")
    baggage = dict(entry.split("=", 1) for entry in sent.params["baggage"].split(","))
    assert unquote(baggage["aws.agentcore.configbundle_arn"]) == BUNDLE_ARN
    assert unquote(baggage["aws.agentcore.configbundle_version"]) == BUNDLE_VERSION
    [trace] = _stored_probed(root)
    [record] = _manifest(_default_manifest(root))
    assert record["trace_id"] == trace.trace_id
    assert (record["config_bundle_arn"], record["config_bundle_version"]) == (
        BUNDLE_ARN,
        BUNDLE_VERSION,
    )


@pytest.mark.parametrize(
    ("bundle", "named"),
    [
        (
            ("--config-bundle-arn", "", "--config-bundle-version", ""),
            "--config-bundle-arn and --config-bundle-version are empty",
        ),
        (("--config-bundle-arn", ""), "--config-bundle-arn is empty"),
        (
            ("--config-bundle-arn", "", "--config-bundle-version", BUNDLE_VERSION),
            "--config-bundle-arn is empty",
        ),
        (
            ("--config-bundle-arn", BUNDLE_ARN, "--config-bundle-version", ""),
            "--config-bundle-version is empty",
        ),
    ],
    ids=["both", "arn-alone", "arn", "version"],
)
def test_an_agentcore_probe_with_an_empty_bundle_flag_is_refused_naming_it_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    capsys: pytest.CaptureFixture[str],
    bundle: tuple[str, ...],
    named: str,
) -> None:
    aws = _AwsSpy(monkeypatch)

    rc = _agentcore_probe(root, "--prompt", _PROMPT, *bundle)

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert f"{named}, so nothing was run" in err
    assert _PLAN not in err
    assert "Traceback" not in err
    assert aws.opened == []
    assert not (root / "records" / "agentcore").exists()


# --- the plan and --yes -----------------------------------------------------------------------


def test_an_agentcore_probe_prints_its_plan_before_any_request(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    printed_before_aws: list[str] = []

    def snapshot() -> None:
        seen = capsys.readouterr()
        printed_before_aws.append(seen.out + seen.err)

    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=1, on_build=snapshot))

    rc = _agentcore_probe(root, "--prompt", _PROMPT)

    assert rc == EXIT_OK, capsys.readouterr().err
    [plan] = printed_before_aws
    assert RUNTIME_ARN in plan
    assert "qualifier DEFAULT" in plan
    assert "up to 1 prompt" in plan
    assert "up to 1 prompts" not in plan
    assert "billed invocation" in plan
    assert "retried when the session is busy or the runtime throttles" in plan
    assert "no spans are read, so the replies are stored as probed, outcome-only traces" in plan
    assert "It stops at the first prompt that fails, and none of the replies are then" in plan
    assert str(_default_manifest(root)) in plan
    assert "configuration bundle" not in plan
    assert runtime.invocations == 1


def test_an_agentcore_probe_plan_names_the_configuration_bundle_it_sends(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    aws = _AwsSpy(monkeypatch)
    bundle = ("--config-bundle-arn", BUNDLE_ARN, "--config-bundle-version", BUNDLE_VERSION)

    rc = _agentcore_probe(root, "--prompt", _PROMPT, *bundle, yes=False)

    plan, refused, _ = capsys.readouterr().err.partition("nothing was sent: ")
    assert rc == EXIT_DOMAIN_ERROR
    assert refused
    assert f"configuration bundle {BUNDLE_ARN} at version {BUNDLE_VERSION}" in plan
    assert aws.opened == []


def test_an_agentcore_probe_without_yes_prints_the_plan_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    aws = _AwsSpy(monkeypatch)

    rc = _agentcore_probe(root, "--prompt", _PROMPT, yes=False)

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    plan, refused, refusal = err.partition("nothing was sent: ")
    assert refused
    assert RUNTIME_ARN in plan
    assert "qualifier DEFAULT" in plan
    assert "up to 1 prompt" in plan
    assert "this probe makes the billed AWS calls listed above, so it needs --yes" in refusal
    assert "Traceback" not in err
    assert aws.opened == []
    assert _stored_probed(root) == []
    assert not (root / "records" / "agentcore").exists()


# --- the billed-call budget -------------------------------------------------------------------


def test_an_agentcore_probe_stops_at_the_billed_call_budget(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(_BUDGET_VARIABLE, "1")
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=2))

    rc = _agentcore_probe(root, "--prompt", _PROMPT, "--prompt", "Perform a simple task.")

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert "BilledCallBudgetExceeded" in err
    assert _BUDGET_VARIABLE in err
    assert "Traceback" not in err
    assert runtime.invocations == 1
    assert _stored_probed(root) == []
    manifest = _default_manifest(root)
    assert str(manifest) in err
    records = _manifest(manifest)
    assert [record["error_code"] for record in records] == [None, "BilledCallBudgetExceeded"]
    assert [record["session_stop"] for record in records] == ["stopped", "not_started"]


def test_probe_preflight_reads_and_stops_do_not_spend_the_budget(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The one billed call is the invocation; the two preflight reads and the stop are free.
    monkeypatch.setenv(_BUDGET_VARIABLE, "1")
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=1))

    rc = _agentcore_probe(root, "--prompt", _PROMPT)

    assert rc == EXIT_OK, capsys.readouterr().err
    [clients] = runtime.built
    assert len(clients.calls) == 4
    assert len(clients.calls_to("StopRuntimeSession")) == 1
    assert len(_stored_probed(root)) == 1


def test_without_the_budget_variable_an_agentcore_probe_sends_every_prompt(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=3))
    prompts = ("one", "two", "three")

    rc = _agentcore_probe(root, *(arg for prompt in prompts for arg in ("--prompt", prompt)))

    assert rc == EXIT_OK, capsys.readouterr().err
    assert runtime.invocations == 3
    stored = _stored_probed(root)
    assert sorted(trace.input["prompt"] for trace in stored) == sorted(prompts)
    assert {trace.output["text"] for trace in stored} == {ANSWER}


@pytest.mark.parametrize("value", ["abc", "-1"])
def test_an_agentcore_probe_with_an_invalid_budget_variable_exits_1_naming_it(
    monkeypatch: pytest.MonkeyPatch, root: Path, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(_BUDGET_VARIABLE, value)
    aws = _AwsSpy(monkeypatch)

    rc = _agentcore_probe(root, "--prompt", _PROMPT)

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert f"{_BUDGET_VARIABLE}={value!r}" in err
    assert "Traceback" not in err
    assert aws.opened == []
    assert not (root / "records" / "agentcore").exists()


def test_an_invalid_budget_variable_is_refused_before_the_probes_store_check_and_plan(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(_BUDGET_VARIABLE, "abc")
    checks: list[str] = []
    _langfuse_store(monkeypatch, _CheckedLangfuse(on_check=lambda: checks.append("check")))
    aws = _AwsSpy(monkeypatch)

    # Without --yes: the variable is checked before the plan asks for it.
    rc = _agentcore_probe(root, "--prompt", _PROMPT, yes=False, backend="langfuse")

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert f"{_BUDGET_VARIABLE}='abc'" in err
    assert checks == []
    assert _PLAN not in err
    assert "needs --yes" not in err
    assert aws.opened == []


# --- the read timeout -------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "-1"])
def test_an_agentcore_probe_with_a_non_positive_read_timeout_exits_1_naming_it(
    monkeypatch: pytest.MonkeyPatch, root: Path, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=1))

    rc = _agentcore_probe(root, "--prompt", _PROMPT, "--read-timeout", value)

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert f"read_timeout_seconds={float(value)!r}" in err
    assert "Traceback" not in err
    for clients in runtime.built:
        assert clients.built == []
        assert clients.calls == []
    assert _stored_probed(root) == []
    assert not (root / "records" / "agentcore").exists()


@pytest.mark.parametrize("value", ["inf", "nan", "1e12"], ids=["infinite", "nan", "overlong"])
def test_an_agentcore_probe_with_an_infinite_or_overlong_read_timeout_exits_1_naming_it(
    monkeypatch: pytest.MonkeyPatch, root: Path, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=1))

    rc = _agentcore_probe(root, "--prompt", _PROMPT, "--read-timeout", value)

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert f"read_timeout_seconds={float(value)!r}" in err
    assert "positive, finite number of seconds" in err
    assert "Traceback" not in err
    assert runtime.invocations == 0
    assert _stored_probed(root) == []


def test_the_read_timeout_flag_reaches_the_probe_invocation_client(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=1))

    rc = _agentcore_probe(root, "--prompt", _PROMPT, "--read-timeout", "60")

    assert rc == EXIT_OK, capsys.readouterr().err
    [clients] = runtime.built
    [invoker] = [built.client for built in clients.built if built.view]
    assert invoker.meta.config.read_timeout == 60
    assert runtime.invocations == 1


# --- cassette modes ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["auto", "record"])
def test_an_agentcore_probe_runs_live_in_auto_and_record_modes(
    monkeypatch: pytest.MonkeyPatch, root: Path, mode: str, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(_MODE_VARIABLE, mode)
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=1))

    rc = _agentcore_probe(root, "--prompt", _PROMPT)

    assert rc == EXIT_OK, capsys.readouterr().err
    assert runtime.invocations == 1
    assert len(_stored_probed(root)) == 1


# --- failures and the run manifest ------------------------------------------------------------


def test_a_failed_agentcore_probe_stores_no_trace_and_prints_the_manifest_path(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=2, fail_at=1))

    rc = _agentcore_probe(root, "--prompt", _PROMPT, "--prompt", "Perform a simple task.")

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert "RuntimeClientError" in err
    assert "Traceback" not in err
    assert _stored_probed(root) == []
    manifest = _default_manifest(root)
    assert f"the run manifest, one record per run started, is {manifest}" in err
    records = _manifest(manifest)
    assert [record["error_code"] for record in records] == [None, "RuntimeClientError"]
    assert [record["session_stop"] for record in records] == ["stopped", "stopped"]
    [clients] = runtime.built
    clients.assert_no_pending_responses()


def test_an_agentcore_probe_whose_invocation_view_fails_to_close_still_stores_every_reply(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=1))
    real_close = StubbedAwsClients.close

    def close(self: StubbedAwsClients) -> None:
        real_close(self)
        if self.is_invocation_view:
            raise OSError("the connection pool could not be closed")

    monkeypatch.setattr(StubbedAwsClients, "close", close)

    rc = _agentcore_probe(root, "--prompt", _PROMPT)

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert "could not close the invocation clients (OSError); the runs' results stand" in err
    assert "Traceback" not in err
    # The prompt answered and its session was stopped before the view was closed.
    [trace] = _stored_probed(root)
    assert trace.output == {"text": ANSWER}
    [record] = _manifest(_default_manifest(root))
    assert record["trace_id"] == trace.trace_id
    assert record["session_stop"] == "stopped"
    assert runtime.invocations == 1


def test_an_agentcore_probe_keeps_its_run_records_under_records_agentcore_and_prints_the_path(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=1))
    assert not (root / "records" / "agentcore").exists()

    rc = _agentcore_probe(root, "--prompt", _PROMPT)

    out = capsys.readouterr().out
    assert rc == EXIT_OK
    manifest = _default_manifest(root)
    [record] = _manifest(manifest)
    assert record["runtime_arn"] == RUNTIME_ARN
    assert record["trace_id"].startswith("probe-")
    assert str(manifest) in out


def test_log_dir_overrides_the_probe_run_record_directory(
    monkeypatch: pytest.MonkeyPatch, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=1))
    logs = tmp_path / "logs"

    rc = _agentcore_probe(root, "--prompt", _PROMPT, "--log-dir", str(logs))

    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert len(_manifest(manifest_path(logs))) == 1
    assert str(manifest_path(logs)) in out
    assert not (root / "records" / "agentcore").exists()


# --- the trace store --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("client", "named"),
    [
        (lambda: _CheckedLangfuse(accepted=False), "Langfuse reachability check failed"),
        (
            lambda: _CheckedLangfuse(unreachable=ConnectionError("connection refused")),
            "Langfuse reachability check failed",
        ),
        (None, "LANGFUSE_SECRET_KEY"),
    ],
    ids=["keys-refused", "host-unreachable", "secret-key-unset"],
)
def test_an_agentcore_probe_with_an_unreachable_trace_store_exits_1_with_no_plan_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    client: Callable[[], _CheckedLangfuse] | None,
    named: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = None if client is None else client()
    if store is None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-test")
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    else:
        _langfuse_store(monkeypatch, store)
    aws = _AwsSpy(monkeypatch)

    rc = _agentcore_probe(root, "--prompt", _PROMPT, backend="langfuse")

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert named in err
    assert "Traceback" not in err
    assert _PLAN not in err
    assert aws.opened == []
    assert not (root / "records" / "agentcore").exists()
    if store is not None:
        assert store.stored_traces() == 0


def test_an_agentcore_probe_checks_its_trace_store_before_the_plan(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[str] = []

    def checked() -> None:
        # What was printed by the time of the check; read here, so the plan shows up after it.
        seen = capsys.readouterr()
        events.append("plan" if _PLAN in seen.err else "check")

    def built() -> None:
        seen = capsys.readouterr()
        if _PLAN in seen.err:
            events.append("plan")
        events.append("aws")

    store = _CheckedLangfuse(on_check=checked)
    _langfuse_store(monkeypatch, store)
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(replies=1, on_build=built))

    rc = _agentcore_probe(root, "--prompt", _PROMPT, backend="langfuse")

    assert rc == EXIT_OK, capsys.readouterr().err
    # The check, then the plan, then the AWS clients; storing the trace checks once more.
    assert events[:3] == ["check", "plan", "aws"]
    assert runtime.invocations == 1
    assert store.stored_traces() == 1
