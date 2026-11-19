# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation generate --run --executor agentcore`: each input goes through an
AgentCore Runtime agent.

Live runs go through stubbed clients over fake credentials and the fake span reader, never AWS.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("boto3")

import boto3.session
import botocore.session

import calibrated_agent_evaluation.agentcore
from _agentcore_runtime_fakes import (
    BUNDLE_ARN,
    BUNDLE_VERSION,
    REGION,
    RUNTIME_ARN,
    StubbedAwsClients,
    add_reader_response,
    register_fake_agentcore_span_reader,
)
from _agentcore_runtime_queue import hide_ambient_aws, queue_preflight, queue_run
from calibrated_agent_evaluation.agentcore import AgentCoreRuntimeExecutor, AwsClients
from calibrated_agent_evaluation.agentcore.run_manifest import RUN_RECORD_STORE, manifest_path
from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend, LangfuseLike
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.args import AGENTCORE_PAYLOAD_FORMATS
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.stages.generate import TemplateRenderer
from calibrated_agent_evaluation.probe.adapter import PAYLOAD_FORMATS
from calibrated_agent_evaluation.schema.errors import BackendError
from calibrated_agent_evaluation.schema.types import Trace
from test_backends_langfuse import StubLangfuseClient

_REPO = Path(__file__).resolve().parent.parent
_CASSETTE = _REPO / "fixtures" / "raw" / "cassettes" / "agentcore_runtime_CLM-4471.cassette.json"
_FRAME = _REPO / "fixtures" / "raw" / "frames" / "agentcore_CLM-4471.frame.json"
_STRANDS_CASSETTE = _REPO / "fixtures" / "raw" / "cassettes" / "claimscrew_strands.cassette.json"
_MODE_VARIABLE = "CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE"
_BUDGET_VARIABLE = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS"
_FORMATS = ("generic_message", "openai", "agentcore_sse", "nested_agentcore")
# The committed frame's one row, the only cell whose prompt the committed cassette holds a run for.
_CELL = {
    "feature": "claim_adjudication",
    "persona": "P2_homeowner_absent",
    "scenario": "S2_vacancy_exclusion",
}
_OTHER_CELL = {
    "feature": "claim_adjudication",
    "persona": "P1_homeowner_present",
    "scenario": "S1_covered_peril",
}
_ANSWER = "Denied: the vacancy exclusion applies."
_RUN = ("generate", "--run", "--executor", "agentcore")
_ARN = ("--runtime-arn", RUNTIME_ARN)
_SSE = ("--payload-format", "agentcore_sse")
_LIVE = (*_RUN, *_ARN, *_SSE)


@pytest.fixture(autouse=True)
def _no_ambient_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hide_ambient_aws(monkeypatch, tmp_path)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "store"


def _generate(root: Path, *flags: str) -> int:
    return main(["--root", str(root), *_LIVE, *flags])


def _generate_into_langfuse(root: Path, *flags: str) -> int:
    return main(["--backend", "langfuse", "--root", str(root), *_LIVE, *flags])


def _replay_committed(root: Path, *flags: str, frame: Path = _FRAME) -> int:
    return _generate(root, "--cassette", str(_CASSETTE), "--frame", str(frame), *flags)


def _frame_file(directory: Path, *cells: dict[str, str], frame: str = "discovery") -> Path:
    path = directory / f"{frame}-{len(cells)}.frame.json"
    rows = [{"cell": dict(cell), "frame": frame, "weight": 1.0} for cell in cells]
    path.write_text(json.dumps({"coverage_strength": 2, "frame": frame, "rows": rows}))
    return path


def _stored(root: Path) -> list[Trace]:
    if not root.exists():
        return []
    with LocalBackend(root=root) as backend:
        return backend.query()


def _manifest(path: Path) -> list[dict[str, Any]]:
    """The run records in the run manifest at ``path``, as JSON."""
    records = RUN_RECORD_STORE.load(path.parent, fmt="json")
    return [record.model_dump(mode="json") for record in records]


def _default_manifest(root: Path) -> Path:
    return manifest_path(root / "records" / "agentcore")


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


def _no_span_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make this install offer no AgentCore span reader, whether or not one is registered."""
    monkeypatch.delattr(calibrated_agent_evaluation.agentcore, "AgentCoreSpanReader", raising=False)
    monkeypatch.delitem(
        calibrated_agent_evaluation.agentcore._LAZY, "AgentCoreSpanReader", raising=False
    )


class _StubbedRuntime:
    """Stands in for ``AwsClients.from_profile`` with stubbed clients that answer like a runtime.

    Each set answers the preflight, then ``runs`` times an invocation, one fake span read and a
    stop. ``on_build`` runs as each set is built, before any call goes through it.
    """

    def __init__(
        self, runs: int, *, qualifier: str = "DEFAULT", on_build: Callable[[], None] | None = None
    ) -> None:
        self.runs = runs
        self.qualifier = qualifier
        self.on_build = on_build
        self.built: list[StubbedAwsClients] = []
        self.profiles: list[str | None] = []

    def __call__(
        self, *, region: str | None = None, profile: str | None = None, **options: Any
    ) -> StubbedAwsClients:
        if self.on_build is not None:
            self.on_build()
        self.profiles.append(profile)
        clients = StubbedAwsClients(region=region or REGION, **options)
        queue_preflight(clients, self.qualifier)
        for _ in range(self.runs):
            queue_run(clients, _ANSWER, self.qualifier)
            add_reader_response(clients)
        self.built.append(clients)
        return clients

    @property
    def invocations(self) -> int:
        return sum(len(clients.calls_to("InvokeAgentRuntime")) for clients in self.built)


def _stubbed_runtime(monkeypatch: pytest.MonkeyPatch, runtime: _StubbedRuntime) -> _StubbedRuntime:
    """Route the CLI's AWS clients through ``runtime`` and offer the fake span reader."""
    monkeypatch.setattr(AwsClients, "from_profile", runtime)
    register_fake_agentcore_span_reader(monkeypatch)
    return runtime


class _ExecutorSpy:
    """Keeps the keyword arguments of each AgentCore Runtime executor built, each built as usual."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.options: list[dict[str, Any]] = []
        real = AgentCoreRuntimeExecutor.__init__

        def spied(executor: AgentCoreRuntimeExecutor, *args: Any, **kwargs: Any) -> None:
            self.options.append(dict(kwargs))
            real(executor, *args, **kwargs)

        monkeypatch.setattr(AgentCoreRuntimeExecutor, "__init__", spied)


class _CheckedLangfuse(StubLangfuseClient):
    """A Langfuse client: ``auth_check`` runs ``on_check``, then returns or raises ``answer``."""

    def __init__(
        self, answer: bool | Exception = True, on_check: Callable[[], None] | None = None
    ) -> None:
        super().__init__()
        self.answer = answer
        self.on_check = on_check
        self.checks = 0

    def auth_check(self) -> bool:
        self.checks += 1
        if self.on_check is not None:
            self.on_check()
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer

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


# --- routing through the runtime --------------------------------------------------------------


def test_a_replay_stores_one_trace_per_frame_row_offline_and_records_it_under_records_agentcore(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(_MODE_VARIABLE, "replay")
    monkeypatch.setenv("AWS_PROFILE", "calibrated-agent-evaluation-no-such-profile")
    _no_span_reader(monkeypatch)
    aws = _AwsSpy(monkeypatch)
    assert not (root / "records" / "agentcore").exists()

    rc = _replay_committed(root)

    captured = capsys.readouterr()
    assert rc == EXIT_OK, captured.err
    [trace] = _stored(root)
    assert trace.frame == "discovery"
    assert trace.ownership == "owned"
    assert trace.taxonomy_cell == _CELL
    assert trace.input == {"prompt": TemplateRenderer().render(_CELL)}
    assert {span.agent for span in trace.spans} >= {"supervisor", "payout"}
    manifest = _default_manifest(root)
    [line] = _manifest(manifest)
    assert line["trace_id"] == trace.trace_id
    assert line["replayed"] is True
    assert line["runtime_arn"] == RUNTIME_ARN
    assert str(manifest) in captured.out
    # A replay needs no span reader, prints no plan and asks for no --yes.
    output = captured.out + captured.err
    assert "up to" not in output
    assert "--yes" not in output
    assert aws.opened == []


def test_the_strands_executor_path_builds_no_agentcore_executor_and_writes_no_manifest(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("strands")
    monkeypatch.setenv(_MODE_VARIABLE, "replay")
    aws = _AwsSpy(monkeypatch)
    executors = _ExecutorSpy(monkeypatch)
    argv = ["--root", str(root), "generate", "--run", "--executor", "strands"]
    argv += ["--agent", "os:getcwd", "--cassette", str(_STRANDS_CASSETTE), "--frame", str(_FRAME)]

    rc = main(argv)

    # The Strands executor ran the task, and refused the agent, which is no topology builder.
    assert rc == EXIT_DOMAIN_ERROR
    assert "strands run failed" in capsys.readouterr().err
    assert executors.options == []
    assert not (root / "records" / "agentcore").exists()
    assert aws.opened == []


@pytest.mark.parametrize("variable", [None, "record"], ids=["mode-unset", "record-variable"])
def test_a_live_cli_run_sends_every_input_through_the_clients_its_flags_build_with_no_cassette(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    tmp_path: Path,
    variable: str | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if variable is not None:
        monkeypatch.setenv(_MODE_VARIABLE, variable)
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(runs=2))
    executors = _ExecutorSpy(monkeypatch)
    frame = _frame_file(tmp_path, _CELL, _OTHER_CELL)

    # No budget variable is set, so no billed-call budget stops the run.
    rc = _generate(root, "--frame", str(frame), "--profile", "p", "--read-timeout", "60", "--yes")

    assert rc == EXIT_OK, capsys.readouterr().err
    [clients] = runtime.built
    assert len(clients.calls_to("InvokeAgentRuntime")) == 2
    stored = _stored(root)
    assert len(stored) == 2
    assert {trace.output["text"] for trace in stored} == {_ANSWER}
    assert {trace.frame for trace in stored} == {"discovery"}
    # The CLI builds the clients from --profile and gives the executor the clients alone.
    assert runtime.profiles == ["p"]
    [options] = executors.options
    assert options["clients"] is clients
    assert options.get("profile") is None
    assert options.get("cassette_path") is None
    assert options.get("cassette_mode") is None
    [invoker] = [built.client for built in clients.built if built.view]
    assert invoker.meta.config.read_timeout == 60
    assert invoker.meta.config.connect_timeout == 10


# --- the billed-call budget -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("budget", "cells", "named", "reads"),
    [
        # One invocation and the reader's one billed read use it up, so the second is refused.
        pytest.param(
            "2", (_CELL, _OTHER_CELL), ("bedrock-agentcore:InvokeAgentRuntime",), 1, id="invocation"
        ),
        # The invocation uses the only billed call, so the span reader's read is refused.
        pytest.param(
            "1", (_CELL,), ("cloudwatchomni:StartTelemetryQuery", "span reader"), 0, id="span-read"
        ),
    ],
)
def test_the_executor_and_the_span_reader_stop_at_one_shared_billed_call_budget(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    tmp_path: Path,
    budget: str,
    cells: tuple[dict[str, str], ...],
    named: tuple[str, ...],
    reads: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(_BUDGET_VARIABLE, budget)
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(runs=len(cells)))

    rc = _generate(root, "--frame", str(_frame_file(tmp_path, *cells)), "--yes")

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    for text in ("BilledCallBudgetExceeded", _BUDGET_VARIABLE, *named):
        assert text in err
    assert "Traceback" not in err
    [clients] = runtime.built
    assert len(clients.calls_to("InvokeAgentRuntime")) == 1
    assert len(clients.calls_to("StartTelemetryQuery")) == reads
    assert len(clients.calls_to("StopRuntimeSession")) == 1
    assert _stored(root) == []


@pytest.mark.parametrize("yes", [False, True], ids=["without-yes", "with-yes"])
@pytest.mark.parametrize(
    "budget", ["abc", "-1", None], ids=["budget-abc", "budget-negative", "no-span-reader"]
)
def test_an_invalid_budget_or_a_missing_span_reader_is_refused_before_the_store_check_and_the_plan(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    budget: str | None,
    yes: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if budget is None:
        # Without a budget variable to refuse, the run has no span reader instead.
        _no_span_reader(monkeypatch)
        named = ["live runs need the AgentCore span reader"]
        named += ["offline runs need a cassette in replay mode"]
    else:
        monkeypatch.setenv(_BUDGET_VARIABLE, budget)
        register_fake_agentcore_span_reader(monkeypatch)
        named = [_BUDGET_VARIABLE]
    store = _CheckedLangfuse()
    _langfuse_store(monkeypatch, store)
    aws = _AwsSpy(monkeypatch)

    rc = _generate_into_langfuse(root, "--frame", str(_FRAME), *(["--yes"] if yes else []))

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    for text in named:
        assert text in err
    assert "Traceback" not in err
    # Known before anything else is asked: no store check, no plan and no call for --yes.
    assert store.checks == 0
    assert "this run can call AWS" not in err
    assert "needs --yes" not in err
    assert aws.opened == []


# --- the trace store check, the plan and --yes ------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [False, ConnectionError("connection refused")],
    ids=["keys-refused", "host-unreachable"],
)
def test_a_live_agentcore_generate_with_an_unreachable_trace_store_exits_1_before_any_aws_request(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    answer: bool | Exception,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = _CheckedLangfuse(answer)
    _langfuse_store(monkeypatch, store)
    register_fake_agentcore_span_reader(monkeypatch)
    aws = _AwsSpy(monkeypatch)

    # Auto mode, with --yes: every call the plan names could go ahead but for the store.
    rc = _generate_into_langfuse(root, "--frame", str(_FRAME), "--yes")

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert "BackendError" in err
    assert "Langfuse reachability check failed" in err
    assert "Traceback" not in err
    assert "this run can call AWS" not in err
    assert aws.opened == []
    assert not (root / "records" / "agentcore").exists()
    assert store.stored_traces() == 0


def test_a_live_run_checks_its_trace_store_then_prints_its_plan_before_any_aws_request(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[str] = []
    plans: list[str] = []

    def record(event: str) -> Callable[[], None]:
        def seen() -> None:
            # Reading the output here puts each print before the first event after it.
            printed = capsys.readouterr()
            if "this run can call AWS" in printed.out + printed.err:
                events.append("plan")
                plans.append(printed.out + printed.err)
            events.append(event)

        return seen

    store = _CheckedLangfuse(on_check=record("check"))
    _langfuse_store(monkeypatch, store)
    runtime = _StubbedRuntime(runs=1, qualifier="live", on_build=record("aws"))
    _stubbed_runtime(monkeypatch, runtime)

    rc = _generate_into_langfuse(root, "--frame", str(_FRAME), "--qualifier", "live", "--yes")

    assert rc == EXIT_OK, capsys.readouterr().err
    # The check, then the plan, then the one set of AWS clients; storing checks the store again.
    assert events[:3] == ["check", "plan", "aws"]
    assert events.count("aws") == 1
    [plan] = plans
    assert RUNTIME_ARN in plan
    assert "qualifier live" in plan
    assert "up to 1 input" in plan
    assert "up to 1 inputs" not in plan
    assert runtime.invocations == 1
    assert store.stored_traces() == 1


def test_a_live_capable_run_without_yes_exits_1_before_any_aws_request(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    aws = _AwsSpy(monkeypatch)
    register_fake_agentcore_span_reader(monkeypatch)

    # Auto mode: every input is in the cassette, but a missing one would go live.
    rc = _replay_committed(root)

    captured = capsys.readouterr()
    assert rc == EXIT_DOMAIN_ERROR
    assert RUNTIME_ARN in captured.err
    assert "up to 1 input" in captured.err
    assert "--yes" in captured.err.split("up to 1 input", 1)[1]
    assert "Traceback" not in captured.err
    assert aws.opened == []
    assert _stored(root) == []
    assert not (root / "records" / "agentcore").exists()


def test_a_replayed_agentcore_generate_skips_the_trace_store_check(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(_MODE_VARIABLE, "replay")
    replayed_before_check: list[bool] = []

    def checked() -> None:
        replayed_before_check.append(_default_manifest(root).exists())

    store = _CheckedLangfuse(on_check=checked)
    _langfuse_store(monkeypatch, store)

    rc = _generate_into_langfuse(root, "--cassette", str(_CASSETTE), "--frame", str(_FRAME))

    assert rc == EXIT_OK, capsys.readouterr().err
    # A replay sends nothing billed, so the store is first checked when the trace is stored.
    assert replayed_before_check == [True]
    assert store.stored_traces() == 1


# --- refused before the executor or any AWS request -------------------------------------------


@pytest.mark.parametrize(
    ("bundle", "named"),
    [
        (
            ("--config-bundle-arn", "", "--config-bundle-version", ""),
            "--config-bundle-arn and --config-bundle-version are empty",
        ),
        (
            ("--config-bundle-arn", BUNDLE_ARN, "--config-bundle-version", ""),
            "--config-bundle-version is empty",
        ),
    ],
    ids=["both", "version"],
)
def test_an_empty_config_bundle_flag_exits_1_naming_it_before_any_aws_request(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    capsys: pytest.CaptureFixture[str],
    bundle: tuple[str, ...],
    named: str,
) -> None:
    aws = _AwsSpy(monkeypatch)
    register_fake_agentcore_span_reader(monkeypatch)

    rc = _generate(root, "--frame", str(_FRAME), *bundle, "--yes")

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert f"{named}, so nothing was run" in err
    assert "up to" not in err
    assert "Traceback" not in err
    assert aws.opened == []
    assert _stored(root) == []
    assert not (root / "records" / "agentcore").exists()


_AGENTCORE_ONLY_FLAGS: list[tuple[str, ...]] = [
    _ARN,
    _SSE,
    ("--qualifier", "live"),
    ("--config-bundle-arn", BUNDLE_ARN),
    ("--config-bundle-version", BUNDLE_VERSION),
    ("--read-timeout", "60"),
    ("--profile", "calibrated-agent-evaluation-test"),
    ("--log-dir", "agentcore-logs"),
    ("--yes",),
]
_AGENT_FLAGS = [("--agent", "myagents.orderbot:build"), ("--agent-model", "bedrock/some-model")]
_STRANDS = ("generate", "--run", "--executor", "strands", "--agent", "os:getcwd")
# Each case: the cassette mode variable, the arguments (all run with --frame) and what stderr names.
_REFUSED = [
    pytest.param(None, (*_RUN, *_SSE), ("--runtime-arn",), id="no-runtime-arn"),
    pytest.param(None, (*_RUN, *_ARN), ("--payload-format", *_FORMATS), id="no-payload-format"),
    *(
        pytest.param(
            "replay",
            (*_LIVE, "--cassette", str(_CASSETTE), *flag),
            (flag[0], "the agent under test is the runtime"),
            id=f"{flag[0][2:]}-with-agentcore",
        )
        for flag in _AGENT_FLAGS
    ),
    *(
        pytest.param(
            None, (*_STRANDS, *flag), (flag[0], "--executor agentcore"), id=f"{flag[0][2:]}-strands"
        )
        for flag in _AGENTCORE_ONLY_FLAGS
    ),
    pytest.param(
        None,
        ("generate", *_ARN, "--agent", "os:getcwd"),
        ("--runtime-arn", "--executor agentcore"),
        id="runtime-arn-without-run",
    ),
    pytest.param(
        None,
        ("generate", "--executor", "agentcore", *_ARN, *_SSE, "--yes"),
        ("--runtime-arn", "--payload-format", "--yes", "without --run"),
        id="runtime-arn-format-yes-without-run",
    ),
    pytest.param(
        None,
        ("generate", "--executor", "agentcore", "--log-dir", "agentcore-logs"),
        ("--log-dir", "without --run"),
        id="log-dir-without-run",
    ),
    pytest.param("replay", (*_LIVE, "--yes"), ("--cassette",), id="replay-without-cassette"),
    pytest.param(
        "rewind", (*_LIVE, "--cassette", str(_CASSETTE)), (_MODE_VARIABLE,), id="invalid-mode"
    ),
]


# A flag the command line gets wrong exits 2; a cassette mode set in the environment that this
# run cannot honor is a refusal, 1.
_ENVIRONMENT_REFUSALS = frozenset({"replay-without-cassette", "invalid-mode"})
_REFUSED_WITH_CODE = [
    pytest.param(
        *case.values,
        EXIT_DOMAIN_ERROR if case.id in _ENVIRONMENT_REFUSALS else EXIT_USAGE,
        id=case.id,
    )
    for case in _REFUSED
]


@pytest.mark.parametrize(("mode", "argv", "named", "code"), _REFUSED_WITH_CODE)
def test_a_generate_it_cannot_run_is_refused_naming_why_before_any_executor_or_aws_request(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    mode: str | None,
    argv: tuple[str, ...],
    named: tuple[str, ...],
    code: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if mode is not None:
        monkeypatch.setenv(_MODE_VARIABLE, mode)
    aws = _AwsSpy(monkeypatch)
    executors = _ExecutorSpy(monkeypatch)

    rc = main(["--root", str(root), *argv, "--frame", str(_FRAME)])

    err = capsys.readouterr().err
    assert rc == code
    for text in named:
        assert text in err
    assert "Traceback" not in err
    assert executors.options == []
    assert aws.opened == []
    assert _stored(root) == []


def test_the_cli_payload_format_names_equal_the_probes() -> None:
    assert tuple(payload.name for payload in PAYLOAD_FORMATS) == AGENTCORE_PAYLOAD_FORMATS


def test_a_production_frame_is_refused_before_any_aws_request(
    monkeypatch: pytest.MonkeyPatch, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    register_fake_agentcore_span_reader(monkeypatch)
    aws = _AwsSpy(monkeypatch)
    frame = _frame_file(tmp_path, _CELL, frame="production-representative")

    rc = _generate(root, "--frame", str(frame), "--yes")

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert str(frame) in err
    assert "discovery" in err
    assert aws.opened == []
    assert _stored(root) == []


# --- the run manifest -------------------------------------------------------------------------


def test_log_dir_overrides_the_manifest_directory(
    monkeypatch: pytest.MonkeyPatch, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(_MODE_VARIABLE, "replay")
    logs = tmp_path / "logs"

    rc = _replay_committed(root, "--log-dir", str(logs))

    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert len(_manifest(manifest_path(logs))) == 1
    assert str(manifest_path(logs)) in out
    assert not (root / "records" / "agentcore").exists()


def _unwritable_store(_backend: LocalBackend, _trace: Trace) -> None:
    raise BackendError("the trace store could not be written")


@pytest.mark.parametrize(
    ("cell", "put_trace", "named", "recorded"),
    [
        # The cassette holds no run for this cell's prompt.
        (_OTHER_CELL, LocalBackend.put_trace, "cassette miss", {"error_code": "CassetteMiss"}),
        (_CELL, _unwritable_store, "the trace store could not be written", {"replayed": True}),
    ],
    ids=["cassette-miss", "traces-not-stored"],
)
def test_a_failed_cli_run_still_prints_the_manifest_path(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    tmp_path: Path,
    cell: dict[str, str],
    put_trace: Callable[[LocalBackend, Trace], None],
    named: str,
    recorded: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(_MODE_VARIABLE, "replay")
    monkeypatch.setattr(LocalBackend, "put_trace", put_trace)

    rc = _replay_committed(root, frame=_frame_file(tmp_path, cell))

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert named in err
    assert "Traceback" not in err
    manifest = _default_manifest(root)
    assert str(manifest) in err
    [line] = _manifest(manifest)
    assert {key: line[key] for key in recorded} == recorded


@pytest.mark.parametrize(
    ("cell", "named"),
    [(_OTHER_CELL, ("cassette miss",)), (_CELL, ())],
    ids=["run-failed", "run-succeeded"],
)
def test_a_cli_run_whose_manifest_cannot_be_written_claims_no_record_and_names_why(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    tmp_path: Path,
    cell: dict[str, str],
    named: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(_MODE_VARIABLE, "replay")
    # A file stands where the log directory would go.
    logs = tmp_path / "logs"
    logs.write_text("not a directory", encoding="utf-8")

    rc = _replay_committed(root, "--log-dir", str(logs), frame=_frame_file(tmp_path, cell))

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert "Traceback" not in err
    for text in named:
        assert text in err
    assert "one record per run started" not in err
    assert "holds no record of the run that started" in err
    assert f"could not create the log directory of the run manifest {manifest_path(logs)}" in err
    assert logs.read_text(encoding="utf-8") == "not a directory"
    # A run whose record can't be kept returns no trace, so nothing is stored.
    assert _stored(root) == []


def test_a_second_cli_run_appends_to_the_manifest_and_dump_still_succeeds(
    monkeypatch: pytest.MonkeyPatch, root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(_MODE_VARIABLE, "replay")

    assert _replay_committed(root) == EXIT_OK
    assert _replay_committed(root) == EXIT_OK

    lines = _manifest(_default_manifest(root))
    assert len(lines) == 2
    assert all(line["replayed"] for line in lines)
    rc = main(["--root", str(root), "dump", str(tmp_path / "dump")])
    assert rc == EXIT_OK, capsys.readouterr().err


def test_a_live_cli_run_whose_invocation_view_fails_to_close_still_stores_every_trace(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _stubbed_runtime(monkeypatch, _StubbedRuntime(runs=1))
    real_close = StubbedAwsClients.close

    def close(self: StubbedAwsClients) -> None:
        real_close(self)
        if self.is_invocation_view:
            raise OSError("the connection pool could not be closed")

    monkeypatch.setattr(StubbedAwsClients, "close", close)

    rc = _generate(root, "--frame", str(_FRAME), "--yes")

    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert "could not close the invocation clients (OSError); the runs' results stand" in err
    assert "Traceback" not in err
    # The run finished and its session was stopped before the view was closed.
    [trace] = _stored(root)
    assert trace.output["text"] == _ANSWER
    [line] = _manifest(_default_manifest(root))
    assert line["trace_id"] == trace.trace_id
    assert line["session_stop"] == "stopped"
    assert runtime.invocations == 1
