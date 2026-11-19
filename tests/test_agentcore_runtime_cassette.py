# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""An AgentCore Runtime run is recorded once into a cassette and replayed offline, field for field,
with no AWS client, credential or connection; a cassette that can't serve a run is refused by
path, and one executor never hands two tasks the same recorded run."""

from __future__ import annotations

import itertools
import json
import math
import os
import socket
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")

import boto3.session
import botocore.session

from _agentcore_runtime_fakes import (
    BUNDLE_ARN,
    BUNDLE_VERSION,
    RUNTIME_ARN,
    FakeRunSpanReader,
    StubbedAwsClients,
    clm_4471_spans,
    json_body,
    sse_body,
    stop_params,
)
from _helpers import load_corpus_trace
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.agentcore import (
    AgentCoreRuntimeExecutor,
    AgentCoreRuntimeTask,
    RunRecord,
    RuntimeInvocationError,
    RuntimeTarget,
)
from calibrated_agent_evaluation.agentcore.run_manifest import RUN_RECORD_STORE
from calibrated_agent_evaluation.agentcore.run_spans import RunSpanQuery
from calibrated_agent_evaluation.agentcore.runtime_cassette import decode_entry, encode_entry
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.execute import ExecutionError, Mode
from calibrated_agent_evaluation.execute.cassette import (
    Cassette,
    CassetteMiss,
    CassetteModelMismatch,
    fingerprint,
)
from calibrated_agent_evaluation.schema.errors import SchemaError
from calibrated_agent_evaluation.schema.serde import to_dict, trace_from_dict
from calibrated_agent_evaluation.schema.types import Span, Trace
from calibrated_agent_evaluation.schema.validate import validate

_REPO = Path(__file__).resolve().parents[1]
_COMMITTED_CASSETTE = (
    _REPO / "fixtures" / "raw" / "cassettes" / "agentcore_runtime_CLM-4471.cassette.json"
)
_COMMITTED_TRACE = _REPO / "fixtures" / "expected" / "agentcore_runtime" / "CLM-4471.trace.json"
_RUNTIME_ID = "calibrated_agent_evaluation_fixture-AbCdEf1234"
_WHEN = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
_CLM_4471 = load_corpus_trace("CLM-4471")
_NARRATIVE = str(_CLM_4471.input["narrative"])
_CELL = dict(_CLM_4471.taxonomy_cell)
_FIVE_AGENTS = {"supervisor", "intake", "policy-checker", "fraud-scorer", "payout"}
_ANSWER_PARTS = ("Denied: ", "the vacancy exclusion applies.")
_ANSWER = "".join(_ANSWER_PARTS)
_ANSWER_EVENTS = [{"response": part} for part in _ANSWER_PARTS]
_RETURNED_TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
_OTHER_BUNDLE_VERSION = "9a1b2c3d-4e5f-4a6b-8c7d-0e1f2a3b4c5d"
_OTHER_PROMPT = "Is water damage from a burst pipe covered while the house is empty?"
_SENTINEL = "SENTINEL-7f3a"
# What the narrative prompt is recorded under in the default payload format.
_NARRATIVE_REQUEST = {
    "executor": "agentcore-runtime",
    "payload_format": "agentcore_sse",
    "builder": None,
    "prompt": _NARRATIVE,
}


@pytest.fixture(autouse=True)
def _no_ambient_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hide the machine's AWS setup (profile, credentials, config files) and cassette mode."""
    for name in list(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", raising=False)
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", raising=False)
    empty = tmp_path / "empty-aws-config"
    empty.write_text("")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(empty))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(empty))


@pytest.fixture
def cassette(tmp_path: Path) -> Path:
    return tmp_path / "runtime.cassette.json"


@pytest.fixture
def recorded(cassette: Path, tmp_path: Path) -> Trace:
    """The narrative's run, recorded into ``cassette``, with its run record in ``record/``."""
    return _record(cassette, tmp_path / "record")


def _clock() -> Callable[[], datetime]:
    """A UTC clock that moves on one second each time it is read."""
    ticks = itertools.count(1)
    return lambda: _WHEN + timedelta(seconds=next(ticks))


def _target(**overrides: object) -> RuntimeTarget:
    fields: dict[str, object] = {"runtime_arn": RUNTIME_ARN, "payload_format": "agentcore_sse"}
    return RuntimeTarget(**(fields | overrides))  # pyright: ignore[reportArgumentType]


def _bundled(version: str = BUNDLE_VERSION) -> RuntimeTarget:
    return _target(config_bundle_arn=BUNDLE_ARN, config_bundle_version=version)


def _task(**overrides: Any) -> AgentCoreRuntimeTask:
    fields: dict[str, Any] = {"prompt": _NARRATIVE, "taxonomy_cell": dict(_CELL)}
    return AgentCoreRuntimeTask(**(fields | overrides))


def _preflight(clients: StubbedAwsClients) -> None:
    """Queue the two read-only preflight answers: endpoint DEFAULT serves version 24."""
    control = clients.stub("bedrock-agentcore-control")
    ready = {"agentRuntimeArn": RUNTIME_ARN, "status": "READY"}
    ready |= {"createdAt": _WHEN, "lastUpdatedAt": _WHEN}
    control.add_response(
        "get_agent_runtime_endpoint",
        {
            **ready,
            "agentRuntimeEndpointArn": f"{RUNTIME_ARN}/runtime-endpoint/DEFAULT",
            "name": "DEFAULT",
            "id": "DEFAULT",
            "liveVersion": "24",
        },
        {"agentRuntimeId": _RUNTIME_ID, "endpointName": "DEFAULT"},
    )
    control.add_response(
        "get_agent_runtime",
        {
            **ready,
            "agentRuntimeName": "calibrated_agent_evaluation_fixture",
            "agentRuntimeId": _RUNTIME_ID,
            "agentRuntimeVersion": "24",
            "roleArn": "arn:aws:iam::111122223333:role/calibrated-agent-evaluation-fixture-runtime",
            "lifecycleConfiguration": {},
            "protocolConfiguration": {"serverProtocol": "HTTP"},
        },
        {"agentRuntimeId": _RUNTIME_ID, "agentRuntimeVersion": "24"},
    )


def _deep_answer(levels: int) -> dict[str, Any]:
    """An answer object nested ``levels`` deep in all: the object, then lists inside it."""
    detail: list[Any] = []
    for _ in range(levels - 2):
        detail = [detail]
    return {"response": _ANSWER, "detail": detail}


def _runs(count: int, *, deep: tuple[int, ...] = ()) -> StubbedAwsClients:
    """Clients queued with the preflight, then one run for each of ``deep`` answered with an
    ``application/json`` object nested that deep, then ``count`` runs answered in two events."""
    clients = StubbedAwsClients()
    _preflight(clients)
    answers = [(json_body(_deep_answer(levels)), "application/json") for levels in deep]
    answers += [(sse_body(*_ANSWER_EVENTS), "text/event-stream") for _ in range(count)]
    for body, content_type in answers:
        clients.stub("bedrock-agentcore", view=True).add_response(
            "invoke_agent_runtime",
            {
                "response": body,
                "contentType": content_type,
                "statusCode": 200,
                "traceParent": _RETURNED_TRACEPARENT,
            },
        )
        clients.stub("bedrock-agentcore").add_response(
            "stop_runtime_session", {"statusCode": 200}, stop_params()
        )
    return clients


def _live_executor(
    clients: StubbedAwsClients,
    cassette: Path | None,
    log_dir: Path,
    *,
    mode: Mode | None = Mode.RECORD,
    **overrides: Any,
) -> AgentCoreRuntimeExecutor:
    """An executor over stubbed clients and the fake reader that may record into ``cassette``."""
    target = overrides.pop("target", None) or _target()
    kwargs: dict[str, Any] = {
        "span_reader": FakeRunSpanReader(clients),
        "clients": clients,
        "cassette_path": cassette,
        "cassette_mode": mode,
        "log_dir": log_dir,
        "sleep": lambda _seconds: None,
        "clock": _clock(),
    }
    return AgentCoreRuntimeExecutor(target, **(kwargs | overrides))


def _replayer(
    cassette: Path, log_dir: Path, *, target: RuntimeTarget | None = None, **overrides: Any
) -> AgentCoreRuntimeExecutor:
    """An executor with no clients and, unless ``overrides`` give one, no reader: it replays."""
    kwargs: dict[str, Any] = {"cassette_mode": Mode.REPLAY, "log_dir": log_dir, "clock": _clock()}
    return AgentCoreRuntimeExecutor(
        target or _target(), cassette_path=cassette, **(kwargs | overrides)
    )


def _executor_in(
    mode: Mode, cassette: Path, log_dir: Path, clients: StubbedAwsClients, **overrides: Any
) -> AgentCoreRuntimeExecutor:
    """A replay-only executor, or one in ``mode`` that may call AWS through ``clients``."""
    if mode is Mode.REPLAY:
        return _replayer(cassette, log_dir, **overrides)
    return _live_executor(clients, cassette, log_dir, mode=mode, **overrides)


def _record(cassette: Path, log_dir: Path, **overrides: Any) -> Trace:
    """Record the narrative's run into ``cassette`` against a stubbed runtime."""
    clients = _runs(1)
    with _live_executor(clients, cassette, log_dir, **overrides) as executor:
        [trace] = executor.run(_task())
    clients.assert_no_pending_responses()
    return trace


def _replay(
    cassette: Path, log_dir: Path, *, target: RuntimeTarget | None = None, **task: Any
) -> Trace:
    """Replay one task with a new replay-only executor."""
    with _replayer(cassette, log_dir, target=target) as executor:
        [trace] = executor.run(_task(**task))
    return trace


def _entries(cassette: Path) -> list[dict[str, Any]]:
    return json.loads(cassette.read_text(encoding="utf-8"))["entries"]


def _manifest(log_dir: Path) -> list[dict[str, Any]]:
    """The run records in ``log_dir``'s run manifest, as the JSON the manifest stores."""
    return [record.model_dump(mode="json") for record in RUN_RECORD_STORE.load(log_dir, fmt="json")]


def _when(text: str) -> datetime:
    return datetime.fromisoformat(text)


class _Network:
    """Records every socket connection, session, client build and credential lookup in
    ``touched``, and refuses the connections, so a test sees whether anything tried to reach AWS."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.touched: list[str] = []
        session_init = boto3.session.Session.__init__
        create_client = botocore.session.Session.create_client
        get_credentials = botocore.session.Session.get_credentials

        def connect(_socket: socket.socket, address: object) -> int:
            self.touched.append(f"a connection to {address}")
            raise OSError("this test allows no network connection")

        def opened(session: boto3.session.Session, *args: Any, **kwargs: Any) -> None:
            self.touched.append("a session")
            session_init(session, *args, **kwargs)

        def built(session: botocore.session.Session, *args: Any, **kwargs: Any) -> Any:
            self.touched.append(f"a {args[0] if args else kwargs.get('service_name')} client")
            return create_client(session, *args, **kwargs)

        def credentials(session: botocore.session.Session) -> Any:
            self.touched.append("a credential lookup")
            return get_credentials(session)

        monkeypatch.setattr(socket.socket, "connect", connect)
        monkeypatch.setattr(socket.socket, "connect_ex", connect)
        monkeypatch.setattr(boto3.session.Session, "__init__", opened)
        monkeypatch.setattr(botocore.session.Session, "create_client", built)
        monkeypatch.setattr(botocore.session.Session, "get_credentials", credentials)


def _no_aws_at_all(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Network:
    """Unset every AWS variable, name a profile that doesn't exist, point ``HOME`` at an empty
    directory, so a lookup of the default AWS files finds none, and watch the network."""
    for name in list(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("AWS_PROFILE", "calibrated-agent-evaluation-no-such-profile")
    return _Network(monkeypatch)


def _restricted(directory: Path, permissions: int) -> Iterator[Path]:
    """Make ``directory`` with ``permissions`` for the test, then make it usable again."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("the superuser reads and writes through any permission")
    directory.mkdir()
    directory.chmod(permissions)
    try:
        yield directory
    finally:
        directory.chmod(0o700)


@pytest.fixture
def unsearchable_directory(tmp_path: Path) -> Iterator[Path]:
    yield from _restricted(tmp_path / "unsearchable", 0o000)


@pytest.fixture
def read_only_directory(tmp_path: Path) -> Iterator[Path]:
    yield from _restricted(tmp_path / "read-only", 0o500)


# --- recording --------------------------------------------------------------------------------


def test_recording_stores_response_spans_version_ids_and_times_under_an_id_free_fingerprint(
    tmp_path: Path, cassette: Path
) -> None:
    clients = _runs(1)
    reader = FakeRunSpanReader(clients)
    with _live_executor(clients, cassette, tmp_path, span_reader=reader) as executor:
        [trace] = executor.run(_task())
    clients.assert_no_pending_responses()

    [entry] = _entries(cassette)
    [sent] = clients.calls_to("InvokeAgentRuntime")
    assert entry["request"] == _NARRATIVE_REQUEST
    assert entry["fingerprint"] == fingerprint(_NARRATIVE_REQUEST)
    assert entry["model"] == {
        "id": f"{RUNTIME_ARN}@DEFAULT",
        "config": {"config_bundle_arn": None, "config_bundle_version": None},
    }
    response = entry["response"]
    assert response["content_type"] == "text/event-stream"
    assert response["status_code"] == 200
    assert response["values"] == _ANSWER_EVENTS
    assert response["spans"] == [to_dict(span) for span in trace.spans]
    assert (response["runtime_version"], response["runtime_version_source"]) == ("24", "endpoint")
    assert response["session_id"] == sent.params["runtimeSessionId"]
    assert response["w3c_trace_id"] == sent.params["traceParent"].split("-")[1]
    assert response["trace_id"] == trace.trace_id
    assert response["response_traceparent"] == _RETURNED_TRACEPARENT
    assert response["attempts"] == 1
    # The entry's times are the invocation's, the ones the reader was given.
    [query] = reader.queries
    assert (response["started_at"], response["ended_at"]) == (query.started_at, query.ended_at)
    assert response["started_at"] == trace.captured_at
    started, ended = _when(response["started_at"]), _when(response["ended_at"])
    assert started < ended
    # The run record spans the whole run, which also builds clients, checks the runtime, reads
    # the spans and stops the session.
    [line] = _manifest(tmp_path)
    assert _when(line["started_at"]) < started
    assert ended < _when(line["ended_at"])
    assert line["duration_ms"] > (ended - started) // timedelta(milliseconds=1)
    # Another recording of the prompt shares the fingerprint, though not the run's IDs.
    other = tmp_path / "other.cassette.json"
    _record(other, tmp_path / "other")
    [again] = _entries(other)
    assert again["fingerprint"] == entry["fingerprint"]
    assert again["response"]["session_id"] != response["session_id"]
    assert again["response"]["trace_id"] != response["trace_id"]


def _body(prompt: str, session_id: str) -> Mapping[str, object]:
    return {"question": prompt, "conversation": session_id}


def test_two_custom_builders_get_separate_entries(tmp_path: Path, cassette: Path) -> None:
    names = ("claims-body", "claims-body-v2")
    for name in names:
        custom = _target(payload_format="custom")
        _record(cassette, tmp_path / name, target=custom, payload_builder=_body, builder_name=name)

    entries = _entries(cassette)
    assert sorted(entry["request"]["builder"] for entry in entries) == list(names)
    assert len({entry["fingerprint"] for entry in entries}) == 2


class _OrphanedReader:
    """A reader whose last span names a parent the tree doesn't hold."""

    def read_spans(self, query: RunSpanQuery) -> list[Span]:
        spans = clm_4471_spans(query.trace_id)
        return [*spans[:-1], replace(spans[-1], parent=f"{query.trace_id}/no-such-span")]


def test_an_invalid_run_is_not_recorded(tmp_path: Path, cassette: Path) -> None:
    executor = _live_executor(_runs(1), cassette, tmp_path, span_reader=_OrphanedReader())

    with executor, pytest.raises(RuntimeInvocationError) as refused:
        executor.run(_task())

    assert isinstance(refused.value.__cause__, SchemaError)
    assert not cassette.exists()
    [record] = executor.run_records
    assert record.session_stop == "stopped"


# --- replaying --------------------------------------------------------------------------------


def test_a_replay_returns_the_recorded_trace_and_records_the_recorded_ids_and_timing(
    tmp_path: Path, cassette: Path, recorded: Trace
) -> None:
    [entry] = _entries(cassette)
    run = entry["response"]

    with _replayer(cassette, tmp_path / "replay") as executor:
        [replayed] = executor.run(_task())

    assert replayed == recorded
    assert to_dict(replayed) == to_dict(recorded)
    assert replayed.captured_at == run["started_at"]
    validate(replayed)
    # The run record reuses the recorded run's IDs and timing, and is marked not started.
    [line] = _manifest(tmp_path / "replay")
    assert line["trace_id"] == run["trace_id"] == replayed.trace_id
    same = ["session_id", "w3c_trace_id", "started_at", "ended_at", "attempts", "status_code"]
    same += ["content_type", "response_traceparent"]
    assert {key: line[key] for key in same} == {key: run[key] for key in same}
    duration = _when(run["ended_at"]) - _when(run["started_at"])
    assert line["duration_ms"] == duration // timedelta(milliseconds=1)
    assert (line["runtime_version"], line["runtime_version_source"]) == ("24", "endpoint")
    assert line["replayed"] is True
    assert (line["session_stop"], line["error_code"]) == ("not_started", None)
    assert [RunRecord.model_validate(line)] == list(executor.run_records)


def test_a_task_trace_id_replaces_the_recorded_one_so_two_replays_share_no_span_id(
    tmp_path: Path, cassette: Path, recorded: Trace
) -> None:
    session = _entries(cassette)[0]["response"]["session_id"]
    # One executor replays an entry once, so each replay gets its own.
    trace_ids = ("CLM-4471-first", "CLM-4471-second")
    replays = [_replay(cassette, tmp_path / trace_id, trace_id=trace_id) for trace_id in trace_ids]

    for trace_id, trace in zip(trace_ids, replays, strict=True):
        assert trace.trace_id == trace_id
        assert {span.trace_id for span in trace.spans} == {trace_id}
        assert trace.output == recorded.output
        [line] = _manifest(tmp_path / trace_id)
        assert (line["trace_id"], line["session_id"]) == (trace_id, session)
    first, second = replays
    assert not {span.span_id for span in first.spans} & {span.span_id for span in second.spans}
    with LocalBackend(root=tmp_path / "store") as backend:
        for trace in replays:
            backend.put_trace(trace)
        stored = [backend.get_trace(trace.trace_id) for trace in replays]
    assert stored == replays
    for trace in stored:
        assert {span.agent for span in trace.spans} == _FIVE_AGENTS
        validate(trace)


class _PinnedReader:
    """A reader whose root and payout spans carry IDs an adapter pinned, without the trace ID."""

    def read_spans(self, query: RunSpanQuery) -> list[Span]:
        prefix = f"{query.trace_id}/"
        pinned = {f"{prefix}supervisor": "supervisor-pinned", f"{prefix}payout": "payout-pinned"}

        def keyed(span_id: str | None) -> str | None:
            return None if span_id is None else pinned.get(span_id, span_id)

        return [
            replace(span, span_id=keyed(span.span_id), parent=keyed(span.parent))
            for span in clm_4471_spans(query.trace_id)
        ]


def _tree(trace: Trace) -> dict[str, tuple[str, str | None, int]]:
    return {span.agent: (span.span_id, span.parent, span.seq) for span in trace.spans}


def test_a_span_id_without_the_recorded_prefix_is_prefixed_only_when_the_trace_id_changes(
    tmp_path: Path, cassette: Path
) -> None:
    recorded = _record(cassette, tmp_path / "record", span_reader=_PinnedReader())

    trace = _replay(cassette, tmp_path / "replay", trace_id="CLM-4471-replayed")
    # One executor replays an entry once, so the replay under the recorded ID gets its own.
    same_id = _replay(cassette, tmp_path / "same-id", trace_id=recorded.trace_id)

    root = "CLM-4471-replayed/supervisor-pinned"
    assert _tree(trace) == {
        "supervisor": (root, None, 0),
        "intake": ("CLM-4471-replayed/intake", root, 1),
        "policy-checker": ("CLM-4471-replayed/policy-checker", root, 2),
        "fraud-scorer": ("CLM-4471-replayed/fraud-scorer", root, 3),
        "payout": ("CLM-4471-replayed/payout-pinned", root, 4),
    }
    assert {span.trace_id for span in trace.spans} == {"CLM-4471-replayed"}
    # Only the IDs move: each span is otherwise the one recorded.
    assert [
        replace(span, trace_id=before.trace_id, span_id=before.span_id, parent=before.parent)
        for span, before in zip(trace.spans, recorded.spans, strict=True)
    ] == recorded.spans
    validate(trace)
    # Under the recorded trace ID nothing moves.
    assert same_id.spans == recorded.spans
    assert _tree(same_id)["supervisor"] == ("supervisor-pinned", None, 0)


def _with_values(cassette: Path, values: list[object]) -> dict[str, Any]:
    """Replace the recorded values of ``cassette``'s one entry; return the entry's response."""
    doc = json.loads(cassette.read_text(encoding="utf-8"))
    [entry] = doc["entries"]
    entry["response"]["values"] = values
    cassette.write_text(json.dumps(doc), encoding="utf-8")
    return entry["response"]


_ERRED, _NO_ANSWER = "reported an error", "no answer could be read"


@pytest.mark.parametrize(
    ("values", "refusal"),
    [
        ([{"error": f"payout down ({_SENTINEL})", "error_type": "ConnectionError"}], _ERRED),
        ([{"response": _ANSWER_PARTS[0]}, {"error": f"halted ({_SENTINEL})"}], _ERRED),
        ([], _NO_ANSWER),
        ([{"tokens": 12}], _NO_ANSWER),
        ([{"response": ""}], _NO_ANSWER),
    ],
    ids=["error-event", "answer-then-error", "no-values", "no-text", "empty-text"],
)
def test_a_replayed_entry_with_an_error_or_no_answer_is_refused_and_records_the_recorded_run(
    tmp_path: Path, cassette: Path, recorded: Trace, values: list[object], refusal: str
) -> None:
    run = _with_values(cassette, values)
    executor = _replayer(cassette, tmp_path / "replay")

    with executor, pytest.raises(RuntimeInvocationError) as refused:
        executor.run(_task())

    message = str(refused.value)
    assert refusal in message
    assert run["session_id"] in message
    assert _SENTINEL not in message
    # The entry decoded, so the record names the recorded run, not IDs made up for this one.
    [line] = _manifest(tmp_path / "replay")
    same = ["session_id", "w3c_trace_id", "trace_id", "started_at", "ended_at", "attempts"]
    assert {key: line[key] for key in same} == {key: run[key] for key in same}
    assert line["replayed"] is True
    # An agent-reported error is recorded by its own type; a missing answer by the general one.
    expected_code = "AgentReportedError" if refusal == _ERRED else "RuntimeInvocationError"
    assert (line["session_stop"], line["error_code"]) == ("not_started", expected_code)
    assert [RunRecord.model_validate(line)] == list(executor.run_records)


# --- values JSON cannot represent -------------------------------------------------------------


def test_a_response_nested_254_levels_records_and_replays_but_257_is_refused_naming_the_session(
    tmp_path: Path, cassette: Path
) -> None:
    clients = _runs(0, deep=(257, 254))
    executor = _live_executor(clients, cassette, tmp_path / "logs")

    with executor:
        with pytest.raises(RuntimeInvocationError) as refused:
            executor.run(_task())
        assert not cassette.exists()
        # A later write would save an unsaved entry still held in memory; it saves only its own.
        # Its value is the deepest a recorded run can carry: the run and its values add two levels.
        [recorded] = executor.run(_task(prompt=_OTHER_PROMPT))
    replayed = _replay(cassette, tmp_path / "replay", prompt=_OTHER_PROMPT)

    deep, _ = clients.calls_to("InvokeAgentRuntime")
    session = deep.params["runtimeSessionId"]
    for name in ("application/json", session, "nests more than 256 levels"):
        assert name in str(refused.value)
    assert _ANSWER not in str(refused.value)
    [entry] = _entries(cassette)
    assert entry["request"]["prompt"] == _OTHER_PROMPT
    assert entry["response"]["values"] == [_deep_answer(254)]
    assert replayed == recorded
    assert to_dict(replayed) == to_dict(recorded)
    assert replayed.output == {"text": _ANSWER}
    failed, _ = executor.run_records
    assert (failed.session_id, failed.session_stop) == (session, "stopped")
    assert failed.error_code == "NestedTooDeepError"
    assert _manifest(tmp_path / "logs")[0] == failed.model_dump(mode="json")
    assert len(clients.calls_to("StopRuntimeSession")) == 2


@pytest.mark.parametrize("mode", [Mode.RECORD, Mode.AUTO], ids=["record", "auto"])
def test_a_run_whose_entry_is_not_standard_json_is_not_recorded_or_run_again(
    tmp_path: Path, cassette: Path, mode: Mode
) -> None:
    # A value nested 255 levels decodes, but inside the recorded run it nests 257 levels deep, so
    # the entry is refused when it is encoded, after the billed run.
    clients = _runs(1, deep=(255,))
    executor = _live_executor(clients, cassette, tmp_path / "logs", mode=mode)

    with executor:
        with pytest.raises(RuntimeInvocationError) as refused:
            executor.run(_task())
        # An answer is still queued, so a repeat that went live again would succeed.
        with pytest.raises(RuntimeInvocationError) as repeat_refused:
            executor.run(_task())
        assert not cassette.exists()
        # A later prompt's write saves only its own entry.
        [other] = executor.run(_task(prompt=_OTHER_PROMPT))

    sent, _ = clients.calls_to("InvokeAgentRuntime")
    session = sent.params["runtimeSessionId"]
    for name in (str(cassette), session, "nests more than 256 levels"):
        assert name in str(refused.value)
    # The repeat names the earlier run and that its entry was not standard JSON.
    repeat_names = (fingerprint(_NARRATIVE_REQUEST), "tried to record", session, str(cassette))
    for name in (*repeat_names, "not standard JSON"):
        assert name in str(repeat_refused.value)
    for message in (str(refused.value), str(repeat_refused.value)):
        assert _ANSWER not in message
    [entry] = _entries(cassette)
    assert entry["request"]["prompt"] == _OTHER_PROMPT
    assert entry["response"]["trace_id"] == other.trace_id
    failed, repeat, _ = executor.run_records
    assert (failed.session_stop, failed.error_code) == ("stopped", "NestedTooDeepError")
    assert _manifest(tmp_path / "logs")[0] == failed.model_dump(mode="json")
    assert (repeat.replayed, repeat.session_stop, repeat.attempts) == (False, "not_started", 0)
    assert repeat.error_code == "RuntimeInvocationError"
    assert len(clients.calls_to("StopRuntimeSession")) == 2


def test_encode_entry_refuses_nan_instead_of_writing_null(cassette: Path, recorded: Trace) -> None:
    [entry] = _entries(cassette)
    run = decode_entry(entry["response"], cassette_path=cassette, fingerprint="fp")
    first, *rest = run.spans
    with_nan = replace(
        run, spans=[replace(first, output={**first.output, "score": math.nan}), *rest]
    )
    before = cassette.read_bytes()
    store = Cassette(cassette, mode=Mode.RECORD)

    with pytest.raises(RuntimeInvocationError) as refused:
        store.play(entry["request"], lambda: encode_entry(with_nan, cassette_path=cassette))

    for name in (str(cassette), run.session_id, "a number JSON cannot represent"):
        assert name in str(refused.value)
    # The entry already recorded stays as it was: no null took the number's place.
    assert cassette.read_bytes() == before
    [kept] = _entries(cassette)
    assert "score" not in kept["response"]["spans"][0]["output"]


class _UnusedReader:
    """A span reader for an auto-mode executor whose every task hits the cassette."""

    def __init__(self) -> None:
        self.queries: list[RunSpanQuery] = []

    def read_spans(self, query: RunSpanQuery) -> list[Span]:
        self.queries.append(query)
        return clm_4471_spans(query.trace_id)


@pytest.mark.parametrize("mode", [Mode.REPLAY, Mode.AUTO], ids=["replay", "auto-hit"])
def test_replay_makes_no_connection_builds_no_client_and_needs_no_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cassette: Path, recorded: Trace, mode: Mode
) -> None:
    network = _no_aws_at_all(monkeypatch, tmp_path)
    reader = _UnusedReader()
    # An auto-mode executor can call AWS, so it needs a reader, though a hit never asks it.
    span_reader = reader if mode is Mode.AUTO else None
    executor = _replayer(cassette, tmp_path / "replay", cassette_mode=mode, span_reader=span_reader)

    with executor:
        [trace] = executor.run(_task())

    assert trace == recorded
    assert network.touched == []
    assert reader.queries == []
    [line] = _manifest(tmp_path / "replay")
    assert (line["replayed"], line["session_stop"]) == (True, "not_started")


def test_an_auto_mode_miss_goes_live(tmp_path: Path, cassette: Path, recorded: Trace) -> None:
    clients = _runs(1)

    with _live_executor(clients, cassette, tmp_path / "auto", mode=Mode.AUTO) as executor:
        [trace] = executor.run(_task(prompt=_OTHER_PROMPT))

    [sent] = clients.calls_to("InvokeAgentRuntime")
    assert (trace.input, trace.output) == ({"prompt": _OTHER_PROMPT}, {"text": _ANSWER})
    assert sorted(entry["request"]["prompt"] for entry in _entries(cassette)) == sorted(
        [_NARRATIVE, _OTHER_PROMPT]
    )
    [line] = _manifest(tmp_path / "auto")
    assert (line["replayed"], line["session_stop"]) == (False, "stopped")
    assert line["session_id"] == sent.params["runtimeSessionId"]


def test_a_replay_miss_is_refused_without_a_network_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cassette: Path, recorded: Trace
) -> None:
    network = _no_aws_at_all(monkeypatch, tmp_path)
    executor = _replayer(cassette, tmp_path / "replay")

    with executor, pytest.raises(ExecutionError) as refused:
        executor.run(_task(prompt=_OTHER_PROMPT))

    assert isinstance(refused.value, RuntimeInvocationError)
    assert "cassette miss" in str(refused.value)
    assert isinstance(refused.value.__cause__, CassetteMiss)
    assert network.touched == []
    [record] = executor.run_records
    assert (record.replayed, record.error_code) == (False, "CassetteMiss")


# --- a cassette that can't serve the run ------------------------------------------------------

_Corruption = Callable[[dict[str, Any]], str | bytes]


def _edited(change: Callable[[dict[str, Any]], object]) -> _Corruption:
    """The cassette's JSON with ``change`` made to its one entry."""

    def corrupt(doc: dict[str, Any]) -> str:
        change(doc["entries"][0])
        return json.dumps(doc)

    return corrupt


def _entries_as(value: Callable[[list[Any]], object]) -> _Corruption:
    return lambda doc: json.dumps({"version": doc["version"], "entries": value(doc["entries"])})


def _padded(padding: str) -> _Corruption:
    """The cassette's JSON with ``padding`` under a key it ignores."""
    return lambda doc: json.dumps(doc)[:-1] + ', "padding": ' + padding + "}"


def _response_part_as(part: str, value: Callable[[Any], object]) -> _Corruption:
    return _edited(lambda entry: entry["response"].update({part: value(entry["response"][part])}))


def _without_its_offset(moment: str) -> str:
    return datetime.fromisoformat(moment).replace(tzinfo=None).isoformat()


def _number_in(place: str, number: str) -> _Corruption:
    """The cassette's JSON with ``number``, as JSON text, in a span's output or in a value."""

    def corrupt(doc: dict[str, Any]) -> str:
        response = doc["entries"][0]["response"]
        if place == "value":
            response["values"].append({"score": "__NUMBER__"})
        else:
            response["spans"][0]["output"]["score"] = "__NUMBER__"
        return json.dumps(doc).replace('"__NUMBER__"', number)

    return corrupt


_CORRUPTIONS: tuple[tuple[str, _Corruption | None], ...] = (
    # The cassette refuses these itself, naming the file.
    ("truncated", lambda doc: json.dumps(doc)[: len(json.dumps(doc)) // 2]),
    ("no-version", lambda doc: json.dumps({"entries": doc["entries"]})),
    ("not-utf-8", lambda doc: json.dumps(doc).replace("Denied", "Deni\u00e9d").encode("latin-1")),
    ("number-too-long-to-read", _padded("9" * 5000)),
    ("nested-too-deep", _padded("[" * 200_000 + "]" * 200_000)),
    # An entry without a key it is read by.
    ("entry-without-fingerprint", _edited(lambda entry: entry.pop("fingerprint"))),
    ("entry-without-request", _edited(lambda entry: entry.pop("request"))),
    ("entry-without-response", _edited(lambda entry: entry.pop("response"))),
    # Entries, or an entry, of the wrong type.
    ("entries-as-an-object", _entries_as(lambda entries: {e["fingerprint"]: e for e in entries})),
    ("entries-as-a-number", _entries_as(lambda _entries: 7)),
    ("entries-as-null", _entries_as(lambda _entries: None)),
    ("entry-as-a-list", _entries_as(lambda _entries: [["fingerprint", "request", "response"]])),
    ("entry-as-a-string", _entries_as(lambda _entries: ["fingerprint"])),
    ("entry-as-a-number", _entries_as(lambda _entries: [7])),
    (
        "fingerprint-as-a-list",
        _edited(lambda entry: entry.update(fingerprint=[entry["fingerprint"]])),
    ),
    # A response a replay can't use.
    ("response-without-spans", _edited(lambda entry: entry["response"].pop("spans"))),
    ("response-without-content-type", _edited(lambda entry: entry["response"].pop("content_type"))),
    ("status-code-as-a-string", _response_part_as("status_code", str)),
    ("time-without-its-utc-offset", _response_part_as("started_at", _without_its_offset)),
    ("span-that-is-not-a-span", _edited(lambda entry: entry["response"]["spans"][0].pop("agent"))),
    (
        "session-id-without-its-prefix",
        _response_part_as(
            "session_id", lambda value: value.replace("calibrated-agent-evaluation-", "session-")
        ),
    ),
    (
        "session-id-in-uppercase-hex",
        _response_part_as(
            "session_id",
            lambda value: (
                "calibrated-agent-evaluation-"
                + value[len("calibrated-agent-evaluation-") :].upper()
            ),
        ),
    ),
    (
        "all-zero-session-id",
        _response_part_as("session_id", lambda _value: "calibrated-agent-evaluation-" + "0" * 32),
    ),
    ("all-zero-w3c-trace-id", _response_part_as("w3c_trace_id", lambda _value: "0" * 32)),
    ("path-does-not-exist", None),
)
_NOT_JSON_NUMBERS = tuple(
    (f"{number}-in-a-{place}", _number_in(place, number))
    for number in ("NaN", "Infinity", "-Infinity", "1e999")
    for place in ("span-output", "value")
)
# Every corruption in replay mode, and in auto mode, which can call AWS, each one but a missing
# file, which auto mode treats as a cassette not recorded yet. A number JSON cannot represent is
# refused saying so.
_CORRUPT_CASES = [
    pytest.param(corrupt, mode, refusal, id=f"{mode.value}-{name}")
    for mode in (Mode.REPLAY, Mode.AUTO)
    for cases, refusal in (
        (_CORRUPTIONS, None),
        (_NOT_JSON_NUMBERS, "a number JSON cannot represent"),
    )
    for name, corrupt in cases
    if corrupt is not None or mode is Mode.REPLAY
]


@pytest.mark.parametrize(("corrupt", "mode", "refusal"), _CORRUPT_CASES)
def test_a_corrupt_cassette_is_refused_naming_its_path(
    tmp_path: Path,
    cassette: Path,
    recorded: Trace,
    corrupt: _Corruption | None,
    mode: Mode,
    refusal: str | None,
) -> None:
    doc = json.loads(cassette.read_text(encoding="utf-8"))
    cassette.unlink()
    if corrupt is not None:
        content = corrupt(doc)
        cassette.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    # Nothing is queued, so any AWS call an auto-mode run made before the refusal would show.
    clients = StubbedAwsClients()
    executor = _executor_in(mode, cassette, tmp_path / "run", clients)

    with pytest.raises(RuntimeInvocationError) as refused, executor:
        executor.run(_task())

    message = str(refused.value)
    assert str(cassette) in message
    assert refusal is None or refusal in message
    assert _ANSWER_PARTS[0] not in message
    assert clients.calls == []
    assert clients.built == []


@pytest.mark.parametrize("mode", [Mode.REPLAY, Mode.AUTO], ids=["replay", "auto"])
def test_a_cassette_in_a_directory_that_cannot_be_searched_is_refused_naming_its_path(
    tmp_path: Path, unsearchable_directory: Path, mode: Mode
) -> None:
    cassette = unsearchable_directory / "runtime.cassette.json"
    clients = StubbedAwsClients()
    executor = _executor_in(mode, cassette, tmp_path / "run", clients)

    with executor, pytest.raises(RuntimeInvocationError) as refused:
        executor.run(_task())

    assert str(cassette) in str(refused.value)
    assert clients.calls == []


@pytest.mark.parametrize(
    "defect", [AttributeError, ValueError, KeyError, TypeError], ids=lambda t: t.__name__
)
def test_a_defect_in_the_cassette_code_is_not_reported_as_a_corrupt_cassette(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cassette: Path, defect: type[Exception]
) -> None:
    def broken(*_args: object, **_kwargs: object) -> None:
        raise defect("a programming error")

    monkeypatch.setattr("calibrated_agent_evaluation.agentcore.runtime_cassette.Cassette", broken)
    cassette.write_text(json.dumps({"version": 1, "entries": []}), encoding="utf-8")

    with _replayer(cassette, tmp_path / "replay") as executor, pytest.raises(defect):
        executor.run(_task())


def test_a_bundled_cassette_replays_under_its_own_version_and_refuses_another_naming_both(
    tmp_path: Path, cassette: Path
) -> None:
    recorded = _record(cassette, tmp_path / "record", target=_bundled())
    assert _replay(cassette, tmp_path / "same", target=_bundled()) == recorded
    [line] = _manifest(tmp_path / "same")
    assert line["config_bundle_arn"] == BUNDLE_ARN
    assert line["config_bundle_version"] == BUNDLE_VERSION
    executor = _replayer(cassette, tmp_path / "other", target=_bundled(_OTHER_BUNDLE_VERSION))

    with executor, pytest.raises(ExecutionError) as refused:
        executor.run(_task())

    assert BUNDLE_VERSION in str(refused.value)
    assert _OTHER_BUNDLE_VERSION in str(refused.value)
    assert isinstance(refused.value.__cause__, CassetteModelMismatch)
    [line] = _manifest(tmp_path / "other")
    assert (line["error_code"], line["replayed"]) == ("CassetteModelMismatch", False)


def test_an_auto_hit_from_another_qualifier_is_refused(
    tmp_path: Path, cassette: Path, recorded: Trace
) -> None:
    clients = StubbedAwsClients()
    target = _target(qualifier="live")
    executor = _live_executor(clients, cassette, tmp_path / "auto", mode=Mode.AUTO, target=target)

    with executor, pytest.raises(RuntimeInvocationError) as refused:
        executor.run(_task())

    assert f"{RUNTIME_ARN}@live" in str(refused.value)
    assert f"{RUNTIME_ARN}@DEFAULT" in str(refused.value)
    assert isinstance(refused.value.__cause__, CassetteModelMismatch)
    assert clients.calls == []
    assert len(_entries(cassette)) == 1


@pytest.mark.parametrize("mode", [Mode.REPLAY, Mode.AUTO], ids=["replay", "auto"])
def test_an_entry_without_its_runtime_stamp_is_refused(
    tmp_path: Path, cassette: Path, recorded: Trace, mode: Mode
) -> None:
    doc = json.loads(cassette.read_text(encoding="utf-8"))
    for entry in doc["entries"]:
        del entry["model"]
    cassette.write_text(json.dumps(doc), encoding="utf-8")
    other = "arn:aws:bedrock-agentcore:eu-west-1:999999999999:runtime/someone_else-ZzZzZzZzZz"
    clients = StubbedAwsClients(region="eu-west-1")
    target = _target(runtime_arn=other, qualifier="prod")
    executor = _executor_in(mode, cassette, tmp_path / "replay", clients, target=target)

    with executor, pytest.raises(RuntimeInvocationError) as refused:
        executor.run(_task())

    assert str(cassette) in str(refused.value)
    assert f"{other}@prod" in str(refused.value)
    assert isinstance(refused.value.__cause__, CassetteModelMismatch)
    assert clients.calls == []
    [line] = _manifest(tmp_path / "replay")
    assert (line["replayed"], line["error_code"]) == (False, "CassetteModelMismatch")


class _BrokenTreeReader:
    """The fake reader's spans, with the first one's parent naming a span not in the run."""

    def __init__(self, clients: StubbedAwsClients) -> None:
        self._reader = FakeRunSpanReader(clients)

    def read_spans(self, query: RunSpanQuery) -> list[Span]:
        first, *rest = self._reader.read_spans(query)
        return [replace(first, parent="no-such-span"), *rest]


def test_a_billed_run_whose_trace_is_not_valid_is_not_run_again_for_the_same_prompt(
    tmp_path: Path,
) -> None:
    cassette = tmp_path / "runtime.cassette.json"
    clients = _runs(2)
    reader = _BrokenTreeReader(clients)
    executor = _live_executor(clients, cassette, tmp_path / "logs", span_reader=reader)

    with executor:
        with pytest.raises(RuntimeInvocationError, match="is not valid"):
            executor.run(_task())
        with pytest.raises(RuntimeInvocationError) as repeat_refused:
            executor.run(_task())

    [sent] = clients.calls_to("InvokeAgentRuntime")
    session = sent.params["runtimeSessionId"]
    repeat_names = (fingerprint(_NARRATIVE_REQUEST), "already tried to record", session)
    for name in (*repeat_names, "not valid"):
        assert name in str(repeat_refused.value)
    assert not cassette.exists()


def test_a_run_whose_cassette_write_failed_keeps_its_run_record_and_is_not_saved_or_run_again(
    tmp_path: Path, read_only_directory: Path
) -> None:
    cassette = read_only_directory / "runtime.cassette.json"
    clients = _runs(2)
    executor = _live_executor(clients, cassette, tmp_path / "logs")

    with executor:
        with pytest.raises(RuntimeInvocationError) as refused:
            executor.run(_task())
        assert not cassette.exists()
        read_only_directory.chmod(0o700)
        with pytest.raises(RuntimeInvocationError) as repeat_refused:
            executor.run(_task())
        [other] = executor.run(_task(prompt=_OTHER_PROMPT))

    sent, _ = clients.calls_to("InvokeAgentRuntime")
    session = sent.params["runtimeSessionId"]
    for name in (str(cassette), "could not be written", "the run is not kept in the cassette"):
        assert name in str(refused.value)
    # The repeat names the earlier run whose recording failed, and how it failed.
    repeat_names = (fingerprint(_NARRATIVE_REQUEST), "already tried to record", session)
    for name in (*repeat_names, "could not be written"):
        assert name in str(repeat_refused.value)
    # The failed run's record is written; the later prompt's write saves only its own entry.
    [line, *_] = _manifest(tmp_path / "logs")
    assert (line["session_id"], line["session_stop"]) == (session, "stopped")
    assert line["error_code"] == "CassetteError"
    [entry] = _entries(cassette)
    assert entry["request"]["prompt"] == _OTHER_PROMPT
    assert entry["response"]["trace_id"] == other.trace_id
    failed, repeat, _ = executor.run_records
    assert (failed.replayed, failed.error_code) == (False, "CassetteError")
    assert (repeat.replayed, repeat.session_stop, repeat.attempts) == (False, "not_started", 0)
    with _replayer(cassette, tmp_path / "replay") as replayer:
        with pytest.raises(RuntimeInvocationError, match="cassette miss"):
            replayer.run(_task())
    [line] = _manifest(tmp_path / "replay")
    assert line["replayed"] is False


# --- one executor, one recorded run per fingerprint -------------------------------------------


def test_a_second_task_replaying_the_same_entry_is_refused_naming_the_fingerprint(
    tmp_path: Path, cassette: Path, recorded: Trace
) -> None:
    [entry] = _entries(cassette)
    executor = _replayer(cassette, tmp_path / "replay")

    with executor:
        [first] = executor.run(_task())
        with pytest.raises(RuntimeInvocationError) as refused:
            executor.run(_task())

    assert entry["fingerprint"] in str(refused.value)
    # It names the earlier task's run, which replayed the recorded session.
    assert f"replayed the run of session {entry['response']['session_id']}" in str(refused.value)
    replayed, repeat = executor.run_records
    assert replayed.trace_id == first.trace_id
    assert (repeat.replayed, repeat.error_code) == (False, "RuntimeInvocationError")
    assert repeat.session_id != replayed.session_id


@pytest.mark.parametrize("mode", [Mode.RECORD, Mode.AUTO], ids=["record", "auto"])
def test_a_repeat_of_a_prompt_this_executor_recorded_is_refused_and_keeps_the_first_entry(
    tmp_path: Path, cassette: Path, mode: Mode
) -> None:
    clients = _runs(2)

    with _live_executor(clients, cassette, tmp_path, mode=mode) as executor:
        [first] = executor.run(_task())
        with pytest.raises(RuntimeInvocationError) as refused:
            executor.run(_task())

    [entry] = _entries(cassette)
    [sent] = clients.calls_to("InvokeAgentRuntime")
    session = sent.params["runtimeSessionId"]
    assert entry["fingerprint"] in str(refused.value)
    assert f"recorded the run of session {session}" in str(refused.value)
    assert entry["response"]["trace_id"] == first.trace_id
    assert entry["response"]["session_id"] == session


def test_without_a_cassette_repeated_prompts_run_independently(tmp_path: Path) -> None:
    clients = _runs(2)

    with _live_executor(clients, None, tmp_path, mode=None) as executor:
        traces = [executor.run(_task())[0] for _ in range(2)]

    assert len({trace.trace_id for trace in traces}) == 2
    sessions = [call.params["runtimeSessionId"] for call in clients.calls_to("InvokeAgentRuntime")]
    assert len(set(sessions)) == 2


# --- the committed fixture --------------------------------------------------------------------


def test_the_committed_cassette_replays_to_the_committed_expected_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    network = _no_aws_at_all(monkeypatch, tmp_path)
    expected = trace_from_dict(json.loads(_COMMITTED_TRACE.read_text(encoding="utf-8")))

    trace = _replay(_COMMITTED_CASSETTE, tmp_path / "replay")

    assert trace == expected
    assert trace.frame == "production-representative"
    assert trace.input == {"prompt": _NARRATIVE}
    assert trace.taxonomy_cell == _CELL
    assert network.touched == []


# --- the session a trace belongs to ----------------------------------------------------------


def test_a_replayed_trace_carries_the_session_it_was_recorded_under(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    network = _no_aws_at_all(monkeypatch, tmp_path)
    entries = _entries(_COMMITTED_CASSETTE)

    with _replayer(_COMMITTED_CASSETTE, tmp_path / "replay") as executor:
        traces = [executor.run(_task(prompt=e["request"]["prompt"]))[0] for e in entries]

    assert [t.session_id for t in traces] == [e["response"]["session_id"] for e in entries]
    assert len({t.session_id for t in traces}) == 2
    assert network.touched == []


def test_a_task_naming_its_trace_id_keeps_the_run_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _no_aws_at_all(monkeypatch, tmp_path)
    plain = _replay(_COMMITTED_CASSETTE, tmp_path / "plain")

    named = _replay(_COMMITTED_CASSETTE, tmp_path / "named", trace_id="CLM-4471-named")

    assert named.trace_id == "CLM-4471-named"
    assert plain.session_id is not None
    assert named.session_id == plain.session_id
