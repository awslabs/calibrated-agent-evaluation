# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pulling Langfuse traces Calibrated Agent Evaluation did not write.

The source is a real ``LangfuseBackend`` over the stub client that mirrors the verified v3 surface
(``tests/test_backends_langfuse.py``). Foreign traces are stored in the shapes the live OSS server
3.202.1 returned with ``langfuse==3.15.0``: one written with the plain SDK (``agent`` / ``tool`` /
``generation`` observations, the root naming a parent that is not in the trace), and one written
by a plain OpenTelemetry exporter to ``/api/public/otel`` (its attributes kept in
``metadata["attributes"]``, its content moved to ``input`` / ``output``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend, LangfuseLike
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli.dispatch import main
from calibrated_agent_evaluation.ingest import ingest_langfuse
from calibrated_agent_evaluation.normalize import StrandsSession, normalize
from calibrated_agent_evaluation.schema.errors import BackendError
from test_backends_langfuse import StubLangfuseClient

_RAW = Path(__file__).resolve().parent.parent / "fixtures" / "raw"
_T0 = datetime(2026, 10, 15, 18, 45, 56, 720000, tzinfo=UTC)
_T1 = datetime(2026, 10, 15, 18, 45, 56, 721000, tzinfo=UTC)
_SDK_META = {
    "resourceAttributes": {"service.name": "unknown_service"},
    "scope": {"name": "langfuse-sdk", "version": "3.15.0"},
}


def _obs(obs_id: str, obs_type: str, name: str, parent: str | None, **fields: Any) -> Any:
    return SimpleNamespace(
        id=obs_id,
        type=obs_type,
        name=name,
        parent_observation_id=parent,
        start_time=fields.pop("start_time", _T0),
        end_time=fields.pop("end_time", _T1),
        input=fields.pop("input", None),
        output=fields.pop("output", None),
        metadata=fields.pop("metadata", _SDK_META),
        model=fields.pop("model", None),
        level=fields.pop("level", "DEFAULT"),
        status_message=None,
    )


def _sdk_written_trace(trace_id: str) -> Any:
    """The shape a plain-SDK trace read back as, live: agent root with a dangling parent id."""
    return SimpleNamespace(
        id=trace_id,
        tags=["support"],
        metadata=_SDK_META,
        input={"message": "refund order A-7781"},
        output={"answer": "Refunded $89.99"},
        observations=[
            _obs(
                "gen1",
                "GENERATION",
                "claude-sonnet",
                "agent1",
                model="claude-sonnet-4-6",
                input=[{"role": "user", "content": "refund order A-7781"}],
                output={"role": "assistant", "content": "calling lookup_order"},
            ),
            _obs(
                "tool1",
                "TOOL",
                "lookup_order",
                "agent1",
                input={"order_id": "A-7781"},
                output={"status": "delivered", "amount_usd": 89.99},
            ),
            _obs(
                "agent1",
                "AGENT",
                "refund-agent",
                "not-in-trace",
                input={"message": "refund order A-7781"},
                output={"answer": "Refunded $89.99"},
            ),
        ],
    )


def _otlp_written_trace(trace_id: str) -> Any:
    """The shape a plain OTel exporter's trace read back as: attributes in metadata."""

    def meta(attrs: dict[str, str]) -> dict[str, Any]:
        return {
            "attributes": attrs,
            "resourceAttributes": {"service.name": "plain-otel-agent"},
            "scope": {"name": "plain.otel", "attributes": {}},
        }

    return SimpleNamespace(
        id=trace_id,
        tags=[],
        metadata=meta({}),
        observations=[
            _obs(
                "a1",
                "AGENT",
                "invoke_agent orderbot",
                None,
                metadata=meta(
                    {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "orderbot"}
                ),
                input=[{"role": "user", "parts": [{"type": "text", "content": "refund A-7781"}]}],
                output=[{"role": "assistant", "parts": [{"type": "text", "content": "refunded"}]}],
            ),
            _obs(
                "t1",
                "TOOL",
                "lookup_order",
                "a1",
                metadata=meta(
                    {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "lookup_order"}
                ),
                input={"order_id": "A-7781"},
                output={"status": "delivered"},
            ),
        ],
    )


def _source(*foreign: Any) -> tuple[LangfuseBackend, StubLangfuseClient]:
    client = StubLangfuseClient()
    client.api.projects = SimpleNamespace(  # type: ignore[attr-defined]
        get=lambda: SimpleNamespace(
            data=[SimpleNamespace(id="calibrated-agent-evaluation-dev", name="Provisioned Project")]
        )
    )
    for trace in foreign:
        client._traces[trace.id] = trace
    return LangfuseBackend(cast("LangfuseLike", client)), client


def test_a_trace_written_with_the_plain_sdk_ingests(tmp_path: Path) -> None:
    """The defect: any trace without the calibrated_agent_evaluation_trace envelope raised out of
    query()."""
    source, _ = _source(_sdk_written_trace("f" * 32))
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = ingest_langfuse(source, backend=dest)

    assert report.exit_code == 0, report.results
    [result] = report.results
    assert result.source == f"langfuse:{'f' * 32}"
    got = dest.get_trace("f" * 32)
    assert got.ownership == "owned"
    [agent] = got.spans
    assert agent.agent == "refund-agent"
    assert agent.input == {"message": "refund order A-7781"}
    assert [(c.name, c.args, c.result) for c in agent.tool_calls] == [
        ("lookup_order", {"order_id": "A-7781"}, {"status": "delivered", "amount_usd": 89.99})
    ]


def test_a_trace_written_by_a_plain_otel_exporter_keeps_its_own_attributes(
    tmp_path: Path,
) -> None:
    """Detection sees the recorded OTel attributes; the content Langfuse moved is put back."""
    source, _ = _source(_otlp_written_trace("e" * 32))
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = ingest_langfuse(source, backend=dest)

    assert report.exit_code == 0, report.results
    [agent] = dest.get_trace("e" * 32).spans
    assert agent.agent == "orderbot"
    assert agent.input, "the content Langfuse moved into `input` must come back"
    assert [c.name for c in agent.tool_calls] == ["lookup_order"]
    assert agent.tool_calls[0].args == {"order_id": "A-7781"}


def test_a_trace_with_no_agent_observation_is_read_as_one_run(tmp_path: Path) -> None:
    """``@observe``-style traces (spans and generations only): the outermost one is the agent."""
    trace = SimpleNamespace(
        id="d" * 32,
        tags=[],
        metadata=_SDK_META,
        observations=[
            _obs("root", "SPAN", "handle_request", None, input={"q": "hi"}, output={"a": "yo"}),
            _obs("g", "GENERATION", "llm", "root", input="hi", output="yo"),
            _obs("t", "TOOL", "search", "root", input={"q": "hi"}, output={"hits": 1}),
        ],
    )
    source, _ = _source(trace)
    dest = LocalBackend(root=str(tmp_path / "store"))
    assert ingest_langfuse(source, backend=dest).exit_code == 0
    [agent] = dest.get_trace("d" * 32).spans
    assert agent.agent == "handle_request"
    assert [c.name for c in agent.tool_calls] == ["search"]


def test_one_unreadable_trace_is_skipped_and_the_pull_carries_on(tmp_path: Path) -> None:
    """Per-trace failures are `skipped` with a reason; the rest of the project still ingests."""
    empty = SimpleNamespace(id="c" * 32, tags=[], metadata={}, observations=[])
    source, _ = _source(empty, _sdk_written_trace("f" * 32))
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = ingest_langfuse(source, backend=dest)

    statuses = {r.source: r.status for r in report.results}
    assert statuses == {f"langfuse:{'c' * 32}": "skipped", f"langfuse:{'f' * 32}": "ingested"}
    skipped = next(r for r in report.results if r.status == "skipped")
    assert "no observations" in (skipped.reason or "")
    assert skipped.adapter == "langfuse"  # a Langfuse pull's unreadable trace still names Langfuse
    assert report.exit_code == 1, "a skipped trace still fails the command"
    assert [t.trace_id for t in dest.query()] == ["f" * 32]


def test_a_failing_get_for_one_trace_is_skipped(tmp_path: Path) -> None:
    source, client = _source(_sdk_written_trace("f" * 32), _sdk_written_trace("b" * 32))
    real_get = client.api.trace.get

    def flaky_get(trace_id: str) -> Any:
        if trace_id == "b" * 32:
            raise TimeoutError("read timed out")
        return real_get(trace_id)

    client.api.trace.get = flaky_get  # type: ignore[method-assign]
    report = ingest_langfuse(source, backend=LocalBackend(root=str(tmp_path / "store")))
    by_source = {r.source: r for r in report.results}
    assert by_source[f"langfuse:{'b' * 32}"].status == "skipped"
    assert "TimeoutError" in (by_source[f"langfuse:{'b' * 32}"].reason or "")
    assert by_source[f"langfuse:{'f' * 32}"].status == "ingested"


def test_calibrated_agent_evaluation_authored_and_foreign_traces_ingest_side_by_side(
    tmp_path: Path,
) -> None:
    """The envelope path is unchanged: the library's own trace still round-trips losslessly."""
    source, _ = _source(_sdk_written_trace("f" * 32))
    own = normalize(StrandsSession(path=_RAW / "strands" / "CLM-4471.otlp.json", trace_id="OWN-1"))
    source.put_trace(own)
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = ingest_langfuse(source, backend=dest)

    assert report.exit_code == 0, report.results
    assert dest.get_trace("OWN-1") == own
    assert dest.get_trace("f" * 32).ownership == "owned"


def test_a_lossy_foreign_trace_is_refused_by_the_same_gate_as_a_file(tmp_path: Path) -> None:
    """Tool spans the tree cannot attach to an agent are refused, not silently dropped."""
    trace = SimpleNamespace(
        id="a" * 32,
        tags=[],
        metadata=_SDK_META,
        observations=[
            _obs("ag", "AGENT", "solo", None, input={"q": 1}, output={"a": 1}),
            # A tool whose parent is a separate root span, not the agent: nothing links it.
            _obs("other", "SPAN", "background", None),
            _obs("tl", "TOOL", "lookup", "other", input={"id": 1}, output={"ok": True}),
        ],
    )
    source, _ = _source(trace)
    dest = LocalBackend(root=str(tmp_path / "store"))
    [result] = ingest_langfuse(source, backend=dest).results
    assert result.status == "refused", result
    assert "tool-execution span" in (result.reason or "")
    assert dest.query() == []


def test_an_unlistable_source_raises_before_anything_is_persisted(tmp_path: Path) -> None:
    """The listing fails as a BackendError and the destination stays empty."""
    source, client = _source(_sdk_written_trace("f" * 32))

    def down(**_: Any) -> Any:
        raise ConnectionError("connection refused")

    client.api.trace.list = down  # type: ignore[method-assign]
    dest = LocalBackend(root=str(tmp_path / "store"))
    with pytest.raises(BackendError, match="pull"):
        ingest_langfuse(source, backend=dest)
    assert dest.query() == []


# --- PROJECT: Langfuse keys are project-scoped, so the value is checked, not used to select ------


@pytest.fixture
def cli_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> LangfuseBackend:
    source, _ = _source(_sdk_written_trace("f" * 32))
    import calibrated_agent_evaluation.cli.backends as cli_backends

    real = cli_backends.backend_from_flag

    def fake(flag: str, **kwargs: Any) -> Any:
        return source if flag == "langfuse" else real(flag, **kwargs)

    monkeypatch.setattr(cli_backends, "backend_from_flag", fake)
    # The project's public key names its write manifest; the stub never reads it.
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.chdir(tmp_path)
    return source


@pytest.mark.parametrize("project", ["Provisioned Project", "calibrated-agent-evaluation-dev"])
def test_from_langfuse_accepts_the_credentials_project_by_name_or_id(
    cli_source: LangfuseBackend, project: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["ingest", "--from-langfuse", project])
    assert code == 0, capsys.readouterr().err
    assert LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).get_trace("f" * 32)


def test_from_langfuse_refuses_a_project_the_credentials_do_not_address(
    cli_source: LangfuseBackend, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The defect: PROJECT was a truthiness check, so a wrong name pulled the env's project."""
    code = main(["ingest", "--from-langfuse", "prod-support"])
    err = capsys.readouterr().err
    assert code == 1
    assert "prod-support" in err
    assert "Provisioned Project" in err
    assert LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).query() == []
