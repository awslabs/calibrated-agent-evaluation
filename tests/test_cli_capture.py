# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture tests: an instrumented in-process agent yields an owned Trace with its tools,
``capture --dry-run`` writes nothing, a missing [strands] extra names itself, and an agent that
emits no span is refused rather than stored as a silently empty trace."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from _helpers import structural_view
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli.dispatch import main
from calibrated_agent_evaluation.ingest import IngestError, capture_and_ingest
from calibrated_agent_evaluation.schema.errors import BackendError
from calibrated_agent_evaluation.schema.types import Trace

# The repo-root `corpus` dev tree (the instrumented reference agent) is not under src/, so put the
# repo root on sys.path — the blessed pattern the flagship tests use (tests/corpus/conftest.py).
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _emit_reference_trace() -> None:
    """An instrumented in-process agent: emit one invoke_agent + execute_tool span into the global
    tracer (reusing the corpus emitter that produced the frozen fixtures)."""
    from corpus.agents.common import ExecutedToolCall, ExecutionRecord
    from corpus.agents.otel_emit import emit_trace
    from opentelemetry import trace as trace_api

    record = ExecutionRecord(
        agent="orderbot",
        role="specialist",
        parent=None,
        seq=0,
        input={"request": "refund order 42"},
        tool_calls=[
            ExecutedToolCall(
                name="lookup_order", args={"order_id": 42}, result={"status": "shipped"}
            )
        ],
        output={"decision": "deny"},
    )
    emit_trace(
        trace_api.get_tracer("f028.capture.test"),
        [record],
        trace_input={"request": "refund order 42"},
    )


def test_capture_agent_emitting_no_span_is_refused(tmp_path: Path) -> None:
    """A no-op agent (emits no OTel span) → IngestError; nothing persisted."""
    dest = LocalBackend(root=str(tmp_path / "store"))
    with pytest.raises(IngestError, match="0 spans"):
        capture_and_ingest(lambda: None, service_name="noop", trace_id="NOOP", backend=dest)
    assert dest.query() == []


def test_calibrated_agent_evaluation_capture_cli_refuses_silent_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Via CLI: `calibrated-agent-evaluation capture --agent os:getcwd` runs a real no-emit callable
    → exit non-zero with the zero-span refusal (never a silent empty ingest)."""
    monkeypatch.chdir(tmp_path)
    code = main(["capture", "--agent", "os:getcwd", "--service", "silent"])
    err = capsys.readouterr().err
    assert code != 0
    assert "0 spans" in err
    assert "'os:getcwd'" in err, "the refusal names the agent, not the OTel service"
    assert LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).query() == []


def test_calibrated_agent_evaluation_capture_rejects_bad_agent_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Negative: an --agent that is not 'module:callable' is a typed CliError (exit 1)."""
    monkeypatch.chdir(tmp_path)
    code = main(["capture", "--agent", "not_a_module_path"])
    err = capsys.readouterr().err
    assert code != 0
    assert "module:callable" in err


def test_the_console_script_imports_an_agent_module_from_the_current_directory(
    tmp_path: Path,
) -> None:
    """The installed `calibrated-agent-evaluation` script has its bin/ directory, not the cwd, on
    sys.path."""
    import subprocess
    import sys

    (tmp_path / "local_silent_agent.py").write_text("def main():\n    return None\n")
    script = Path(sys.executable).parent / "calibrated-agent-evaluation"
    if not script.exists():
        pytest.skip(
            "the calibrated-agent-evaluation console script is not installed in this environment"
        )
    result = subprocess.run(
        [str(script), "capture", "--agent", "local_silent_agent:main", "--dry-run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert "cannot load agent" not in result.stderr
    # Loaded and run: the agent emits no span, so the capture is refused for that reason.
    assert result.returncode != 0
    assert "0 spans" in result.stderr


def test_an_agent_that_does_not_resolve_is_a_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["capture", "--agent", "no_such_agent_module:main"]) != 0
    err = capsys.readouterr().err
    assert "cannot load agent" in err
    assert "Traceback" not in err


def test_capture_reference_agent_ingests_owned_trace_with_tools(tmp_path: Path) -> None:
    """Capturing an instrumented in-process agent yields an owned Trace whose recovered
    tool calls match what the agent emitted."""
    pytest.importorskip("corpus.agents.otel_emit")  # needs the corpus dev tree
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = capture_and_ingest(
        _emit_reference_trace, service_name="orderbot", trace_id="CAP-1", backend=dest
    )
    assert report.exit_code == 0
    assert report.ingested == 1
    got = dest.get_trace("CAP-1")
    assert got.ownership == "owned"
    assert [tc.name for span in got.spans for tc in span.tool_calls] == ["lookup_order"]


def test_capture_dry_run_reports_but_writes_nothing(tmp_path: Path) -> None:
    """`capture --dry-run` computes the recovery summary but persists no Trace."""
    pytest.importorskip("corpus.agents.otel_emit")
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = capture_and_ingest(
        _emit_reference_trace,
        service_name="orderbot",
        trace_id="CAP-1",
        backend=dest,
        dry_run=True,
    )
    assert report.exit_code == 0
    assert report.results[0].tool_call_count == 1
    assert dest.query() == []


def test_capture_without_strands_extra_names_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative: if the OTel SDK ([strands]) cannot be imported, capture raises a typed error
    naming the extra — never a bare ImportError."""
    monkeypatch.setitem(sys.modules, "calibrated_agent_evaluation.execute.strands_capture", None)
    dest = LocalBackend(root=str(tmp_path / "store"))
    with pytest.raises(IngestError, match="strands") as info:
        capture_and_ingest(lambda: None, service_name="x", trace_id="X", backend=dest)
    # the message names the module, so a gap in the extra itself is diagnosable.
    assert "calibrated_agent_evaluation.execute.strands_capture" in str(info.value)


# --- an unspecified --trace-id must not collide with the last capture --------


def test_two_captures_with_no_trace_id_both_persist(tmp_path: Path) -> None:
    """The defect: `--trace-id` defaulted to `--service`, which defaults to the constant "agent".

    So every capture on the machine that specified neither resolved to the trace id `"agent"`, and
    the store persists with `INSERT OR REPLACE` keyed on trace id — so capture 2 silently replaced
    capture 1 while both runs reported `1 ingested` and exit 0. A capture cannot be re-run, so the
    replaced trace was gone.

    The derived default is the captured OTel trace id, which is unique per run by construction.
    """
    pytest.importorskip("corpus.agents.otel_emit")
    dest = LocalBackend(root=str(tmp_path / "store"))

    first = capture_and_ingest(_emit_reference_trace, service_name="orderbot", backend=dest)
    second = capture_and_ingest(_emit_reference_trace, service_name="orderbot", backend=dest)

    assert first.exit_code == 0, first.results[0].reason
    assert second.exit_code == 0, second.results[0].reason
    ids = {r.trace_id for r in (*first.results, *second.results)}
    assert len(ids) == 2, f"two captures must not share a trace id, got {ids}"
    assert "agent" not in ids, "the service name must not become the trace id"
    assert len(dest.query()) == 2, "both captures must survive in the store"


def test_an_explicit_repeated_trace_id_is_refused_without_overwrite(tmp_path: Path) -> None:
    """Negative: re-using an id the store already holds refuses rather than clobbering.

    Read before write. An id collision is at least as likely to be a mistake as an intention, and
    only the caller knows which — so the caller has to say.
    """
    pytest.importorskip("corpus.agents.otel_emit")
    dest = LocalBackend(root=str(tmp_path / "store"))
    capture_and_ingest(
        _emit_reference_trace, service_name="orderbot", trace_id="CAP-DUP", backend=dest
    )

    with pytest.raises(IngestError) as excinfo:
        capture_and_ingest(
            _emit_reference_trace, service_name="orderbot", trace_id="CAP-DUP", backend=dest
        )
    assert "CAP-DUP" in str(excinfo.value)
    assert "--overwrite" in str(excinfo.value), "the message must name the opt-in"
    assert len(dest.query()) == 1, "the first capture is intact"


class _UnreadableBackend(LocalBackend):
    """A store whose reads fail the way an unreachable or unauthorized backend's do."""

    def get_trace(self, trace_id: str) -> Trace:
        raise BackendError(f"get_trace({trace_id!r}) failed: 503 Service Unavailable")


def test_a_failed_existence_check_refuses_the_capture(tmp_path: Path) -> None:
    """Negative: a read that fails is not proof the id is free, so nothing is replaced."""
    pytest.importorskip("corpus.agents.otel_emit")
    root = str(tmp_path / "store")
    capture_and_ingest(
        _emit_reference_trace,
        service_name="orderbot",
        trace_id="CAP-DUP",
        backend=LocalBackend(root=root),
    )
    before = LocalBackend(root=root).query()

    with pytest.raises(IngestError, match="could not check"):
        capture_and_ingest(
            _emit_reference_trace,
            service_name="orderbot",
            trace_id="CAP-DUP",
            backend=_UnreadableBackend(root=root),
        )
    assert LocalBackend(root=root).query() == before, "the stored trace is untouched"


def test_overwrite_permits_the_replacement(tmp_path: Path) -> None:
    """The opt-in works: with `overwrite=True` the same id is replaced, deliberately."""
    pytest.importorskip("corpus.agents.otel_emit")
    dest = LocalBackend(root=str(tmp_path / "store"))
    capture_and_ingest(
        _emit_reference_trace, service_name="orderbot", trace_id="CAP-DUP", backend=dest
    )
    report = capture_and_ingest(
        _emit_reference_trace,
        service_name="orderbot",
        trace_id="CAP-DUP",
        backend=dest,
        overwrite=True,
    )
    assert report.exit_code == 0
    assert len(dest.query()) == 1


def test_a_dry_run_capture_never_refuses_on_an_existing_id(tmp_path: Path) -> None:
    """A dry run writes nothing, so it has nothing to overwrite and must not be blocked."""
    pytest.importorskip("corpus.agents.otel_emit")
    dest = LocalBackend(root=str(tmp_path / "store"))
    capture_and_ingest(
        _emit_reference_trace, service_name="orderbot", trace_id="CAP-DUP", backend=dest
    )
    report = capture_and_ingest(
        _emit_reference_trace,
        service_name="orderbot",
        trace_id="CAP-DUP",
        backend=dest,
        dry_run=True,
    )
    assert report.exit_code == 0
    assert len(dest.query()) == 1


# --- an agent that raises, and spans that close during teardown -------------------


def _agent_that_raises() -> None:
    raise RuntimeError("tool backend unreachable")


def test_agent_that_raises_is_a_typed_error_naming_the_agent(tmp_path: Path) -> None:
    """A user agent that raises mid-run is an IngestError naming the agent; nothing is persisted.

    It used to escape as the agent's own RuntimeError, outside the CLI's domain-error boundary, so
    `calibrated-agent-evaluation capture` printed a raw traceback instead of a one-line diagnostic.
    """
    dest = LocalBackend(root=str(tmp_path / "store"))
    with pytest.raises(IngestError, match=r"mymod:run_agent.*RuntimeError.*unreachable") as info:
        capture_and_ingest(
            _agent_that_raises,
            service_name="x",
            backend=dest,
            agent_label="mymod:run_agent",
        )
    assert "nothing was persisted" in str(info.value)
    assert isinstance(info.value.__cause__, RuntimeError)
    assert dest.query() == []


def test_calibrated_agent_evaluation_capture_cli_reports_a_raising_agent_in_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Via the CLI: exit 1, the diagnostic names the --agent spec, and the store stays empty."""
    monkeypatch.chdir(tmp_path)
    module = tmp_path / "raising_agent.py"
    module.write_text("def run():\n    raise RuntimeError('boom')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    code = main(["capture", "--agent", "raising_agent:run"])
    err = capsys.readouterr().err
    assert code == 1
    assert "raising_agent:run" in err
    assert "Traceback" not in err
    assert LocalBackend(root=str(tmp_path / ".calibrated-agent-evaluation")).query() == []


def _agent_whose_root_span_closes_on_flush() -> None:
    """An agent whose instrumentation ends its root span when the provider is flushed.

    Models a library that keeps a supervising span open across the call and closes it from its
    flush hook. That span reaches the capture's exporter only during the capture's teardown
    force-flush, so a read taken before teardown loses it.
    """
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.trace import SpanProcessor, TracerProvider

    tracer = trace_api.get_tracer("f028.capture.teardown")
    supervisor = tracer.start_span(
        "invoke_agent supervisor",
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "supervisor",
            "gen_ai.input.messages": '[{"role": "user", "parts": [{"type": "text", '
            '"content": "refund order 42"}]}]',
        },
    )

    class _EndOnFlush(SpanProcessor):
        def force_flush(self, timeout_millis: int = 30000) -> bool:
            if supervisor.is_recording():
                supervisor.end()
            return True

    provider = trace_api.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    provider.add_span_processor(_EndOnFlush())
    with trace_api.use_span(supervisor, end_on_exit=False):
        _emit_reference_trace()


def test_a_span_closing_during_teardown_is_captured(tmp_path: Path) -> None:
    """Spans are read after the capture's force-flush, so one that closes in teardown is kept."""
    pytest.importorskip("corpus.agents.otel_emit")
    dest = LocalBackend(root=str(tmp_path / "store"))
    report = capture_and_ingest(
        _agent_whose_root_span_closes_on_flush,
        service_name="orderbot",
        trace_id="CAP-TEARDOWN",
        backend=dest,
    )
    assert report.exit_code == 0, report.results
    agents = {span.agent for span in dest.get_trace("CAP-TEARDOWN").spans}
    assert "supervisor" in agents, f"the span that closed in teardown was dropped: {agents}"


# --- a captured run equals the oracle on the structural standard ----------------


def test_captured_reference_run_equals_the_oracle_structurally(tmp_path: Path) -> None:
    """Capture an in-process agent that replays ORD-1001's recorded trajectory
    through the corpus emitter, and the persisted Trace equals ``fixtures/normalized/ORD-1001.json``
    on the structural standard (ownership, tree, tool name/args/result).

    The replay makes no model call; what is under test is the capture path (in-process OTel spans,
    OTLP-JSON encoding, detection, normalize, persist), which must not lose or reshape anything.
    """
    pytest.importorskip("corpus.agents.otel_emit")
    from corpus.agents.common import ExecutedToolCall, ExecutionRecord
    from corpus.agents.otel_emit import emit_trace
    from opentelemetry import trace as trace_api

    oracle = json.loads((_REPO / "fixtures" / "normalized" / "ORD-1001.json").read_text("utf-8"))
    (span,) = oracle["spans"]

    def replay() -> None:
        record = ExecutionRecord(
            agent=span["agent"],
            role=span["role"],
            parent=None,
            seq=span["seq"],
            input=span["input"],
            tool_calls=[
                ExecutedToolCall(name=tc["name"], args=tc["args"], result=tc["result"])
                for tc in span["tool_calls"]
            ],
            output=span["output"],
        )
        emit_trace(trace_api.get_tracer("f028.t102"), [record], trace_input=oracle["input"])

    dest = LocalBackend(root=str(tmp_path / "store"))
    report = capture_and_ingest(replay, service_name="orderbot", trace_id="ORD-1001", backend=dest)

    assert report.exit_code == 0
    assert report.results[0].tool_call_count == len(span["tool_calls"])
    assert structural_view(dest.get_trace("ORD-1001")) == structural_view(oracle)
