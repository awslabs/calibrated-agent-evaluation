# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingest refuses silent loss and silent replacement: id-less calls, nested tools, reused ids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli.dispatch import main
from calibrated_agent_evaluation.ingest import run_ingest


def _kv(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _write_otlp(path: Path, spans: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}), encoding="utf-8"
    )
    return path


def _gen_ai_span(
    span_id: str, parent: str | None, op: str, name: str, *attrs: dict[str, Any]
) -> dict[str, Any]:
    span: dict[str, Any] = {
        "traceId": "cd" * 16,
        "spanId": span_id,
        "name": name,
        "startTimeUnixNano": str(len(span_id) + ord(span_id[0])),
        "attributes": [_kv("gen_ai.operation.name", op), *attrs],
    }
    if parent:
        span["parentSpanId"] = parent
    return span


def _agent_trace(
    path: Path, *, message: str, nested_tool: bool = True, wire_frame: str | None = None
) -> Path:
    agent = [_kv("gen_ai.agent.name", "bot"), _kv("gen_ai.input.content", message)]
    if wire_frame:
        agent.append(_kv("calibrated_agent_evaluation.frame", wire_frame))
    spans = [
        _gen_ai_span("a" * 16, None, "invoke_agent", "invoke_agent bot", *agent),
        _gen_ai_span(
            "b" * 16,
            "a" * 16,
            "execute_tool",
            "lookup",
            _kv("gen_ai.tool.name", "lookup"),
            _kv("gen_ai.tool.call.arguments", '{"id": 1}'),
        ),
    ]
    if nested_tool:
        spans += [
            _gen_ai_span("c" * 16, "a" * 16, "chat", "chat"),
            _gen_ai_span(
                "d" * 16,
                "c" * 16,
                "execute_tool",
                "refund",
                _kv("gen_ai.tool.name", "refund"),
                _kv("gen_ai.tool.call.arguments", '{"amt": 500}'),
            ),
        ]
    return _write_otlp(path, spans)


def test_an_id_less_openinference_call_ingests_as_a_tool_call(tmp_path: Path) -> None:
    attr = "llm.output_messages.0.message.tool_calls.0.tool_call.function."
    span = {
        "traceId": "ab" * 16,
        "spanId": "01" * 8,
        "startTimeUnixNano": "1",
        "endTimeUnixNano": "2",
        "attributes": [
            _kv("openinference.span.kind", "LLM"),
            _kv("input.value", "refund order 7"),
            _kv("output.value", "done"),
            _kv("llm.output_messages.0.message.role", "assistant"),
            _kv(attr + "name", "issue_refund"),
            _kv(attr + "arguments", '{"amount": 500}'),
        ],
    }
    report = run_ingest(
        [_write_otlp(tmp_path / "oi.json", [span])], backend=LocalBackend(":memory:")
    )
    (result,) = report.results
    assert (result.status, result.tool_call_count) == ("ingested", 1)


def test_a_tool_under_an_intermediate_span_is_recovered(tmp_path: Path) -> None:
    path = _agent_trace(tmp_path / "nested.otlp.json", message="q")
    with LocalBackend(tmp_path / "store") as store:
        report = run_ingest([path], backend=store)
        (result,) = report.results
        assert (result.status, result.tool_call_count, result.unrecovered_tool_spans) == (
            "ingested",
            2,
            0,
        )
        calls = [c.name for s in store.get_trace("nested").spans for c in s.tool_calls]
    assert calls == ["lookup", "refund"]


def test_a_partially_recovered_trace_is_refused_unless_partial_tools_are_allowed(
    tmp_path: Path,
) -> None:
    # A tool span whose parent is outside the file links to no agent: one of two calls is lost.
    path = _agent_trace(tmp_path / "partial.otlp.json", message="q", nested_tool=False)
    doc = json.loads(path.read_text())
    orphan = _gen_ai_span("e" * 16, "f" * 16, "execute_tool", "refund")
    orphan["attributes"].append(_kv("gen_ai.tool.name", "refund"))
    doc["resourceSpans"][0]["scopeSpans"][0]["spans"].append(orphan)
    path.write_text(json.dumps(doc))

    refused = run_ingest([path], backend=LocalBackend(":memory:"))
    (result,) = refused.results
    assert result.status == "refused"
    assert "recovered 1" in str(result.reason)
    assert "--allow-partial-tools" in str(result.reason)
    assert refused.exit_code == 1

    allowed = run_ingest([path], backend=LocalBackend(":memory:"), allow_partial_tools=True)
    (result,) = allowed.results
    assert (result.status, result.unrecovered_tool_spans) == ("ingested", 1)


def test_a_second_run_reusing_a_file_name_does_not_replace_the_first(tmp_path: Path) -> None:
    monday = _agent_trace(tmp_path / "mon" / "trace.otlp.json", message="refund order 1")
    tuesday = _agent_trace(tmp_path / "tue" / "trace.otlp.json", message="cancel order 2")
    with LocalBackend(tmp_path / "store") as store:
        assert run_ingest([monday], backend=store).exit_code == 0
        report = run_ingest([tuesday], backend=store)
        (result,) = report.results
        assert result.status == "refused"
        assert "--overwrite" in str(result.reason)
        assert "refund order 1" in json.dumps(store.get_trace("trace").input)


def test_re_ingesting_the_identical_trace_is_idempotent(tmp_path: Path) -> None:
    path = _agent_trace(tmp_path / "trace.otlp.json", message="q")
    with LocalBackend(tmp_path / "store") as store:
        assert run_ingest([path], backend=store).exit_code == 0
        assert run_ingest([path], backend=store).exit_code == 0


def test_overwrite_replaces_a_stored_trace(tmp_path: Path) -> None:
    monday = _agent_trace(tmp_path / "mon" / "trace.otlp.json", message="refund order 1")
    tuesday = _agent_trace(tmp_path / "tue" / "trace.otlp.json", message="cancel order 2")
    with LocalBackend(tmp_path / "store") as store:
        run_ingest([monday], backend=store)
        assert run_ingest([tuesday], backend=store, overwrite=True).exit_code == 0
        assert "cancel order 2" in json.dumps(store.get_trace("trace").input)


def test_a_file_with_no_declared_frame_ingests_as_discovery(tmp_path: Path) -> None:
    undeclared = _agent_trace(tmp_path / "plain.otlp.json", message="q")
    declared = _agent_trace(
        tmp_path / "tagged.otlp.json", message="q", wire_frame="production-representative"
    )
    with LocalBackend(tmp_path / "store") as store:
        run_ingest([undeclared, declared], backend=store)
        assert store.get_trace("plain").frame == "discovery"
        assert store.get_trace("tagged").frame == "production-representative"


def test_the_frame_flag_sets_the_frame_of_undeclared_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _agent_trace(tmp_path / "plain.otlp.json", message="q")
    assert main(["ingest", "--frame", "production-representative", str(path)]) == 0
    with LocalBackend(tmp_path / ".calibrated-agent-evaluation") as store:
        assert store.get_trace("plain").frame == "production-representative"


def test_the_cli_refuses_a_reused_id_and_overwrite_replaces_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monday = _agent_trace(tmp_path / "mon" / "trace.otlp.json", message="refund order 1")
    tuesday = _agent_trace(tmp_path / "tue" / "trace.otlp.json", message="cancel order 2")
    assert main(["ingest", str(monday)]) == 0
    assert main(["ingest", str(tuesday)]) == 1
    assert "already holds a different trace" in capsys.readouterr().err
    assert main(["ingest", "--overwrite", str(tuesday)]) == 0


def test_content_the_normalizer_cannot_handle_is_skipped_not_raised(tmp_path: Path) -> None:
    good = _agent_trace(tmp_path / "good.otlp.json", message="q")
    bad = _agent_trace(tmp_path / "bad.otlp.json", message="q")
    doc = json.loads(bad.read_text())
    doc["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"] = 7  # not a list
    bad.write_text(json.dumps(doc))

    report = run_ingest([bad, good], backend=LocalBackend(":memory:"))
    statuses = {Path(r.source).name: r.status for r in report.results}
    assert statuses == {"bad.otlp.json": "skipped", "good.otlp.json": "ingested"}
