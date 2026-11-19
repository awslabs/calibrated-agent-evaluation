# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parquet dataset ingestion → normalized Traces.

Runs against the committed real `Exgentic/agent-llm-traces` slice
(`fixtures/raw/hf/swebench-slice.parquet`), so the loader is tested on genuine captured output.
Covers: one owned Trace per row with reconstructed+paired tool calls,
multi-row-group streaming (an edge case), corrupt-file typed error, the flat one-span
collapse, and the same collapse on real multi-span sessions.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from calibrated_agent_evaluation.normalize import SessionSpans, normalize
from calibrated_agent_evaluation.normalize.datasets import load_traces
from calibrated_agent_evaluation.normalize.errors import NormalizationError
from calibrated_agent_evaluation.normalize.sources import ParquetDataset
from calibrated_agent_evaluation.schema.validate import validate

_REPO = Path(__file__).resolve().parents[2]
_SLICE = _REPO / "fixtures" / "raw" / "hf" / "swebench-slice.parquet"
_CLAUDE_CODE_SESSION = "e3bfdef4aca3_c0427304"  # a claude_code swebench session in the slice
_TOOL_CALLING_SESSION = "c0d1d84c7898_35f42bfd"  # 12 chat spans, structured tool calls
_CODE_ACTION_SESSION = "140646846ae7_8d10c195"  # smolagents_code: tool use in assistant text


def _load() -> list:
    return list(load_traces(ParquetDataset(_SLICE)))


def test_parquet_yields_one_owned_trace_per_row_with_paired_tools() -> None:
    traces = _load()
    assert len(traces) == 3  # the committed slice has 3 whole sessions

    session = next(t for t in traces if t.trace_id == _CLAUDE_CODE_SESSION)
    validate(session)
    assert session.ownership == "owned"

    # Collapse-first one-span tree: a single root Span, parent null, seq 0.
    assert len(session.spans) == 1
    root = session.spans[0]
    assert root.parent is None
    assert root.seq == 0

    # The TodoWrite call is reconstructed with non-empty args and paired to its result by id.
    todo = next((c for c in root.tool_calls if c.name == "TodoWrite"), None)
    assert todo is not None, "TodoWrite tool call not reconstructed"
    assert todo.args.get("todos"), "TodoWrite args.todos should be non-empty"
    assert todo.result, "TodoWrite result should be paired (present across the session's spans)"
    assert len(root.tool_calls) > 1, "the session's multi-step tool sequence should be recovered"


def test_streams_across_all_row_groups_not_only_the_first() -> None:
    # The slice stores one session per row group, so a first-row-group-only reader would drop
    # two of the three sessions.
    assert pq.read_metadata(_SLICE).num_row_groups == 3
    ids = {t.trace_id for t in _load()}
    assert ids == {_CLAUDE_CODE_SESSION, _TOOL_CALLING_SESSION, _CODE_ACTION_SESSION}


def test_corrupt_parquet_raises_typed_error_not_raw_exception(tmp_path: Path) -> None:
    # A non-parquet file must surface as a typed NormalizationError, never a raw pyarrow/OS error.
    bogus = tmp_path / "not-a-parquet.parquet"
    bogus.write_text("this is definitely not parquet\n")
    with pytest.raises(NormalizationError):
        list(load_traces(ParquetDataset(bogus)))


def test_missing_parquet_file_raises_typed_error(tmp_path: Path) -> None:
    with pytest.raises(NormalizationError):
        list(load_traces(ParquetDataset(tmp_path / "does-not-exist.parquet")))


def test_flat_multi_span_session_collapses_to_exactly_one_root_span() -> None:
    # a flat sequence of ≥2 chat spans (no parent_span_id) collapses to ONE root Span
    # carrying the tool calls in call order — not N sibling spans, not a synthesized parent.
    spans = [
        {
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.output.messages": (
                    '[{"role":"assistant","parts":[{"type":"tool_call","id":"a",'
                    '"name":"lookup","arguments":{"q":1}}]}]'
                ),
            }
        },
        {
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.input.messages": (
                    '[{"role":"tool","parts":[{"type":"tool_call_response","id":"a",'
                    '"result":{"ok":true}}]}]'
                ),
                "gen_ai.output.messages": (
                    '[{"role":"assistant","parts":[{"type":"tool_call","id":"b",'
                    '"name":"issue","arguments":{"amt":5}}]}]'
                ),
            }
        },
    ]
    trace = normalize(SessionSpans(spans=spans, trace_id="FLAT-1"))
    validate(trace)
    assert trace.ownership == "owned"
    assert len(trace.spans) == 1, "a flat session must collapse to exactly one root span"
    root = trace.spans[0]
    assert root.parent is None
    assert root.seq == 0
    assert [c.name for c in root.tool_calls] == ["lookup", "issue"], "call order preserved"
    assert root.tool_calls[0].result == {"ok": True}  # 'a' paired
    assert root.tool_calls[1].result == {}  # 'b' unpaired → empty, not mis-joined


def test_real_multi_span_sessions_collapse_to_one_root_span() -> None:
    # The real sessions are flat lists of chat spans with no parent_span_id and no
    # invoke_agent/execute_tool boundary spans. Each must collapse to ONE root span, not N
    # root-less siblings that would fail validate().
    table = pq.read_table(_SLICE, columns=["session_id", "spans"]).to_pylist()
    raw = {row["session_id"]: row["spans"] for row in table}
    traces = {t.trace_id: t for t in _load()}
    for session_id in (_TOOL_CALLING_SESSION, _CODE_ACTION_SESSION):
        spans = raw[session_id]
        assert len(spans) > 1
        assert all(not span.get("parent_span_id") for span in spans)
        trace = traces[session_id]
        validate(trace)
        assert trace.ownership == "owned"
        assert len(trace.spans) == 1, f"{session_id} must not produce N sibling spans"

    tool_calling = traces[_TOOL_CALLING_SESSION].spans[0]
    assert [c.name for c in tool_calling.tool_calls].count("bash") == 8
    # The code-action session has no structured tool_call parts, so nothing is invented.
    assert traces[_CODE_ACTION_SESSION].spans[0].tool_calls == []
