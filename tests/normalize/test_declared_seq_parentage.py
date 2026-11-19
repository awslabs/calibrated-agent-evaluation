# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A declared ``calibrated_agent_evaluation.span.seq`` never moves a child to another parent.

Parents used to be matched by order value, so a declared seq that equalled another span's fallback
position attached that span's children to the wrong agent. Parents are now resolved by span
identity, and two spans declaring the same seq are refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.ingest.core import run_ingest
from calibrated_agent_evaluation.normalize import OTelSpans, normalize
from calibrated_agent_evaluation.normalize.errors import NormalizationError

_TRACE = "0af7651916cd43dd8448eb211c80319c"
_SEQ = "calibrated_agent_evaluation.span.seq"


def _kv(key: str, value: dict[str, Any]) -> dict[str, Any]:
    return {"key": key, "value": value}


def _agent(span_id: str, parent: str | None, name: str, start: int, seq: int | None = None) -> Any:
    attrs = [
        _kv("gen_ai.operation.name", {"stringValue": "invoke_agent"}),
        _kv("gen_ai.agent.name", {"stringValue": name}),
        _kv("gen_ai.input.content", {"stringValue": f"in {name}"}),
        _kv("gen_ai.output.content", {"stringValue": f"out {name}"}),
    ]
    if seq is not None:
        attrs.append(_kv(_SEQ, {"intValue": str(seq)}))
    span: dict[str, Any] = {
        "traceId": _TRACE,
        "spanId": span_id,
        "name": f"invoke_agent {name}",
        "startTimeUnixNano": str(1_700_000_000_000_000_000 + start * 10**9),
        "endTimeUnixNano": str(1_700_000_000_000_000_000 + (start + 10) * 10**9),
        "attributes": attrs,
    }
    if parent:
        span["parentSpanId"] = parent
    return span


def _write(tmp_path: Path, spans: list[dict[str, Any]]) -> Path:
    path = tmp_path / "seq.otlp.json"
    doc = {"resourceSpans": [{"scopeSpans": [{"scope": {"name": "x"}, "spans": spans}]}]}
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_a_declared_seq_equal_to_a_fallback_position_keeps_the_true_parent(tmp_path: Path) -> None:
    """alpha declares 2, and beta's fallback position is 2; gamma is beta's child."""
    path = _write(
        tmp_path,
        [
            _agent("0000000000000001", None, "sup", 0),
            _agent("0000000000000002", "0000000000000001", "alpha", 1, seq=2),
            _agent("0000000000000003", "0000000000000001", "beta", 2),
            _agent("0000000000000004", "0000000000000003", "gamma", 3),
        ],
    )
    store = LocalBackend(root=str(tmp_path / "store"))
    (result,) = run_ingest([path], backend=store).results
    assert result.status == "ingested", result.reason
    assert result.trace_id is not None
    spans = store.get_trace(result.trace_id).spans
    agent_of = {s.span_id: s.agent for s in spans}
    parents = {s.agent: agent_of.get(s.parent) if s.parent else None for s in spans}
    assert parents == {"sup": None, "alpha": "sup", "beta": "sup", "gamma": "beta"}


def test_two_spans_declaring_the_same_seq_are_refused(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            _agent("0000000000000001", None, "sup", 0, seq=0),
            _agent("0000000000000002", "0000000000000001", "alpha", 1, seq=1),
            _agent("0000000000000003", "0000000000000001", "beta", 2, seq=1),
        ],
    )
    with pytest.raises(NormalizationError, match=r"declare the same .*seq 1.*alpha.*beta"):
        normalize(OTelSpans(path))
    store = LocalBackend(root=str(tmp_path / "store"))
    (result,) = run_ingest([path], backend=store).results
    assert result.status == "skipped"
    assert "seq 1" in (result.reason or "")
