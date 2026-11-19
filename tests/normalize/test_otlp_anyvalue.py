# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTLP ``AnyValue`` arrays and key-value lists decode to Python lists and dicts.

``scalar()`` used to return the wire shape for ``arrayValue`` and ``kvlistValue``, so a tool call
whose arguments rode as a kvlist was stored with ``args={'values': [...]}``. The binary path decodes
through ``MessageToDict`` into the same JSON shape, so both reach the same decoder.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.ingest.core import run_ingest
from calibrated_agent_evaluation.normalize.otlp_json import scalar

_TRACE = "0af7651916cd43dd8448eb211c80319c"


def _kv(key: str, value: dict[str, Any]) -> dict[str, Any]:
    return {"key": key, "value": value}


def test_scalars_arrays_and_kvlists_decode_recursively() -> None:
    raw = {
        "kvlistValue": {
            "values": [
                _kv("city", {"stringValue": "Paris"}),
                _kv("days", {"intValue": "3"}),
                _kv(
                    "units", {"arrayValue": {"values": [{"stringValue": "C"}, {"boolValue": True}]}}
                ),
                _kv("nested", {"kvlistValue": {"values": [_kv("x", {"doubleValue": 1.5})]}}),
                _kv("unset", {}),
            ]
        }
    }
    assert scalar(raw) == {
        "city": "Paris",
        "days": 3,
        "units": ["C", True],
        "nested": {"x": 1.5},
        "unset": None,
    }
    assert scalar({"arrayValue": {}}) == []
    assert scalar({"kvlistValue": {}}) == {}


def test_bytes_stay_in_their_base64_json_encoding() -> None:
    encoded = base64.b64encode(b"\x00\x01binary").decode()
    assert scalar({"bytesValue": encoded}) == encoded
    assert scalar({"arrayValue": {"values": [{"bytesValue": encoded}]}}) == [encoded]


def _tool_call_trace(arguments: dict[str, Any]) -> dict[str, Any]:
    agent = {
        "traceId": _TRACE,
        "spanId": "b7ad6b7169203331",
        "name": "invoke_agent a",
        "startTimeUnixNano": "1700000000000000000",
        "endTimeUnixNano": "1700000005000000000",
        "attributes": [
            _kv("gen_ai.operation.name", {"stringValue": "invoke_agent"}),
            _kv("gen_ai.agent.name", {"stringValue": "a"}),
            _kv("gen_ai.input.content", {"stringValue": "weather in Paris?"}),
            _kv("gen_ai.output.content", {"stringValue": "sunny"}),
        ],
    }
    tool = {
        "traceId": _TRACE,
        "spanId": "c7ad6b7169203331",
        "parentSpanId": "b7ad6b7169203331",
        "name": "execute_tool weather",
        "startTimeUnixNano": "1700000001000000000",
        "endTimeUnixNano": "1700000002000000000",
        "attributes": [
            _kv("gen_ai.operation.name", {"stringValue": "execute_tool"}),
            _kv("gen_ai.tool.name", {"stringValue": "weather"}),
            _kv("gen_ai.tool.call.arguments", arguments),
            _kv("gen_ai.tool.call.result", {"stringValue": "sunny"}),
        ],
    }
    return {"resourceSpans": [{"scopeSpans": [{"scope": {"name": "x"}, "spans": [agent, tool]}]}]}


def _binary(doc: dict[str, Any], dest: Path) -> Path:
    from google.protobuf.json_format import ParseDict
    from opentelemetry.proto.trace.v1.trace_pb2 import TracesData

    wire = json.loads(json.dumps(doc))
    for resource in wire["resourceSpans"]:
        for scope in resource["scopeSpans"]:
            for span in scope["spans"]:
                for key in ("traceId", "spanId", "parentSpanId"):
                    if span.get(key):
                        span[key] = base64.b64encode(bytes.fromhex(span[key])).decode()
    dest.write_bytes(ParseDict(wire, TracesData(), ignore_unknown_fields=True).SerializeToString())
    return dest


@pytest.mark.parametrize("binary", [False, True], ids=["json", "protobuf"])
def test_kvlist_tool_arguments_are_stored_as_a_dict(tmp_path: Path, binary: bool) -> None:
    pytest.importorskip("opentelemetry.proto")
    kvlist = {"kvlistValue": {"values": [_kv("city", {"stringValue": "Paris"})]}}
    doc = _tool_call_trace(kvlist)
    if binary:
        path = _binary(doc, tmp_path / "kv.otlp")
    else:
        path = tmp_path / "kv.otlp.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
    store = LocalBackend(root=str(tmp_path / "store"))
    (result,) = run_ingest([path], backend=store).results
    assert result.status == "ingested", result.reason
    assert result.trace_id is not None
    calls = [c for s in store.get_trace(result.trace_id).spans for c in s.tool_calls]
    assert [(c.name, c.args) for c in calls] == [("weather", {"city": "Paris"})]


_RAW = Path(__file__).resolve().parents[2] / "fixtures" / "raw"


def _hex_ids(doc: dict[str, Any]) -> bool:
    try:
        for resource in doc.get("resourceSpans", []):
            for scope in resource.get("scopeSpans", []):
                for span in scope.get("spans", []):
                    for key in ("traceId", "spanId", "parentSpanId"):
                        if span.get(key):
                            bytes.fromhex(span[key])
    except ValueError:
        return False
    return True


_BINARY_FIXTURES = sorted(
    path
    for path in _RAW.rglob("*.otlp.json")
    if _hex_ids(json.loads(path.read_text(encoding="utf-8")))
)


@pytest.mark.parametrize("fixture", _BINARY_FIXTURES, ids=lambda p: str(p.relative_to(_RAW)))
def test_a_fixture_normalizes_the_same_from_json_and_from_protobuf(
    fixture: Path, tmp_path: Path
) -> None:
    pytest.importorskip("opentelemetry.proto")
    stem = fixture.name.split(".")[0]
    as_json = tmp_path / "json" / f"{stem}.otlp.json"
    as_json.parent.mkdir()
    as_json.write_bytes(fixture.read_bytes())
    as_binary = (tmp_path / "bin").joinpath(f"{stem}.otlp")
    as_binary.parent.mkdir()
    _binary(json.loads(fixture.read_text(encoding="utf-8")), as_binary)

    def ingest(path: Path, name: str) -> Any:
        store = LocalBackend(root=str(tmp_path / name))
        results = run_ingest([path], backend=store).results
        return [(r.status, r.trace_id) for r in results], [
            store.get_trace(str(r.trace_id)) for r in results if r.status == "ingested"
        ]

    assert ingest(as_json, "a") == ingest(as_binary, "b")
