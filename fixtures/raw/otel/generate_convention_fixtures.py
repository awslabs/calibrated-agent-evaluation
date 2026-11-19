# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the per-convention OTLP-JSON fixtures for dataset ingestion from REAL vendor sources.

The attribute **keys and values** below are genuine vendor output, not invented (repo rule:
"fixture traces are genuine vendor output, never hand-shaped" — see `fixtures/README.md`):

  * ``openinference/`` — the exact attribute set published in the Arize OpenInference repo
    (`js/.../openinference_client_span_tool_calls.json`, Apache-2.0, commit
    930e41adaab8318e093ed24a68cbeea098e41b13). Copied verbatim.
  * ``openllmetry/`` — the dotted-index attribute grammar OpenLLMetry emits, verified against its
    `v0.33.12` instrumentation source + `opentelemetry-semantic-conventions-ai` constants, filled
    with the **real recorded values** from its own VCR cassette
    (`tests/traces/cassettes/test_functions/test_open_ai_function_calls_tools.yaml`, Apache-2.0):
    user "What's the weather like in Boston?", tool_call `call_b4lU3wKOqwO7Xb2Ksok2XWyB`
    `get_current_weather({"location":"Boston"})`, model `gpt-4-0613`.
  * ``messages/``: extracted from the REAL committed dataset slice
    (`fixtures/raw/hf/swebench-slice.parquet`), so this convention's fixture is a genuine capture.

This script only **wraps** those real attribute sets in the canonical OTLP-JSON envelope
(`resourceSpans → scopeSpans → spans`, `KeyValue`/`AnyValue`) that an OTLP/HTTP exporter emits —
the envelope is transport, not content. Re-run to regenerate; provenance is in `PROVENANCE.md`.

    uv run python fixtures/raw/otel/generate_convention_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent


def _any_value(v: Any) -> dict[str, Any]:
    """Encode a Python scalar as an OTLP ``AnyValue`` (the real wire encoding)."""
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}  # OTLP encodes int64 as a decimal string
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}


def _kv(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _any_value(v)} for k, v in attrs.items()]


def _otlp(span_attrs: dict[str, Any], *, name: str, service: str) -> dict[str, Any]:
    """Wrap one span's real attribute map in the canonical OTLP-JSON envelope."""
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _kv({"service.name": service})},
                "scopeSpans": [
                    {
                        "scope": {"name": service},
                        "spans": [
                            {
                                "traceId": "f026f026f026f026f026f026f026f026",
                                "spanId": "f026000000000001",
                                "name": name,
                                "kind": "SPAN_KIND_CLIENT",
                                "startTimeUnixNano": "1",
                                "endTimeUnixNano": "2",
                                "attributes": _kv(span_attrs),
                                "status": {"code": 1},
                            }
                        ],
                    }
                ],
            }
        ]
    }


# --- OpenInference: verbatim real attribute set (Arize openinference, Apache-2.0) -----------
# Copied exactly from the published fixture (see module docstring); no key/value altered.
_OPENINFERENCE_ATTRS: dict[str, Any] = {
    "openinference.span.kind": "LLM",
    "llm.model_name": "gpt-4-0613",
    "llm.system": "openai",
    "llm.provider": "openai",
    "llm.input_messages.0.message.role": "user",
    "llm.input_messages.0.message.contents.0.message_content.type": "text",
    "llm.input_messages.0.message.contents.0.message_content.text": "Weather in Paris?",
    "llm.input_messages.1.message.role": "assistant",
    "llm.input_messages.1.message.contents.0.message_content.type": "text",
    "llm.input_messages.1.message.contents.0.message_content.text": (
        "Absolutely! Let me check the weather for you."
    ),
    "llm.input_messages.1.message.tool_calls.0.tool_call.id": "1234",
    "llm.input_messages.1.message.tool_calls.0.tool_call.function.name": "get_weather",
    "llm.input_messages.1.message.tool_calls.0.tool_call.function.arguments": (
        '{"location":"Paris"}'
    ),
    "llm.input_messages.2.message.role": "tool",
    "llm.input_messages.2.message.content": "rainy, 57°F",
    "llm.input_messages.2.message.tool_call_id": "1234",
    "input.value": '{"model":"gpt-4","messages":[{"role":"user"}]}',
    "input.mime_type": "application/json",
    "llm.output_messages.0.message.role": "assistant",
    "llm.output_messages.0.message.contents.0.message_content.type": "text",
    "llm.output_messages.0.message.contents.0.message_content.text": (
        "The weather in Paris is currently rainy with a temperature of 57°F."
    ),
    "output.value": '{"text":"The weather in Paris is currently rainy, 57°F."}',
    "output.mime_type": "application/json",
    "llm.token_count.completion": 52,
    "llm.token_count.prompt": 97,
    "llm.token_count.total": 149,
}


# --- OpenLLMetry legacy: verified dotted-index grammar + real recorded cassette values ------
# Grammar verified against traceloop/openllmetry v0.33.12 instrumentation source; values are the
# real recorded API interaction from that repo's own VCR cassette (see module docstring).
_OPENLLMETRY_ATTRS: dict[str, Any] = {
    "gen_ai.system": "openai",
    "gen_ai.request.model": "gpt-4",
    "gen_ai.response.model": "gpt-4-0613",
    "llm.request.type": "chat",
    "gen_ai.prompt.0.role": "user",
    "gen_ai.prompt.0.content": "What's the weather like in Boston?",
    "gen_ai.completion.0.role": "assistant",
    "gen_ai.completion.0.finish_reason": "tool_calls",
    "gen_ai.completion.0.tool_calls.0.id": "call_b4lU3wKOqwO7Xb2Ksok2XWyB",
    "gen_ai.completion.0.tool_calls.0.name": "get_current_weather",
    "gen_ai.completion.0.tool_calls.0.arguments": '{\n  "location": "Boston"\n}',
}


# The smallest structured-tool-call session in the committed slice.
_MESSAGES_SESSION = "e3bfdef4aca3_c0427304"


def _messages_span_from_slice() -> tuple[dict[str, Any], str]:
    """Extract one real ``gen_ai.*.messages`` span from the committed dataset slice.

    Returns the span attributes and the ``service.name`` the capture recorded. Requires the
    ``[datasets]`` extra (pyarrow) to read the parquet; the produced OTLP-JSON fixture itself is
    then read with the stdlib.
    """
    import pyarrow.parquet as pq

    slice_path = _HERE.parent / "hf" / "swebench-slice.parquet"
    rows = pq.read_table(slice_path).to_pylist()
    (row,) = [r for r in rows if r["session_id"] == _MESSAGES_SESSION]
    for span in row["spans"]:
        attrs = span.get("attributes") or {}
        out = attrs.get("gen_ai.output.messages") or ""
        if '"tool_call"' in out:
            model = attrs["gen_ai.request.model"]
            service = (span.get("resource_attributes") or {})["service.name"]
            return {
                "gen_ai.operation.name": "chat",
                "gen_ai.system": _system_from_model(model, service),
                "gen_ai.request.model": model,
                "gen_ai.input.messages": attrs.get("gen_ai.input.messages") or "[]",
                "gen_ai.output.messages": out,
            }, service
    raise SystemExit(f"{_MESSAGES_SESSION}: no tool-bearing gen_ai/messages span in the slice")


def _system_from_model(model: str, service: str) -> str:
    """The provider from a LiteLLM ``provider/model`` string; the rows carry no provider attribute.

    The spans were emitted by LiteLLM (``service.name``), whose model strings name the provider
    before the slash (``azure/DeepSeek-V3.2`` is served through Azure).
    """
    provider, sep, _ = model.partition("/")
    if service != "litellm" or not sep or not provider:
        raise SystemExit(f"cannot derive gen_ai.system from model {model!r} (service {service!r})")
    return provider


def main() -> None:
    (_HERE / "openinference").mkdir(exist_ok=True)
    (_HERE / "openllmetry").mkdir(exist_ok=True)
    (_HERE / "messages").mkdir(exist_ok=True)

    _write(
        _HERE / "openinference" / "weather_tool_call.otlp.json",
        _otlp(_OPENINFERENCE_ATTRS, name="ChatCompletion", service="openinference-openai"),
    )
    _write(
        _HERE / "openllmetry" / "weather_tool_call.otlp.json",
        _otlp(_OPENLLMETRY_ATTRS, name="openai.chat", service="openllmetry-openai"),
    )
    messages_attrs, messages_service = _messages_span_from_slice()
    _write(
        _HERE / "messages" / "swebench_tool_call.otlp.json",
        _otlp(messages_attrs, name="chat", service=messages_service),
    )
    print("wrote 3 convention fixtures under", _HERE)


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("  ", path.relative_to(_HERE.parent.parent), f"({path.stat().st_size} B)")


if __name__ == "__main__":
    main()
