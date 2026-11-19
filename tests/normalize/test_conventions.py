# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared OTel convention layer + hybrid detection.

Against the real per-convention OTLP-JSON fixtures (see `fixtures/raw/otel/PROVENANCE.md`): each
convention auto-detects and reconstructs its tool calls, the original grammars still
normalize under the auto default, and the honest-failure paths (undetectable
convention, malformed args) raise typed errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrated_agent_evaluation.normalize import OTelSpans, SessionSpans, normalize
from calibrated_agent_evaluation.normalize.conventions.detect import detect
from calibrated_agent_evaluation.normalize.errors import NormalizationError
from calibrated_agent_evaluation.schema.validate import validate

_REPO = Path(__file__).resolve().parents[2]
_OTEL = _REPO / "fixtures" / "raw" / "otel"


def _spans(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return [s for rs in doc["resourceSpans"] for ss in rs["scopeSpans"] for s in ss["spans"]]


def _only(conv_dir: str) -> Path:
    return next((_OTEL / conv_dir).glob("*.otlp.json"))


CONVENTIONS = [
    ("messages", "gen_ai/messages", "TodoWrite"),
    ("openinference", "openinference", "get_weather"),
    ("openllmetry", "openllmetry", "get_current_weather"),
]


@pytest.mark.parametrize(("conv_dir", "expected_kind", "tool_name"), CONVENTIONS)
def test_each_convention_auto_detects_and_reconstructs(
    conv_dir: str, expected_kind: str, tool_name: str
) -> None:
    path = _only(conv_dir)
    # Detection picks the right convention with NO declared semconv...
    assert detect(_spans(path)) == expected_kind
    # ...and normalize() (default semconv="auto") reconstructs an owned Trace with the tool call.
    trace = normalize(OTelSpans(path))
    validate(trace)
    assert trace.ownership == "owned"
    names = [c.name for s in trace.spans for c in s.tool_calls]
    assert tool_name in names, f"{conv_dir}: expected tool {tool_name!r} in {names}"
    # Every reconstructed call carries non-empty args (the reconstruction actually recovered them).
    for span in trace.spans:
        for call in span.tool_calls:
            assert call.args, f"{conv_dir}: {call.name} args empty"


def test_grammars_still_normalize_under_the_auto_default() -> None:
    # Regression: the own OTLP fixtures (invoke_agent/execute_tool grammar) must keep
    # normalizing to owned Traces under the new auto default — the shared layer detects them as
    # gen_ai/latest and routes to the reader, unchanged.
    events = normalize(OTelSpans(_OTEL / "ORD-1001.events.otlp.json", trace_id="ORD-1001"))
    attrs = normalize(OTelSpans(_OTEL / "ORD-1001.attrs.otlp.json", trace_id="ORD-1001"))
    for trace in (events, attrs):
        validate(trace)
        assert trace.ownership == "owned"
        assert len(trace.spans) == 1
        assert len(trace.spans[0].tool_calls) == 3  # the three OrderBot tools, as before


def test_openinference_recovers_terminal_io_from_value_blobs() -> None:
    # OpenInference carries input.value/output.value JSON blobs; terminal I/O is recovered.
    trace = normalize(OTelSpans(_only("openinference")))
    validate(trace)
    assert trace.input, "terminal input should be recovered (input.value or first user message)"
    assert trace.output, "terminal output should be recovered (output.value or assistant message)"


def test_messages_pairs_result_under_response_key_and_leaves_unpaired_empty() -> None:
    # A messages session whose tool result rides the `response` key (latest-OpenLLMetry dialect) is
    # paired; a tool_call with no matching response reconstructs empty (never mis-joined). Own path.
    spans = [
        {
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.output.messages": (
                    '[{"role":"assistant","parts":['
                    '{"type":"tool_call","id":"p","name":"paid","arguments":{"x":1}},'
                    '{"type":"tool_call","id":"u","name":"unpaired","arguments":{"y":2}}]}]'
                ),
                "gen_ai.input.messages": (
                    '[{"role":"tool","parts":[{"type":"tool_call_response","id":"p",'
                    '"response":{"ok":true}}]}]'
                ),
            }
        }
    ]
    trace = normalize(SessionSpans(spans=spans, semconv="gen_ai/messages", trace_id="RESP-1"))
    validate(trace)
    calls = {c.name: c for s in trace.spans for c in s.tool_calls}
    assert calls["paid"].result == {"ok": True}, "result under 'response' key should pair"
    assert calls["unpaired"].result == {}, "unpaired call must reconstruct empty, not mis-joined"


def test_openllmetry_unpaired_completion_tool_and_latest_fallthrough() -> None:
    # Legacy dotted-index: a completion tool_call with no matching prompt tool_call_id → empty
    # result (not mis-paired).
    legacy = normalize(OTelSpans(_only("openllmetry")))
    validate(legacy)
    tool = next(c for s in legacy.spans for c in s.tool_calls if c.name == "get_current_weather")
    assert tool.result == {}, "unpaired legacy completion tool_call → empty result"

    # Latest fall-through: an openllmetry source in the latest JSON messages shape is handled by the
    # shared messages reconstruction (result under `response`), via the openllmetry adapter.
    spans = [
        {
            "attributes": {
                "gen_ai.system": "openai",
                "gen_ai.output.messages": (
                    '[{"role":"assistant","parts":[{"type":"tool_call","id":"z",'
                    '"name":"z_tool","arguments":{}}]}]'
                ),
                "gen_ai.input.messages": (
                    '[{"role":"tool","parts":[{"type":"tool_call_response","id":"z",'
                    '"response":{"done":1}}]}]'
                ),
            }
        }
    ]
    latest = normalize(SessionSpans(spans=spans, semconv="openllmetry", trace_id="OLM-LATEST"))
    validate(latest)
    z = next(c for s in latest.spans for c in s.tool_calls if c.name == "z_tool")
    assert z.result == {"done": 1}, "latest-JSON openllmetry falls through to messages pairing"


def test_openinference_non_json_arguments_wrapped_not_raw() -> None:
    # A tool_call arguments value that is not JSON must not crash the adapter with a raw error;
    # content coercion wraps it. (The adapter's read boundary never leaks a raw parser exception.)
    spans = [
        {
            "attributes": {
                "openinference.span.kind": "LLM",
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.tool_calls.0.tool_call.id": "t1",
                "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "f",
                "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": (
                    "not-json-at-all"
                ),
            }
        }
    ]
    trace = normalize(SessionSpans(spans=spans, semconv="openinference", trace_id="OI-BAD"))
    validate(trace)
    f = next(c for s in trace.spans for c in s.tool_calls if c.name == "f")
    assert f.args == {"answer": "not-json-at-all"}, "non-JSON args are wrapped, not dropped/raised"


def test_undetectable_convention_raises_naming_candidates() -> None:
    # A span whose attributes match no known convention (and no override) → typed raise, never a
    # silent guess. The message names what was considered.
    with pytest.raises(NormalizationError) as exc:
        normalize(OTelSpans(_OTEL / "unknown" / "unrecognized.otlp.json"))
    assert "convention" in str(exc.value).lower()
