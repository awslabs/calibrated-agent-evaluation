# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OTLP traceId encoding is detected, not guessed.

`trace_id_hex` is the last layer of header resolution — reached only when no
`calibrated_agent_evaluation.trace_id` rides on the wire and the descriptor carries no override. Its
job is to turn the transport's own id into a stable string, and the one thing it must never do is
invent a value: a `Trace.trace_id` that matches nothing in the user's Jaeger/Tempo/Honeycomb breaks
the only link between the library's output and the trace it describes.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.normalize import OTelSpans, StrandsSession, normalize
from calibrated_agent_evaluation.normalize.otlp_json import trace_id_hex

# The exact id from the ticket's Evidence, and what the base64-first path turned it into.
_REAL_HEX = "d18aa7f7a211a5f9ae6b86ed58a27bd0"
_CORRUPTED = "775f1a6bb7fb6b6d756b97fd69ee9bf3a79de7c6b6edb774"


def test_a_standard_hex_trace_id_is_returned_unchanged() -> None:
    """The regression. A 32-hex-char collector id must survive as itself.

    `base64.b64decode` without `validate=True` discards non-alphabet characters rather than raising,
    and 32 hex characters are themselves valid base64 — so the old base64-first order decoded this
    id to 24 junk bytes and re-hexed it to a different value of a different length. The base64
    branch was simultaneously unreachable for real base64 ids and corrupting for hex ones.
    """
    assert trace_id_hex(_REAL_HEX) == _REAL_HEX
    assert trace_id_hex(_REAL_HEX) != _CORRUPTED


def test_hex_is_lowercased_so_the_same_id_never_yields_two_keys() -> None:
    assert trace_id_hex(_REAL_HEX.upper()) == _REAL_HEX


def test_a_genuine_base64_trace_id_still_decodes() -> None:
    """The case the base64 branch was written for, which must keep working.

    16 raw bytes whose base64 encoding contains non-hex characters, so there is no ambiguity about
    which encoding this is.
    """
    raw = bytes(range(16))
    encoded = base64.b64encode(raw).decode("ascii")
    assert not all(c in "0123456789abcdefABCDEF" for c in encoded), "must be unambiguously base64"
    assert trace_id_hex(encoded) == raw.hex()


def test_an_8_byte_span_id_is_recognized_as_hex_too() -> None:
    assert trace_id_hex("0102030405060708") == "0102030405060708"


@pytest.mark.parametrize("junk", ["not-an-id!!", "", "zz", "AAAA"])
def test_an_unrecognized_id_is_returned_raw_rather_than_mangled(junk: str) -> None:
    """Neither hex nor a plausible 8/16-byte base64 id: return it unchanged, never a re-encoding.

    `"AAAA"` is the case that needs the length check — it IS valid base64, decoding to 3 bytes,
    which is not an OTel id length. Hexing those 3 bytes would be exactly the invention this
    function exists to avoid.
    """
    assert trace_id_hex(junk) == junk


# --- the hint: a synthesized header now announces itself ------------------------------------------


def _otlp_with_trace_id(raw_trace_id: str, *, scope: str, op: str) -> dict[str, Any]:
    """A minimal one-agent OTLP payload carrying `raw_trace_id` and no calibrated_agent_evaluation.*
    convention."""
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {
                        "scope": {"name": scope},
                        "spans": [
                            {
                                "traceId": raw_trace_id,
                                "spanId": "0102030405060708",
                                "name": op,
                                "startTimeUnixNano": "1700000000000000000",
                                "endTimeUnixNano": "1700000001000000000",
                                "attributes": [
                                    {"key": "gen_ai.operation.name", "value": {"stringValue": op}},
                                    {
                                        "key": "gen_ai.agent.name",
                                        "value": {"stringValue": "solo"},
                                    },
                                    {
                                        "key": "gen_ai.input.content",
                                        "value": {"stringValue": '{"q": "hi"}'},
                                    },
                                    {
                                        "key": "gen_ai.output.content",
                                        "value": {"stringValue": '{"a": "yes"}'},
                                    },
                                ],
                                "events": [
                                    {
                                        "name": "gen_ai.user.message",
                                        "attributes": [
                                            {
                                                "key": "content",
                                                "value": {"stringValue": '{"q": "hi"}'},
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


def test_the_generic_adapter_uses_the_wire_id_and_says_it_synthesized_the_header(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """End to end: the trace_id is the real hex, and the user is told how to stop synthesizing it.

    `guidance.synthesized_header_hint` was written for this path and called from nowhere — it had
    one reference, in its own unit test. So a synthesized header was silent even though the module
    exists to explain it.
    """
    path = tmp_path / "wire.otlp.json"
    path.write_text(
        json.dumps(_otlp_with_trace_id(_REAL_HEX, scope="app", op="invoke_agent")),
        encoding="utf-8",
    )

    with caplog.at_level(logging.INFO, logger="calibrated_agent_evaluation.normalize"):
        trace = normalize(OTelSpans(path=path, semconv="gen_ai/latest"))

    assert trace.trace_id == _REAL_HEX, "the id must be the one on the wire"
    assert "calibrated_agent_evaluation.trace_id" in caplog.text, (
        "the hint names the attribute that would fix it"
    )
    assert _REAL_HEX in caplog.text, "and the value it had to synthesize"


def test_the_strands_adapter_shares_the_same_implementation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Both adapters had the identical defect; the fix is one function, so assert both callers."""
    payload = _otlp_with_trace_id(_REAL_HEX, scope="strands.telemetry", op="invoke_agent")
    path = tmp_path / "strands.otlp.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="calibrated_agent_evaluation.normalize"):
        trace = normalize(StrandsSession(path=path))

    assert trace.trace_id == _REAL_HEX
    assert "calibrated_agent_evaluation.trace_id" in caplog.text


def test_an_explicit_trace_id_wins_and_no_hint_is_emitted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Negative: the hint is for the fallback layer only.

    A descriptor override means nothing was synthesized, so emitting the hint would train the reader
    to ignore it — the same reasoning as the conditional probed/contentless counts in the ingest
    report.
    """
    path = tmp_path / "override.otlp.json"
    path.write_text(
        json.dumps(_otlp_with_trace_id(_REAL_HEX, scope="app", op="invoke_agent")),
        encoding="utf-8",
    )

    with caplog.at_level(logging.INFO, logger="calibrated_agent_evaluation.normalize"):
        trace = normalize(OTelSpans(path=path, semconv="gen_ai/latest", trace_id="ORD-9"))

    assert trace.trace_id == "ORD-9"
    assert "calibrated_agent_evaluation.trace_id" not in caplog.text
