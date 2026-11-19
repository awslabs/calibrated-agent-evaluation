# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Honest degradation to outcome-only.

A session with a detectable convention but no recoverable per-step content degrades to a `probed`
outcome-only Trace (`spans == []`) with a surfaced reason — never an empty `owned` trace, never a
silent drop.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from calibrated_agent_evaluation.normalize import OTelSpans, SessionSpans, normalize
from calibrated_agent_evaluation.schema.validate import validate

_REPO = Path(__file__).resolve().parents[2]
_DEGRADED = _REPO / "fixtures" / "raw" / "otel" / "degraded_v2.otlp.json"


def test_detectable_but_empty_messages_session_degrades_to_probed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        trace = normalize(OTelSpans(_DEGRADED))
    validate(trace)
    assert trace.ownership == "probed"
    assert trace.spans == [], "a degraded trace has no spans (never an empty owned tree)"
    # The degradation reason is surfaced (a logged warning naming what could not be recovered).
    assert any("recover" in r.message.lower() or "no" in r.message.lower() for r in caplog.records)


def test_terminal_io_preserved_when_present_but_no_steps() -> None:
    # A messages session with a top-level user/assistant text but no tool calls stays owned only if
    # there IS per-step content; with only empty message lists it degrades and preserves nothing
    # fabricated. Here we give a terminal user message but no tools → still owned (has content),
    # confirming the gate distinguishes "has content, no tools" from "no content at all".
    spans = [
        {
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.input.messages": (
                    '[{"role":"user","parts":[{"type":"text","content":"just answer"}]}]'
                ),
                "gen_ai.output.messages": (
                    '[{"role":"assistant","parts":[{"type":"text","content":"ok"}]}]'
                ),
            }
        }
    ]
    trace = normalize(SessionSpans(spans=spans, semconv="gen_ai/messages", trace_id="NOTOOLS-1"))
    validate(trace)
    assert trace.ownership == "owned"
    assert trace.output == {"answer": "ok"}
