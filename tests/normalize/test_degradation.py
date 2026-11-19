# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Honest-degradation tests.

Against the real ``fixtures/raw/otel/ORD-DEGRADED.otlp.json`` (an under-instrumented run — top-
level I/O only): it degrades to a ``probed`` outcome-only trace with a surfaced reason,
the warning is emitted only on the degrade path (not on a successful ``owned`` normalize), and a
source is never returned as ``owned`` with empty spans (no invalid in-between).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from calibrated_agent_evaluation.normalize import OTelSpans, normalize
from calibrated_agent_evaluation.schema.validate import validate


def _degraded(raw_dir: Path) -> OTelSpans:
    return OTelSpans(raw_dir / "otel" / "ORD-DEGRADED.otlp.json", trace_id="ORD-1001")


def test_underinstrumented_trace_degrades_to_probed_outcome_only(raw_dir: Path) -> None:
    trace = normalize(_degraded(raw_dir))
    validate(trace)
    assert trace.ownership == "probed"
    assert trace.spans == []
    # Terminal input/output preserved (outcome-only, not lost).
    assert trace.input
    assert trace.output


def test_degradation_surfaces_a_reason_only_on_the_degrade_path(
    raw_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A WARNING naming what could not be recovered is emitted on the degrade path...
    with caplog.at_level(logging.WARNING, logger="calibrated_agent_evaluation.normalize"):
        normalize(_degraded(raw_dir))
    degrade_warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert degrade_warnings
    assert any("probed" in r.getMessage() for r in degrade_warnings)

    caplog.clear()
    # ...and NOT emitted on a successful owned normalization.
    events_src = OTelSpans(raw_dir / "otel" / "ORD-1001.events.otlp.json", trace_id="ORD-1001")
    with caplog.at_level(logging.WARNING, logger="calibrated_agent_evaluation.normalize"):
        owned = normalize(events_src)
    assert owned.ownership == "owned"
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_never_owned_with_empty_spans(raw_dir: Path) -> None:
    # A partially-recovered source is never returned as `owned` with empty spans — it is either
    # fully owned (spans populated) or probed (spans empty), never the invalid in-between.
    for name in ("ORD-1001.events.otlp.json", "ORD-DEGRADED.otlp.json"):
        trace = normalize(OTelSpans(raw_dir / "otel" / name, trace_id="ORD-1001"))
        if trace.ownership == "owned":
            assert trace.spans, f"{name}: owned but empty spans (invalid in-between)"
        else:
            assert trace.ownership == "probed"
            assert trace.spans == [], f"{name}: probed but non-empty spans"
