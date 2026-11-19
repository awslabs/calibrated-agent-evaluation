# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""An Inspect eval over N samples yields N Traces, not 1.

`inspect_adapter._read_single_sample` took the first sample and then drained the generator to close
the file handle — so samples 2..N were read and thrown away. Every real Inspect eval runs a dataset
of many samples, so an external project's `.eval` was evaluated at 1/N of its data, and every
downstream rate, count and gate decision came from that one sample. Nothing reported it.

The fixture is `multisample-3.eval`, three genuine recorded samples in one container (see
`fixtures/raw/inspect/generate_multisample.py` for why it is assembled rather than freshly minted).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrated_agent_evaluation.normalize import (
    InspectEvalLog,
    NormalizationError,
    normalize,
    normalize_all,
)
from calibrated_agent_evaluation.schema.validate import validate

_RAW = Path(__file__).resolve().parent.parent.parent / "fixtures" / "raw" / "inspect"
_MULTI = _RAW / "multisample-3.eval"
_SINGLE = _RAW / "ORD-1001.eval"
_EXPECTED_IDS = ["ORD-1001", "ORD-1002", "CLM-4471"]

pytest.importorskip("inspect_ai")


def test_normalize_all_returns_one_trace_per_sample() -> None:
    """3 samples in, 3 Traces out — one per sample id, in log order.

    Before the fix this entry point did not exist, and `normalize` returned one trace for the log.
    """
    traces = normalize_all(InspectEvalLog(path=_MULTI))

    assert [t.trace_id for t in traces] == _EXPECTED_IDS
    for trace in traces:
        validate(trace)


def test_every_recovered_sample_carries_its_own_content() -> None:
    """Not just N traces: N *distinct* traces. Copying one sample N times would pass a count."""
    traces = normalize_all(InspectEvalLog(path=_MULTI))

    assert len({str(t.input) for t in traces}) == 3, "each sample's input is its own"
    for trace in traces:
        assert trace.spans, f"{trace.trace_id} recovered no span tree"
        assert sum(len(s.tool_calls) for s in trace.spans) > 0, (
            f"{trace.trace_id} recovered no tool calls"
        )


def test_each_trace_keeps_its_samples_taxonomy_cell() -> None:
    """The cell rides the sample metadata, so a per-sample read must keep them distinct."""
    cells = [t.taxonomy_cell for t in normalize_all(InspectEvalLog(path=_MULTI))]
    assert all(cells), "every sample carried a taxonomy cell"
    assert len({tuple(sorted(c.items())) for c in cells}) > 1, "cells must not be shared"


def test_normalize_refuses_a_multi_sample_log_rather_than_reading_one_of_n() -> None:
    """The negative path: `normalize` returns ONE trace, so it must not silently pick one.

    This is the honest half of the fix. `normalize`'s contract is one Trace, and a multi-sample log
    has no single answer — so rather than returning 1/N of the data it says so and names both ways
    forward.
    """
    with pytest.raises(NormalizationError) as excinfo:
        normalize(InspectEvalLog(path=_MULTI))

    message = str(excinfo.value)
    assert "3 samples" in message
    assert "normalize_all" in message, "the message must name the entry point that reads them all"
    assert "sample_id" in message, "and the selector for reading one deliberately"


def test_an_explicit_sample_id_reads_exactly_that_sample() -> None:
    """The targeted single-sample read the ticket allowed for, made explicit not implicit."""
    trace = normalize(InspectEvalLog(path=_MULTI, sample_id="ORD-1002"))
    assert trace.trace_id == "ORD-1002"


def test_an_unknown_sample_id_refuses_and_lists_what_is_available() -> None:
    """A typo'd selector must not silently fall back to the first sample."""
    with pytest.raises(NormalizationError) as excinfo:
        normalize(InspectEvalLog(path=_MULTI, sample_id="ORD-9999"))

    message = str(excinfo.value)
    assert "ORD-9999" in message
    assert "ORD-1001" in message, "the available ids must be listed"


def test_a_single_sample_log_still_normalizes_through_both_entry_points() -> None:
    """Regression guard: the one-sample case, which is every committed fixture."""
    single = normalize(InspectEvalLog(path=_SINGLE, trace_id="ORD-1001"))
    assert single.trace_id == "ORD-1001"

    (only,) = normalize_all(InspectEvalLog(path=_SINGLE, trace_id="ORD-1001"))
    assert only.trace_id == "ORD-1001"
    assert only == single, "the two entry points agree on a single-sample log"


def test_a_multi_sample_log_does_not_take_the_descriptor_trace_id_for_every_sample() -> None:
    """The store is keyed on trace_id, so N samples sharing one id would collapse to 1 on write.

    `calibrated-agent-evaluation ingest` derives the descriptor's trace_id from the FILE name, which
    names the log and not
    any sample — so using it per sample would swap this defect for a store-level one.
    """
    traces = normalize_all(InspectEvalLog(path=_MULTI, trace_id="multisample-3"))
    ids = [t.trace_id for t in traces]

    assert len(set(ids)) == 3, f"every sample needs its own trace id, got {ids}"
    assert ids == _EXPECTED_IDS, "the sample metadata's own trace_id wins"
