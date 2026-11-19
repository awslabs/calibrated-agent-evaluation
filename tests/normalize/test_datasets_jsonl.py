# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSONL dataset ingestion, one Trace per line.

One JSON object per line → one Trace; blank lines are skipped; a malformed JSON line raises a typed
error chained from JSONDecodeError — never a raw exception, never a silent drop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrated_agent_evaluation.normalize.datasets import load_traces
from calibrated_agent_evaluation.normalize.errors import NormalizationError
from calibrated_agent_evaluation.normalize.sources import JSONLDataset
from calibrated_agent_evaluation.schema.validate import validate

_REPO = Path(__file__).resolve().parents[2]
_REAL_JSONL = _REPO / "fixtures" / "raw" / "hf" / "swebench-session.jsonl"


def test_one_trace_per_line_on_the_real_slice() -> None:
    traces = list(load_traces(JSONLDataset(_REAL_JSONL)))
    assert len(traces) == 1  # the committed single-session JSONL twin
    validate(traces[0])


def test_blank_lines_skipped_and_valid_lines_all_yielded(tmp_path: Path) -> None:
    # Reuse the real record so the content is genuine; interleave blank lines around it.
    record = json.loads(_REAL_JSONL.read_text(encoding="utf-8").strip())
    f = tmp_path / "with_blanks.jsonl"
    f.write_text("\n" + json.dumps(record) + "\n\n   \n" + json.dumps(record) + "\n")
    traces = list(load_traces(JSONLDataset(f)))
    assert len(traces) == 2, "blank/whitespace lines skipped; both real records yielded"


def test_malformed_json_line_raises_typed_chained(tmp_path: Path) -> None:
    # The malformed line is line 1 so the JSONL reader's own parse-error boundary is what fires
    # (not a downstream convention error on a later record): typed, chained, and naming the line.
    f = tmp_path / "bad.jsonl"
    f.write_text("{ this is not json\n")
    with pytest.raises(NormalizationError) as exc:
        list(load_traces(JSONLDataset(f)))
    assert "line 1" in str(exc.value)
    assert isinstance(exc.value.__cause__, json.JSONDecodeError)
