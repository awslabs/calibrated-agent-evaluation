# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column mapping for differently-shaped datasets.

A dataset whose columns are named differently ingests the same via a `DatasetMapping`, and a
mapping naming a nonexistent column fails loudly (typed), never silently drops the record.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrated_agent_evaluation.normalize.datasets import load_traces
from calibrated_agent_evaluation.normalize.errors import NormalizationError
from calibrated_agent_evaluation.normalize.sources import DatasetMapping, ParquetDataset

_REPO = Path(__file__).resolve().parents[2]
_CANONICAL = _REPO / "fixtures" / "raw" / "hf" / "swebench-slice.parquet"
_RENAMED = _REPO / "fixtures" / "raw" / "hf" / "renamed-columns-slice.parquet"


def _tool_total(traces: list) -> int:
    return sum(len(s.tool_calls) for t in traces for s in t.spans)


def test_renamed_columns_yield_same_traces_as_default_mapping() -> None:
    # The renamed fixture is the SAME data with spans→otel_events, session_id→trace_key. With a
    # matching DatasetMapping it must reconstruct identically to the default-mapping canonical run.
    default = list(load_traces(ParquetDataset(_CANONICAL)))
    renamed = list(
        load_traces(
            ParquetDataset(
                _RENAMED,
                mapping=DatasetMapping(spans_column="otel_events", session_id_column="trace_key"),
            )
        )
    )
    assert len(renamed) == len(default)
    assert {t.trace_id for t in renamed} == {t.trace_id for t in default}
    assert _tool_total(renamed) == _tool_total(default)


def test_mapping_naming_a_nonexistent_column_raises_typed() -> None:
    with pytest.raises(NormalizationError) as exc:
        list(
            load_traces(
                ParquetDataset(_CANONICAL, mapping=DatasetMapping(spans_column="does_not_exist"))
            )
        )
    assert "does_not_exist" in str(exc.value)
