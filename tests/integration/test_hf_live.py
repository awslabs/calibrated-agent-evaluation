# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live HuggingFace-hub ingestion.

Marked `integration` (not run in the default hermetic suite): it reaches the live HuggingFace hub,
normalizes a couple of real records from `Exgentic/agent-llm-traces`, and keeps the committed
frozen slice honest against dataset/SDK drift. The resolver downloads every shard before it yields
a row, so the first run fetches the whole dataset (39 shards, about 984 MB) into the HF cache.
Run with `just test-integration -k dataset` (needs the `[hf]` + `[datasets]` extras and network).
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.normalize.datasets import load_traces
from calibrated_agent_evaluation.normalize.sources import HFDataset
from calibrated_agent_evaluation.schema.validate import validate

pytestmark = pytest.mark.integration


def test_live_hf_dataset_normalizes_a_bounded_sample() -> None:
    traces = list(load_traces(HFDataset("Exgentic/agent-llm-traces"), limit=2))
    assert len(traces) == 2, "the record limit bounds the live fetch"
    for trace in traces:
        validate(trace)
        assert trace.ownership in {"owned", "probed"}
