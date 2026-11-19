# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline, deterministic ingestion of the committed dataset slice.

Proves a downstream feature can load a real dataset slice for its tests with zero network and a
bounded record count. Runs with the network hard-disabled to prove the offline guarantee,
and checks the `limit` bound.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from calibrated_agent_evaluation.normalize.datasets import load_traces
from calibrated_agent_evaluation.normalize.errors import NormalizationError
from calibrated_agent_evaluation.normalize.sources import HFDataset, ParquetDataset
from calibrated_agent_evaluation.schema.validate import validate

_SLICE = Path(__file__).resolve().parents[2] / "fixtures" / "raw" / "hf" / "swebench-slice.parquet"


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hard-disable outbound sockets so any network access raises (proves offline operation)."""

    def _blocked(*_a: object, **_k: object) -> object:
        raise AssertionError("network access attempted during an offline frozen-slice load")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


def test_frozen_slice_loads_offline_within_limit(no_network: None) -> None:
    traces = list(load_traces(ParquetDataset(_SLICE), limit=3))
    assert len(traces) == 3
    for trace in traces:
        validate(trace)  # every yielded trace is schema-valid


def test_live_hf_fetch_offline_raises_naming_the_extra_or_network(
    no_network: None,
) -> None:
    # Under the same offline condition, an HFDataset fetch must fail loudly (typed), never hang or
    # silently return nothing. (Either the [hf] extra is absent → typed extra error, or the network
    # block trips inside the resolver and is wrapped typed.)
    with pytest.raises((NormalizationError, AssertionError)):
        list(load_traces(HFDataset("Exgentic/agent-llm-traces"), limit=1))


def test_limit_zero_yields_nothing_and_limit_over_count_yields_all() -> None:
    assert list(load_traces(ParquetDataset(_SLICE), limit=0)) == []
    # limit greater than the slice size yields every record (3 committed sessions), not an error.
    assert len(list(load_traces(ParquetDataset(_SLICE), limit=999))) == 3


def test_a_missing_slice_file_raises_typed(tmp_path: Path) -> None:
    with pytest.raises(NormalizationError):
        list(load_traces(ParquetDataset(tmp_path / "no-such-slice.parquet")))


# --- a benchmark slice must not arrive pre-authorized ------------------------


def test_load_traces_does_not_default_to_the_production_frame() -> None:
    """A dataset ingested without a stated frame must not be tagged `production-representative`.

    `_record_to_session` built its `SessionSpans` with no `frame=` argument, so every record
    inherited `SessionSpans.frame`'s own default — the single value
    `sampling.assert_production_frame` accepts. A public benchmark slice therefore arrived carrying
    an analyst's claim nobody had made, and the guard that exists to stop a headline rate being
    computed off non-representative data waved it through.
    """
    traces = list(load_traces(ParquetDataset(_SLICE), limit=2))
    assert traces, "precondition: the frozen slice yields records"
    for trace in traces:
        assert trace.frame == "discovery"
        assert trace.frame != "production-representative"


def test_load_traces_still_honors_an_explicit_production_frame() -> None:
    """The default is a default, not a ceiling: an analyst who *can* make the claim still can."""
    traces = list(load_traces(ParquetDataset(_SLICE), limit=1, frame="production-representative"))
    assert traces[0].frame == "production-representative"


def test_the_production_frame_guard_refuses_the_default_output() -> None:
    """The end the fix exists for: the guard now fires on a default-frame dataset trace.

    Asserted through `assert_production_frame` itself rather than by reading `.frame`, because the
    defect was only harmful via that authorization path.
    """
    from calibrated_agent_evaluation.sampling import assert_production_frame
    from calibrated_agent_evaluation.sampling.errors import FrameContaminationError

    trace = next(iter(load_traces(ParquetDataset(_SLICE), limit=1)))
    with pytest.raises(FrameContaminationError, match="discovery"):
        assert_production_frame([trace])  # type: ignore[list-item]  # duck-typed on .frame
