# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AgentCore source configuration types."""

from __future__ import annotations

from typing import Any

import pytest

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreError,
    AgentCoreIndexingTooLow,
    AgentCorePartialResult,
    AgentCoreTruncatedResult,
)
from calibrated_agent_evaluation.agentcore.sources import (
    AgentCoreSourceConfig,
    TimeWindow,
    build_omni_sql,
    refuse_indexing_below_full,
    refuse_partial_result,
    refuse_truncated_result,
    xray_id_batches,
)

_T0 = "2026-10-01T00:00:00+00:00"
_T1 = "2026-10-01T01:00:00+00:00"


def _cfg(start: str = _T0, end: str = _T1, **over: Any) -> AgentCoreSourceConfig:
    return AgentCoreSourceConfig(
        runtime="nl_sql_agent", window=TimeWindow(start=start, end=end), **over
    )


def test_config_round_trips_its_fields_and_defaults() -> None:
    cfg = AgentCoreSourceConfig(runtime="nl_sql_agent", window=TimeWindow(start=_T0, end=_T1))
    assert cfg.runtime == "nl_sql_agent"
    assert cfg.window.end == _T1
    assert cfg.rekey_sessions is False
    assert cfg.with_scores is False


def test_the_omni_flag_selects_the_omni_path() -> None:
    without = AgentCoreSourceConfig(runtime="r", window=TimeWindow(start=_T0, end=_T1))
    withomni = AgentCoreSourceConfig(runtime="r", window=TimeWindow(start=_T0, end=_T1), omni=True)
    assert without.uses_omni is False
    assert withomni.uses_omni is True


def test_a_naive_window_bound_is_refused() -> None:
    with pytest.raises(AgentCoreError):
        TimeWindow(start="2026-10-01T00:00:00", end=_T1)  # no timezone offset


def test_a_window_whose_start_is_not_before_its_end_is_refused() -> None:
    # Swapped --since/--until would read an empty window and report "0 ingested" with no error,
    # reading as "nothing happened" rather than "you asked for an empty window".
    with pytest.raises(AgentCoreError):
        TimeWindow(start=_T1, end=_T0)  # end before start
    with pytest.raises(AgentCoreError):
        TimeWindow(start=_T0, end=_T0)  # zero-width window


def test_omni_sql_uses_to_timestamp_and_session_id_map_access() -> None:
    sql = build_omni_sql(_cfg(), session_name="calibrated-agent-evaluation-abc")
    assert "to_timestamp('2026-10-01T00:00:00+00:00')" in sql
    assert "attributes['session.id']" in sql
    # Never the backtick form, which returns an empty string with no error.
    assert "`attributes.session.id`" not in sql


def test_omni_sql_refuses_a_window_of_a_year_or_more() -> None:
    with pytest.raises(AgentCoreError):
        build_omni_sql(
            _cfg(start="2025-01-01T00:00:00+00:00", end="2026-06-01T00:00:00+00:00"),
            session_name="calibrated-agent-evaluation-abc",
        )


def test_a_partial_omni_result_is_refused() -> None:
    with pytest.raises(AgentCorePartialResult):
        refuse_partial_result({"partialResults": {"partialResultsDetected": True}})
    refuse_partial_result({"partialResults": {"partialResultsDetected": False}})  # does not raise


def test_a_logs_insights_match_larger_than_the_read_is_refused() -> None:
    with pytest.raises(AgentCoreTruncatedResult):
        refuse_truncated_result(records_matched=120_000, rows_retrieved=100_000)
    refuse_truncated_result(records_matched=42, rows_retrieved=42)  # complete read, no raise


def test_xray_indexing_below_full_is_refused() -> None:
    one_percent = [{"Name": "Default", "Rule": {"Probabilistic": {"DesiredSamplingPercentage": 1}}}]
    with pytest.raises(AgentCoreIndexingTooLow):
        refuse_indexing_below_full(one_percent)
    full = [{"Name": "Default", "Rule": {"Probabilistic": {"DesiredSamplingPercentage": 100}}}]
    refuse_indexing_below_full(full)  # 100% does not raise


def test_xray_ids_are_batched_in_hundreds() -> None:
    batches = list(xray_id_batches([f"id-{i}" for i in range(250)]))
    assert [len(b) for b in batches] == [100, 100, 50]
    assert max(len(b) for b in batches) <= 100
