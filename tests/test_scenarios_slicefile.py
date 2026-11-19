# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slice files and answer keys round-trip through disk, and unreadable ones are refused."""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrated_agent_evaluation.scenarios.errors import ScenarioError
from calibrated_agent_evaluation.scenarios.slicefile import read_answer_key, read_slice, write_slice
from calibrated_agent_evaluation.scenarios.types import (
    AnswerEntry,
    CalibratedAgentEvaluationScenarioMeta,
    DatasetSlice,
    PredefinedScenario,
    PublishedVersion,
    Turn,
)


def _slice() -> DatasetSlice:
    return DatasetSlice(
        form="predefined",
        scenarios=[
            PredefinedScenario(
                scenario_id="ORD-1001",
                turns=[Turn(input={"message": "Refund order A-7781"}, expected_response="{}")],
                assertions=["Agent denies a refund outside the return window"],
                calibrated_agent_evaluation=CalibratedAgentEvaluationScenarioMeta(
                    source_trace_id="ORD-1001", expected={"note": "é"}
                ),
            )
        ],
        published=PublishedVersion(
            dataset_arn="arn:aws:bedrock-agentcore:us-west-2:111122223333:dataset/orderbot-AbCdEf1234",
            dataset_id="orderbot-AbCdEf1234",
            dataset_name="orderbot_regressions",
            version="1",
            region="us-west-2",
            content_digest="0" * 64,
            published_at="2026-10-27T19:00:00Z",
        ),
    )


def test_slice_file_and_answer_key_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "slice.json"
    write_slice(path, _slice())

    assert read_slice(path) == _slice()

    key = tmp_path / "key.json"
    key.write_text(
        '{"by_failure_mode": {"trusted_broken_policy_tool": {"assertions": '
        '["Agent denies a refund outside the return window"], "expected_trajectory": '
        '["lookup_order", "check_refund_policy"]}}, "by_trace": {}}'
    )
    loaded = read_answer_key(key)
    assert loaded.by_failure_mode["trusted_broken_policy_tool"] == AnswerEntry(
        assertions=["Agent denies a refund outside the return window"],
        expected_trajectory=["lookup_order", "check_refund_policy"],
    )


@pytest.mark.parametrize(
    ("content", "named"),
    [
        ('{"schema_version": "2", "form": "predefined", "scenarios": []}', "schema_version"),
        ('{"form": "predefined", "scenarios": [', "line 1"),
        ("[1, 2]", "JSON object"),
    ],
)
def test_unreadable_slice_files_are_refused(tmp_path: Path, content: str, named: str) -> None:
    path = tmp_path / "slice.json"
    path.write_text(content)

    with pytest.raises(ScenarioError, match=named) as refused:
        read_slice(path)
    assert str(path) in str(refused.value)


def test_a_missing_slice_file_is_refused_naming_the_path(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError, match="cannot read"):
        read_slice(tmp_path / "absent.json")


def test_existing_files_are_not_replaced_unless_forced(tmp_path: Path) -> None:
    path = tmp_path / "slice.json"
    path.write_text("keep me")

    with pytest.raises(ScenarioError, match="already exists"):
        write_slice(path, _slice())
    assert path.read_text() == "keep me"

    write_slice(path, _slice(), force=True)
    assert read_slice(path) == _slice()
