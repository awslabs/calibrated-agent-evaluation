# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the JudgeRunRecord round-trip + provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrated_agent_evaluation.judges.errors import JudgeRuntimeError
from calibrated_agent_evaluation.judges.run_record import (
    JudgeRunRecord,
    dump_run_records,
    load_run_records,
)


def _record() -> JudgeRunRecord:
    return JudgeRunRecord(
        run_id="run-1",
        judge_id="orderbot-refund-resolved",
        model="bedrock/global.anthropic.claude-sonnet-4-6",
        prompt_version="v1",
        calib_id="calib-1",
        level="workflow",
        trace_ids=("ORD-1001", "ORD-1002"),
        span_ids=(None, None),
        n_cached=1,
        n_computed=1,
    )


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_run_record_round_trips(tmp_path: Path, fmt: str) -> None:
    """A dumped run record reloads field-for-field (both CSV and JSON)."""
    original = _record()
    dump_run_records([original], tmp_path, fmt=fmt)
    (reloaded,) = load_run_records(tmp_path, fmt=fmt)
    assert reloaded == original
    # The frozen judge version + calib_id + scored ids are what make the run reproducible.
    assert reloaded.model == "bedrock/global.anthropic.claude-sonnet-4-6"
    assert reloaded.prompt_version == "v1"
    assert reloaded.calib_id == "calib-1"
    assert reloaded.trace_ids == ("ORD-1001", "ORD-1002")


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_a_session_level_run_record_round_trips(tmp_path: Path, fmt: str) -> None:
    """An AgentCore session-level evaluator's run is recorded at the session level."""
    original = _record().model_copy(update={"level": "session"})
    dump_run_records([original], tmp_path, fmt=fmt)
    (reloaded,) = load_run_records(tmp_path, fmt=fmt)
    assert reloaded.level == "session"


def test_a_run_record_at_an_unknown_level_is_refused() -> None:
    with pytest.raises(JudgeRuntimeError):
        JudgeRunRecord.model_validate({**_record().model_dump(), "level": "conversation"})


def test_run_record_refuses_version_mismatch(tmp_path: Path) -> None:
    """A stored schema_version != running is refused (no coercion)."""
    dump_run_records([_record()], tmp_path, fmt="json")
    file = tmp_path / "judge_run_record.json"
    docs = json.loads(file.read_text())
    docs[0]["schema_version"] = "999"
    file.write_text(json.dumps(docs))
    with pytest.raises(JudgeRuntimeError, match="schema_version"):
        load_run_records(tmp_path, fmt="json")


def test_run_record_refuses_missing_calib_id() -> None:
    """A run record with no calib_id cannot be constructed — a run must name the
    calibration it cleared, or it is not reproducible."""
    with pytest.raises(JudgeRuntimeError, match="calib_id"):
        JudgeRunRecord(
            run_id="run-x",
            judge_id="j",
            model="m",
            prompt_version="v1",
            calib_id="",
            level="workflow",
            trace_ids=("ORD-1001",),
            span_ids=(None,),
        )
