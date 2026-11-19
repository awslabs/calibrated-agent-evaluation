# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the self-contained enforcement-artifact round-trip.

Proves field-for-field CSV+JSON round-trip for EnforcementDecision and JudgeDefinition, and the
wrap-external-error / refuse-on-version-mismatch disciplines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrated_agent_evaluation.enforce import (
    Axis,
    ChecklistItem,
    EnforcementDecision,
    EnforcementStoreError,
    JudgeDefinition,
    dump_decisions,
    dump_definitions,
    load_decisions,
    load_definitions,
)


def _decisions() -> list[EnforcementDecision]:
    return [
        EnforcementDecision(
            rule="ENF-1",
            tier="hard_refusal",
            inputs={"judge_id": "j1", "record_result": "REFUSED"},
            outcome="refused",
            message="no certifying calibration",
        ),
        EnforcementDecision(
            rule="ENF-3",
            tier="warning",
            inputs={"judge_id": "j1", "binary_axes": "a,b"},
            outcome="warned",
            message="multi-criteria",
            nudge="split into N single-axis judges",
        ),
    ]


def _definition() -> JudgeDefinition:
    return JudgeDefinition(
        judge_id="orderbot-refund",
        axes=(Axis(name="resolved"), Axis(name="tone", scale="likert")),
        output_fields=("critique", "verdict"),
        scorer="exact_match",
        checklist=(ChecklistItem(name="policy_cited", critical=True),),
    )


# --- round-trip equality ----------------------------------------------------


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_decisions_round_trip_field_for_field(tmp_path: Path, fmt: str) -> None:
    original = _decisions()
    dump_decisions(original, tmp_path, fmt=fmt)
    assert load_decisions(tmp_path, fmt=fmt) == original


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_judge_definition_round_trips_field_for_field(tmp_path: Path, fmt: str) -> None:
    original = [_definition()]
    dump_definitions(original, tmp_path, fmt=fmt)
    assert load_definitions(tmp_path, fmt=fmt) == original


def test_empty_decisions_round_trip(tmp_path: Path) -> None:
    dump_decisions([], tmp_path, fmt="csv")
    assert load_decisions(tmp_path, fmt="csv") == []


# --- negative: corrupt JSON + version mismatch wrap into the typed family ---


def test_corrupt_json_raises_typed_store_error(tmp_path: Path) -> None:
    # Negative: a truncated JSON dump surfaces as EnforcementStoreError, not a raw parse error.
    (tmp_path / "enforcement_decision.json").write_text('[{"rule": "ENF-1", ')
    with pytest.raises(EnforcementStoreError):
        load_decisions(tmp_path, fmt="json")


def test_version_mismatch_refuses_load(tmp_path: Path) -> None:
    # Negative: a bumped schema_version makes load refuse rather than coerce.
    doc = [
        {
            "rule": "ENF-1",
            "tier": "hard_refusal",
            "inputs": {"judge_id": "j1"},
            "outcome": "refused",
            "message": "m",
            "nudge": None,
            "schema_version": "999",
        }
    ]
    (tmp_path / "enforcement_decision.json").write_text(json.dumps(doc))
    with pytest.raises(EnforcementStoreError):
        load_decisions(tmp_path, fmt="json")


def test_malformed_but_valid_json_raises_typed_error(tmp_path: Path) -> None:
    # Negative: valid JSON missing a required field must not let a raw KeyError escape.
    (tmp_path / "enforcement_decision.json").write_text(json.dumps([{"rule": "ENF-1"}]))
    with pytest.raises(EnforcementStoreError):
        load_decisions(tmp_path, fmt="json")
