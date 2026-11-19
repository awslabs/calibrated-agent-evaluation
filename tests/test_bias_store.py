# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bias-battery record round-trip + refuse-on-corruption."""

from __future__ import annotations

from pathlib import Path

import pytest

from _helpers import TEST_STAMP
from calibrated_agent_evaluation.bias import (
    BiasBatteryRecord,
    BiasStoreError,
    ProbeResult,
    dump_battery,
    load_battery,
)
from calibrated_agent_evaluation.bias.version import BIAS_SCHEMA_VERSION


def _record() -> BiasBatteryRecord:
    return BiasBatteryRecord(
        provenance=TEST_STAMP,
        battery_id="bat-1",
        judge_id="judge-1",
        model="test-model",
        prompt_version="v1",
        is_pairwise=False,
        results=(
            ProbeResult(
                bias="length",
                status="run",
                n=4,
                biased_count=1,
                magnitude=0.25,
                magnitude_ub95=0.05,
                threshold=0.2,
                passed=True,
                is_must_pass=True,
            ),
            ProbeResult(
                bias="positional",
                status="not_applicable",
                n=0,
                biased_count=0,
                magnitude=0.0,
                magnitude_ub95=0.0,
                threshold=None,
                passed=None,
                is_must_pass=False,
                detail="pairwise-only",
            ),
        ),
        verdict="CLEAR",
    )


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_round_trips_field_for_field(tmp_path: Path, fmt: str) -> None:
    original = _record()
    dump_battery([original], tmp_path, fmt=fmt)
    (loaded,) = load_battery(tmp_path, fmt=fmt)
    assert loaded == original


def test_corrupt_json_raises_typed_store_error(tmp_path: Path) -> None:
    (tmp_path / "bias_battery_record.json").write_text("{ this is not valid json ]")
    with pytest.raises(BiasStoreError):
        load_battery(tmp_path, fmt="json")


def test_schema_version_mismatch_refuses_to_load(tmp_path: Path) -> None:
    dump_battery([_record()], tmp_path, fmt="json")
    path = tmp_path / "bias_battery_record.json"
    path.write_text(
        path.read_text().replace(
            f'"schema_version": "{BIAS_SCHEMA_VERSION}"', '"schema_version": "999"'
        )
    )
    with pytest.raises(BiasStoreError):
        load_battery(tmp_path, fmt="json")
