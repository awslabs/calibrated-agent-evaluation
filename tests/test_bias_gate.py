# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bias promotion gate + calibration link."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from _helpers import TEST_STAMP
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.bias import (
    BiasBatteryRecord,
    BiasedJudgeError,
    MissingBiasBatteryError,
    StaleBiasBatteryError,
    enforce_bias_gate,
    link_battery_to_calibration,
)
from calibrated_agent_evaluation.calibrate import Bound, CalibrationRecord


@dataclass(frozen=True)
class FakeJudge:
    judge_id: str = "judge-1"
    model: str = "test-model"
    prompt_version: str = "v1"


def _calib(*, ref: str | None) -> CalibrationRecord:
    return CalibrationRecord(
        provenance=TEST_STAMP,
        calib_id="c1",
        judge_id="judge-1",
        model="test-model",
        prompt_version="v1",
        hunted_set_id="h1",
        n_pos=80,
        n_neg=80,
        tpr=Bound(point=0.95, lb95=0.90, n=80),
        tnr=Bound(point=0.95, lb95=0.90, n=80),
        youden_j=0.9,
        result="CERTIFIED",
        bias_battery_ref=ref,
    )


def _battery(
    *, verdict: str, model: str = "test-model", prompt_version: str = "v1", bid: str = "bat-1"
) -> BiasBatteryRecord:
    return BiasBatteryRecord(
        provenance=TEST_STAMP,
        battery_id=bid,
        judge_id="judge-1",
        model=model,
        prompt_version=prompt_version,
        is_pairwise=False,
        results=(),
        verdict=verdict,  # type: ignore[arg-type]
        failed_biases=(("length",) if verdict == "BIASED" else ()),  # type: ignore[arg-type]
    )


def test_link_sets_bias_battery_ref_for_clear_and_biased() -> None:
    base = _calib(ref=None)
    for verdict in ("CLEAR", "BIASED"):
        battery = _battery(verdict=verdict, bid=f"bat-{verdict}")
        linked = link_battery_to_calibration(base, battery)
        assert linked.bias_battery_ref == battery.battery_id
        # the returned record differs ONLY in bias_battery_ref (forbidden side-effect check).
        assert replace(linked, bias_battery_ref=None) == base


def test_clear_fresh_battery_permits_biased_refuses() -> None:
    judge = FakeJudge()
    clear = _battery(verdict="CLEAR")
    enforce_bias_gate(_calib(ref=clear.battery_id), judge, clear)  # returns cleanly

    biased = _battery(verdict="BIASED")
    with pytest.raises(BiasedJudgeError):
        enforce_bias_gate(_calib(ref=biased.battery_id), judge, biased)


def test_null_ref_refuses_missing_battery() -> None:
    with pytest.raises(MissingBiasBatteryError):
        enforce_bias_gate(_calib(ref=None), FakeJudge(), _battery(verdict="CLEAR"))


def test_ref_mismatch_is_stale() -> None:
    # calibration record points at a different battery id than the one supplied.
    battery = _battery(verdict="CLEAR", bid="bat-1")
    with pytest.raises(StaleBiasBatteryError):
        enforce_bias_gate(_calib(ref="a-different-battery"), FakeJudge(), battery)


def test_battery_pinned_to_other_version_is_absent() -> None:
    # A battery run against a different (model, prompt_version) is treated as absent (stale).
    stale = _battery(verdict="CLEAR", prompt_version="v2")
    with pytest.raises(StaleBiasBatteryError):
        enforce_bias_gate(_calib(ref=stale.battery_id), FakeJudge(), stale)
    # ...while a matching pin is recognized.
    fresh = _battery(verdict="CLEAR")
    enforce_bias_gate(_calib(ref=fresh.battery_id), FakeJudge(), fresh)
