# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the auditable ProbeFinding envelope."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.probe import LimitsFraming, ProbeError, ProbeFinding


def _finding(**over: object) -> ProbeFinding:
    base: dict[str, object] = {
        "kind": "capability_map",
        "captured_at": "2026-07-07T00:00:00+00:00",
        "endpoint_fingerprint": "ep-abc123",
        "framing": LimitsFraming(),
        "payload": {"observed_tools": "lookup"},
    }
    base.update(over)
    return ProbeFinding(**base)  # type: ignore[arg-type]


def test_finding_carries_timestamp_fingerprint_and_framing() -> None:
    f = _finding()
    assert f.captured_at
    assert f.endpoint_fingerprint
    assert f.framing.tool_inventory_is_lower_bound is True  # structured framing present


def test_finding_without_fingerprint_is_rejected() -> None:
    # Negative: an unfingerprinted finding is not auditable.
    with pytest.raises(ProbeError):
        _finding(endpoint_fingerprint="")


def test_finding_without_timestamp_is_rejected() -> None:
    # Negative/edge: no capture time -> not attributable to a run.
    with pytest.raises(ProbeError):
        _finding(captured_at="")


def test_framing_defaults_are_all_set() -> None:
    fr = LimitsFraming()
    assert fr.tool_inventory_is_lower_bound is True
    assert fr.refusal_is_not_inability is True
    assert fr.verdict_bounded_by_judge_ceiling is True
