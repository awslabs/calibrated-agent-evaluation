# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bias-battery type invariants.

The must-pass / advisory partition is load-bearing: only a must-pass bias may gate,
and positional is neither (it is pairwise-only).
"""

from __future__ import annotations

from calibrated_agent_evaluation.bias import ADVISORY, MUST_PASS, ProbeResult


def test_must_pass_and_advisory_partition_is_exact() -> None:
    assert {"length", "self_preference", "sycophancy"} == MUST_PASS
    assert {"distraction", "bandwagon", "compassion_fade"} == ADVISORY
    # positional is in NEITHER set — it is pairwise-only and never gates.
    assert "positional" not in MUST_PASS
    assert "positional" not in ADVISORY
    assert MUST_PASS.isdisjoint(ADVISORY)


def test_probe_result_carries_both_point_and_bounded_magnitude() -> None:
    # "no bare number": every probe result carries the point AND its lower bound.
    names = set(ProbeResult.model_fields)
    assert {"magnitude", "magnitude_ub95", "threshold", "passed", "is_must_pass"} <= names
