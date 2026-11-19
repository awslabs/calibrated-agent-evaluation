# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-relative threshold lookup, fail-closed for must-pass."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.bias import BiasThresholds, MissingBiasThresholdError


def _thresholds() -> BiasThresholds:
    return BiasThresholds(by_model={"model-a": {"length": 0.15, "compassion_fade": 0.2}})


def test_returns_configured_threshold_for_must_pass() -> None:
    assert _thresholds().threshold_for("model-a", "length") == 0.15
    assert _thresholds().has_threshold("model-a", "length") is True


def test_missing_must_pass_threshold_fails_closed() -> None:
    # self_preference is must-pass but unconfigured for model-a -> refuse (fail-closed).
    with pytest.raises(MissingBiasThresholdError):
        _thresholds().threshold_for("model-a", "self_preference")
    # a completely unknown model likewise fails closed for a must-pass bias.
    with pytest.raises(MissingBiasThresholdError):
        _thresholds().threshold_for("unknown-model", "length")


def test_advisory_without_threshold_is_reported_absent_not_defaulted() -> None:
    # An advisory bias with no configured threshold reports has_threshold False (the battery then
    # records it without a pass/fail) rather than silently defaulting to a permissive value.
    assert _thresholds().has_threshold("model-a", "bandwagon") is False
