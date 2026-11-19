# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-relative bias thresholds, fail-closed for must-pass biases.

Bias magnitude varies ~10x across judge models, so there is no single fixed
bar -- each bias is compared to a threshold configured *for that judge model*. A must-pass bias
with no configured threshold **fails closed**: the battery refuses rather than applying a borrowed
or permissive default ("enforcement, not enablement"). An advisory bias with no threshold is
recorded without a pass/fail (:meth:`has_threshold` is ``False``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from calibrated_agent_evaluation.bias.errors import MissingBiasThresholdError
from calibrated_agent_evaluation.bias.record import MUST_PASS, BiasName


@dataclass(frozen=True)
class BiasThresholds:
    """Per-model, per-bias maximum-allowed magnitude (the upper bound must not exceed it)."""

    by_model: Mapping[str, Mapping[BiasName, float]]

    def has_threshold(self, model: str, bias: BiasName) -> bool:
        """Whether a threshold is configured for ``(model, bias)`` (gates advisory biases)."""
        return bias in self.by_model.get(model, {})

    def threshold_for(self, model: str, bias: BiasName) -> float:
        """Return the configured threshold for ``(model, bias)``.

        Raises :class:`MissingBiasThresholdError` when ``bias`` is a must-pass bias with no
        configured threshold for ``model`` (fail-closed). For an advisory bias, callers
        gate on :meth:`has_threshold` first; calling this for an unconfigured advisory bias also
        raises (the caller bug of asking for a threshold that was never set).
        """
        model_thresholds = self.by_model.get(model, {})
        if bias in model_thresholds:
            return model_thresholds[bias]
        if bias in MUST_PASS:
            raise MissingBiasThresholdError(
                f"no bias threshold configured for must-pass bias {bias!r} on model {model!r}; "
                "refusing to produce a verdict (fail-closed). Configure a per-model "
                "threshold before running the battery."
            )
        raise MissingBiasThresholdError(
            f"no threshold configured for advisory bias {bias!r} on model {model!r}; "
            "check has_threshold() before calling threshold_for()"
        )
