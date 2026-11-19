# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Statistical utilities for the sampling engine."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

from calibrated_agent_evaluation.sampling.errors import DeffWarning, SamplingConfigError

DEFAULT_DEFF_THRESHOLD = 5.0


def kish_deff(
    weights: Sequence[float], *, threshold: float = DEFAULT_DEFF_THRESHOLD
) -> tuple[float, float]:
    """Compute Kish design effect and effective sample size.

    Deff = 1 + CV²(weights) = (sum(w²) * n) / (sum(w))²
    n_eff = n / Deff

    Emits ``DeffWarning`` when deff exceeds *threshold*.
    """
    n = len(weights)
    if n == 0:
        raise SamplingConfigError("Cannot compute Kish deff on an empty weight list")
    sum_w = sum(weights)
    if sum_w == 0.0:
        raise SamplingConfigError("Cannot compute Kish deff: all weights are zero")

    sum_w_sq = sum(w * w for w in weights)
    deff = (sum_w_sq * n) / (sum_w * sum_w)
    n_eff = n / deff

    if deff > threshold:
        warnings.warn(
            f"Deff={deff:.2f}, n_eff={n_eff:.1f}; effective sample critically low",
            DeffWarning,
            stacklevel=2,
        )

    return (deff, n_eff)
