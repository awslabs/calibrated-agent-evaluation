# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for calibrated_agent_evaluation.sampling.stats."""

from __future__ import annotations

import warnings

import pytest

from calibrated_agent_evaluation.sampling.errors import DeffWarning, SamplingConfigError
from calibrated_agent_evaluation.sampling.stats import kish_deff


class TestKishDeff:
    """Kish design effect and effective sample size."""

    def test_uniform_weights_deff_equals_1(self) -> None:
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        deff, n_eff = kish_deff(weights)
        assert deff == pytest.approx(1.0)
        assert n_eff == pytest.approx(5.0)

    def test_known_unequal_weights(self) -> None:
        # Hand-computed: weights = [1, 1, 1, 3]
        # n=4, sum_w=6, sum_w_sq=1+1+1+9=12
        # deff = (12 * 4) / (6^2) = 48/36 = 4/3
        # n_eff = 4 / (4/3) = 3
        weights = [1.0, 1.0, 1.0, 3.0]
        deff, n_eff = kish_deff(weights)
        assert deff == pytest.approx(4.0 / 3.0)
        assert n_eff == pytest.approx(3.0)

    def test_single_weight(self) -> None:
        weights = [5.0]
        deff, n_eff = kish_deff(weights)
        assert deff == pytest.approx(1.0)
        assert n_eff == pytest.approx(1.0)


class TestKishDeffWarning:
    """DeffWarning emitted when deff > threshold."""

    def test_high_deff_emits_warning(self) -> None:
        # Weights chosen to give deff ≈ 6.0 (> the default 5.0 threshold). For a single
        # outlier w=[a, 1, …, 1] over n items, deff = (a²+n-1)·n / (a+n-1)²; solving that
        # for n=10, deff=6 gives a ≈ 30.27.
        weights = [30.27] + [1.0] * 9
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            deff, n_eff = kish_deff(weights)
            assert deff > 5.0
            assert n_eff < 10
            assert len(w) == 1
            assert issubclass(w[0].category, DeffWarning)
            assert "effective sample critically low" in str(w[0].message)

    def test_below_threshold_no_warning(self) -> None:
        weights = [1.0, 1.0, 2.0, 1.0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kish_deff(weights)
            deff_warnings = [x for x in w if issubclass(x.category, DeffWarning)]
            assert len(deff_warnings) == 0


class TestKishDeffNegative:
    """Edge cases — empty and all-zero weights."""

    def test_empty_weights_raises(self) -> None:
        with pytest.raises(SamplingConfigError, match="empty weight list"):
            kish_deff([])

    def test_all_zero_weights_raises(self) -> None:
        with pytest.raises(SamplingConfigError, match="all weights are zero"):
            kish_deff([0.0, 0.0, 0.0])
