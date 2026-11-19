# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the multiplicity wrapper.

Behavioural: apply_multiplicity states the method + comparison count m per family and delegates to
the primitive; the forbidden fdr_bh (and any unknown method) is refused for correlated slices.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.calibrate.errors import StatisticsInputError
from calibrated_agent_evaluation.run.multi import apply_multiplicity


def test_reports_method_and_comparison_count() -> None:
    """The result states its family, method, and m == number of comparisons."""
    result = apply_multiplicity([0.001, 0.2, 0.9, 0.04], family="improve_guard", method="holm")
    assert result.family == "improve_guard"
    assert result.method == "holm"
    assert result.m == 4
    assert len(result.reject) == 4


def test_fdr_by_and_holm_both_produce_reject_vectors() -> None:
    """Both accepted methods return a boolean reject vector aligned to pvals."""
    pvals = [0.001, 0.01, 0.3, 0.8]
    by = apply_multiplicity(pvals, family="drift_flags", method="fdr_by")
    holm = apply_multiplicity(pvals, family="improve_guard", method="holm")
    assert by.method == "fdr_by"
    assert holm.method == "holm"
    assert all(isinstance(r, bool) for r in by.reject)
    # A strongly-significant p-value survives both corrections.
    assert by.reject[0] is True
    assert holm.reject[0] is True


def test_fdr_bh_is_refused() -> None:
    """Plain fdr_bh is forbidden (slices are correlated) -> typed refusal."""
    with pytest.raises(StatisticsInputError):
        apply_multiplicity([0.01, 0.5], family="drift_flags", method="fdr_bh")  # type: ignore[arg-type]


def test_unknown_method_is_refused() -> None:
    """An unknown correction method raises rather than passing through."""
    with pytest.raises(StatisticsInputError):
        apply_multiplicity([0.01, 0.5], family="improve_guard", method="bonferroni")  # type: ignore[arg-type]


def test_empty_pvals_refused() -> None:
    """An empty p-value family raises rather than reporting m=0 silently."""
    with pytest.raises(StatisticsInputError):
        apply_multiplicity([], family="improve_guard", method="holm")
