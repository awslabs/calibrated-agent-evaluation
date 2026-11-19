# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the exploratory cross-slice flagging.

Behavioural: flag_slices applies BY-FDR at q=0.10 (not BH), records m = the slice count, and is
never more permissive than uncorrected flagging.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.calibrate.errors import StatisticsInputError
from calibrated_agent_evaluation.run.flags import FLAG_FDR_Q, flag_slices
from calibrated_agent_evaluation.run.multi import apply_multiplicity


def test_applies_fdr_by_and_records_slice_count() -> None:
    """Exploratory flags use fdr_by (dependence-aware) and record m = number of slices."""
    pvals = [0.001, 0.02, 0.3, 0.7, 0.9]
    result = flag_slices(pvals)
    assert result.method == "fdr_by"
    assert result.m == 5
    assert len(result.reject) == 5


def test_by_fdr_no_more_permissive_than_uncorrected() -> None:
    """BY correction rejects no MORE than a raw p < q=0.10 rule (a control, not a boost).

    BY at q=0.10 is stricter than the naive "flag anything with p < 0.10" rule -- so the number of
    flags it raises is <= the naive count. This is the whole point: it curbs the ~64%-per-run false
    alarm rate at 20 slices.
    """
    pvals = [0.001, 0.04, 0.06, 0.08, 0.3, 0.5, 0.7, 0.9]
    naive_flags = sum(1 for p in pvals if p < FLAG_FDR_Q)
    corrected = sum(flag_slices(pvals).reject)
    assert corrected <= naive_flags


def test_strong_signal_survives_correction() -> None:
    """A strongly-significant slice survives BY correction (a genuine flag)."""
    result = flag_slices([0.0001, 0.5, 0.6, 0.7])
    assert result.reject[0] is True


def test_fdr_bh_is_not_reachable_through_flags() -> None:
    """The flag family is BY-only; the primitive refuses fdr_bh (correlated).

    flag_slices exposes no method arg; the guard lives one level down in apply_multiplicity,
    exercised here to prove the forbidden method cannot slip through the flag family.
    """
    with pytest.raises(StatisticsInputError):
        apply_multiplicity([0.01, 0.5], family="drift_flags", method="fdr_bh")  # type: ignore[arg-type]
