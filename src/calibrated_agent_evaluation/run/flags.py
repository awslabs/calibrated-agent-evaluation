# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exploratory cross-slice flagging with dependence-aware multiplicity control.

The run stage watches many slices (per taxonomy cell, per agent, per suite) for drift and regression
signals. Watch 20 slices and one "fails" by pure chance ~64% of the time,
so the exploratory *flags* are corrected with **Benjamini-Yekutieli FDR at q = 0.10** -- the
dependence-robust variant, because slices are built from overlapping traces (one trace feeds several
slices) so their p-values are correlated in arbitrary directions and plain Benjamini-Hochberg's
independence assumption does not hold (``fdr_bh`` is refused by the underlying primitive).

The CI gate (``run.gate``) is not a multiplicity family: it passes only when every mode passes, an
intersection-union test that needs no correction. Holm is for a family where one rejection blocks
the change (the improve guard panel); BY is for flags. Pure computation; the number of comparisons
``m`` is recorded on the result.
"""

from __future__ import annotations

from collections.abc import Sequence

from calibrated_agent_evaluation.run.multi import apply_multiplicity
from calibrated_agent_evaluation.run.report import MultiplicityResult

# The exploratory-flag false-discovery-rate target (BY-FDR at q = 0.10).
FLAG_FDR_Q = 0.10


def flag_slices(pvals: Sequence[float], *, family: str = "drift_flags") -> MultiplicityResult:
    """Flag exploratory cross-slice signals via Benjamini-Yekutieli FDR at q = 0.10.

    ``pvals`` are the per-slice p-values (one per watched slice); ``family`` names the flag family
    ("drift_flags", "regression_flags", ...). Returns a
    :class:`~calibrated_agent_evaluation.run.report.MultiplicityResult` whose ``reject[i]`` is True
    for a slice that survives the BY correction (a genuine flag) and whose ``m`` == the number of
    slices watched (stated per run). Delegates to the multiplicity primitive via
    :func:`apply_multiplicity`; ``fdr_bh`` is not offered here -- correlated slices forbid it.
    """
    return apply_multiplicity(pvals, family=family, method="fdr_by", alpha=FLAG_FDR_Q)
