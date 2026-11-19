# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multiple-comparison control.

Watch 20 slices and one fails by pure chance ~64% of the time. The framework corrects for
this, and -- crucially -- uses the RIGHT correction per use:

  * **Holm** for a guardrail family in which any one rejection blocks the change (the improve
    guard panel). A gate that passes only when every comparison passes (the CI gate across modes,
    judge certification across TPR and TNR) is an intersection-union test and takes no correction.
  * **Benjamini-Yekutieli (fdr_by)** for exploratory cross-slice flags -- the dependence-aware FDR
    (NOT ``fdr_bh``, which assumes independence the slices don't have).

Wraps the verified ``statsmodels.stats.multitest.multipletests`` surface; only
these two methods are accepted, and the number of comparisons ``m`` is returned so it can be
recorded
per run (so the number of comparisons m is stated per run).
"""

from __future__ import annotations

from collections.abc import Sequence

from statsmodels.stats.multitest import multipletests

from calibrated_agent_evaluation.calibrate.errors import StatisticsInputError

_METHODS = frozenset({"holm", "fdr_by"})


def multiplicity(
    pvals: Sequence[float],
    method: str = "holm",
    *,
    alpha: float = 0.05,
) -> tuple[list[bool], int]:
    """Apply a multiplicity correction; return ``(reject, m)``.

    ``reject[i]`` is True when hypothesis ``i`` survives the correction at level ``alpha``; ``m`` is
    the number of comparisons (``len(pvals)``), recorded per run so the gate is auditable.

    Only ``"holm"`` (a hard-reject guardrail family) and ``"fdr_by"`` (exploratory dependence-aware
    FDR) are accepted -- an unknown method raises :class:`StatisticsInputError` rather than silently
    passing through to statsmodels (which would accept e.g. the independence-assuming ``fdr_bh`` we
    forbid).
    """
    if method not in _METHODS:
        raise StatisticsInputError(
            f"multiplicity: method must be one of {sorted(_METHODS)}, got {method!r} "
            "(Holm for hard-reject guardrails, fdr_by for exploratory flags --)"
        )
    if not pvals:
        raise StatisticsInputError("multiplicity: pvals must be non-empty")
    if any(not 0.0 <= p <= 1.0 for p in pvals):
        raise StatisticsInputError(
            f"multiplicity: all p-values must be in [0, 1], got {list(pvals)}"
        )
    reject = multipletests(list(pvals), alpha=alpha, method=method)[0]
    return [bool(r) for r in reject], len(pvals)
