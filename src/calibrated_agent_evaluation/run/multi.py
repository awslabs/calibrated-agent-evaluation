# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multiplicity application across a run's comparison families.

The stats layer ships the *primitive* -- :func:`calibrated_agent_evaluation.stats.multiplicity`
(Holm / Benjamini-Yekutieli, ``fdr_bh`` forbidden). This wrapper applies it to a run's families (the
drift flags, the improve guard slices) and packages the result as a
:class:`~calibrated_agent_evaluation.run.report.MultiplicityResult` that states the method +
comparison count ``m`` per family. The CI gate across modes and judge certification are
intersection-union tests and are not corrected here. It re-implements no correction; an invalid
method / empty pvals raise the shared
:class:`~calibrated_agent_evaluation.calibrate.errors.StatisticsInputError` unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

from calibrated_agent_evaluation.run.report import MultiplicityResult, MultMethod
from calibrated_agent_evaluation.stats import multiplicity


def apply_multiplicity(
    pvals: Sequence[float],
    *,
    family: str,
    method: MultMethod,
    alpha: float = 0.05,
) -> MultiplicityResult:
    """Correct ``pvals`` for multiplicity and package the method + ``m`` + reject vector.

    ``family`` names the correction site ("drift_flags", "improve_guard", ...). ``method`` is
    ``"holm"`` (a guardrail family where one rejection blocks the change) or ``"fdr_by"``
    (exploratory dependence-aware FDR); ``fdr_bh`` and any other method are refused by the
    underlying primitive (:class:`StatisticsInputError`), because slices are built from overlapping
    traces and BH assumes an independence they do not have. ``m`` == ``len(pvals)`` is recorded so
    the run is auditable.
    """
    reject, m = multiplicity(pvals, method, alpha=alpha)
    return MultiplicityResult(family=family, method=method, m=m, reject=tuple(reject))
