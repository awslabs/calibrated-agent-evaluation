# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive the minimum rate-sample size for a corrected rate from measured coverage.

The coverage floor (:mod:`calibrated_agent_evaluation.stats.studies.planner_floor`) bounds the
calibration side: how many gold labels per class the interval needs. This module bounds the other
sample, the production-representative verdicts the corrected rate is read from.

Input is coverage scans of the shipped interval at several seeds. Cells of the same configuration
are pooled across seeds. Only cells the shipped floor admits are kept: TPR = TNR
= ``p_hat`` with ``n_class`` at or above ``COVERAGE_FLOOR[p_hat]``, the population a corrected rate
is ever computed for.

Each cell gets a one-sided exact binomial test for coverage below the acceptance band's lower edge
(0.9405), and Holm corrects all kept cells together. A configuration ``(p_hat, n_class,
prevalence)`` is eval-limited when it has a rejected cell but passes at every larger scanned
``n_eval``: more traces restore its coverage. Its minimum is the smallest scanned ``n_eval`` from
which every larger scanned one passes. A configuration still rejected at the largest scanned
``n_eval`` is calibration-limited: more evaluation traces cannot help it, and it is reported as a
known limit, not folded into this minimum. The shipped minimum is the largest eval-limited minimum,
so it is one constant that holds for every prevalence scanned (0.02 to 0.50).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from scipy.stats import binomtest

from calibrated_agent_evaluation.calibrate.plan import COVERAGE_FLOOR
from calibrated_agent_evaluation.stats.multiplicity import multiplicity

__all__ = ["ARM", "FAMILY_ALPHA", "LOWER_EDGE", "Config", "cells_from", "derive", "pool"]

LOWER_EDGE = 0.9405
FAMILY_ALPHA = 0.05
ARM = "jeffreys"

Config = tuple[float, int, float]


def _admitted(p_hat: float, n_class: int) -> bool:
    floor = dict(COVERAGE_FLOOR).get(round(p_hat, 2))
    return floor is not None and n_class >= floor


def pool(rows: Sequence[dict[str, Any]]) -> dict[tuple[Config, int], list[int]]:
    """``(config, n_eval) -> [covered, replicates]`` over admitted symmetric cells, seeds pooled."""
    cells: dict[tuple[Config, int], list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        if "tnr" in r or "tpr" in r:
            continue
        p, n = float(r["p_hat"]), int(r["n_class"])
        if not _admitted(p, n):
            continue
        arm = r["arms"][ARM]
        reps = int(r["replicates"])
        cell = cells[((p, n, float(r["prevalence"])), int(r["n_eval"]))]
        cell[0] += reps - int(arm["miss_above"]) - int(arm["miss_below"])
        cell[1] += reps
    return dict(cells)


def derive(cells: dict[tuple[Config, int], list[int]]) -> dict[str, Any]:
    """The minimum rate sample and the evidence it was read from.

    Returns the minimum, the Holm-rejected cells, each eval-limited configuration's minimum, the
    calibration-limited configurations, and the worst kept cell at or above the minimum.
    """
    keys = sorted(cells)
    pvals = [
        binomtest(cells[k][0], cells[k][1], LOWER_EDGE, alternative="less").pvalue for k in keys
    ]
    reject, m = multiplicity(pvals, alpha=FAMILY_ALPHA, method="holm")
    rejected = {k for k, r in zip(keys, reject, strict=True) if r}
    by_config: dict[Config, list[int]] = defaultdict(list)
    for config, n_eval in keys:
        by_config[config].append(n_eval)
    eval_limited: dict[Config, int] = {}
    calibration_limited: list[Config] = []
    for config, evals in by_config.items():
        evals.sort()
        bad = [e for e in evals if (config, e) in rejected]
        if not bad:
            continue
        if bad[-1] == evals[-1]:
            calibration_limited.append(config)
            continue
        eval_limited[config] = next(e for e in evals if e > bad[-1])

    def cov(k: tuple[Config, int]) -> float:
        return round(cells[k][0] / cells[k][1], 4)

    replicates = sorted({c[1] for c in cells.values()})
    minimum = max(eval_limited.values()) if eval_limited else min(e for _, e in keys)
    return {
        "minimum": minimum,
        "tests": m,
        "replicates_per_cell": replicates,
        "rejected_cells": [
            {
                "p_hat": c[0],
                "n_class": c[1],
                "prevalence": c[2],
                "n_eval": e,
                "coverage": cov((c, e)),
                "kind": "eval-limited" if c in eval_limited else "calibration-limited",
            }
            for c, e in sorted(rejected)
        ],
        "eval_limited": [
            {"p_hat": c[0], "n_class": c[1], "prevalence": c[2], "minimum": v}
            for c, v in sorted(eval_limited.items())
        ],
        "calibration_limited": [
            {"p_hat": c[0], "n_class": c[1], "prevalence": c[2]}
            for c in sorted(calibration_limited)
        ],
        "worst_admitted_at_or_above_minimum": _worst(cells, minimum, set(calibration_limited)),
    }


def _worst(
    cells: dict[tuple[Config, int], list[int]], minimum: int, excluded: set[Config]
) -> dict[str, Any]:
    """The lowest-coverage cell at or above ``minimum`` outside the calibration-limited configs."""
    kept = [k for k in cells if k[1] >= minimum and k[0] not in excluded]
    worst = min(kept, key=lambda k: cells[k][0] / cells[k][1])
    (p, n, prev), n_eval = worst
    return {
        "p_hat": p,
        "n_class": n,
        "prevalence": prev,
        "n_eval": n_eval,
        "coverage": round(cells[worst][0] / cells[worst][1], 4),
    }


def cells_from(doc: dict[str, Any]) -> dict[tuple[Config, int], list[int]]:
    """Rebuild the pooled cells from a derived file's ``cell_counts``."""
    return {
        ((float(p), int(n), float(prev)), int(e)): [int(cov), int(reps)]
        for p, n, prev, e, cov, reps in doc["cell_counts"]
    }
