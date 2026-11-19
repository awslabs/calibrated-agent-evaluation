# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive the calibration planner's coverage floor from measured coverage.

``plan_calibration_size`` inverts the gate's Wilson bound, which answers "how many labels certify
this judge". The corrected rate's interval needs its own minimum, and that minimum is read off
measured coverage here rather than picked.

Input is one or more coverage scans of the shipped interval over a ``p_hat x n_class`` cross, each
row (one ``p_hat`` and one ``n_class``) spanning prevalence 0.02 to 0.50 and the scanned eval
sizes.

A row passes when no cell's coverage is shown to lie outside the study's acceptance band, 0.9405
to 0.9595 (:func:`calibrated_agent_evaluation.stats.studies.coverage.acceptance_band` at its 2000
replicates). Each cell gets two one-sided exact binomial tests, below the band's lower edge and
above its upper edge, and Holm corrects the row's tests together: a row has dozens of cells and
uncorrected tests would fail some of an exactly nominal interval's. The band, not 0.95 itself, is
the null because the acceptance criterion tolerates that much, and a test against the point would
eventually reject an interval that misses nominal by 0.003. Over-coverage fails a row as well as
under-coverage, as in the acceptance criterion.

``floor(p_hat)`` is the smallest scanned ``n_class`` whose row and every larger scanned row pass.
Larger rows are included because coverage is not monotone in ``n_class``: it dips where the
expected error count per class is about 2 to 3. When no scanned size qualifies, the floor is
``None`` and the planner refuses.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from scipy.stats import binomtest

from calibrated_agent_evaluation.stats.multiplicity import multiplicity

__all__ = ["ARM", "FAMILY_ALPHA", "NOMINAL", "accuracy_key", "derive", "pool"]

NOMINAL = 0.95
FAMILY_ALPHA = 0.05
ARM = "jeffreys"


def accuracy_key(p: float) -> str:
    """A floor-table key: two decimals, or three where two would merge it with a neighbor."""
    two = f"{p:.2f}"
    return two if abs(float(two) - p) < 1e-9 else f"{p:.3f}"


def _cell_tests(cell: dict[str, Any], band: tuple[float, float], *, one_sided: bool) -> list[float]:
    """Exact binomial p-values that a cell's coverage is below the band, and above it if tested."""
    reps = int(cell["replicates"])
    covered = reps - cell["arms"][ARM]["miss_above"] - cell["arms"][ARM]["miss_below"]
    tests = [binomtest(covered, reps, band[0], alternative="less").pvalue]
    if not one_sided:
        tests.append(binomtest(covered, reps, band[1], alternative="greater").pvalue)
    return tests


def pool(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge the seeds of each cell: replicates and misses add, as independent replicates do."""
    pooled: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in rows:
        key = (float(r["p_hat"]), r.get("tnr"), int(r["n_class"]), r["prevalence"], r["n_eval"])
        cell = pooled.setdefault(
            key,
            {
                "p_hat": key[0],
                "tnr": key[1],
                "n_class": key[2],
                "prevalence": key[3],
                "n_eval": key[4],
                "replicates": 0,
                "arms": {ARM: {"miss_above": 0, "miss_below": 0}},
            },
        )
        cell["replicates"] += int(r["replicates"])
        cell["arms"][ARM]["miss_above"] += r["arms"][ARM]["miss_above"]
        cell["arms"][ARM]["miss_below"] += r["arms"][ARM]["miss_below"]
    for cell in pooled.values():
        misses = cell["arms"][ARM]["miss_above"] + cell["arms"][ARM]["miss_below"]
        cell["arms"][ARM]["coverage"] = round(1 - misses / cell["replicates"], 4)
    return list(pooled.values())


def derive(
    rows: Sequence[dict[str, Any]],
    band: tuple[float, float],
    *,
    one_sided_above: float | None = None,
) -> tuple[dict[float, int | None], list[dict[str, Any]]]:
    """The floor per accuracy, and the per-row verdicts it was read from.

    A row (one judge, ``p_hat`` and ``tnr``, at one ``n_class``) is the configuration a caller
    runs. Cells scanned under several seeds are pooled first. A row passes when Holm across its
    cells' one-sided tests rejects nothing. A row whose higher rate is above ``one_sided_above`` is
    a boundary row: the interval's limits there are set to the boundary, which over-covers by
    design, so only under-coverage is tested and over-coverage is accepted. Every other row is
    tested on both sides, as before.

    A judge type's floor is the smallest scanned ``n_class`` whose row and every larger scanned row
    pass, so a plan never sits just below a dip further up the scan. The floor at an accuracy is
    the largest floor over the judge types whose higher rate is that accuracy (a symmetric judge
    and the asymmetric ones scanned there), and ``None`` when any of them has none.
    """
    grouped: dict[tuple[float, float | None, int], list[dict[str, Any]]] = defaultdict(list)
    for r in pool(rows):
        tnr = None if r["tnr"] is None else float(r["tnr"])
        grouped[(r["p_hat"], tnr, r["n_class"])].append(r)
    report = []
    passes: dict[tuple[float, float | None, int], bool] = {}
    for (p, tnr, n), cells in sorted(
        grouped.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1, kv[0][2])
    ):
        top = max(p, p if tnr is None else tnr)
        one_sided = one_sided_above is not None and top > one_sided_above + 1e-12
        tests = [_cell_tests(c, band, one_sided=one_sided) for c in cells]
        pvals = [pv for t in tests for pv in t]
        reject, _ = multiplicity(pvals, alpha=FAMILY_ALPHA, method="holm")
        width = 1 if one_sided else 2
        under = sum(reject[0::width])
        over = 0 if one_sided else sum(reject[1::2])
        passes[(p, tnr, n)] = under == 0 and over == 0
        covs = [c["arms"][ARM]["coverage"] for c in cells]
        report.append(
            {
                "p_hat": p,
                **({} if tnr is None else {"tnr": tnr}),
                "n_class": n,
                "cells": len(cells),
                "replicates": cells[0]["replicates"],
                "one_sided": one_sided,
                "holm_under": under,
                "holm_over": over,
                "passes": passes[(p, tnr, n)],
                "min_coverage": min(covs),
                "max_coverage": max(covs),
            }
        )
    type_floor: dict[tuple[float, float | None], int | None] = {}
    for judge_type in {(p, tnr) for p, tnr, _ in grouped}:
        floor: int | None = None
        sizes = sorted((n for p, t, n in grouped if (p, t) == judge_type), reverse=True)
        for n in sizes:
            if not passes[(*judge_type, n)]:
                break
            floor = n
        type_floor[judge_type] = floor
    table: dict[float, int | None] = {}
    by_accuracy: dict[float, list[int | None]] = defaultdict(list)
    for (p, tnr), floor in type_floor.items():
        by_accuracy[round(max(p, p if tnr is None else tnr), 6)].append(floor)
    for acc in sorted(by_accuracy):
        floors = by_accuracy[acc]
        table[acc] = (
            None if any(f is None for f in floors) else max(f for f in floors if f is not None)
        )
    return table, report
