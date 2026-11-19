# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the re-test oracles: the re-test verdict each replay set must give, computed without
Calibrated Agent Evaluation.

Reads the H1 (``abtest/replay``) and H2 (``recommendations/replay``) roots that
``fixtures/raw/agentcore/make_replay_roots.py`` wrote, pairs each trace's before and after human
verdicts for the target and the guard mode, and applies the re-test's documented rule with scipy and
statsmodels:

* McNemar two-sided: mid-p (``2 * binom.cdf(k) - binom.pmf(k)``, ``k = min(b, c)``, capped at 1)
  below 25 discordant pairs; ``statsmodels`` ``mcnemar(exact=False, correction=False)`` at 25 or
  more.
* The effect is the after-minus-before fail rate with its own Newcombe (1998) method-10 95%
  interval, checked against the NCSS "Two Correlated Proportions" Example 1.
* The guard's one-sided regression p is half the two-sided p when regressions outnumber
  improvements, else one minus that half, Holm-corrected at 0.05 across the panel.
* Accepted when the two-sided p is below 0.05, the effect is negative, its upper bound is at most
  ``-0.10`` (the registered minimum reduction) and no guard regressed.

Run from the repository root: ``python fixtures/expected/audit/make_oracles.py``.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

from scipy.stats import binom, norm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

_RAW = Path("fixtures/raw/agentcore")
_OUT = Path("fixtures/expected/audit")
TARGET = "missing_date_range"
GUARD = "wrong_join"
MIN_REDUCTION = 0.10
Z = float(norm.ppf(0.975))


def wilson(x: int, n: int) -> tuple[float, float]:
    p = x / n
    den = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """``(p1 - p2, lower, upper)``, ``p1 = (a + b)/n`` before, ``p2 = (a + c)/n`` after."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if den == 0:
        phi = 0.0
    elif a * d > b * c:
        phi = max(a * d - b * c - n / 2, 0) / den
    else:
        phi = (a * d - b * c) / den
    lo = (p1 - p2) - math.sqrt(
        max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    )
    hi = (p1 - p2) + math.sqrt(
        max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    )
    return p1 - p2, lo, hi


def two_sided(b: int, c: int) -> tuple[str, float]:
    m = b + c
    if m < 25:
        k = min(b, c)
        return "midp", min(1.0, float(2 * binom.cdf(k, m, 0.5) - binom.pmf(k, m, 0.5)))
    result = mcnemar([[0, b], [c, 0]], exact=False, correction=False)
    return "asymptotic", float(result.pvalue)  # pyright: ignore[reportAttributeAccessIssue]  # an untyped bunch


def verdicts(root: str, mode: str) -> dict[str, bool]:
    plain = _RAW / f"{root}.json"
    gzipped = plain.with_name(plain.name + ".gz")
    doc = json.loads(
        gzip.decompress(gzipped.read_bytes()) if gzipped.exists() else plain.read_bytes()
    )
    return {
        lb["trace_id"]: lb["outcome"] == "FAIL"
        for lb in doc["labels"]
        if lb["kind"] == "human" and lb["failure_mode"] == mode
    }


def cells(before: dict[str, bool], after: dict[str, bool]) -> tuple[int, int, int, int]:
    a = sum(before[t] and after[t] for t in before)
    b = sum(before[t] and not after[t] for t in before)
    c = sum(not before[t] and after[t] for t in before)
    return a, b, c, len(before) - a - b - c


def oracle(before_root: str, after_root: str) -> dict:
    target = cells(verdicts(before_root, TARGET), verdicts(after_root, TARGET))
    guard = cells(verdicts(before_root, GUARD), verdicts(after_root, GUARD))
    a, b, c, d = target
    method, p = two_sided(b, c)
    diff, lo, hi = newcombe(a, b, c, d)
    effect = {"point": -diff, "lo": -hi, "hi": -lo, "n": a + b + c + d}
    _, gb, gc, _ = guard
    _, gp = two_sided(gb, gc)
    regression_p = gp / 2 if gc > gb else 1 - gp / 2
    rejected, _, _, _ = multipletests([regression_p], alpha=0.05, method="holm")
    regressed = [GUARD] if bool(rejected[0]) else []
    reasons = []
    if not (p < 0.05 and effect["point"] < 0):
        reasons.append("not a significant reduction")
    if effect["hi"] > -MIN_REDUCTION:
        reasons.append("threshold")
    if regressed:
        reasons.append("guard regressed")
    return {
        "target_cells": list(target),
        "guard_cells": list(guard),
        "mcnemar": {"method": method, "n_discordant": b + c, "p_value": p, "effect": effect},
        "panel": {
            "method": "holm",
            "m": 1,
            "slices": [{"name": GUARD, "regression_p": regression_p}],
            "regressed": regressed,
        },
        "min_reduction": MIN_REDUCTION,
        "verdict": "rejected" if reasons else "accepted",
        "reasons": reasons,
    }


def main() -> None:
    diff, lo, hi = newcombe(30, 6, 5, 9)
    if (round(lo, 4), round(hi, 4)) != (-0.1145, 0.1537):
        raise SystemExit(f"Newcombe {diff, lo, hi} does not match NCSS Example 1")
    for before, after, name in (
        ("abtest/replay/control", "abtest/replay/treatment", "retest_improve.json"),
        (
            "recommendations/replay/input",
            "recommendations/replay/result",
            "retest_recommendation_improve.json",
        ),
    ):
        doc = oracle(before, after)
        (_OUT / name).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
