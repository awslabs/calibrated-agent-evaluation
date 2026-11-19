# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""How many disagreeing trials a bias ceiling lets through, by probe-set size.

The bias battery gates each must-pass bias on the one-sided 95% upper bound of how much more often
its trials disagree than their placebo repeats: Newcombe's paired interval at the 90% two-sided
level, the bound ``calibrated_agent_evaluation.bias.run_battery`` computes. This prints it with no
placebo disagreeing, and the most disagreeing trials a ceiling of 0.35 passes. Run from the
repository root with the ``stats`` extra.

    uv run --extra stats python docs/site/examples/bias_ceiling.py
"""

from __future__ import annotations

from calibrated_agent_evaluation.bias.battery import DIRECTION_THRESHOLD
from calibrated_agent_evaluation.stats.proportions import newcombe_paired_diff

CEILING = 0.35


def upper_bound(n: int, disagreeing: int) -> float:
    """The bound for ``disagreeing`` of ``n`` trials when no placebo disagrees."""
    _, _, upper = newcombe_paired_diff(0, disagreeing, 0, n - disagreeing, alpha=0.10)
    return max(upper, 0.0)


print(f"ceiling {CEILING:.2f}; no placebo disagrees\n")
print("trials   upper bound at 0, 1, 2, 3 disagreeing          most disagreeing that pass")
for n in (6, 12, 30):
    bounds = [upper_bound(n, k) for k in range(n + 1)]
    passing = [k for k, ub in enumerate(bounds) if ub <= CEILING]
    most = str(max(passing)) if passing else "none"
    shown = ", ".join(f"{ub:.3f}" for ub in bounds[:4])
    print(f"  {n:>2}     {shown:<42}  {most}")

fewest = next(n for n in range(1, 200) if upper_bound(n, 0) <= DIRECTION_THRESHOLD)
print(f"\ndirection bound {DIRECTION_THRESHOLD:.2f}: needs at least {fewest} trials with no moves")
