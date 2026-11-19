# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""How many traces a failure mode needs before ``calibrated-agent-evaluation run --ci`` can pass it.

The gate passes a mode when the 95% lower bound on its pass rate clears the threshold (0.80 by
default), computed with the same ``wilson_lower`` the gate calls. A small sample cannot pass even
with no failures, because its lower bound is too wide. Run from the repository root with the
``stats`` extra.

    uv run --extra stats python docs/site/examples/integrate/gate_sample_size.py
"""

from __future__ import annotations

import math

from calibrated_agent_evaluation.stats import wilson_lower

THRESHOLD = 0.80
SEARCH_LIMIT = 5000


def fewest_with_failures(failures: int) -> int:
    """Smallest n where a mode with exactly ``failures`` failing traces clears the gate."""
    n = failures + 1
    while wilson_lower(n - failures, n) < THRESHOLD:
        n += 1
    return n


def fewest_at_rate(pass_rate: float) -> int | None:
    """Smallest n from which a mode passing at ``pass_rate`` clears the gate at every larger n.

    The pass count is rounded down, so the rate is never overstated, and the result holds for
    every sample size up to the search limit, not only at one lucky size.
    """
    first = None
    for n in range(1, SEARCH_LIMIT + 1):
        if wilson_lower(math.floor(pass_rate * n + 1e-9), n) >= THRESHOLD:
            first = n if first is None else first
        else:
            first = None
    return first


print(f"gate threshold: pass-rate 95% lower bound >= {THRESHOLD:.2f}\n")
print("failing traces in the mode   traces needed")
for failures in (0, 1, 2, 5):
    print(f"  {failures:>2}                          {fewest_with_failures(failures):>4}")

print("\ntrue pass rate of the mode   traces needed")
for rate in (1.00, 0.97, 0.95, 0.90, 0.85, 0.80):
    needed = fewest_at_rate(rate)
    shown = f"{needed:>4}" if needed is not None else "never passes"
    print(f"  {rate:.2f}                        {shown}")
