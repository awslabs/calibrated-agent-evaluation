# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""How many double-coded traces the agreement gate needs, from the function the gate's minimum uses.

For raters expected to agree at alpha 0.90 on calibration gold balanced between PASS and FAIL,
this prints the expected counts at a few overlap sizes, alpha's bootstrap lower bound on them,
whether the gate clears (with the margin for its 2000 resamples), and the minimum. Run from the
repository root with the ``stats`` extra.

    uv run --extra stats python docs/site/examples/guides/overlap_size.py
"""

from __future__ import annotations

from calibrated_agent_evaluation.review.overlap import (
    alpha_lower_bound_at,
    clears_planned,
    plan_overlap,
    plug_in_counts,
)

plan = plan_overlap()
print(f"raters expected at alpha {plan.expected_alpha}, {plan.prevalence:.0%} FAIL\n")
print("shared traces   both FAIL / both PASS / split   lower bound   clears the gate")
for n in (50, 100, 110, 122, 130, 150, 200):
    counts = plug_in_counts(n)
    shown = " / ".join(str(c) for c in counts)
    clears = "yes" if clears_planned(counts) else "no"
    print(f"  {n:>3}           {shown:<29}   {alpha_lower_bound_at(counts):.3f}         {clears}")
print(f"\nminimum overlap: {plan.n_units} (clears at every size from there to {plan.scan_limit})")
