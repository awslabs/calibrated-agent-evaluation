# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``plan --gate``: how many traces each failure mode needs before ``run`` can pass.

``calibrated-agent-evaluation run`` passes a mode when the 95% lower bound of its pass rate reaches
``--threshold``. A suite that truly passes ``--expected-pass`` of the time needs enough verdicts per
mode for that bound to clear, and the bound is a sawtooth in the count: a size that clears can be
followed by larger ones that do not. :func:`calibrated_agent_evaluation.calibrate.gate_set_size`
returns the smallest size from which every larger one clears too, the same guarantee the label
planner gives calibration. Each gated mode is measured over the production-representative sample, so
the size is the number of those traces that need a verdict for every mode.
"""

from __future__ import annotations

import argparse

from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError


def plan_handler(args: argparse.Namespace) -> StageOutcome:
    """Print the per-mode CI set size for the gate at ``--threshold``."""
    if not getattr(args, "gate", False):
        raise UsageError(
            "calibrated-agent-evaluation plan needs --gate: it sizes the CI gate's sample (the "
            "calibration plan is "
            "printed by `calibrated-agent-evaluation calibrate`)"
        )
    from calibrated_agent_evaluation.cli.config import require_extras

    require_extras("statsmodels")
    from calibrated_agent_evaluation.calibrate import gate_fails_allowed, gate_set_size
    from calibrated_agent_evaluation.stats.proportions import wilson_lower

    threshold = float(args.threshold)
    expected = float(args.expected_pass)
    if not 0.0 < threshold < 1.0:
        raise UsageError(f"--threshold must be in (0, 1), got {threshold}")
    if not 0.0 < expected <= 1.0:
        raise UsageError(f"--expected-pass must be in (0, 1], got {expected}")
    if expected <= threshold:
        raise CliError(
            f"a suite expected to pass {expected:.2f} of the time cannot clear a gate at "
            f"{threshold:.2f} at any size: the gate reads the lower bound of the pass rate, which "
            "stays below the rate itself. Lower --threshold or improve the agent first"
        )
    n = gate_set_size(expected, threshold)
    fails = gate_fails_allowed(n, threshold)
    first = next(
        k for k in range(1, n + 1) if wilson_lower(k - fails_at(expected, k), k) >= threshold
    )
    summary = [
        f"Each gated failure mode needs verdicts on {n} production-representative traces for "
        f"`calibrated-agent-evaluation run --threshold {threshold:g}` to pass a suite that passes "
        f"{expected:.0%} of "
        f"the time; at {n} the gate tolerates up to {fails} FAIL verdict(s) per mode.",
        "Next: sample that many traces at random (not hand-picked ones), score them with "
        "`calibrated-agent-evaluation judge` or `calibrated-agent-evaluation score`, then run the "
        "gate.",
    ]
    detail = [
        f"gate: a mode passes when the 95% one-sided Wilson lower bound of its pass rate reaches "
        f"{threshold:g}",
        f"  plan at an expected pass rate of {expected:g}: {n} trace(s) per mode, the smallest "
        "size from which the bound clears at every larger size too (counts rounded down)",
        f"  the bound first clears at {first}"
        + (
            f", but some sizes between {first} and {n} fall short again, so {first} is not enough"
            if first < n
            else ""
        ),
        f"  at {n}: at most {fails} FAIL verdict(s) per mode keep the bound at or above "
        f"{threshold:g}",
    ]
    return StageOutcome(
        stage="plan", exit_code=EXIT_OK, summary="\n".join(summary), detail="\n".join(detail)
    )


def fails_at(expected: float, n: int) -> int:
    """The FAIL count a suite passing ``expected`` of the time is planned to show on ``n``."""
    import math

    return n - math.floor(expected * n + 1e-9)
