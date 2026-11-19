# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Did the fix work? Two fixes with the same before/after numbers get different verdicts.

Both fixes take a held-out slice of 40 traces from 18 failures to 4. The OrderBot fix is accepted.
The ClaimsCrew fix is rejected: the agent it touched fails less, but the crew's workflow outcome
rate did not move, so the failure moved somewhere else instead of going away.

    uv run --extra stats python docs/site/examples/improve_guard.py
"""

from __future__ import annotations

import sys

from calibrated_agent_evaluation.attribution import CrewDefinition, CrewOutcomes, ExpectedProcess
from calibrated_agent_evaluation.calibrate import Interval
from calibrated_agent_evaluation.improve import (
    GuardSlice,
    ImproveRequest,
    ImproveVerdict,
    PairedObs,
    PreRegEntry,
    PreRegLedger,
    evaluate_improvement,
    register_prereg,
)


def rate(point: float) -> Interval:
    """A guard slice's failure rate, unchanged by the fix in both examples."""
    return Interval(
        point=point,
        lo=point - 0.05,
        hi=point + 0.05,
        refusal_reason=None,
        method="illustrative",
        level=0.95,
        sided="two-sided",
        n=100,
    )


def paired(prefix: str) -> tuple[PairedObs, ...]:
    """40 held-out traces: 18 failed before the fix, 4 fail after."""
    return tuple(
        PairedObs(f"{prefix}-{i}", before_fail=i < 18, after_fail=i < 4) for i in range(40)
    )


DETAIL = "--detail" in sys.argv[1:]


def report(name: str, verdict: ImproveVerdict) -> None:
    if not DETAIL:
        if verdict.verdict == "accepted":
            print(
                f"{name}: accepted. The failure rate dropped by at least the registered amount, "
                "more than chance would explain."
            )
        elif any(r.startswith("failure relocation") for r in verdict.reasons):
            print(
                f"{name}: rejected. The agent it touched fails less, but the crew's outcome did "
                "not improve: the failures moved instead of going away."
            )
        else:
            print(f"{name}: rejected. {'; '.join(verdict.reasons)}")
        return
    effect = verdict.paired.effect.point
    print(f"{name}: {verdict.verdict}")
    print(f"  McNemar ({verdict.paired.method}) p = {verdict.paired.p_value:.2g}")
    print(
        f"  change in failure rate {effect:+.3f}" if effect is not None else "  no effect estimate"
    )
    for reason in verdict.reasons:
        print(f"  reason: {reason}")


orderbot = ImproveRequest(
    # The target and the minimum reduction are registered before the fix is measured.
    ledger=register_prereg(
        PreRegLedger(),
        PreRegEntry(
            prereg_id="ob-1", target_metric="trusted_broken_policy_tool", success_threshold=0.10
        ),
    ),
    target_metric="trusted_broken_policy_tool",
    pairs=paired("ORD"),
    confirmation_ids=tuple(f"ORD-{i}" for i in range(40)),
    development_ids=("ORD-1001", "ORD-1002"),
    slices=(
        GuardSlice(
            name="misread_amount",
            kind="other",
            regression_p=0.7,
            rate_before=rate(0.06),
            rate_after=rate(0.06),
        ),
    ),
)
report("OrderBot fix", evaluate_improvement(orderbot))

crew = CrewDefinition(
    agents=("intake", "policy-checker"),
    handoffs=("intake->policy-checker",),
    scenarios={
        "claim": ExpectedProcess(
            scenario="claim",
            expected_agents=("intake", "policy-checker"),
            terminal_decision="approve",
            payout_allowed=True,
            escalation_required=False,
        )
    },
    rules=(),
)
claimscrew = ImproveRequest(
    ledger=register_prereg(
        PreRegLedger(),
        PreRegEntry(
            prereg_id="clm-1",
            target_metric="workflow_outcome_rate",
            success_threshold=0.10,
            is_crew=True,
        ),
    ),
    target_metric="workflow_outcome_rate",
    pairs=paired("CLM"),
    confirmation_ids=tuple(f"CLM-{i}" for i in range(40)),
    development_ids=("CLM-4471", "CLM-4473"),
    slices=(
        GuardSlice(
            name="policy-checker",
            kind="agent",
            regression_p=0.8,
            rate_before=rate(0.15),
            rate_after=rate(0.15),
        ),
    ),
    crew=crew,
    crew_before=CrewOutcomes(
        workflow_failures=60, workflow_total=200, per_agent_failures={"intake": 60}
    ),
    crew_after=CrewOutcomes(
        workflow_failures=60, workflow_total=200, per_agent_failures={"intake": 20}
    ),
    touched_agent="intake",
)
print()
report("ClaimsCrew fix to intake", evaluate_improvement(claimscrew))
