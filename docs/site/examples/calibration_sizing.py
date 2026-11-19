# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""How many gold labels a judge needs before "90% accurate" means anything.

The certification gate reads the one-sided 95% Wilson lower bound of TPR and TNR, not the point
estimate, and both must reach 0.80. About 90% observed accuracy fails on 20 labels per class and
passes on 35.

    uv run --extra stats python docs/site/examples/calibration_sizing.py
"""

from __future__ import annotations

from dataclasses import dataclass

from calibrated_agent_evaluation.calibrate import (
    CalibrationSet,
    InMemoryLockedTestLedger,
    LabelPair,
    certify,
    plan_calibration_size,
    wilson_calibration_size,
)
from calibrated_agent_evaluation.schema import Label, Outcome
from calibrated_agent_evaluation.stats import wilson_lower

ledger = InMemoryLockedTestLedger()

GATE = 0.80


@dataclass(frozen=True)
class Judge:
    judge_id: str
    model: str
    prompt_version: str


def locked_test(n_per_class: int, correct_per_class: int, set_id: str) -> CalibrationSet:
    """Gold labels and judge verdicts with exactly this accuracy on each class."""
    pairs = []
    classes: tuple[tuple[Outcome, Outcome], ...] = (("FAIL", "PASS"), ("PASS", "FAIL"))
    for gold, other in classes:
        for i in range(n_per_class):
            trace_id = f"{set_id}-{gold.lower()}-{i}"
            verdict = gold if i < correct_per_class else other
            pairs.append(
                LabelPair(
                    gold=Label(
                        trace_id=trace_id, span_id=None, kind="human", outcome=gold, critique="gold"
                    ),
                    verdict=Label(
                        trace_id=trace_id,
                        span_id=None,
                        kind="llm-judge",
                        outcome=verdict,
                        critique="v",
                    ),
                )
            )
    return CalibrationSet(tuple(pairs), split="locked_test", set_id=set_id)


print("about 90% observed accuracy, by labels per class:")
for n in (20, 35, 50):
    k = round(0.9 * n)
    print(f"  {k}/{n} correct ({k / n:.1%}): lower bound {wilson_lower(k, n, alpha=0.10):.3f}")

print(f"\nlabels per class to clear the {GATE:.2f} gate:")
for accuracy in (0.95, 0.90, 0.85):
    certify_at = wilson_calibration_size(accuracy, GATE)
    plan = plan_calibration_size(accuracy, GATE)
    print(f"  expected accuracy {accuracy:.2f}: clears from {certify_at:>3} on, plan {plan:>3}")

print("\nthe same judge, certified twice:")
judge = Judge(
    "refund-policy", model="bedrock/global.anthropic.claude-sonnet-4-6", prompt_version="v3"
)
for n in (20, 35):
    k = round(0.9 * n)
    record = certify(judge, locked_test(n, k, set_id=f"locked-{n}"), ledger=ledger)
    print(
        f"  {k}/{n} per class: TPR lower bound {record.tpr.lb95:.3f}, "
        f"TNR lower bound {record.tnr.lb95:.3f}: {record.result}"
    )
