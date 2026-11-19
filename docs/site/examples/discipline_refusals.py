# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three ways to fool yourself with a judge, and the error Calibrated Agent Evaluation raises for
each.

1. Certifying on the few-shot examples the judge prompt was written from.
2. Certifying twice on the same locked test, after tuning against its first result.
3. Measuring the failure rate on the hunted calibration set, which was picked to be fail-heavy.

    uv run --extra stats python docs/site/examples/discipline_refusals.py
"""

from __future__ import annotations

from dataclasses import dataclass

from calibrated_agent_evaluation.calibrate import (
    CalibrationError,
    CalibrationSet,
    InMemoryLockedTestLedger,
    LabelPair,
    certify,
    corrected_rate,
)
from calibrated_agent_evaluation.schema import Label, Outcome


@dataclass(frozen=True)
class Judge:
    judge_id: str
    model: str
    prompt_version: str


ledger = InMemoryLockedTestLedger()


def pairs(n_per_class: int, correct_per_class: int) -> tuple[LabelPair, ...]:
    """Gold labels and judge verdicts with exactly this accuracy on each class."""
    out = []
    classes: tuple[tuple[Outcome, Outcome], ...] = (("FAIL", "PASS"), ("PASS", "FAIL"))
    for gold, other in classes:
        for i in range(n_per_class):
            trace_id = f"{gold.lower()}-{i}"
            verdict = gold if i < correct_per_class else other
            out.append(
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
    return tuple(out)


def attempt(description: str, action: object) -> None:
    try:
        action()  # type: ignore[operator]
    except CalibrationError as exc:
        print(f"{description}\n  refused: {type(exc).__name__}: {exc}\n")
    else:
        print(f"{description}\n  allowed\n")


judge = Judge(
    "refund-policy", model="bedrock/global.anthropic.claude-sonnet-4-6", prompt_version="v3"
)
labels = pairs(n_per_class=80, correct_per_class=74)

attempt(
    "certify on the few-shot pool",
    lambda: certify(
        judge, CalibrationSet(labels, split="few_shot", set_id="prompt-examples"), ledger=ledger
    ),
)

locked = CalibrationSet(labels, split="locked_test", set_id="locked-q4")
record = certify(judge, locked, ledger=ledger)
print(f"first certification on locked-q4: {record.result}\n")
attempt("certify again on locked-q4", lambda: certify(judge, locked, ledger=ledger))

attempt(
    "measure the rate on the hunted set",
    lambda: corrected_rate(
        record,
        [1] * 37 + [0] * 43,
        rate_set_id=record.hunted_set_id,
        rate_trace_ids=list(record.trace_ids),
    ),
)
