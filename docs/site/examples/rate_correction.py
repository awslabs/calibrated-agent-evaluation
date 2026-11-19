# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A judge that is right 90% of the time still reports the wrong failure rate.

Uses the synthetic 90/90 fixture: the judge is right on 90% of failures and 90% of passes, and the
true failure rate of the 1,000-trace production sample is known to be 6%, which a real deployment
never knows. Certify the judge on its 80-per-class gold set, then correct the rate it reports. A
second sample shows the refusal when the measured rate sits below the judge's own false-positive
floor.

    uv run --extra stats python docs/site/examples/rate_correction.py [--explain | --panel]

``--explain`` adds the statistical detail; ``--panel`` prints the homepage value panel.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from calibrated_agent_evaluation.calibrate import (
    CalibrationRecord,
    CalibrationSet,
    InMemoryLockedTestLedger,
    Interval,
    LabelPair,
    certify,
    corrected_rate,
)
from calibrated_agent_evaluation.schema import Label, Outcome

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "calibration" / "synthetic_9090.json"


@dataclass(frozen=True)
class Judge:
    judge_id: str
    model: str
    prompt_version: str


def gold_pairs(verdicts: list[int], gold: Outcome) -> list[LabelPair]:
    """Pair each gold label with the judge's verdict on the same trace (1 means it said FAIL)."""
    return [
        LabelPair(
            gold=Label(
                trace_id=f"{gold}-{i}", span_id=None, kind="human", outcome=gold, critique="gold"
            ),
            verdict=Label(
                trace_id=f"{gold}-{i}",
                span_id=None,
                kind="llm-judge",
                outcome="FAIL" if v else "PASS",
                critique="v",
            ),
        )
        for i, v in enumerate(verdicts)
    ]


def certified_judge(doc: dict) -> CalibrationRecord:
    hunted = doc["hunted_set"]
    locked = CalibrationSet(
        tuple(gold_pairs(hunted["cal_pos"], "FAIL") + gold_pairs(hunted["cal_neg"], "PASS")),
        split="locked_test",
        set_id=hunted["set_id"],
    )
    return certify(Judge(**doc["judge"]), locked, ledger=InMemoryLockedTestLedger())


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def panel(flagged: float, rate: Interval, true_rate: float, n: int) -> str:
    likely = f"likely between {pct(rate.lo)} and {pct(rate.hi)}"
    return f"""<div class="value-panel" markdown>

<div class="value-panel__figure value-panel__figure--raw" markdown>
<span class="value-panel__label">The judge flagged</span>
<span class="value-panel__number">{pct(flagged)}</span>
<span class="value-panel__note">of {n:,} traces</span>
</div>

<div class="value-panel__figure value-panel__figure--honest" markdown>
<span class="value-panel__label">Calibrated Agent Evaluation reports</span>
<span class="value-panel__number">{pct(rate.require_point())}</span>
<span class="value-panel__note">{likely}</span>
</div>

<div class="value-panel__figure value-panel__figure--key" markdown>
<span class="value-panel__label">The true rate is</span>
<span class="value-panel__number">{pct(true_rate)}</span>
<span class="value-panel__note">known here because the sample is synthetic</span>
</div>

<div class="value-panel__why" markdown>
The judge is right on 90% of failures and 90% of passes. On a sample where few traces fail, the
10% of passes it gets wrong outnumber the real failures, so its raw rate is more than double the
truth. The toolkit measures the judge's mistakes against your labels and takes them back out.
</div>

</div>"""


def main(mode: str) -> None:
    doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    record = certified_judge(doc)
    verdicts = doc["rate_sample"]["verdicts"]
    flagged = sum(verdicts) / len(verdicts)
    rate = corrected_rate(
        record,
        verdicts,
        rate_set_id=doc["rate_sample"]["set_id"],
        rate_trace_ids=[f"prod-{i}" for i in range(len(verdicts))],
    ).corrected_rate
    if rate is None:
        raise SystemExit("corrected_rate returned no interval")
    true_rate = doc["params"]["true_prevalence"]
    if mode == "--panel":
        print(panel(flagged, rate, true_rate, len(verdicts)))
        return

    caught, cleared = pct(record.tpr.point), pct(record.tnr.point)
    print(
        f"Judge certified: it caught {caught} of the failures and cleared {cleared} of the passes."
    )
    print(f"It flagged {sum(verdicts)} of {len(verdicts):,} production traces: {pct(flagged)}.")
    likely = f"likely between {pct(rate.lo)} and {pct(rate.hi)}"
    print(f"Failure rate: {pct(rate.require_point())} ({likely}).")
    print(f"The true rate in this synthetic sample: {pct(true_rate)}.")

    low = [1] * 50 + [0] * 950
    low_rate = corrected_rate(
        record,
        low,
        rate_set_id="synthetic-prod-low",
        rate_trace_ids=[f"prod-low-{i}" for i in range(len(low))],
    ).corrected_rate
    if low_rate is None:
        raise SystemExit("corrected_rate returned no interval")
    print(f"\nOn another sample it flagged {sum(low)} of {len(low):,}: {pct(sum(low) / len(low))}.")
    print("That is fewer than the judge's false alarms alone would produce,")
    print("so there is no honest single number.")
    print(f"Failure rate: likely between {pct(low_rate.lo)} and {pct(low_rate.hi)}.")

    if mode == "--explain":
        print("\nDetail:")
        print(f"  TPR {record.tpr.point:.3f} (95% lower bound {record.tpr.lb95:.3f})")
        print(f"  TNR {record.tnr.point:.3f} (95% lower bound {record.tnr.lb95:.3f})")
        print(f"  Youden's J {record.youden_j:.3f}")
        print(f"  Rogan-Gladen corrected rate {rate.require_point():.4f},")
        print(f"    95% interval [{rate.lo:.4f}, {rate.hi:.4f}]")
        print(f"  second sample: point estimate refused ({low_rate.refusal_reason}),")
        print(f"    95% interval [{low_rate.lo:.4f}, {low_rate.hi:.4f}]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
