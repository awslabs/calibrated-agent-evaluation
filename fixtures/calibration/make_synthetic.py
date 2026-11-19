#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the deterministic known-prevalence calibration fixture for judge calibration.

The spec requires verifying the Rogan-Gladen correction "against a known-prevalence synthetic
fixture", because a real ~80-fails-per-class calibration set does not exist in the seeded corpus
(there are only a handful of real gold labels), and the whole point of the correction is that you
know the true prevalence a-priori only in a synthetic. This script writes that fixture:

  * a HUNTED, fail-heavy calibration set — a judge with TPR = TNR = 0.90 over 80 true-FAIL and 80
    true-PASS items (the balanced set whose lower bounds clear the 0.80 gate); and
  * a SEPARATE production-representative RATE sample — a true 6% failure prevalence scored by the
    same 90/90 judge (so its raw positive fraction reads ~0.148 and the corrected rate recovers
    ~0.06). Two samples, two jobs: the rate is never read off the hunted set.

Deterministic: the counts are exact (no RNG), so re-running writes byte-identical JSON. The stated
parameters (prevalence 0.06, TPR/TNR 0.90) are the ORACLE the acceptance checklist compares against.

Run:  uv run python fixtures/calibration/make_synthetic.py
"""

from __future__ import annotations

import json
from pathlib import Path

_OUT = Path(__file__).resolve().parent / "synthetic_9090.json"

# Stated parameters (the oracle). A judge that catches 90% of true fails (TPR) and correctly clears
# 90% of true passes (TNR), and a production stream whose true failure rate is 6%.
TPR = 0.90
TNR = 0.90
TRUE_PREVALENCE = 0.06
N_POS = 80  # hunted true-FAIL items (well past the 35/class gate minimum, so bounds clear)
N_NEG = 80  # hunted true-PASS items
RATE_SAMPLE_N = 1000  # a separate production-representative sample


def _verdicts(n_true: int, judge_positive_rate: float) -> list[int]:
    """An exact 1/0 verdict list of length ``n_true`` whose mean is ``judge_positive_rate``.

    1 = the judge said FAIL. For the true-FAIL class this rate is TPR; for the true-PASS class it is
    the false-positive rate (1 - TNR).
    """
    k = round(judge_positive_rate * n_true)
    return [1] * k + [0] * (n_true - k)


def build() -> dict[str, object]:
    """Build the fixture document (pure — no RNG, no clock, no network)."""
    # Hunted calibration set: verdicts on each class.
    cal_pos = _verdicts(N_POS, TPR)  # judge-FAIL on true-FAIL items → mean = TPR = 0.90
    cal_neg = _verdicts(N_NEG, 1.0 - TNR)  # judge-FAIL on true-PASS items → mean = 1-TNR = 0.10

    # Separate production-representative rate sample: a 6%-prevalence stream scored by the 90/90
    # judge. Its raw positive fraction ≈ 0.06*0.90 + 0.94*0.10 = 0.148 (the "reports ~15%" floor).
    n_true_fail = round(TRUE_PREVALENCE * RATE_SAMPLE_N)  # 60
    n_true_pass = RATE_SAMPLE_N - n_true_fail  # 940
    rate_verdicts = _verdicts(n_true_fail, TPR) + _verdicts(n_true_pass, 1.0 - TNR)

    return {
        "description": (
            # The committed synthetic_9090.json carries this text; changing it breaks byte-identity.
            "Synthetic known-prevalence calibration fixture. "
            "Judge TPR=TNR=0.90; true failure prevalence=0.06. Stated params are the oracle."
        ),
        "judge": {
            "judge_id": "synthetic-9090",
            "model": "synthetic/known-9090",
            "prompt_version": "v1",
        },
        "params": {"tpr": TPR, "tnr": TNR, "true_prevalence": TRUE_PREVALENCE},
        "hunted_set": {
            "set_id": "synthetic-hunted-9090",
            "cal_pos": cal_pos,  # verdicts on the true-FAIL class (mean = TPR)
            "cal_neg": cal_neg,  # verdicts on the true-PASS class (mean = 1-TNR)
        },
        "rate_sample": {
            "set_id": "synthetic-prod-6pct",
            "verdicts": rate_verdicts,  # 1/0, mean ≈ 0.148 raw
        },
    }


def main() -> int:
    doc = build()
    _OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    verdicts: list[int] = doc["rate_sample"]["verdicts"]  # type: ignore[index,assignment]
    raw = sum(verdicts) / len(verdicts)
    print(f"wrote {_OUT}")
    print(f"  hunted: {N_POS} pos + {N_NEG} neg at TPR=TNR={TPR}")
    print(f"  rate sample: {RATE_SAMPLE_N} items, prevalence {TRUE_PREVALENCE} -> raw ~= {raw:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
