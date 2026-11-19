# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the same verdicts three ways and see the CI decision change.

A gate fails a build when more than 20% of traces fail. For each failure mode in an illustrative
scenario, the same verdicts are read as a bare number, as a number with its likely range, and the
way Calibrated Agent Evaluation reads them: with the judge's measured mistakes taken out, or not at
all when the judge isn't trusted. The arithmetic is the shipped estimators', through
``calibrated_agent_evaluation.stats.studies.baseline_arm``.

    uv run --extra stats python docs/site/examples/decision_flips.py
"""

from __future__ import annotations

import json
from pathlib import Path

from calibrated_agent_evaluation.stats.studies.baseline_arm import evaluate_mode

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO = REPO_ROOT / "fixtures" / "expected" / "run" / "baseline_arm_example.json"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    doc = json.loads(SCENARIO.read_text(encoding="utf-8"))
    limit = float(doc["max_fail_rate"])
    print(f"The gate fails a build when more than {pct(limit)} of traces fail.\n")
    for mode in doc["modes"]:
        row = evaluate_mode(mode, max_fail_rate=limit, seed=0, n_resamples=9999)
        print(f"{row['mode']}")
        print(f"  bare number:     {pct(row['raw_point'])}, so {row['naive_point']}")
        print(f"  with its range:  up to {pct(row['raw_hi'])}, so {row['naive_wilson']}")
        if row["calibrated"].startswith("refused"):
            print("  corrected rate:  no decision, because the judge isn't trusted yet")
        else:
            reach = f"up to {pct(row['corrected_hi'])} after removing the judge's mistakes"
            print(f"  corrected rate:  {reach}, so {row['calibrated']}")


if __name__ == "__main__":
    main()
