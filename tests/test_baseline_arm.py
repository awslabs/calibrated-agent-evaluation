# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The three-arm baseline: naive, naive+Wilson, calibrated.

Runs the study's functions over the committed example input, so the arms and the flip count are
pinned to the shipped estimators rather than to numbers typed into a test.
"""

from __future__ import annotations

import json
from pathlib import Path

from calibrated_agent_evaluation.stats.studies.baseline_arm import evaluate_mode, flips

_REPO = Path(__file__).resolve().parent.parent
EXAMPLE = _REPO / "fixtures" / "expected" / "run" / "baseline_arm_example.json"


def test_each_arm_reads_the_same_sample_differently() -> None:
    doc = json.loads(EXAMPLE.read_text())
    strong, weak = (
        evaluate_mode(m, max_fail_rate=doc["max_fail_rate"], seed=0, n_resamples=999)
        for m in doc["modes"]
    )
    # 8/50 = 0.16 is under the 0.20 ceiling as a bare number, but its Wilson upper edge is not.
    assert strong["naive_point"] == "pass"
    assert strong["naive_wilson"] == "fail"
    assert strong["corrected_point"] < strong["raw_point"]
    # A 70/70 judge does not certify, so the calibrated arm refuses instead of deciding.
    assert weak["calibrated"].startswith("refused")


def test_flips_count_disagreements_and_a_refusal_never_counts_as_a_pass() -> None:
    rows = [
        {"naive_point": "pass", "naive_wilson": "fail", "calibrated": "pass"},
        {"naive_point": "pass", "naive_wilson": "pass", "calibrated": "refused: x"},
    ]
    assert flips(rows) == {
        "naive_point_vs_naive_wilson": 1,
        "naive_point_vs_calibrated": 1,
        "naive_wilson_vs_calibrated": 2,
    }
