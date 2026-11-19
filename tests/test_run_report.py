# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the multiplicity-family reporting on the assembled RunRecord.

Behavioural: a run that corrects two exploratory flag families (BY-FDR) records both
MultiplicityResults, each stating its method and comparison count m.
"""

from __future__ import annotations

from _helpers import TEST_STAMP
from calibrated_agent_evaluation.run.flags import flag_slices
from calibrated_agent_evaluation.run.multi import apply_multiplicity
from calibrated_agent_evaluation.run.report import RunRecord


def test_run_record_states_method_and_m_per_family() -> None:
    """The RunRecord carries every corrected family with its method + m stated per run."""
    slice_family = apply_multiplicity([0.01, 0.5, 0.7], family="slice_flags", method="fdr_by")
    flag_family = flag_slices([0.001, 0.2, 0.4, 0.9], family="drift_flags")

    record = RunRecord(
        provenance=TEST_STAMP,
        run_id="run-mult",
        mode="cadence",
        gate_passed=False,
        judge_ids=("j1",),
        calib_ids=("c1",),
        scorer_lanes=(),
        trend=(),
        input_drift=None,
        judge_drift=(),
        multiplicity=(slice_family, flag_family),
        decisions=(),
        inputs={"run_id": "run-mult"},
    )

    by_family = {m.family: m for m in record.multiplicity}
    assert by_family["slice_flags"].method == "fdr_by"
    assert by_family["slice_flags"].m == 3
    assert by_family["drift_flags"].method == "fdr_by"
    assert by_family["drift_flags"].m == 4
