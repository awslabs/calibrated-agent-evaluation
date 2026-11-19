# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-axis inter-coder attribution agreement.

A high outcome-axis κ with a low culprit-axis κ is the responsibility-boundary signal, not noise.
Every test embeds its FR and asserts the computed agreement (or the capability-absent None).
"""

from __future__ import annotations

from typing import Literal

import pytest

from _helpers import make_label
from calibrated_agent_evaluation.attribution import attribution_agreement

# --- agree on outcome, disagree on culprit → high outcome κ, low culprit κ ---


def _outcome(i: int) -> Literal["PASS", "FAIL"]:
    return "FAIL" if i < 7 else "PASS"


def test_culprit_disagreement_is_the_signal() -> None:
    # Both coders agree on every outcome (7 FAIL, 3 PASS) but split on the culprit — the
    # CLM-4477-style undefined responsibility boundary (intake vs policy-checker).
    coder_a = [
        make_label(trace_id=f"CLM-{i}", outcome=_outcome(i), culprit_span=f"CLM-{i}/intake")
        for i in range(10)
    ]
    coder_b = [
        make_label(
            trace_id=f"CLM-{i}",
            outcome=_outcome(i),
            culprit_span=f"CLM-{i}/intake" if i < 3 else f"CLM-{i}/policy-checker",
        )
        for i in range(10)
    ]
    report = attribution_agreement(coder_a, coder_b)
    assert report is not None
    assert report.n_dual_coded == 10
    assert report.outcome_kappa is not None
    assert report.culprit_kappa is not None
    assert report.outcome_kappa.point == 1.0  # perfect outcome agreement
    # The boundary signal: low culprit agreement, with an interval that excludes the outcome's.
    assert (report.culprit_kappa.point or 0.0) < (report.outcome_kappa.point or 0.0)
    assert report.culprit_kappa.hi < 1.0


def test_constant_axis_kappa_is_undefined_not_one() -> None:
    """Both coders mark every trace FAIL: outcome kappa is 0/0, reported as such, not 1.0."""
    coder_a = [make_label(trace_id=f"CLM-{i}", outcome="FAIL") for i in range(10)]
    coder_b = [make_label(trace_id=f"CLM-{i}", outcome="FAIL") for i in range(10)]
    report = attribution_agreement(coder_a, coder_b)
    assert report is not None
    assert report.outcome_kappa is None
    assert "0/0" in (report.outcome_undefined_reason or "")


# --- (capability-absent) no second coder overlap → None, no fabricated agreement ---


def test_no_dual_coded_overlap_returns_none() -> None:
    coder_a = [make_label(trace_id="CLM-1", outcome="FAIL")]
    coder_b = [make_label(trace_id="CLM-9", outcome="FAIL")]  # disjoint traces
    assert attribution_agreement(coder_a, coder_b) is None
    # And an entirely empty second coder is likewise capability-absent, not agreement=1.0.
    assert attribution_agreement(coder_a, []) is None


def _coded(i: int, outcome: str, culprit: str | None) -> object:
    return make_label(trace_id=f"t{i}", outcome=outcome, culprit_span=culprit)


def test_culprit_kappa_is_read_over_co_fail_traces_only() -> None:
    """80 shared PASS traces used to lift random culprits on 20 FAILs to culprit kappa 0.54."""
    import random

    rnd = random.Random(0)
    coder_a = [_coded(i, "PASS", None) for i in range(80)]
    coder_b = [_coded(i, "PASS", None) for i in range(80)]
    for i in range(80, 100):
        coder_a.append(_coded(i, "FAIL", rnd.choice("abcd")))
        coder_b.append(_coded(i, "FAIL", rnd.choice("abcd")))
    report = attribution_agreement(coder_a, coder_b)  # type: ignore[arg-type]
    assert report is not None
    assert report.n_dual_coded == 100
    assert report.n_culprit_coded == 20
    assert report.culprit_kappa is not None
    assert (report.culprit_kappa.point or 0.0) < 0.2
    assert report.culprit_kappa.n == 20


def test_no_co_fail_trace_leaves_the_culprit_axis_undefined() -> None:
    coder_a = [_coded(0, "FAIL", "a"), _coded(1, "PASS", None)]
    coder_b = [_coded(0, "PASS", None), _coded(1, "PASS", None)]
    report = attribution_agreement(coder_a, coder_b)  # type: ignore[arg-type]
    assert report is not None
    assert report.culprit_kappa is None
    assert report.n_culprit_coded == 0
    assert "both coders" in (report.culprit_undefined_reason or "")


def test_span_level_labels_are_refused_with_a_typed_error() -> None:
    """A trace keyed both at trace and span level used to raise a raw TypeError from sorting."""
    from calibrated_agent_evaluation.attribution.errors import AttributionError

    mixed = [
        make_label(trace_id="t", span_id=None, outcome="PASS"),
        make_label(trace_id="t", span_id="s", outcome="PASS"),
    ]
    with pytest.raises(AttributionError, match="span-level"):
        attribution_agreement(mixed, mixed)


def test_a_coder_labeling_a_trace_twice_is_refused() -> None:
    from calibrated_agent_evaluation.attribution.errors import AttributionError

    twice = [make_label(trace_id="t", outcome="PASS"), make_label(trace_id="t", outcome="FAIL")]
    with pytest.raises(AttributionError, match="more than once"):
        attribution_agreement(twice, [make_label(trace_id="t", outcome="PASS")])
