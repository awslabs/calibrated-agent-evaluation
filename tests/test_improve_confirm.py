# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the confirmation-slice guard.

Behavioural: a held-out slice disjoint from the development set passes and pairs on the same inputs;
a slice that overlaps the dev set, or an empty slice, or a before/after set covering different
inputs, is refused with a typed error.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.improve import (
    ConfirmationContaminationError,
    ImproveError,
    assert_held_out,
    pair_observations,
)


def test_disjoint_slice_pairs_on_same_inputs() -> None:
    """A held-out slice disjoint from dev passes and pairs one PairedObs per input."""
    assert_held_out(["c1", "c2", "c3"], ["d1", "d2"])  # no raise
    pairs = pair_observations(
        {"c1": True, "c2": False, "c3": True}, {"c1": False, "c2": False, "c3": True}
    )
    assert [p.trace_id for p in pairs] == ["c1", "c2", "c3"]
    assert (pairs[0].before_fail, pairs[0].after_fail) == (True, False)


def test_overlap_with_development_refused() -> None:
    """A confirmation slice sharing a trace with the dev set is contamination -> refused."""
    with pytest.raises(ConfirmationContaminationError):
        assert_held_out(["c1", "c2", "shared"], ["d1", "shared"])


def test_empty_confirmation_slice_refused() -> None:
    """Edge: an empty confirmation slice has nothing to measure -> typed refusal."""
    with pytest.raises(ImproveError):
        assert_held_out([], ["d1"])


def test_mismatched_before_after_inputs_refused() -> None:
    """Before/after covering different inputs breaks pairing -> refused (not dropped)."""
    with pytest.raises(ImproveError):
        pair_observations({"c1": True, "c2": False}, {"c1": False, "c3": True})


def test_empty_before_or_after_refused() -> None:
    """An empty before or after label set cannot form a paired slice."""
    with pytest.raises(ImproveError):
        pair_observations({}, {"c1": True})
    with pytest.raises(ImproveError):
        pair_observations({"c1": True}, {})
