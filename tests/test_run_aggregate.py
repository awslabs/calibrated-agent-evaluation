# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-mode aggregator.

Behavioural: a flat verdict Label collection reduces to correct per-mode counts; crew labels key per
(agent, failure_type); an empty label set yields no modes (the gate turns that into a refusal).
"""

from __future__ import annotations

from _helpers import make_label
from calibrated_agent_evaluation.run.aggregate import aggregate_by_mode
from calibrated_agent_evaluation.run.report import ModeKey
from calibrated_agent_evaluation.schema.types import Label

GoldMap = dict[tuple[str, str | None], ModeKey]


def _lbl(trace_id: str, outcome: str, span_id: str | None = None) -> Label:
    return make_label(trace_id=trace_id, span_id=span_id, kind="llm-judge", outcome=outcome)


def test_reduces_to_per_mode_counts() -> None:
    """Labels reduce to per-mode totals + FAIL counts (FAIL is the positive class)."""
    mode = ModeKey(failure_type="over_refund")
    labels = [_lbl("t1", "PASS"), _lbl("t2", "FAIL"), _lbl("t3", "PASS")]
    gold: GoldMap = {("t1", None): mode, ("t2", None): mode, ("t3", None): mode}

    counts = aggregate_by_mode(labels, gold)

    assert set(counts) == {mode}
    assert counts[mode].n_total == 3
    assert counts[mode].n_judged_fail == 1


def test_crew_labels_key_per_agent_and_failure_type() -> None:
    """A crew's modes are keyed per (origin agent, failure type), corrected separately."""
    intake = ModeKey(failure_type="dropped_fact", agent="intake")
    payout = ModeKey(failure_type="misread_amount", agent="payout")
    labels = [
        _lbl("c1", "FAIL", span_id="c1/intake"),
        _lbl("c2", "PASS", span_id="c2/intake"),
        _lbl("c3", "FAIL", span_id="c3/payout"),
    ]
    gold: GoldMap = {
        ("c1", "c1/intake"): intake,
        ("c2", "c2/intake"): intake,
        ("c3", "c3/payout"): payout,
    }

    counts = aggregate_by_mode(labels, gold)

    assert set(counts) == {intake, payout}
    assert counts[intake].n_total == 2
    assert counts[intake].n_judged_fail == 1
    assert counts[payout].n_judged_fail == 1


def test_records_calib_id_per_mode() -> None:
    """Each mode carries the calibration-record id that corrects it."""
    mode = ModeKey(failure_type="over_refund")
    labels = [_lbl("t1", "PASS")]
    gold: GoldMap = {("t1", None): mode}

    counts = aggregate_by_mode(labels, gold, calib_by_mode={mode: "calib-42"})

    assert counts[mode].calib_id == "calib-42"


def test_empty_label_set_yields_no_modes() -> None:
    """An empty label set produces no modes (the gate refuses on that)."""
    assert aggregate_by_mode([], {}) == {}


def test_label_absent_from_gold_is_skipped_not_misfiled() -> None:
    """A label with no gold mode is skipped, never folded into a wrong mode."""
    mode = ModeKey(failure_type="over_refund")
    labels = [_lbl("t1", "FAIL"), _lbl("unmapped", "FAIL")]
    gold: GoldMap = {("t1", None): mode}

    counts = aggregate_by_mode(labels, gold)

    assert set(counts) == {mode}
    assert counts[mode].n_total == 1
