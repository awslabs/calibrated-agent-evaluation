# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the imbalance-aware agreement display.

Under imbalance, raw agreement is hidden and Cohen's κ + confusion cells surface instead; a
balanced set surfaces raw agreement. Reuses the imbalance helpers (no reimplementation).
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.enforce import agreement_display


def test_imbalanced_set_shows_kappa_and_hides_raw_agreement() -> None:
    # 9 PASS + 1 FAIL: minority fraction 0.1 < 0.20 threshold -> hide raw agreement.
    gold = ["PASS"] * 9 + ["FAIL"]
    verdict = ["PASS"] * 9 + ["FAIL"]
    display = agreement_display(gold, verdict)
    assert display.scalar_name == "cohen_kappa"
    assert display.raw_agreement_hidden is True
    assert display.confusion_cells is not None


def test_balanced_set_shows_raw_agreement() -> None:
    # Negative/edge: a balanced set (minority fraction 0.5) surfaces raw agreement normally.
    gold = ["PASS", "FAIL", "PASS", "FAIL"]
    verdict = ["PASS", "FAIL", "FAIL", "FAIL"]
    display = agreement_display(gold, verdict)
    assert display.scalar_name == "raw_agreement"
    assert display.raw_agreement_hidden is False
    assert display.value is not None
    assert display.value.point == 0.75  # 3 of 4 agree
    # Wilson 95% for 3 of 4: (k + z^2/2 -+ z sqrt(k(n-k)/n + z^2/4)) / (n + z^2)
    # = [0.300642, 0.954413].
    assert display.value.lo == pytest.approx(0.300642, abs=1e-6)
    assert display.value.hi == pytest.approx(0.954413, abs=1e-6)
    assert display.value.method == "wilson"


def test_imbalanced_kappa_carries_its_interval() -> None:
    # 5 FAIL / 45 PASS gold: FAIL is the 10% minority, so raw agreement is hidden.
    gold = ["FAIL"] * 5 + ["PASS"] * 45
    verdict = ["FAIL"] * 4 + ["PASS"] + ["FAIL"] * 3 + ["PASS"] * 42
    display = agreement_display(gold, verdict)
    assert display.scalar_name == "cohen_kappa"
    assert display.value is not None
    assert display.value.method == "cohen_kappa_fce_ase_union_wilson_agreement"
    assert display.value.lo < (display.value.point or 0.0) < display.value.hi


def test_all_fail_gold_hides_raw_agreement_and_refuses_kappa() -> None:
    """The fully degenerate set is the most imbalanced, not exempt.

    A judge that answers FAIL to everything scores 100% raw agreement on an all-FAIL gold set (the
    Who&When shape). Raw agreement is hidden, and kappa is 0/0 there, so no number is shown.
    """
    gold = ["FAIL"] * 184
    display = agreement_display(gold, ["FAIL"] * 184)
    assert display.raw_agreement_hidden is True
    assert display.scalar_name == "cohen_kappa"
    assert display.value is None
    assert "0/0" in (display.undefined_reason or "")
    assert display.confusion_cells == {"tp": 184, "fp": 0, "tn": 0, "fn": 0}
