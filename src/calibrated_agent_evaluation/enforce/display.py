# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Imbalance-aware agreement display default.

Under class imbalance, raw percent agreement flatters a do-nothing judge (a judge that always says
PASS "agrees" 94% of the time on a 6%-fail stream while catching nothing). The display hides raw
agreement and surfaces Cohen's κ with its confusion cells instead. This layer does not re-decide the
imbalance rule -- it reuses the shared ``should_hide_raw_agreement`` / ``minority_fraction``.

Either scalar carries its interval: κ its Fleiss-Cohen-Everitt 95% interval
(:func:`calibrated_agent_evaluation.stats.kappa.cohen_kappa_interval`), raw agreement its 95% Wilson
interval. When κ is 0/0 (gold and verdict both one constant label, e.g. an all-FAIL set and a judge
that always says FAIL) no number is shown: ``value`` is ``None``, ``undefined_reason`` says why, and
the confusion cells stay visible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from calibrated_agent_evaluation.calibrate.record import METHOD_WILSON, Interval
from calibrated_agent_evaluation.enforce.errors import EnforcementError
from calibrated_agent_evaluation.stats import should_hide_raw_agreement, wilson_lower, wilson_upper
from calibrated_agent_evaluation.stats.agreement import UndefinedAgreementError
from calibrated_agent_evaluation.stats.kappa import cohen_kappa_interval


@dataclass(frozen=True)
class AgreementDisplay:
    """What to surface for an agreement computation.

    Under imbalance: ``scalar_name == "cohen_kappa"`` with ``confusion_cells`` populated and
    ``raw_agreement_hidden`` true. Otherwise: ``scalar_name == "raw_agreement"`` with the raw
    fraction and no cells. ``value`` is the scalar with its 95% interval, or ``None`` with
    ``undefined_reason`` set when κ is 0/0.
    """

    scalar_name: str
    value: Interval | None
    raw_agreement_hidden: bool
    confusion_cells: dict[str, int] | None = None
    undefined_reason: str | None = None


def _raw_agreement(gold: Sequence[object], verdict: Sequence[object]) -> Interval:
    """Raw agreement over a non-empty pairing; refuses an empty one.

    `agreement_display` is a public surface, and this divided by `len(gold)` unguarded -- so an
    empty pairing raised a bare `ZeroDivisionError` from inside the enforcement plane instead of a
    typed, named error. There is also no honest number to return: 0.0 and 1.0 are both claims about
    agreement that was never measured.
    """
    if not gold:
        raise EnforcementError(
            "agreement_display: cannot compute agreement over an empty gold/verdict pairing "
            "(no items were compared); check the slice that produced it"
        )
    n = len(gold)
    k = sum(1 for g, v in zip(gold, verdict, strict=True) if g == v)
    return Interval(
        point=k / n,
        lo=wilson_lower(k, n, alpha=0.05),
        hi=wilson_upper(k, n, alpha=0.05),
        method=METHOD_WILSON,
        level=0.95,
        sided="two-sided",
        n=n,
    )


def _confusion_cells(gold: Sequence[object], verdict: Sequence[object]) -> dict[str, int]:
    """2x2 cells with FAIL as the positive class (mirrors calibrate.pairing vocabulary)."""
    cells = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for g, v in zip(gold, verdict, strict=True):
        gold_fail, verdict_fail = g == "FAIL", v == "FAIL"
        if gold_fail and verdict_fail:
            cells["tp"] += 1
        elif not gold_fail and verdict_fail:
            cells["fp"] += 1
        elif not gold_fail and not verdict_fail:
            cells["tn"] += 1
        else:
            cells["fn"] += 1
    return cells


def agreement_display(gold: Sequence[object], verdict: Sequence[object]) -> AgreementDisplay:
    """Select the agreement scalar to surface, hiding raw agreement under imbalance.

    When the gold labels are class-imbalanced (per ``should_hide_raw_agreement``), returns
    Cohen's κ with confusion cells and ``raw_agreement_hidden=True``; otherwise returns the raw
    percent-agreement scalar.
    """
    if should_hide_raw_agreement(gold):
        cells = _confusion_cells(gold, verdict)
        try:
            kappa = cohen_kappa_interval(gold, verdict)
        except UndefinedAgreementError as exc:
            return AgreementDisplay(
                scalar_name="cohen_kappa",
                value=None,
                raw_agreement_hidden=True,
                confusion_cells=cells,
                undefined_reason=str(exc),
            )
        return AgreementDisplay(
            scalar_name="cohen_kappa",
            value=kappa,
            raw_agreement_hidden=True,
            confusion_cells=cells,
        )
    return AgreementDisplay(
        scalar_name="raw_agreement",
        value=_raw_agreement(gold, verdict),
        raw_agreement_hidden=False,
    )
