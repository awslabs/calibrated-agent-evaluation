# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dual-coding inter-rater agreement over human labels, and the gold-set agreement gate.

The labeling stage's IAA path (design settled in). Given the
stored labels, the raters whose agreement is being measured, and one ``codebook_version``, it
builds reliability data keyed on ``(trace_id, span_id)`` and ``rater_id`` and reports:

* Krippendorff's alpha, the headline, because it handles incomplete overlap natively;
* Cohen's kappa alongside when exactly two raters are named, for comparison with published
  kappas;
* per-class specific agreement, always, because pooled agreement on imbalanced failure labels
  hides a rare class the raters never agree on (Feinstein and Cicchetti 1990);

each with a unit-bootstrap interval, plus the gate verdict on alpha.

The agreement set is the *pre-adjudication* labels: only the named raters' labels enter, so an
adjudicator's resolving label (a third ``kind="human"`` row under its own ``rater_id``) never
lifts the number. Labels under any other codebook version are left out, because pooling across
versions measures agreement under a moving definition; ``None`` is its own version, the
unversioned bucket, never a wildcard. A rater who labeled one unit more than once (a relabel) is
represented by their newest label there, the same rule the run gate applies to verdicts, so the
labels must arrive oldest first, the order ``get_scores`` returns them. Pure: no I/O.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from calibrated_agent_evaluation.review.errors import AgreementDataError
from calibrated_agent_evaluation.review.gate import (
    ALPHA_LOWER_BOUND_GATE,
    ALPHA_REVISE_FLOOR,
    MINIMUM_OVERLAP,
    AgreementAxis,
    GateVerdict,
)
from calibrated_agent_evaluation.review.relabel import newest_per_rater_unit, unit_of
from calibrated_agent_evaluation.schema import Label
from calibrated_agent_evaluation.stats.agreement import (
    AgreementInterval,
    UndefinedAgreementError,
    Units,
    bootstrap_agreement,
    cohen_kappa,
    krippendorff_alpha,
    specific_agreement,
)

__all__ = [
    "ALPHA_LOWER_BOUND_GATE",
    "ALPHA_REVISE_FLOOR",
    "MINIMUM_OVERLAP",
    "NO_FAILURE_MODE",
    "AgreementAxis",
    "GateVerdict",
    "InterRaterReport",
    "alpha_gate",
    "inter_rater_agreement",
    "newest_per_rater_unit",
]

_Statistic: TypeAlias = Callable[[Units], float]

#: The ``failure_mode`` value for a label that names no failure mode (a PASS). It is a category,
#: not a gap: one rater naming a mode where the other names none is a disagreement.
NO_FAILURE_MODE = "(none)"


@dataclass(frozen=True)
class InterRaterReport:
    """Agreement among the named raters on one axis, within one codebook version.

    ``n_units`` counts units at least one named rater coded; ``n_pairable_units`` the ones two or
    more coded, which are the only units alpha uses. ``cohen_kappa`` is ``None`` unless exactly two
    raters are named. ``per_class`` has an entry for every category observed on a pairable unit.
    ``n_superseded`` counts the older labels a rater's relabel replaced. ``minimum_overlap`` is the
    pairable-unit count below which ``gate`` cannot be ``usable``.
    """

    axis: AgreementAxis
    codebook_version: str | None
    raters: tuple[str, ...]
    n_units: int
    n_pairable_units: int
    alpha: AgreementInterval
    cohen_kappa: AgreementInterval | None
    per_class: Mapping[str, AgreementInterval]
    gate: GateVerdict
    n_superseded: int = 0
    minimum_overlap: int = MINIMUM_OVERLAP


def alpha_gate(
    alpha: AgreementInterval, *, n_pairable_units: int, minimum_overlap: int = MINIMUM_OVERLAP
) -> GateVerdict:
    """Return the gold-set verdict for an alpha interval over ``n_pairable_units`` shared units.

    Below ``minimum_overlap`` units coded by two or more raters the interval cannot show the
    gold is usable (raters who agree on all 20 shared traces get a lower bound of 1.0), but it
    can show it is not:

    * ``codebook_broken``: alpha's upper bound is below :data:`ALPHA_REVISE_FLOOR`.
    * ``revise_codebook``: alpha's upper bound is below :data:`ALPHA_LOWER_BOUND_GATE`.
    * ``top_up_overlap``: otherwise; double-code more of the calibration gold.

    From ``minimum_overlap`` up:

    * ``usable``: the lower bound is >= :data:`ALPHA_LOWER_BOUND_GATE`. The bound gates, not the
      point estimate.
    * ``revise_codebook``: the point estimate is in [:data:`ALPHA_REVISE_FLOOR`, gate).
    * ``codebook_broken``: the point estimate is below :data:`ALPHA_REVISE_FLOOR`.
    * ``top_up_overlap``: the point estimate clears the gate but the lower bound does not. The
      interval is too wide to tell, which is a sample-size problem, not a codebook one.

    :data:`MINIMUM_OVERLAP` comes from
    :func:`calibrated_agent_evaluation.review.overlap.plan_overlap`.
    """
    if n_pairable_units < minimum_overlap:
        # More shared units narrow the interval but cannot lift a band its upper end is under.
        if alpha.upper < ALPHA_REVISE_FLOOR:
            return "codebook_broken"
        if alpha.upper < ALPHA_LOWER_BOUND_GATE:
            return "revise_codebook"
        return "top_up_overlap"
    if alpha.lower >= ALPHA_LOWER_BOUND_GATE:
        return "usable"
    if alpha.point >= ALPHA_LOWER_BOUND_GATE:
        return "top_up_overlap"
    if alpha.point >= ALPHA_REVISE_FLOOR:
        return "revise_codebook"
    return "codebook_broken"


def _value(label: Label, axis: AgreementAxis) -> str:
    if axis == "outcome":
        return label.outcome
    return label.failure_mode if label.failure_mode is not None else NO_FAILURE_MODE


def _reliability_data(
    labels: Sequence[Label],
    raters: tuple[str, ...],
    codebook_version: str | None,
    axis: AgreementAxis,
) -> tuple[list[list[Hashable | None]], int]:
    named = set(raters)
    in_set = [
        label
        for label in labels
        if label.rater_id in named and label.codebook_version == codebook_version
    ]
    for label in in_set:
        if label.kind != "human":
            raise AgreementDataError(
                f"inter_rater_agreement: rater {label.rater_id!r} has a kind={label.kind!r} label "
                f"on trace {label.trace_id!r}; inter-rater agreement is between human raters"
            )
    newest, superseded = newest_per_rater_unit(in_set)
    by_unit: dict[tuple[str, str], dict[str, str]] = {}
    for label in newest:
        by_unit.setdefault(unit_of(label), {})[label.rater_id or ""] = _value(label, axis)
    if not by_unit:
        raise AgreementDataError(
            f"inter_rater_agreement: no human label by {list(raters)} in "
            f"{_version_name(codebook_version)}"
        )
    return [[by_unit[u].get(r) for r in raters] for u in sorted(by_unit)], superseded


def _version_name(codebook_version: str | None) -> str:
    if codebook_version is None:
        return "the unversioned bucket (labels with no codebook version)"
    return f"codebook version {codebook_version!r}"


def _two_rater_kappa(units: Units) -> float:
    pairs = [(a, b) for a, b in units if a is not None and b is not None]
    if not pairs or len({v for pair in pairs for v in pair}) < 2:
        # cohen_kappa's own convention returns 1.0 here; the agreement set refuses, as alpha does.
        raise UndefinedAgreementError("cohen_kappa: undefined with no shared unit or one category")
    return cohen_kappa([a for a, _ in pairs], [b for _, b in pairs])


def _class_agreement(category: Hashable) -> _Statistic:
    def statistic(units: Units) -> float:
        per_class = specific_agreement(units)
        if category not in per_class:
            raise UndefinedAgreementError(f"specific_agreement: {category!r} not observed")
        return per_class[category]

    return statistic


def inter_rater_agreement(
    labels: Sequence[Label],
    *,
    raters: Collection[str],
    codebook_version: str | None,
    axis: AgreementAxis = "outcome",
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    minimum_overlap: int = MINIMUM_OVERLAP,
) -> InterRaterReport:
    """Return the named raters' agreement on ``axis`` within ``codebook_version``.

    Only ``kind="human"`` labels whose ``rater_id`` is in ``raters`` and whose ``codebook_version``
    matches enter (``None`` selects the unversioned labels); every other label (an adjudicator's,
    an unattributed one, another version's) is outside this agreement set by construction. A unit
    is a ``(trace_id, span_id)``; a rater who labeled a unit twice counts with the newest label, so
    ``labels`` must be oldest first. Every statistic is bootstrapped over the same seeded unit
    resamples. Below ``minimum_overlap`` pairable units the gate is ``top_up_overlap`` unless
    alpha's upper bound already rules the gold out (see :func:`alpha_gate`).

    Raises :class:`AgreementDataError` for fewer than two distinct raters, a non-human label under
    a named rater, or no matching label, and
    :class:`~calibrated_agent_evaluation.stats.agreement.UndefinedAgreementError` when the raters
    share no unit or only one category is observed.
    """
    ordered = tuple(dict.fromkeys(raters))
    if len(ordered) < 2:
        raise AgreementDataError(
            f"inter_rater_agreement: needs at least two distinct raters, got {list(ordered)}"
        )
    units, superseded = _reliability_data(labels, ordered, codebook_version, axis)

    def interval(statistic: _Statistic) -> AgreementInterval:
        return bootstrap_agreement(
            units, statistic, n_resamples=n_resamples, confidence=confidence, seed=seed
        )

    alpha = interval(krippendorff_alpha)
    kappa = interval(_two_rater_kappa) if len(ordered) == 2 else None
    per_class = {
        str(category): interval(_class_agreement(category))
        for category in sorted(specific_agreement(units), key=str)
    }
    n_pairable = sum(1 for unit in units if sum(v is not None for v in unit) >= 2)
    return InterRaterReport(
        axis=axis,
        codebook_version=codebook_version,
        raters=ordered,
        n_units=len(units),
        n_pairable_units=n_pairable,
        alpha=alpha,
        cohen_kappa=kappa,
        per_class=per_class,
        gate=alpha_gate(alpha, n_pairable_units=n_pairable, minimum_overlap=minimum_overlap),
        n_superseded=superseded,
        minimum_overlap=minimum_overlap,
    )
