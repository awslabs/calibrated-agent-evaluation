# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static lint of a ``JudgeDefinition`` under an ``EnforcementContext``.

The tiered enforcement point: hard refusals raise a typed error carrying their audit
decision; directional smells and display defaults are returned as :class:`EnforcementDecision`s the
caller records. One function, all the judge-definition rules:

  * hard: >1 non-binary (Likert) axis in one call -> ``MultiAxisLikertError``.
  * hard: a Likert axis while ``allow_likert`` is off -> ``LikertNotAllowedError``.
  * hard: ``critique`` missing or after ``verdict`` -> ``CritiqueOrderError``.
  * hard: a weighted-threshold collapse rule -> ``WeightedCollapseError``.
  * hard in CI: a generic generation-quality scorer while ``ctx.ci`` ->
    ``GenericMetricInCIError``; outside CI it is a ``display_default`` exploration-only decision.
  * display: a single Likert axis with ``allow_likert`` -> a ``display_default``
    warning.
  * warning: general multi-criteria (>1 binary axis) -> a ``warning`` + nudge, never
    a refusal.
"""

from __future__ import annotations

from collections.abc import Sequence

from calibrated_agent_evaluation.enforce.context import EnforcementContext
from calibrated_agent_evaluation.enforce.decision import EnforcementDecision
from calibrated_agent_evaluation.enforce.errors import (
    CritiqueOrderError,
    EnforcementError,
    GenericMetricInCIError,
    LikertNotAllowedError,
    MultiAxisLikertError,
    WeightedCollapseError,
)
from calibrated_agent_evaluation.enforce.judgedef import (
    JudgeDefinition,
    critique_before_verdict,
    likert_axes,
)

# Generic *generation-quality* metrics -- correlate poorly with human judgment on open-ended
# output, so they are exploration-only and banned in CI. Structured-target
# scorers (exact_match, pass@k, sql, f1, recall@k, mrr) are first-class CI gates and NOT listed.
GENERIC_GENERATION_METRICS: frozenset[str] = frozenset(
    {"rouge", "bertscore", "bleu", "meteor", "helpfulness"}
)


def _refuse(
    error_cls: type[EnforcementError], rule: str, inputs: dict[str, str], message: str
) -> None:
    """Build the audit decision, attach it to a typed refusal, and raise."""
    decision = EnforcementDecision(
        rule=rule, tier="hard_refusal", inputs=inputs, outcome="refused", message=message
    )
    raise error_cls(message, decision=decision)


def lint_judge_definition(
    defn: JudgeDefinition, ctx: EnforcementContext
) -> list[EnforcementDecision]:
    """Lint ``defn`` under ``ctx``; return warning/display decisions or raise on a hard refusal.

    Hard refusals (multi-Likert, Likert-not-allowed, critique-order, weighted-collapse,
    generic-metric-in-CI) raise a typed error carrying the recorded decision. Non-fatal outcomes
    (Likert-allowed, exploration-only, multi-criteria nudge) are returned as
    :class:`EnforcementDecision`s in evaluation order.
    """
    decisions: list[EnforcementDecision] = []

    # --- Likert axes: multiple axes are always invalid; a single axis is opt-in ---------------
    likert = likert_axes(defn)
    if len(likert) > 1:
        _refuse(
            MultiAxisLikertError,
            "ENF-2",
            {"judge_id": defn.judge_id, "likert_axes": ",".join(ax.name for ax in likert)},
            f"judge {defn.judge_id!r} scores {len(likert)} non-binary (Likert) axes in one call "
            f"({', '.join(ax.name for ax in likert)}); this is refused at lint time. "
            "Decompose into N single-axis binary judges or a single-call binary checklist.",
        )
    if len(likert) == 1 and not ctx.allow_likert:
        _refuse(
            LikertNotAllowedError,
            "ENF-6",
            {"judge_id": defn.judge_id, "likert_axis": likert[0].name},
            f"judge {defn.judge_id!r} scores {likert[0].name!r} on a Likert (non-binary) scale; "
            "Likert is refused unless the EnforcementContext sets allow_likert. Rewrite the axis "
            "as one binary PASS/FAIL question with a critique.",
        )

    # --- Critique-first --------------------------------------------------------
    if not critique_before_verdict(defn):
        _refuse(
            CritiqueOrderError,
            "PRIN-4",
            {"judge_id": defn.judge_id, "output_fields": ",".join(defn.output_fields)},
            f"judge {defn.judge_id!r} output schema must place a non-empty 'critique' before "
            f"'verdict'; got fields {list(defn.output_fields)} (critique-first).",
        )

    # --- Checklist weighted-threshold collapse is refused -----------------------------
    if defn.weighted_collapse:
        _refuse(
            WeightedCollapseError,
            "ENF-4",
            {"judge_id": defn.judge_id, "weighted_collapse": "true"},
            f"judge {defn.judge_id!r} declares a weighted-threshold collapse rule; only "
            "critical/non-critical tagging is offered -- tunable weights reintroduce vibe-scoring. "
            "Tag each item critical or non-critical instead.",
        )

    # --- Generic generation-quality scorer: banned in CI, exploration-only otherwise --
    if defn.scorer is not None and defn.scorer.lower() in GENERIC_GENERATION_METRICS:
        if ctx.ci:
            _refuse(
                GenericMetricInCIError,
                "ENF-5",
                {"judge_id": defn.judge_id, "scorer": defn.scorer, "ci": "true"},
                f"judge {defn.judge_id!r} uses the generic generation-quality metric "
                f"{defn.scorer!r} as a CI gate; generic metrics are banned in --ci and labeled "
                "exploration-only. Use a domain-specific or structured-target scorer.",
            )
        decisions.append(
            EnforcementDecision(
                rule="ENF-5",
                tier="display_default",
                inputs={"judge_id": defn.judge_id, "scorer": defn.scorer, "ci": "false"},
                outcome="shaped",
                message=(
                    f"scorer {defn.scorer!r} is a generic generation-quality metric: "
                    "exploration-only, not a CI gate."
                ),
            )
        )

    # --- Single Likert axis, allowed with a warning (display default) --------------------
    if len(likert) == 1 and ctx.allow_likert:
        decisions.append(
            EnforcementDecision(
                rule="ENF-6",
                tier="display_default",
                inputs={"judge_id": defn.judge_id, "likert_axis": likert[0].name},
                outcome="warned",
                message=(
                    f"judge {defn.judge_id!r} uses a Likert axis {likert[0].name!r} behind "
                    "allow_likert; 1-5 scoring is noisy -- prefer binary."
                ),
                nudge="Consider a binary axis or a single-call binary checklist instead.",
            )
        )

    # --- General multi-criteria (>1 binary axis): warn + nudge, never refuse ----------
    binary_axes = tuple(ax for ax in defn.axes if ax.scale == "binary")
    if len(binary_axes) > 1:
        decisions.append(
            EnforcementDecision(
                rule="ENF-3",
                tier="warning",
                inputs={
                    "judge_id": defn.judge_id,
                    "binary_axes": ",".join(ax.name for ax in binary_axes),
                },
                outcome="warned",
                message=(
                    f"judge {defn.judge_id!r} scores {len(binary_axes)} criteria in one call; "
                    "general multi-criteria judging is allowed but directional."
                ),
                nudge=(
                    "Prefer N single-axis judges or a single-call binary checklist "
                    "(each item binary, tagged critical/non-critical)."
                ),
            )
        )

    return decisions


def lint_scorer_lanes(scorers: Sequence[str], ctx: EnforcementContext) -> list[EnforcementDecision]:
    """Refuse, in CI, a gate over verdicts that a generic generation-quality metric produced.

    ``scorers`` names the scorers whose verdicts a run would gate (the scorer name in each
    recorded lane). In CI a generic metric among them raises :class:`GenericMetricInCIError`
    carrying the decision; outside CI each one is returned as an exploration-only
    ``display_default`` decision. Structured-target scorers pass with no decision.
    """
    generic = sorted({s for s in scorers if s.lower() in GENERIC_GENERATION_METRICS})
    if not generic:
        return []
    names = ", ".join(repr(s) for s in generic)
    if ctx.ci:
        _refuse(
            GenericMetricInCIError,
            "ENF-5",
            {"scorers": ",".join(generic), "ci": "true"},
            f"the scorer verdicts include ones from {names}, a generic generation-quality "
            "metric; such metrics track human judgment poorly on open-ended output, so they "
            "cannot gate CI. Gate on a structured-target scorer (exact match, a JSON or numeric "
            "comparison) or a calibrated judge, and keep the generic metric for exploration.",
        )
    return [
        EnforcementDecision(
            rule="ENF-5",
            tier="display_default",
            inputs={"scorer": name, "ci": "false"},
            outcome="shaped",
            message=(
                f"scorer {name!r} is a generic generation-quality metric: exploration-only, it "
                "cannot gate CI."
            ),
        )
        for name in generic
    ]
