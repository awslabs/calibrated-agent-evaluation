# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The paired before/after guard -- McNemar + discordant adjudication.

Turns the paired confirmation-slice observations into the paired significance, and enforces the
human-gold boundary on the discordant pairs:

  * :func:`paired_mcnemar` -- builds the 2x2 table from :class:`PairedObs` and runs McNemar (reusing
    :func:`calibrated_agent_evaluation.stats.mcnemar_paired`: auto exact/mid-p under 25 discordant,
    asymptotic above), carrying the change in fail-rate (after - before) as an :class:`Interval`. A
    zero-discordant table has no paired information, so it is surfaced as an explicit
    ``method="undefined"`` result with ``p_value=None`` -- never a fabricated p-value.
  * :func:`discordant_pairs` -- the flipped (b + c) pairs McNemar counts.
  * :func:`discovery_enqueue` -- routes the discordant pairs into the labeling queue tagged
    ``frame="discovery"`` (the :class:`~calibrated_agent_evaluation.run.DiscoveryEnqueue` precedent)
    for human adjudication.
  * :func:`require_adjudication` -- when the labels are judge-produced, refuses McNemar until every
    discordant pair carries a human's trace-level verdict on the target metric. McNemar assumes
    ground-truth labels; a judge's own run-to-run flip-flops land in exactly the discordant cells
    it counts, so a judge-vs-judge before/after is refused, not certified.
  * :func:`adjudicate_pairs` -- puts each human verdict into its pair's after outcome, so the test
    runs on what the human saw. The improve stage writes no label; it reuses the
    ``kind="human"`` ratification mechanism.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace

from calibrated_agent_evaluation.calibrate.record import METHOD_NEWCOMBE_PAIRED, Interval
from calibrated_agent_evaluation.improve.errors import AdjudicationRequiredError, ImproveError
from calibrated_agent_evaluation.improve.report import McNemarResult, PairedObs
from calibrated_agent_evaluation.run import DiscoveryEnqueue
from calibrated_agent_evaluation.schema import Label
from calibrated_agent_evaluation.stats import mcnemar_paired
from calibrated_agent_evaluation.stats.proportions import newcombe_paired_diff


def _counts(pairs: Sequence[PairedObs]) -> tuple[int, int, int, int]:
    """Reduce paired observations to the 2x2 McNemar cells ``(a, b, c, d)``.

    ``a`` both fail, ``b`` before-fail/after-pass (an improvement), ``c`` before-pass/after-fail (a
    regression), ``d`` both pass. The discordant cells ``b`` and ``c`` are what McNemar tests.
    """
    repeated = sorted(t for t, n in Counter(p.trace_id for p in pairs).items() if n > 1)
    if repeated:
        raise ImproveError(
            f"paired_mcnemar: {len(repeated)} trace id(s) appear more than once in the pairs "
            f"({', '.join(repeated[:5])}); each input is paired once, and a repeated pair would "
            "count as another input and inflate the test. Remove the duplicates."
        )
    a = b = c = d = 0
    for p in pairs:
        if p.before_fail and p.after_fail:
            a += 1
        elif p.before_fail and not p.after_fail:
            b += 1
        elif not p.before_fail and p.after_fail:
            c += 1
        else:
            d += 1
    return a, b, c, d


def _paired_effect_interval(a: int, b: int, c: int, d: int) -> Interval:
    """95% Newcombe (1998, method 10) interval for the change in fail-rate (after - before).

    ``point = (c - b) / N`` (negative == fewer failures after == an improvement). The matched-pairs
    Wald interval this replaced has zero variance when a discordant cell is empty, so 5 improved
    and 0 regressed out of 5 read as ``[-1.00, -1.00]``; the hybrid-score interval gives
    ``[-1.00, -0.39]`` there and never leaves ``[-1, 1]``.
    """
    n = a + b + c + d
    # newcombe_paired_diff orients the difference as p1 - p2 = before - after; negate it.
    diff, lower, upper = newcombe_paired_diff(a, b, c, d)
    return Interval(
        point=0.0 - diff,
        lo=0.0 - upper,
        hi=0.0 - lower,
        method=METHOD_NEWCOMBE_PAIRED,
        level=0.95,
        sided="two-sided",
        n=n,
    )


def paired_mcnemar(pairs: Sequence[PairedObs]) -> McNemarResult:
    """Run the paired McNemar test over ``pairs``.

    Auto-selects the exact/mid-p variant when the discordant count is < 25 and the asymptotic one
    otherwise (via :func:`calibrated_agent_evaluation.stats.mcnemar_paired`). A **zero-discordant**
    slice is surfaced as ``McNemarResult(method="undefined", p_value=None, ...)`` with the effect
    still reported (a genuine 0.0 change) -- never a fabricated p-value. Raises
    :class:`ImproveError` on empty input or a trace id that appears in more than one pair.
    """
    if not pairs:
        raise ImproveError(
            "paired_mcnemar: no paired observations; refusing to run a before/after test over an "
            "empty confirmation slice"
        )
    a, b, c, d = _counts(pairs)
    effect = _paired_effect_interval(a, b, c, d)
    n_discordant = b + c
    if n_discordant == 0:
        # No flipped pairs: McNemar has no information. Report an honest undefined result rather
        # than fabricate a p-value; the effect is a real 0.0 change and is still carried.
        return McNemarResult(
            method="undefined",
            n_discordant=0,
            p_value=None,
            effect=effect,
            undefined_reason=(
                "zero discordant pairs (no before/after flips); McNemar's test is undefined"
            ),
        )
    # midp is meaningful only on the small-sample exact path; mcnemar_paired ignores it above 25.
    use_midp = n_discordant < 25
    stat = mcnemar_paired([[a, b], [c, d]], midp=use_midp)
    return McNemarResult(
        method=stat.method,
        n_discordant=stat.n_discordant,
        p_value=stat.p_value,
        effect=effect,
        undefined_reason=None,
    )


def discordant_pairs(pairs: Sequence[PairedObs]) -> tuple[PairedObs, ...]:
    """Return the discordant (flipped) pairs -- the ``b`` + ``c`` cells McNemar counts."""
    return tuple(p for p in pairs if p.before_fail != p.after_fail)


def discovery_enqueue(discordant: Sequence[PairedObs]) -> tuple[DiscoveryEnqueue, ...]:
    """Route the discordant pairs into the labeling queue tagged ``frame="discovery"``.

    Reuses :class:`~calibrated_agent_evaluation.run.DiscoveryEnqueue`: a discordant pair a judge
    disagreed with itself on is non-random augmentation (it grows the discovery set) and must never
    enter a production-frame rate (the ``assert_production_frame`` guard enforces that downstream).
    """
    return tuple(DiscoveryEnqueue(trace_id=p.trace_id) for p in discordant)


def require_adjudication(
    discordant: Sequence[PairedObs],
    adjudications: Mapping[str, Label],
    *,
    labels_are_judge_produced: bool,
    target_metric: str,
) -> None:
    """Refuse McNemar until every judge-labeled discordant pair has a usable human verdict.

    When ``labels_are_judge_produced`` is False the before/after labels are human gold and no
    adjudication is required (returns immediately). When True, every discordant ``trace_id`` must
    map to a :class:`Label` that is a human's trace-level verdict on that trace's after-run output
    for ``target_metric``: a FAIL for ``target_metric``, or a PASS for it or for no mode. Anything
    else raises :class:`AdjudicationRequiredError`, naming the traces: a missing entry, a non-human
    (unratified) label, a span-level label, a label for another trace, or one about another failure
    mode, which says nothing about whether the target was fixed.
    """
    if not labels_are_judge_produced:
        return
    problems: dict[str, list[str]] = {}
    for pair in discordant:
        reason = adjudication_problem(
            pair.trace_id, adjudications.get(pair.trace_id), target_metric
        )
        if reason is not None:
            problems.setdefault(reason, []).append(pair.trace_id)
    if problems:
        found = "; ".join(
            f"{reason} for {len(ids)} trace(s) ({', '.join(sorted(ids)[:5])})"
            for reason, ids in problems.items()
        )
        raise AdjudicationRequiredError(
            "require_adjudication: the before/after labels are judge-produced, so every discordant "
            "pair needs a human verdict on the after run before McNemar runs (a judge's run-to-run "
            f"flips land in exactly the discordant cells the test counts). Found {found}. Next: "
            f"route them to review (frame='discovery') and label each trace PASS or FAIL for "
            f"{target_metric!r} at trace level."
        )


def adjudicate_pairs(
    pairs: Sequence[PairedObs], adjudications: Mapping[str, Label], *, target_metric: str
) -> tuple[PairedObs, ...]:
    """Return ``pairs`` with each judge-labeled discordant pair's after outcome set by its human.

    The human's verdict replaces the judge's after-run verdict, so McNemar and the effect interval
    count what the human saw: a flip the human says did not happen becomes a concordant pair. Call
    :func:`require_adjudication` first; a pair this cannot adjudicate raises
    :class:`AdjudicationRequiredError`. Concordant pairs and human-gold pairs are kept as given.
    """
    out: list[PairedObs] = []
    for pair in pairs:
        if not pair.labels_are_judge_produced or pair.before_fail == pair.after_fail:
            out.append(pair)
            continue
        label = adjudications.get(pair.trace_id)
        reason = adjudication_problem(pair.trace_id, label, target_metric)
        if label is None or reason is not None:
            raise AdjudicationRequiredError(
                f"adjudicate_pairs: discordant pair {pair.trace_id!r} has no usable human verdict "
                f"({reason}); run require_adjudication first"
            )
        out.append(replace(pair, after_fail=label.outcome == "FAIL"))
    return tuple(out)


def adjudication_problem(trace_id: str, label: Label | None, target_metric: str) -> str | None:
    """Why ``label`` cannot adjudicate ``trace_id`` for ``target_metric``, or ``None`` if it can.

    A usable adjudication is a human's trace-level label on that trace that speaks to the target:
    a FAIL for ``target_metric``, or a PASS for it or for no mode.
    """
    if label is None:
        return "no human label"
    if label.kind != "human":
        return f"a {label.kind!r} label, not a human one"
    if label.trace_id != trace_id:
        return f"a label for another trace ({label.trace_id!r})"
    if label.span_id is not None:
        return "a span-level label, not a verdict on the whole trace"
    if label.outcome == "FAIL" and label.failure_mode == target_metric:
        return None
    if label.outcome == "PASS" and label.failure_mode in (None, target_metric):
        return None
    return f"a human {label.outcome} about another failure mode ({label.failure_mode!r})"
