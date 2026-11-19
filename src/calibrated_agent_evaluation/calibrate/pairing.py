# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pair (human-gold, judge-verdict) observations into 2x2 confusion counts.

The calibration gate is computed from a confusion matrix, so the first job is to turn a set of
:class:`~calibrated_agent_evaluation.calibrate.record.LabelPair` observations into
``(tp, fn, tn, fp)`` counts -- honestly:

  * **FAIL is the positive class.** The failure mode we hunt is the positive; a true
    positive is a real FAIL the judge also called FAIL. So ``tpr = tp/(tp+fn)`` measures "of the
    real failures, how many did the judge catch".
  * **Human gold only.** Every gold label must have ``kind == "human"`` -- an
    LLM-produced label in the gold set means certifying a judge against a judge. Refused.
  * **Exact pairing.** Each ``LabelPair``'s gold and verdict must reference the SAME
    ``(trace_id, span_id)``; a mismatch is a caller bug surfaced as a typed error, never dropped.
  * **One gold label per target.** Two raters' labels on one target are one observation; a
    repeated ``(trace_id, span_id)`` is refused, so double-coded gold must be adjudicated first.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from calibrated_agent_evaluation.calibrate.errors import (
    DuplicateGoldError,
    UnpairedObservationError,
    UnratifiedGoldError,
)
from calibrated_agent_evaluation.calibrate.record import LabelPair


@dataclass(frozen=True)
class ConfusionCounts:
    """2x2 confusion counts with FAIL as the positive class.

    ``tp`` -- gold FAIL, verdict FAIL (caught failure).      ``fn`` -- gold FAIL, verdict PASS
    (missed).
    ``tn`` -- gold PASS, verdict PASS (correct clear).       ``fp`` -- gold PASS, verdict FAIL
    (false alarm).
    """

    tp: int
    fn: int
    tn: int
    fp: int

    @property
    def n_pos(self) -> int:
        """Number of real positives (gold FAIL) -- the denominator of TPR."""
        return self.tp + self.fn

    @property
    def n_neg(self) -> int:
        """Number of real negatives (gold PASS) -- the denominator of TNR."""
        return self.tn + self.fp


_SHOWN_DUPLICATES = 5


def refuse_duplicate_targets(targets: Iterable[tuple[str, str | None]]) -> None:
    """Refuse a gold set in which any ``(trace_id, span_id)`` target appears more than once.

    Raises:
        DuplicateGoldError: A target appears more than once; the error names the repeats.
    """
    counts = Counter(targets)
    repeated = tuple(
        sorted((k for k, n in counts.items() if n > 1), key=lambda k: (k[0], k[1] or ""))
    )
    if not repeated:
        return
    shown = ", ".join(
        f"{trace_id}"
        + (f"/{span_id}" if span_id is not None else "")
        + f" (x{counts[(trace_id, span_id)]})"
        for trace_id, span_id in repeated[:_SHOWN_DUPLICATES]
    )
    more = len(repeated) - _SHOWN_DUPLICATES
    raise DuplicateGoldError(
        f"{len(repeated)} target(s) carry more than one gold label: {shown}"
        + (f" and {more} more" if more > 0 else "")
        + ". Each extra label would count as an independent observation and inflate the sample "
        "behind the bounds. Keep one adjudicated label per target (for double-coded gold, resolve "
        "the raters' disagreements first) and calibrate again.",
        duplicates=repeated,
    )


def pair_observations(pairs: Sequence[LabelPair]) -> ConfusionCounts:
    """Reduce paired observations to 2x2 confusion counts.

    Raises :class:`UnratifiedGoldError` if any gold label is not ``kind == "human"``,
    :class:`UnpairedObservationError` if a pair's gold and verdict do not reference the same
    ``(trace_id, span_id)``, and :class:`DuplicateGoldError` if a target appears in more than one
    pair. All are refusals, never silent drops.

    Args:
        pairs: The (judge verdict, gold label) pairs.

    Returns:
        The 2x2 confusion counts.

    Raises:
        UnratifiedGoldError: A gold label is not human.
        UnpairedObservationError: A verdict and its gold label name different targets.
        DuplicateGoldError: A target appears in more than one pair.
    """
    refuse_duplicate_targets((p.gold.trace_id, p.gold.span_id) for p in pairs)
    tp = fn = tn = fp = 0
    for pair in pairs:
        gold, verdict = pair.gold, pair.verdict
        if gold.kind != "human":
            raise UnratifiedGoldError(
                f"gold label for {gold.trace_id} has kind={gold.kind!r}, not 'human'; "
                "calibrating from a non-human gold set certifies a judge against a judge "
                ""
            )
        if gold.trace_id != verdict.trace_id or gold.span_id != verdict.span_id:
            raise UnpairedObservationError(
                f"observation mispaired: gold {(gold.trace_id, gold.span_id)} != "
                f"verdict {(verdict.trace_id, verdict.span_id)}"
            )
        gold_fail = gold.outcome == "FAIL"
        verdict_fail = verdict.outcome == "FAIL"
        if gold_fail and verdict_fail:
            tp += 1
        elif gold_fail and not verdict_fail:
            fn += 1
        elif not gold_fail and not verdict_fail:
            tn += 1
        else:
            fp += 1
    return ConfusionCounts(tp=tp, fn=fn, tn=tn, fp=fp)
