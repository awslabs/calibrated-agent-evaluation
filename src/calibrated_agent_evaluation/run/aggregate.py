# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate flat per-trace verdict Labels into per-failure-mode counts.

The run stage's first job: reduce the flat ``Label`` collections that ``score_traces`` and
``run_judge`` emit -- neither produces any dataset-level aggregate -- into per-failure-mode
:class:`~calibrated_agent_evaluation.run.report.ModeCounts`. A single-agent mode is keyed by
``failure_type`` alone; a crew mode is keyed per ``(origin agent, failure type)`` so each agent's
rate is corrected by its own judge downstream.

The mode a verdict belongs to comes from ``gold``: either a map (the human-ratified ``(trace_id,
span_id) -> ModeKey`` assignment produced upstream in the pipeline) or a function of the label
itself. ``FAIL`` is the positive class throughout
(the same convention calibration pairs on), so ``n_judged_fail`` counts the verdicts the
judge/scorer marked ``FAIL``. Pure computation, no I/O, no vendor type.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from calibrated_agent_evaluation.run.report import ModeCounts, ModeKey
from calibrated_agent_evaluation.schema.types import Label

#: Where a verdict's failure mode comes from: a ``(trace_id, span_id) -> ModeKey`` map, or a
#: function that reads it off the label and returns ``None`` for a label no mode measures.
ModeSource = Mapping[tuple[str, str | None], ModeKey] | Callable[[Label], ModeKey | None]


def aggregate_by_mode(
    labels: Sequence[Label],
    gold: ModeSource,
    *,
    calib_by_mode: Mapping[ModeKey, str] | None = None,
) -> dict[ModeKey, ModeCounts]:
    """Reduce verdict/scorer ``labels`` to per-:class:`ModeKey` :class:`ModeCounts`.

    ``gold`` maps each ``(trace_id, span_id)`` to the failure mode it belongs to (the human-ratified
    assignment), or is a function giving each label's mode. A map holds one mode per target, so
    when several judges for different modes score one trace, pass the function: a map would put
    every verdict on the target under whichever mode it holds. A label with no mode (absent from
    the map, or ``None`` from the function) is skipped -- never silently folded into a wrong mode.
    ``FAIL`` is the positive class, so ``n_judged_fail`` is the count of ``outcome == "FAIL"``
    verdicts per mode.

    ``calib_by_mode`` optionally names the ``CalibrationRecord`` id that corrects each mode (its own
    judge); recorded on the counts so the trend stage can look up the right record.
    Returns an empty dict for an empty label set (the gate turns that into a refusal).
    """
    calib_by_mode = calib_by_mode or {}
    totals: dict[ModeKey, int] = {}
    fails: dict[ModeKey, int] = {}
    mode_of = gold if callable(gold) else _from_map(gold)
    for label in labels:
        mode = mode_of(label)
        if mode is None:
            continue
        totals[mode] = totals.get(mode, 0) + 1
        if label.outcome == "FAIL":
            fails[mode] = fails.get(mode, 0) + 1

    return {
        mode: ModeCounts(
            mode=mode,
            n_total=n_total,
            n_judged_fail=fails.get(mode, 0),
            calib_id=calib_by_mode.get(mode),
        )
        for mode, n_total in totals.items()
    }


def _from_map(gold: Mapping[tuple[str, str | None], ModeKey]) -> Callable[[Label], ModeKey | None]:
    return lambda label: gold.get((label.trace_id, label.span_id))
