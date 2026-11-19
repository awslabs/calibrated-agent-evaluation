# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired before/after verdicts for ``improve --evaluate``, read from two stores.

A fix is measured on the same inputs run through the agent before it and after it. The CLI takes
the two runs as two local stores (``--before-root`` and ``--after-root``) in which the same input
has the same ``trace_id``, and pairs them per trace. Two stores rather than one store with
before/after tags, because a trace id is unique within a store: pairing inside one store would
need a second key the schema does not have (a tag dimension and a pair id), while two stores pair
on the id every stage already uses. Both stores are local; a Langfuse project holds one id once,
so its runs are brought into two roots with ``calibrated-agent-evaluation ingest --from-langfuse``.

For each failure mode a trace's verdict is read from one source: ``judge`` (the LLM-judge verdicts
stamped with that mode), ``scorer`` (the code-scorer verdicts stamped with it) or ``human`` (a
FAIL for the mode, or a human label that names no failure of it). Judge and scorer verdicts are
never pooled (the split ``calibrated-agent-evaluation run`` has had since). Every gap is a
typed refusal naming the traces: a source with no verdict for the target in either store, a trace
in one store only, a trace without a verdict for the target mode on either side, a guard mode
judged on one side only, and verdicts on one trace that disagree with each other. McNemar's test
pairs every observation, so a dropped trace would bias it silently.

With ``human`` verdicts, a trace whose only label on a side is a FAIL for another failure mode is
excluded from a mode's pairs rather than read as a PASS for it: it failed on something else, and
nothing says whether it failed this mode too. That holds for the target and for
each guard slice. The excluded traces are counted and named per mode. An exclusion
that leaves the target no pair is a typed refusal; a guard mode left with no pair has no slice.

Every other failure mode with verdicts on both sides is a guard slice: its pairs are tested with
McNemar for a regression, one-sided (how likely at least as many before-pass/after-fail flips
would be if flips in each direction were equally likely), and the panel is then Holm corrected at
0.05 by ``calibrated_agent_evaluation.improve.guard_panel``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from calibrated_agent_evaluation.cli.errors import CliError, UsageError
from calibrated_agent_evaluation.schema.types import Label

__all__ = [
    "SOURCE_KINDS",
    "PairedStores",
    "excluded_note",
    "fails_another_mode_only",
    "guard_slice",
    "pair_stores",
    "source_name",
    "verdicts_for",
]

# The label kinds each verdict source reads. A code scorer is its own source.
SOURCE_KINDS: dict[str, tuple[str, ...]] = {
    "judge": ("llm-judge",),
    "scorer": ("code",),
    "human": ("human",),
}
_NAMES = {"judge": "judge verdicts", "scorer": "code-scorer verdicts", "human": "human verdicts"}
_HINTS = {
    "judge": "score both stores with `calibrated-agent-evaluation judge`",
    "scorer": (
        "store code-scorer verdicts in both stores with a "
        "`calibrated_agent_evaluation.scorers` scorer"
    ),
    "human": "label both stores",
}
# Names shown in a refusal, at most this many per gap; the count is always given in full.
_SHOW = 5


@dataclass(frozen=True)
class PairedStores:
    """The target's paired observations and each guard mode's, from one verdict source.

    ``target`` and each value of ``guards`` map a trace id to its ``(before_fail, after_fail)``.
    ``excluded`` are the traces left out of the target's human pairs because a side's only
    evidence is a FAIL for another failure mode. ``guard_excluded`` holds the same
    for each guard mode that had any; a guard mode whose every trace was excluded
    maps to no pairs in ``guards``.
    """

    target: dict[str, tuple[bool, bool]]
    guards: dict[str, dict[str, tuple[bool, bool]]]
    excluded: tuple[str, ...] = ()
    guard_excluded: dict[str, tuple[str, ...]] = field(default_factory=dict)


def source_name(source: str) -> str:
    """How the output names a verdict source's verdicts."""
    return _NAMES[source]


def verdicts_for(labels: Sequence[Label], mode: str, source: str) -> tuple[bool | None, bool]:
    """``(fail, conflicting)`` for one trace and mode: ``fail`` is ``None`` with no verdict.

    ``judge``: the trace-level LLM-judge verdicts stamped with ``mode``; ``scorer``: the
    code-scorer verdicts stamped with it. ``human``: a FAIL for ``mode`` is a failure; a PASS (with
    no mode, or for ``mode``) or a FAIL a labeler gave another mode says the trace did not fail on
    it; a PASS for another mode alone says nothing about it. ``conflicting`` is set when the
    verdicts on the trace disagree with each other.

    The pairing does not take a FAIL for another mode as a pass: a trace whose only evidence on a
    side is such a FAIL is excluded from the mode's pairs (func:`fails_another_mode_only`). It is
    still a label here, so a trace labelled on one side only is
    refused as one-sided rather than skipped.
    """
    trace_level = [lb for lb in labels if lb.span_id is None]
    if source != "human":
        kinds = SOURCE_KINDS[source]
        outcomes = {
            lb.outcome == "FAIL"
            for lb in trace_level
            if lb.kind in kinds and lb.failure_mode == mode
        }
    else:
        human = [lb for lb in trace_level if lb.kind == "human"]
        failed = any(lb.outcome == "FAIL" and lb.failure_mode == mode for lb in human)
        # A PASS with no mode passes the trace outright; one with a mode passes that mode only.
        passed = any(lb.outcome == "PASS" and lb.failure_mode in (None, mode) for lb in human)
        failed_other = any(lb.outcome == "FAIL" and lb.failure_mode != mode for lb in human)
        if failed and passed:
            return True, True
        if failed:
            return True, False
        return (False, False) if passed or failed_other else (None, False)
    if not outcomes:
        return None, False
    return (True in outcomes), len(outcomes) > 1


def fails_another_mode_only(labels: Sequence[Label], mode: str) -> bool:
    """Whether a trace's only human evidence on ``mode`` is a FAIL for another mode (or no mode).

    Such a trace failed on something else, and no label says whether it failed on ``mode`` too, so
    it is not evidence about ``mode`` either way.
    """
    human = [lb for lb in labels if lb.span_id is None and lb.kind == "human"]
    speaks = any(
        (lb.outcome == "FAIL" and lb.failure_mode == mode)
        or (lb.outcome == "PASS" and lb.failure_mode in (None, mode))
        for lb in human
    )
    return not speaks and any(lb.outcome == "FAIL" for lb in human)


def modes_in(labels_by_trace: Mapping[str, Sequence[Label]], source: str) -> set[str]:
    """The failure modes ``source`` has verdicts for, over every trace."""
    kinds = SOURCE_KINDS[source]
    return {
        lb.failure_mode
        for labels in labels_by_trace.values()
        for lb in labels
        if lb.span_id is None and lb.kind in kinds and lb.failure_mode is not None
    }


def pair_stores(
    before: Mapping[str, Sequence[Label]],
    after: Mapping[str, Sequence[Label]],
    metric: str,
    source: str,
) -> PairedStores:
    """Pair ``before`` and ``after`` (trace id -> labels) for ``metric`` and every guard mode."""
    if source not in SOURCE_KINDS:
        raise UsageError(f"--verdicts must be one of {sorted(SOURCE_KINDS)}, got {source!r}")
    only_before = sorted(set(before) - set(after))
    only_after = sorted(set(after) - set(before))
    if only_before or only_after:
        raise CliError(
            "the before and after stores hold different traces, so they are not one input set run "
            f"twice: {_names(only_before)} only before, {_names(only_after)} only after; a paired "
            "test needs every input on both sides"
        )
    modes = modes_in(before, source) | modes_in(after, source)
    sides = (before, after)
    if all(verdicts_for(side[t], metric, source)[0] is None for side in sides for t in side):
        # The run gate's rule: an empty source is refused, never swapped.
        others = " or ".join(f"--verdicts {s}" for s in SOURCE_KINDS if s != source)
        raise CliError(
            f"no {source_name(source)} for the target {metric!r} in either store (--verdicts "
            f"{source}); {_HINTS[source]} first, or pair another source with {others}"
        )
    target, excluded = _pair_mode(before, after, metric, source, required=True)
    guards: dict[str, dict[str, tuple[bool, bool]]] = {}
    guard_excluded: dict[str, tuple[str, ...]] = {}
    for mode in sorted(modes - {metric}):
        guards[mode], left_out = _pair_mode(before, after, mode, source, required=False)
        if left_out:
            guard_excluded[mode] = tuple(left_out)
    return PairedStores(
        target=target, guards=guards, excluded=tuple(excluded), guard_excluded=guard_excluded
    )


def _pair_mode(
    before: Mapping[str, Sequence[Label]],
    after: Mapping[str, Sequence[Label]],
    mode: str,
    source: str,
    *,
    required: bool,
) -> tuple[dict[str, tuple[bool, bool]], list[str]]:
    """The pairs for ``mode`` and, for human verdicts, the traces excluded from them."""
    pairs: dict[str, tuple[bool, bool]] = {}
    missing: dict[str, list[str]] = {"before": [], "after": []}
    conflicts: list[str] = []
    excluded: list[str] = []
    # For the target and for each guard mode, a human FAIL for another mode is not
    # read as a PASS for this one.
    exclude_other_fails = source == "human"
    for trace_id in sorted(before):
        b, b_conflict = verdicts_for(before[trace_id], mode, source)
        a, a_conflict = verdicts_for(after[trace_id], mode, source)
        if b_conflict or a_conflict:
            conflicts.append(trace_id)
            continue
        if b is None and a is None and not required:
            continue
        if b is None:
            missing["before"].append(trace_id)
        if a is None:
            missing["after"].append(trace_id)
        if b is None or a is None:
            continue
        if exclude_other_fails and any(
            fails_another_mode_only(side[trace_id], mode) for side in (before, after)
        ):
            excluded.append(trace_id)
            continue
        pairs[trace_id] = (b, a)
    what = f"the target {mode!r}" if required else f"the guard mode {mode!r}"
    if conflicts:
        raise CliError(
            f"{len(conflicts)} trace(s) carry {source_name(source)} for {what} that disagree "
            f"with each other ({_names(conflicts)}); adjudicate them before pairing"
        )
    gaps = [
        f"{len(ids)} without one {side} ({_names(ids)})" for side, ids in missing.items() if ids
    ]
    if gaps:
        raise CliError(
            f"the {source_name(source)} for {what} are one-sided: {'; '.join(gaps)}. Every trace "
            f"needs a verdict on both sides; {_HINTS[source]} (or gate another source with "
            "--verdicts)"
        )
    if required and not pairs:
        why = (
            f": {len(excluded)} trace(s) were excluded because a side's only human label is a "
            f"FAIL for another failure mode, which is not evidence about it ({_names(excluded)}). "
            f"Label {mode!r} on those traces in both stores"
            if excluded
            else ""
        )
        raise CliError(f"no trace has a {source} verdict for {what} on both sides{why}")
    return pairs, excluded


def excluded_note(paired: PairedStores) -> str:
    """Why each guard mode with no pair left has no slice, for the no-guard refusal."""
    emptied = [mode for mode, pairs in paired.guards.items() if not pairs]
    return "".join(
        f"; the guard mode {mode!r} has no pair left: all {len(paired.guard_excluded[mode])} of "
        "its traces were excluded because a side's only human label is a FAIL for another "
        f"failure mode ({_names(list(paired.guard_excluded[mode]))})"
        for mode in emptied
        if mode in paired.guard_excluded
    )


def guard_slice(name: str, pairs: Mapping[str, tuple[bool, bool]]) -> Any:
    """A :class:`calibrated_agent_evaluation.improve.GuardSlice` of a guard mode's paired verdicts.

    ``regression_p`` is McNemar's p-value for a regression, one-sided: half the two-sided p of
    :func:`calibrated_agent_evaluation.improve.paired_mcnemar` (mid-p under 25 flips, the
    uncorrected asymptotic test above) when the slice has more before-pass/after-fail flips than the
    reverse, else one minus that half. Under 25 flips that is the one-sided mid-p tail. A slice with
    no flips has no evidence of a regression (p = 1). The rates are the slice's fail rates with
    their two-sided 95% Wilson intervals.
    """
    from calibrated_agent_evaluation.improve import GuardSlice, PairedObs, paired_mcnemar

    obs = [PairedObs(trace_id=t, before_fail=b, after_fail=a) for t, (b, a) in pairs.items()]
    result = paired_mcnemar(obs)
    regressions = sum(1 for _, (b, a) in pairs.items() if not b and a)
    improvements = sum(1 for _, (b, a) in pairs.items() if b and not a)
    if result.p_value is None:
        p = 1.0
    elif regressions > improvements:
        p = result.p_value / 2
    else:
        p = 1.0 - result.p_value / 2
    return GuardSlice(
        name=name,
        kind="other",
        regression_p=p,
        rate_before=_wilson([b for b, _ in pairs.values()]),
        rate_after=_wilson([a for _, a in pairs.values()]),
    )


def _wilson(fails: list[bool]) -> Any:
    from calibrated_agent_evaluation.calibrate.record import METHOD_WILSON, Interval
    from calibrated_agent_evaluation.stats.proportions import wilson_lower, wilson_upper

    n, k = len(fails), sum(fails)
    return Interval(
        point=k / n,
        lo=wilson_lower(k, n, alpha=0.05),
        hi=wilson_upper(k, n, alpha=0.05),
        method=METHOD_WILSON,
        level=0.95,
        sided="two-sided",
        n=n,
    )


def _names(ids: list[str]) -> str:
    if not ids:
        return "none"
    shown = ", ".join(repr(i) for i in ids[:_SHOW])
    return shown + (f" and {len(ids) - _SHOW} more" if len(ids) > _SHOW else "")
