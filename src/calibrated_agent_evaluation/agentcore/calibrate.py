# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate an AgentCore judge: run it over the locked test and pair its verdicts with the gold.

:func:`agentcore_locked_test_set` is the AgentCore counterpart of the prompt judge's locked-test
runner. It refuses an advisory-only evaluator and malformed gold before any billed call, scores the
locked test through the AgentCore judge (one billed ``Evaluate`` call per trace, inside the budget
and after the estimate is confirmed), and pairs each verdict with its gold. The calibration engine's
``certify`` then certifies or refuses the judge exactly as it does a prompt judge; nothing in the
engine knows the verdicts came from AgentCore.

The run also returns the session ids its requests carried, read from the requests as sent, which
the online read-back skips, and
:func:`gold_source` names where the gold came from (the quickstart's is rule-derived), so a
summary never presents rule-derived gold as human review.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
from calibrated_agent_evaluation.agentcore.evaluate import run_agentcore_judge_with_sessions
from calibrated_agent_evaluation.agentcore.judge import AgentCoreJudge, require_calibratable
from calibrated_agent_evaluation.calibrate.errors import (
    UnpairedObservationError,
    UnratifiedGoldError,
)
from calibrated_agent_evaluation.calibrate.pairing import refuse_duplicate_targets
from calibrated_agent_evaluation.calibrate.record import CalibrationSet, LabelPair
from calibrated_agent_evaluation.schema.types import Label, Trace

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.pricing import RunEstimate
    from calibrated_agent_evaluation.agentcore.session import AwsClients

__all__ = ["SYNTHETIC_RULE_PREFIX", "LockedTestRun", "agentcore_locked_test_set", "gold_source"]

SYNTHETIC_RULE_PREFIX = "synthetic-rule"
"""Gold whose rater id starts with this was derived by a rule, not labeled by a person."""


@dataclass(frozen=True)
class LockedTestRun:
    """The locked test an AgentCore judge was scored on, and the sessions the scoring sent.

    Attributes:
        calibration_set: The ``locked_test`` set of (verdict, gold) pairs ``certify`` takes.
        session_ids: The session id each ``Evaluate`` call carried, one per trace, in order, read
            from the requests as sent.
    """

    calibration_set: CalibrationSet
    session_ids: tuple[str, ...]


def gold_source(labels: Iterable[Label]) -> str:
    """Where the gold came from: ``synthetic-rule``, ``human``, or both joined by ``and``.

    Gold whose rater id starts with ``synthetic-rule`` (the quickstart's ``synthetic-rule-oracle``)
    is rule-derived; any other gold is human.
    """
    sources = {
        SYNTHETIC_RULE_PREFIX
        if (label.rater_id or "").startswith(SYNTHETIC_RULE_PREFIX)
        else "human"
        for label in labels
    }
    return " and ".join(sorted(sources)) if sources else "none"


def agentcore_locked_test_set(
    judge: AgentCoreJudge,
    gold: Sequence[tuple[Trace, Label]],
    *,
    clients: AwsClients,
    confirm: Callable[[RunEstimate], bool],
    set_id: str,
    max_concurrency: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> LockedTestRun:
    """Score the locked test with ``judge`` and pair each verdict with the trace's gold.

    Args:
        judge: The AgentCore judge to calibrate.
        gold: Each trace of the locked test with its trace-level human gold label.
        clients: The AWS clients; their billed-call budget must fit one call per trace.
        confirm: Receives the run's estimate; returns whether to make the billed calls.
        set_id: The locked test's id.
        max_concurrency: The most ``Evaluate`` calls in flight at once (1 to 10).
        progress: Called with ``(scored, total)`` after each call ends.

    Returns:
        The ``locked_test`` calibration set and the session ids the calls sent.

    Raises:
        AgentCoreAdvisoryOnly: The evaluator is ``Builtin`` or ``ThirdParty`` (or otherwise
            advisory-only), so it can never be certified; before any billed call.
        UnpairedObservationError: The gold is empty, a label names another trace, or a label is
            span-level (an AgentCore judge grades whole traces); before any billed call.
        UnratifiedGoldError: A gold label is not human; before any billed call.
        DuplicateGoldError: A trace is graded more than once; before any billed call.
        BilledCallBudgetExceeded: The budget cannot fit one call per trace; before any call.
        AgentCoreRunDeclined: ``confirm`` declined the estimate; no billed call was made.
        AgentCoreError: The scoring failed, or a trace did not get exactly one trace-level
            verdict, so it cannot be paired; nothing is certified from a partial set.
    """
    require_calibratable(judge)
    if not gold:
        raise UnpairedObservationError(
            f"judge {judge.judge_id!r}: no gold to calibrate against; a locked test needs at least "
            "one human-labeled trace"
        )
    for trace, label in gold:
        if label.kind != "human":
            raise UnratifiedGoldError(
                f"gold label for {label.trace_id} has kind={label.kind!r}, not 'human'; a judge is "
                "certified only against human gold"
            )
        if label.trace_id != trace.trace_id:
            raise UnpairedObservationError(
                f"trace {trace.trace_id!r} is graded against gold for {label.trace_id!r}; a trace "
                "and its gold must name the same trace"
            )
        if label.span_id is not None:
            raise UnpairedObservationError(
                f"gold for trace {label.trace_id!r} names span {label.span_id!r}; an AgentCore "
                "judge grades whole traces, so it is calibrated on trace-level gold only"
            )
    refuse_duplicate_targets((label.trace_id, label.span_id) for _, label in gold)

    traces = [trace for trace, _ in gold]
    scored = run_agentcore_judge_with_sessions(
        judge,
        traces,
        clients=clients,
        confirm=confirm,
        max_concurrency=max_concurrency,
        progress=progress,
    )
    verdicts = scored.labels
    by_trace: dict[str, list[Label]] = {}
    for verdict in verdicts:
        by_trace.setdefault(verdict.trace_id, []).append(verdict)
    pairs: list[LabelPair] = []
    for trace, label in gold:
        found = by_trace.get(trace.trace_id, [])
        if len(found) != 1 or found[0].span_id is not None:
            raise AgentCoreError(
                f"judge {judge.judge_id!r}: trace {trace.trace_id!r} got {len(found)} verdict(s)"
                + (", one of them for a span" if any(v.span_id for v in found) else "")
                + " from AgentCore, not one trace-level verdict, so it cannot be paired with its "
                "gold; nothing is certified from this run"
            )
        pairs.append(LabelPair(gold=label, verdict=found[0]))
    return LockedTestRun(
        calibration_set=CalibrationSet(pairs=tuple(pairs), split="locked_test", set_id=set_id),
        session_ids=scored.session_ids,
    )
