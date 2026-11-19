# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-agent Rogan-Gladen-corrected rates.

Each agent's failure rate is corrected by **its own** judge's TPR/TNR -- rows do not share a fate. A
weak per-agent judge (small Youden's J) honestly gets a wide bound (or a refused point), which is
the signal, not something to hide. This is one
:func:`calibrated_agent_evaluation.calibrate.engine.corrected_rate` call per agent (reusing the
calibration gate, not re-implementing it); this layer only wires the per-agent loop, labels the
denominator (share-of-failures vs rate-over-traffic), and enforces that an agent's record belongs to
that agent's judge. The calibration layer has no "agent" concept, so the agent -> judge_id binding
is this layer's to hold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from calibrated_agent_evaluation.attribution.errors import AgentRecordMismatchError
from calibrated_agent_evaluation.attribution.types import AgentRate, Denominator
from calibrated_agent_evaluation.calibrate.engine import corrected_rate
from calibrated_agent_evaluation.calibrate.record import CalibrationRecord


def per_agent_rates(
    agent_records: Mapping[str, CalibrationRecord],
    agent_verdicts: Mapping[str, tuple[Sequence[int], str, Sequence[str]]],
    *,
    agent_judges: Mapping[str, str],
    denominator: Denominator,
    seed: int = 0,
) -> list[AgentRate]:
    """Return one independent :class:`AgentRate` per agent.

    ``agent_records`` maps each agent to its own judge's certified :class:`CalibrationRecord`;
    ``agent_verdicts`` maps each agent to ``(verdicts, rate_set_id, trace_ids)`` -- that agent's
    judge verdicts (1=FAIL, 0=PASS) over a production-representative sample, the id of that
    sample, and the trace id of each verdict (checked against the calibration gold).
    ``agent_judges`` maps each agent to the ``judge_id`` that legitimately judges it.

    Each agent's rate is corrected via ``corrected_rate`` (Rogan-Gladen with its interval); the
    resulting :class:`~calibrated_agent_evaluation.calibrate.record.Interval` is attached with the
    stated ``denominator``. The calibration refusals propagate unchanged -- an uncertified agent
    judge raises ``UncalibratedJudgeError``; a J<0.5 or non-identifiable correction yields an
    interval whose point is ``None`` (the widening bound is the signal), never a silently fabricated
    number.

    Raises :class:`AgentRecordMismatchError` if an agent has no declared judge in ``agent_judges``
    (the binding cannot be verified, so it is refused rather than silently trusted), or if an
    agent's record's ``judge_id`` is not that declared judge -- a policy-checker rate must not be
    computed from intake's judge.
    """
    rates: list[AgentRate] = []
    for agent, record in agent_records.items():
        expected_judge = agent_judges.get(agent)
        if expected_judge is None:
            # No declared judge => the own-judge binding cannot be checked. Refuse, never wave it
            # through: an unverifiable binding is exactly the cross-agent contamination we bar.
            raise AgentRecordMismatchError(
                f"agent {agent!r} is being rated but has no declared judge in agent_judges; the "
                f"own-judge binding cannot be verified, so the rate is refused."
            )
        if record.judge_id != expected_judge:
            raise AgentRecordMismatchError(
                f"agent {agent!r} was handed calibration record for judge {record.judge_id!r}, "
                f"but its declared judge is {expected_judge!r}; a per-agent rate must be corrected "
                f"by that agent's own judge."
            )
        if agent not in agent_verdicts:
            # No silent KeyError across the public surface: name the missing rate sample.
            raise AgentRecordMismatchError(
                f"agent {agent!r} has a calibration record but no (verdicts, rate_set_id, "
                "trace_ids) entry in agent_verdicts; every agent being rated needs its own rate "
                "sample."
            )
        verdicts, rate_set_id, trace_ids = agent_verdicts[agent]
        scored = corrected_rate(record, verdicts, rate_set_id, rate_trace_ids=trace_ids, seed=seed)
        interval = scored.corrected_rate
        if interval is None:  # corrected_rate always attaches an Interval on the success path
            raise AgentRecordMismatchError(
                f"agent {agent!r}: corrected_rate returned no interval for record "
                f"{record.calib_id!r} (unexpected)."
            )
        rates.append(AgentRate(agent=agent, interval=interval, denominator=denominator))
    return rates
