# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the bias battery against an AgentCore judge, scoring every probe in one concurrent run.

The battery (:func:`calibrated_agent_evaluation.bias.run_battery`) asks its
:class:`~calibrated_agent_evaluation.bias.probe.ProbeRunner` for one verdict at a time, which would
leave an AgentCore judge's battery sequential. It asks for every input of every probe, and every
placebo repeat, in a fixed order whatever the verdicts, so :func:`run_agentcore_battery` runs it
twice: first with a runner that only records what it is asked for, then, once every recorded input
has been scored by AgentCore in one run of up to ``max_concurrency`` calls at a time, with
:class:`AgentCoreProbeRunner`, which answers each call with the next scored verdict and refuses one
asked for out of that order. ``bias/`` is unchanged.

Each probe input is sent as a one-span trace: the question and any context the bias varies as the
user turn, and the answer as the agent's reply, so the evaluator judges the same content a prompt
judge is shown. The thresholds are looked up for the evaluator's ARN, or else for the model it
runs on (:func:`bias_thresholds_for`).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreError,
    AgentCoreRunDeclined,
    BilledCallBudgetExceeded,
)
from calibrated_agent_evaluation.agentcore.judge import AgentCoreJudge
from calibrated_agent_evaluation.bias import (
    BiasBatteryRecord,
    BiasProbe,
    BiasThresholds,
    ProbeInput,
    ProbeVerdict,
    run_battery,
)
from calibrated_agent_evaluation.bias.errors import (
    IncompleteBatteryError,
    MissingBiasThresholdError,
    ProbeExecutionError,
)
from calibrated_agent_evaluation.calibrate.record import CalibrationRecord
from calibrated_agent_evaluation.schema.types import Span, Trace

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.pricing import RunEstimate
    from calibrated_agent_evaluation.agentcore.session import AwsClients

__all__ = [
    "AgentCoreBatteryRun",
    "AgentCoreProbeRunner",
    "bias_thresholds_for",
    "probe_trace",
    "run_agentcore_battery",
]

# The routing prefix Calibrated Agent Evaluation model ids carry for a Bedrock model; an evaluator
# names the model without it.
_BEDROCK_PREFIX = "bedrock/"


@dataclass(frozen=True)
class AgentCoreBatteryRun:
    """An AgentCore judge's battery record, the sessions its calls named, and its thresholds.

    Attributes:
        record: The ``CLEAR`` or ``BIASED`` battery record, pinned to the evaluator's ARN, its
            ``updatedAt`` and the judge's fingerprint.
        session_ids: The session id each ``Evaluate`` call carried, one per probe input.
        thresholds_key: The thresholds file entry the battery was gated against.
    """

    record: BiasBatteryRecord
    session_ids: tuple[str, ...]
    thresholds_key: str


def bias_thresholds_for(
    judge: AgentCoreJudge, thresholds: BiasThresholds
) -> tuple[BiasThresholds, str]:
    """The thresholds the battery reads for ``judge``, and the file entry they came from.

    The entry for the evaluator's ARN is used when the file has one; otherwise the entry for the
    model the evaluator runs on, ``bedrock/<pinned model>``. The whole entry is taken, never merged
    bias by bias with the other. The result is keyed by the ARN, the ``model`` the battery reads.

    Raises:
        MissingBiasThresholdError: The file has neither entry; the message names the keys looked
            for (the ARN, and the model's when the evaluator has a pinned model).
    """
    keys = [judge.model]
    if judge.pinned_model:
        keys.append(f"{_BEDROCK_PREFIX}{judge.pinned_model}")
    for key in keys:
        if key in thresholds.by_model:
            return BiasThresholds(by_model={judge.model: thresholds.by_model[key]}), key
    looked = " or ".join(repr(key) for key in keys)
    raise MissingBiasThresholdError(
        f"the bias thresholds file has no entry for AgentCore judge {judge.judge_id!r}: it was "
        f"looked up as {looked}. Add an entry under one of them; nothing was sent"
    )


def probe_trace(probe_input: ProbeInput) -> Trace:
    """``probe_input`` as the one-span trace AgentCore scores: the user turn and the reply.

    The user turn is the question and any context the bias varies (a framing, an assertion); the
    reply is the answer, headed with its claimed author when the probe names one. Control keys,
    the bias name and the probe id are never in the content. Each placebo repeat is its own trace,
    so each is its own session.

    Raises:
        ProbeExecutionError: The payload has no answer, or a key the prompt judge's renderer does
            not know (it may be a control label), or the probe is pairwise.
    """
    from calibrated_agent_evaluation.judges.probe_runner import render_probe_input

    payload = probe_input.payload
    if probe_input.bias == "positional" or "better" in payload:
        raise ProbeExecutionError(
            f"probe {probe_input.probe_id!r} is pairwise (a candidate pick); an AgentCore judge "
            "gives a single verdict"
        )
    # Refuses an unknown key or a missing answer, exactly as for a prompt judge.
    render_probe_input(probe_input)
    asked = [str(payload[key]) for key in ("question", "framing", "assertion") if key in payload]
    answer = str(payload["answer"])
    if "attributed_model" in payload:
        answer = f"Response (written by {payload['attributed_model']}):\n{answer}"
    trace_id = f"{probe_input.probe_id}~{probe_input.repeat}"
    span = Span(
        span_id=f"{trace_id}/agent",
        trace_id=trace_id,
        agent="agent",
        role="specialist",
        parent=None,
        seq=0,
        status="completed",
        input={"message": "\n\n".join(asked)},
        output={"answer": answer},
    )
    return Trace(
        trace_id=trace_id,
        frame="discovery",
        ownership="owned",
        input={"message": "\n\n".join(asked)},
        output={"answer": answer},
        taxonomy_cell={"bias_probe": probe_input.bias},
        spans=[span],
    )


class AgentCoreProbeRunner:
    """A :class:`~calibrated_agent_evaluation.bias.probe.ProbeRunner` over AgentCore's verdicts.

    ``scored`` holds each probe input with its verdict label, in the order the battery asks for
    them. A call naming another judge version, or an input other than the next one, is refused.
    """

    def __init__(self, judge: AgentCoreJudge, scored: Sequence[tuple[ProbeInput, str]]) -> None:
        self._judge = judge
        self._scored = list(scored)
        self._next = 0

    @property
    def exhausted(self) -> bool:
        """Whether the battery asked for every scored input."""
        return self._next == len(self._scored)

    def run(self, judge_model: str, prompt_version: str, probe_input: ProbeInput) -> ProbeVerdict:
        """The verdict AgentCore gave ``probe_input``.

        Raises:
            ProbeExecutionError: The call names another judge version, or asks for an input other
                than the next scored one (or for more than were scored).
        """
        if (judge_model, prompt_version) != (self._judge.model, self._judge.prompt_version):
            raise ProbeExecutionError(
                f"battery asked for {judge_model}@{prompt_version} but this runner is pinned to "
                f"{self._judge.model}@{self._judge.prompt_version}; refusing to record one judge's "
                "probes under another's identity"
            )
        if self._next >= len(self._scored) or self._scored[self._next][0] != probe_input:
            raise ProbeExecutionError(
                f"the battery asked for probe {probe_input.probe_id!r} (repeat "
                f"{probe_input.repeat}) out of the order its inputs were scored in, so no verdict "
                "can be given for it; the battery is refused"
            )
        label = self._scored[self._next][1]
        self._next += 1
        return ProbeVerdict(label=label)


class _Recorder:
    """Records what the battery asks for, answering every call ``PASS`` so it runs through.

    It refuses a call naming another judge version, as the replaying runner does, so a mismatch
    is caught before any billed call.
    """

    def __init__(self, judge: AgentCoreJudge) -> None:
        self._pin = (judge.model, judge.prompt_version)
        self.asked: list[ProbeInput] = []

    def run(self, judge_model: str, prompt_version: str, probe_input: ProbeInput) -> ProbeVerdict:
        if (judge_model, prompt_version) != self._pin:
            raise ProbeExecutionError(
                f"battery asked for {judge_model}@{prompt_version} but the judge is "
                f"{self._pin[0]}@{self._pin[1]}; refusing before any probe is sent"
            )
        self.asked.append(probe_input)
        return ProbeVerdict(label="PASS")


def run_agentcore_battery(
    judge: AgentCoreJudge,
    calib_record: CalibrationRecord,
    thresholds: BiasThresholds,
    probes: Sequence[BiasProbe],
    *,
    clients: AwsClients,
    confirm: Callable[[RunEstimate], bool],
    max_concurrency: int = 1,
    progress: Callable[[int, int], None] | None = None,
    battery_id: str | None = None,
) -> AgentCoreBatteryRun:
    """Run the bias battery for a certified AgentCore judge.

    Args:
        judge: The AgentCore judge; ``calib_record`` must certify it.
        calib_record: The judge's certifying calibration record.
        thresholds: The thresholds file; its entry for the judge is chosen by
            :func:`bias_thresholds_for`.
        probes: The probe set.
        clients: The AWS clients; their billed-call budget must fit one call per probe input.
        confirm: Receives the run's estimate; returns whether to make the billed calls.
        max_concurrency: The most ``Evaluate`` calls in flight at once (1 to 10).
        progress: Called with ``(scored, total)`` after each call ends.
        battery_id: The record's id; generated when omitted.

    Returns:
        The battery record, the session ids sent and the thresholds entry used.

    Raises:
        UncalibratedScoreError: ``calib_record`` does not certify ``judge``; before any call.
        MissingBiasThresholdError: No thresholds entry for the judge, or its entry lacks a
            must-pass bias; before any call.
        EmptyProbeError: A must-pass probe has no trials; before any call.
        IncompleteBatteryError: ``probes`` is empty (before any call), or lacks a must-pass bias
            and no measured must-pass bias was exceeded (after the billed run, since only the
            verdicts can show an exceedance).
        AgentCoreRunDeclined: ``confirm`` declined the estimate; no call was made.
        BilledCallBudgetExceeded: The budget cannot fit the run; before any call.
        ProbeExecutionError: A probe input cannot be sent or two share an id (before any call),
            scoring failed (the AgentCore error is chained), or the battery asked for its inputs
            in another order than recorded.
    """
    from calibrated_agent_evaluation.agentcore.evaluate import run_agentcore_judge_with_sessions
    from calibrated_agent_evaluation.enforce.gate import enforce_score_gate

    # The prompt path scores only a certified judge (run_judge checks this gate); so does this one.
    enforce_score_gate(calib_record, judge)
    keyed, key = bias_thresholds_for(judge, thresholds)
    recorder = _Recorder(judge)
    # The recording pass answers everything PASS, so it can look like an unearned CLEAR; only the
    # list of inputs is kept, and the real pass decides the verdict.
    with contextlib.suppress(IncompleteBatteryError):
        run_battery(judge, recorder, keyed, probes, sampling=None)
    asked = recorder.asked
    traces = [probe_trace(probe_input) for probe_input in asked]
    seen: set[str] = set()
    for trace in traces:
        if trace.trace_id in seen:
            # Two inputs would be one trace and one verdict; refused before anything is billed.
            raise ProbeExecutionError(
                f"two bias probe inputs share the id {trace.trace_id!r}, so their verdicts could "
                "not be told apart; give each probe input its own probe_id. Nothing was sent"
            )
        seen.add(trace.trace_id)
    try:
        scored = run_agentcore_judge_with_sessions(
            judge,
            traces,
            clients=clients,
            confirm=confirm,
            max_concurrency=max_concurrency,
            progress=progress,
        )
    except (AgentCoreRunDeclined, BilledCallBudgetExceeded):
        # Not a failed probe: nothing was sent, and the caller handles both.
        raise
    except AgentCoreError as exc:
        raise ProbeExecutionError(
            f"scoring the bias probes through AgentCore failed: {exc}"
        ) from exc
    by_trace: dict[str, list[str]] = {}
    for label in scored.labels:
        by_trace.setdefault(label.trace_id, []).append(label.outcome)
    answers: list[tuple[ProbeInput, str]] = []
    for probe_input, trace in zip(asked, traces, strict=True):
        found = by_trace.get(trace.trace_id, [])
        if len(found) != 1:
            raise ProbeExecutionError(
                f"probe {probe_input.probe_id!r} (repeat {probe_input.repeat}) got {len(found)} "
                "verdict(s) from AgentCore, not one, so the battery cannot read it"
            )
        answers.append((probe_input, found[0]))
    replay = AgentCoreProbeRunner(judge, answers)
    record = run_battery(judge, replay, keyed, probes, battery_id=battery_id, sampling=None)
    if not replay.exhausted:
        raise ProbeExecutionError(
            "the battery asked for fewer probe inputs than were scored, so the order it asks in "
            "has changed and its verdicts cannot be trusted; the battery is refused"
        )
    return AgentCoreBatteryRun(record=record, session_ids=scored.session_ids, thresholds_key=key)
