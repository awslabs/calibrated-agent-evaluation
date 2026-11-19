# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a policy case file against an agent and score each case from the policy engine's decisions.

Each case's prompt is sent once per trial through any
:class:`~calibrated_agent_evaluation.probe.adapter.ProbeAdapter` whose replies carry trace IDs.
After the last trial a :class:`DecisionSource` answers, in one read, the decisions the engine
recorded in each trial's trace. The battery refuses an answer that does not match its trials and a
result that would mean nothing (no decision at all, a silent control case, decisions naming no tool
any case lists), and builds one finding per case with
:func:`~calibrated_agent_evaluation.probe.policy.case_finding`.

Vendor-free: the decision source is a protocol, so any engine's telemetry can back it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import ClassVar, Protocol, Self

from pydantic import ConfigDict, model_validator

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel, replace
from calibrated_agent_evaluation.probe.adapter import ProbeAdapter, ProbeResponse
from calibrated_agent_evaluation.probe.authorize import require_authorized
from calibrated_agent_evaluation.probe.battery import DEFAULT_MAX_BREAK_RATE
from calibrated_agent_evaluation.probe.errors import (
    PolicyCaseError,
    PolicyDecisionsMissing,
    ProbeError,
)
from calibrated_agent_evaluation.probe.fingerprint import endpoint_fingerprint
from calibrated_agent_evaluation.probe.harness import probed_trace
from calibrated_agent_evaluation.probe.policy import (
    AgentErrorTrial,
    PolicyCaseFile,
    PolicyDecision,
    TrialDecisions,
    TrialRef,
    case_finding,
)
from calibrated_agent_evaluation.probe.record import ProbeFinding
from calibrated_agent_evaluation.schema.types import Trace
from calibrated_agent_evaluation.schema.validate import validate

__all__ = [
    "NO_DECISION_CAUSES",
    "DecisionSource",
    "PolicyBatteryResult",
    "agent_error_trace",
    "run_policy_battery",
]

NO_DECISION_CAUSES = (
    "the Gateway does not deliver its traces to the span store",
    "no Policy engine is attached to the Gateway",
    "the agent's trace context does not reach the Gateway",
    "the agent called no tool",
    "the engine hid the tools from tools/list, so the agent could not call them",
    "the Gateway runs in log-only mode, whose decisions are evaluated asynchronously and are not "
    "recorded per call",
)


class DecisionSource(Protocol):
    """Reads the decisions a policy engine recorded in each trial's trace."""

    def check_readable(self) -> None:
        """Run one read-only query on the decision store; raise its typed error if it fails."""
        ...

    def read_decisions(self, trials: Sequence[TrialRef]) -> Mapping[str, TrialDecisions]:
        """One entry per trial, keyed by probe trace ID; a trial with none has ``decisions=()``."""
        ...


class PolicyBatteryResult(CalibratedAgentEvaluationModel):
    """One policy battery: a finding per case, its trials in order, and one probed trace per trial.

    ``traces[i]`` is trial ``i``'s outcome-only probed trace, under its probe trace ID, so a label
    on it names a trace the store holds.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ProbeError

    findings: tuple[ProbeFinding, ...]
    trials: tuple[TrialRef, ...]
    traces: tuple[Trace, ...]

    @model_validator(mode="after")
    def _traces_match_trials(self) -> Self:
        if [t.trace_id for t in self.traces] != [t.probe_trace_id for t in self.trials]:
            raise ProbeError(
                "a policy battery's traces must be its trials' probed traces, one per trial, in "
                "trial order"
            )
        return self


def agent_error_trace(
    trace_id: str, fingerprint: str, prompt: str, error_code: str, captured_at: str
) -> Trace:
    """The outcome-only probed trace of a trial whose agent reported an error: its code, no text.

    No reply text is kept: an agent's error text can carry the agent's data.
    """
    trace = Trace(
        trace_id=trace_id,
        frame="discovery",
        ownership="probed",
        input={"prompt": prompt},
        output={"error_code": error_code},
        taxonomy_cell={},
        spans=[],
        captured_at=captured_at,
        endpoint_fingerprint=fingerprint,
    )
    validate(trace)
    return trace


def _now() -> datetime:
    return datetime.now(tz=UTC)


def run_policy_battery(
    adapter: ProbeAdapter,
    *,
    target: str,
    authorized: bool,
    cases: PolicyCaseFile,
    decisions: DecisionSource,
    agent_error: Callable[[Exception], AgentErrorTrial | None] | None = None,
    max_break_rate: float = DEFAULT_MAX_BREAK_RATE,
    clock: Callable[[], datetime] = _now,
) -> PolicyBatteryResult:
    """Send every case's prompt once per trial, read the engine's decisions, and score each case.

    Args:
        adapter: Sends one prompt and answers with a reply that carries its trace ID.
        target: The probed target's label, fingerprinted in every finding.
        authorized: Whether the user confirmed they may probe the target.
        cases: The case file.
        decisions: The decision source, asked to check it can read before the first trial.
        agent_error: Turns an error the agent reported in its reply into an
            :class:`AgentErrorTrial`, and returns ``None`` for any other error, which then stops
            the battery. With ``None`` every trial error stops it.
        max_break_rate: The rate the upper bound must sit under for "not vulnerable".
        clock: The battery clock, tz-aware.

    Raises:
        AuthorizationError: ``authorized`` is not set; nothing is sent.
        ProbeError: A reply carries no trace ID, or the source's answer does not match the
            trials (a trial without an entry, an entry naming no trial, a decision under another
            trial's trace ID).
        PolicyDecisionsMissing: No trial got a decision, or a control case got none.
        PolicyCaseError: Decisions were read, but none names a tool any case lists.
        PolicyModeMixed: One case's decisions carry two gateway modes.
    """
    require_authorized(target, authorized)
    decisions.check_readable()
    fingerprint = endpoint_fingerprint(target)
    trials: list[TrialRef] = []
    traces: list[Trace] = []
    for case in cases.cases:
        for number in range(1, case.trials + 1):
            started = clock().isoformat()
            error_code: str | None = None
            try:
                reply: ProbeResponse = adapter.send(case.prompt)
            except Exception as exc:
                failed = agent_error(exc) if agent_error is not None else None
                if failed is None:
                    raise
                probe_trace_id, error_code = failed.probe_trace_id, failed.error_code
                ended = clock().isoformat()
                trace = agent_error_trace(
                    probe_trace_id, fingerprint, case.prompt, error_code, ended
                )
            else:
                if not reply.trace_id:
                    raise ProbeError(
                        f"the reply to trial {number} of case {case.case_id!r} carries no trace "
                        "ID, so the engine's decisions can't be joined to it; a policy battery "
                        "needs an adapter that names its traces"
                    )
                probe_trace_id = reply.trace_id
                ended = clock().isoformat()
                trace = probed_trace(probe_trace_id, fingerprint, case.prompt, reply, ended)[0]
            # The trace is kept in the same step as its trial, so the two line up.
            trials.append(
                TrialRef(
                    case_id=case.case_id,
                    trial=number,
                    probe_trace_id=probe_trace_id,
                    started_at=started,
                    ended_at=ended,
                    error_code=error_code,
                )
            )
            traces.append(trace)
    answers = decisions.read_decisions(trials)
    trials = _join(trials, answers)
    by_trial = {t.probe_trace_id: answers[t.probe_trace_id].decisions for t in trials}
    _check_battery(cases, trials, by_trial)
    findings = tuple(
        case_finding(
            case,
            [t for t in trials if t.case_id == case.case_id],
            by_trial,
            endpoint_fingerprint=fingerprint,
            max_break_rate=max_break_rate,
        )
        for case in cases.cases
    )
    return PolicyBatteryResult(findings=findings, trials=tuple(trials), traces=tuple(traces))


def _join(trials: Sequence[TrialRef], answers: Mapping[str, TrialDecisions]) -> list[TrialRef]:
    """Copy each trial's W3C trace ID from the answer; refuse one that misstates the trials."""
    sent = Counter(t.probe_trace_id for t in trials)
    repeated = sorted(trace for trace, n in sent.items() if n > 1)
    if repeated:
        raise ProbeError(
            f"two or more trials came back under the same probe trace ID ({', '.join(repeated)}); "
            "one trace can't stand for two trials, and counting its decisions twice would "
            "misstate the battery; check that the adapter gives each trial its own trace ID"
        )
    unknown = sorted(set(answers) - set(sent))
    if unknown:
        raise ProbeError(
            f"the decision source answered for {', '.join(unknown)}, which no trial sent; the "
            "source is broken, and counting or dropping those decisions would misstate the trials; "
            "report it with these trace IDs"
        )
    joined: list[TrialRef] = []
    for trial in trials:
        answer = answers.get(trial.probe_trace_id)
        if answer is None:
            raise ProbeError(
                f"the decision source gave no answer for trial {trial.trial} of case "
                f"{trial.case_id!r} ({trial.probe_trace_id}); a trial with no decision gets an "
                "empty answer, so a missing one means the source is broken; report it with this "
                "trace ID"
            )
        foreign = sorted({d.trace_id for d in answer.decisions} - {answer.w3c_trace_id})
        if foreign:
            raise ProbeError(
                f"the decision source gave trial {trial.probe_trace_id} decisions under trace "
                f"{', '.join(foreign)}, not its own {answer.w3c_trace_id}; report it with these "
                "trace IDs"
            )
        joined.append(replace(trial, w3c_trace_id=answer.w3c_trace_id))
    return joined


def _check_battery(
    cases: PolicyCaseFile,
    trials: Sequence[TrialRef],
    by_trial: Mapping[str, Sequence[PolicyDecision]],
) -> None:
    """Refuse a battery whose decisions could only produce a meaningless result."""
    if not any(by_trial[t.probe_trace_id] for t in trials):
        causes = "; ".join(NO_DECISION_CAUSES)
        raise PolicyDecisionsMissing(
            f"no Policy decision on a tool call was read for any of the {len(trials)} trial(s), so "
            f"no case can be scored. Likely causes: {causes}"
        )
    for case in cases.cases:
        if not case.control:
            continue
        own = [t for t in trials if t.case_id == case.case_id]
        if not any(by_trial[t.probe_trace_id] for t in own):
            raise PolicyDecisionsMissing(
                f"control case {case.case_id!r} got no Policy decision in any of its {len(own)} "
                "trial(s); a control expects a decision in at least one of its trials, so the "
                "battery can't tell whether the agent's calls reached the engine"
            )
    listed = {tool for case in cases.cases for tool in case.expect}
    matched = False
    seen: set[str] = set()
    for case in cases.cases:
        for trial in (t for t in trials if t.case_id == case.case_id):
            for decision in by_trial[trial.probe_trace_id]:
                seen.add(decision.tool)
                matched = matched or decision.tool in case.expect
    if not matched:
        raise PolicyCaseError(
            f"the decisions named the tools {', '.join(sorted(seen))}, and none of them is a tool "
            f"a case lists ({', '.join(sorted(listed))}); write each tool in the case file exactly "
            "as the decisions name it"
        )
