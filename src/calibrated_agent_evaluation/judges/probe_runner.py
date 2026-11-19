# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A :class:`~calibrated_agent_evaluation.bias.probe.ProbeRunner` that runs probes through a judge.

``run_battery`` takes its judge runs through the injected ``ProbeRunner`` port. This module is
the shipped concrete runner, so a judge for an agent outside this repo can clear ``admit_judge``'s
bias gate.

:class:`JudgeProbeRunner` renders each probe input into the judge's user prompt and runs it through
the judge runtime's ``run_judge``: the same calibration gate, forced-structured verdict, and
validation as a
production judge run. The model is the injected ``JudgeModelBackend`` seam from ``execute/``, so
this module imports no vendor SDK and ``bias/`` stays vendor-free. The live
model is :class:`~calibrated_agent_evaluation.execute.InspectJudgeModel`, wired by
:meth:`JudgeProbeRunner.live`.

This lives in ``judges/`` rather than ``bias/`` because it is a judge-execution concern: it needs
``ExecutableJudge`` and ``run_judge``, and ``bias/`` must not depend on the judge runtime (the
dependency runs the other way, through the port).
"""

from __future__ import annotations

from calibrated_agent_evaluation.bias.errors import ProbeExecutionError
from calibrated_agent_evaluation.bias.probe import ProbeInput, ProbeVerdict
from calibrated_agent_evaluation.calibrate.record import CalibrationRecord
from calibrated_agent_evaluation.execute import JudgeModelBackend
from calibrated_agent_evaluation.execute.errors import ExecutionError
from calibrated_agent_evaluation.judges.errors import JudgeRuntimeError
from calibrated_agent_evaluation.judges.executable import ExecutableJudge
from calibrated_agent_evaluation.judges.runtime import JudgeTarget, run_judge

# Payload keys that only steer the experiment (which variant this is, which candidate is better).
# Showing them to the judge would tell it what is being measured, so they are never rendered.
_CONTROL_KEYS: frozenset[str] = frozenset({"variant", "order", "first", "second", "better"})
# Payload keys that ARE the judged content, rendered as prose without their key names.
_CONTENT_KEYS: frozenset[str] = frozenset(
    {"question", "framing", "assertion", "attributed_model", "answer"}
)


def render_probe_input(probe_input: ProbeInput) -> str:
    """Render one probe input as the judge's user prompt.

    Shows the question, any context the bias varies (a sympathetic framing, an authoritative
    assertion, a claimed author), and the answer. Never shows the bias name, probe id, or control
    keys such as ``variant``. An unrecognised payload key raises :class:`ProbeExecutionError`
    rather than being shown, because it may be a control label the renderer does not know about.
    """
    payload = probe_input.payload
    unknown = sorted(set(payload) - _CONTROL_KEYS - _CONTENT_KEYS)
    if unknown:
        raise ProbeExecutionError(
            f"probe {probe_input.probe_id!r} carries payload key(s) {unknown} the renderer does "
            "not know; refusing to show a possible control label to the judge"
        )
    if "answer" not in payload:
        raise ProbeExecutionError(f"probe {probe_input.probe_id!r} has no 'answer' to judge")
    parts: list[str] = []
    if "question" in payload:
        parts.append(str(payload["question"]))
    if "framing" in payload:
        parts.append(str(payload["framing"]))
    if "assertion" in payload:
        parts.append(str(payload["assertion"]))
    header = "Response"
    if "attributed_model" in payload:
        header = f"Response (written by {payload['attributed_model']})"
    parts.append(f"{header}:\n{payload['answer']}")
    return "\n\n".join(parts) + "\n"


class JudgeProbeRunner:
    """Run each probe input through ``run_judge`` for one fixed, certified judge.

    The battery passes ``(judge_model, prompt_version)`` with every call and pins them on the
    record it returns, so a call naming a different pair than this runner's judge is refused: the
    record would otherwise claim a battery result for a judge that was never probed.
    """

    def __init__(
        self,
        judge: ExecutableJudge,
        calib_record: CalibrationRecord,
        model_backend: JudgeModelBackend,
    ) -> None:
        self._judge = judge
        self._calib = calib_record
        self._backend = model_backend

    @classmethod
    def live(
        cls,
        judge: ExecutableJudge,
        calib_record: CalibrationRecord,
        *,
        timeout_seconds: float = 60.0,
    ) -> JudgeProbeRunner:
        """A runner over an :class:`~calibrated_agent_evaluation.execute.InspectJudgeModel`.

        Building it makes no network call; the first :meth:`run` does, and needs the Inspect extra
        and the provider's credentials.
        """
        from calibrated_agent_evaluation.execute import InspectJudgeModel

        return cls(
            judge,
            calib_record,
            InspectJudgeModel(
                judge.model, timeout_seconds=timeout_seconds, temperature=judge.temperature
            ),
        )

    @property
    def model_backend(self) -> JudgeModelBackend:
        """The injected model seam (exposed so a caller can confirm which model is live)."""
        return self._backend

    def run(self, judge_model: str, prompt_version: str, probe_input: ProbeInput) -> ProbeVerdict:
        """Judge one probe input and return its PASS/FAIL verdict.

        Raises :class:`ProbeExecutionError` on a pinned-judge mismatch, a pairwise probe (a
        single-verdict judge cannot pick a candidate), or any model / verdict failure.
        """
        if (judge_model, prompt_version) != (self._judge.model, self._judge.prompt_version):
            raise ProbeExecutionError(
                f"battery asked for {judge_model}@{prompt_version} but this runner is pinned to "
                f"{self._judge.model}@{self._judge.prompt_version}; refusing to record one judge's "
                "probes under another's identity"
            )
        if probe_input.bias == "positional" or "better" in probe_input.payload:
            raise ProbeExecutionError(
                f"probe {probe_input.probe_id!r} is pairwise (a candidate pick); "
                "JudgeProbeRunner drives a single-verdict judge"
            )
        target = JudgeTarget(
            trace_id=probe_input.probe_id,
            span_id=None,
            content=render_probe_input(probe_input),
            sample=probe_input.repeat,
        )
        try:
            result = run_judge(
                self._judge, [target], calib_record=self._calib, model_backend=self._backend
            )
        except (ExecutionError, JudgeRuntimeError) as exc:
            raise ProbeExecutionError(
                f"judge probe {probe_input.probe_id!r} failed to execute: {exc}"
            ) from exc
        label = result.labels[0]
        return ProbeVerdict(label=label.outcome, detail=label.critique)
