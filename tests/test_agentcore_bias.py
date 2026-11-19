# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bias battery run against an AgentCore judge: one concurrent scoring run, then a replay."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")

import boto3
from botocore.awsrequest import AWSResponse
from botocore.config import Config

from calibrated_agent_evaluation.agentcore.bias import (
    AgentCoreProbeRunner,
    bias_thresholds_for,
    probe_trace,
    run_agentcore_battery,
)
from calibrated_agent_evaluation.agentcore.evaluate_payload import aws_trace_id
from calibrated_agent_evaluation.agentcore.judge import AgentCoreJudge, judge_from_evaluator
from calibrated_agent_evaluation.agentcore.pricing import EvaluationRates
from calibrated_agent_evaluation.agentcore.session import AwsClients
from calibrated_agent_evaluation.bias import (
    BiasProbe,
    BiasThresholds,
    ProbeInput,
    load_probe_set,
    load_thresholds,
)
from calibrated_agent_evaluation.bias.errors import MissingBiasThresholdError, ProbeExecutionError
from calibrated_agent_evaluation.bias.record import BiasName
from cli_store_helpers import certified_calibration

_ID = "orderbot_refund_policy_v2-AbCdEf1234"
_ARN = f"arn:aws:bedrock-agentcore:us-east-1:111122223333:evaluator/{_ID}"
_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
_CERTIFY = (
    Path(__file__).resolve().parents[1] / "src/calibrated_agent_evaluation/_quickstart/data/certify"
)


@pytest.fixture(autouse=True)
def _published_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    rates = EvaluationRates(
        region="us-east-1",
        builtin_input_per_million=Decimal("2.40"),
        builtin_output_per_million=Decimal("12.00"),
        custom_per_evaluation=Decimal("0.0015"),
    )
    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.evaluate.lookup_evaluation_rates", lambda _c: rates
    )


def _evaluator() -> dict[str, Any]:
    return {
        "evaluatorArn": _ARN,
        "evaluatorId": _ID,
        "evaluatorName": "orderbot_refund_policy_v2",
        "evaluatorType": "Custom",
        "level": "TRACE",
        "status": "ACTIVE",
        "updatedAt": "2026-11-05T16:19:46+00:00",
        "evaluatorConfig": {
            "llmAsAJudge": {
                "instructions": "Judge it. {context} {assistant_turn}",
                "ratingScale": {
                    "numerical": [
                        {"value": 0.0, "label": "FAIL", "definition": "fails"},
                        {"value": 1.0, "label": "PASS", "definition": "passes"},
                    ]
                },
                "modelConfig": {"bedrockEvaluatorModelConfig": {"modelId": _MODEL}},
            }
        },
    }


def _judge() -> AgentCoreJudge:
    return judge_from_evaluator("orderbot-refund-policy", _evaluator(), level="workflow")


def _calib(judge: AgentCoreJudge) -> Any:
    return certified_calibration(
        judge.judge_id,
        model=judge.model,
        prompt_version=judge.prompt_version,
        fingerprint=judge.fingerprint,
    )


def _inputs(probes: list[BiasProbe]) -> list[ProbeInput]:
    """Every input the battery asks a judge about: each trial's inputs and its placebo repeats."""
    out: list[ProbeInput] = []
    for probe in probes:
        if probe.pairwise:
            continue
        for trial in probe.trials:
            out.extend(trial)
            out.extend(replace(trial[0], repeat=i) for i in range(1, len(trial)))
    return out


class _Account:
    """Answers each Evaluate call from botocore's own hooks, the verdict chosen per probe input.

    ``verdict(probe_input)`` returns ``"PASS"`` or ``"FAIL"``. ``peak`` is the most calls ever
    open at once; ``calls`` counts them.
    """

    def __init__(
        self, clients: AwsClients, probes: list[BiasProbe], verdict: Any, *, error: str = ""
    ) -> None:
        self.by_target = {aws_trace_id(probe_trace(p).trace_id): p for p in _inputs(probes)}
        self.verdict = verdict
        self.error = error
        self.lock = threading.Lock()
        self.calls = self.open = self.peak = 0
        self.sessions: list[str] = []
        events = clients.client("bedrock-agentcore").meta.events
        events.register("before-parameter-build.bedrock-agentcore.Evaluate", self._capture)
        events.register("before-call.bedrock-agentcore.Evaluate", self._respond)

    def _capture(self, params: dict[str, Any], context: dict[str, Any], **_: Any) -> None:
        context["target"] = params["evaluationTarget"]["traceIds"][0]
        with self.lock:
            self.sessions.append(
                params["evaluationInput"]["sessionSpans"][0]["attributes"]["session.id"]
            )

    def _respond(self, context: dict[str, Any], **_: Any) -> Any:
        target = context["target"]
        with self.lock:
            self.calls += 1
            self.open += 1
            self.peak = max(self.peak, self.open)
        time.sleep(0.002)
        with self.lock:
            self.open -= 1
        if self.error:
            error = {"Error": {"Code": self.error, "Message": "refused by the stub"}}
            return AWSResponse("https://stub", 400, {}, None), error
        outcome = self.verdict(self.by_target[target])
        result = {
            "evaluatorArn": _ARN,
            "evaluatorId": _ID,
            "evaluatorName": "orderbot_refund_policy_v2",
            "explanation": "graded",
            "value": 1.0 if outcome == "PASS" else 0.0,
            "label": outcome,
            "context": {"spanContext": {"sessionId": target, "traceId": target}},
        }
        return AWSResponse("https://stub", 200, {}, None), {"evaluationResults": [result]}


def _clients() -> AwsClients:
    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    return AwsClients(session).with_config(Config(retries={"total_max_attempts": 1}))


def _approve(_estimate: Any) -> bool:
    return True


def _quickstart() -> tuple[list[BiasProbe], BiasThresholds]:
    return load_probe_set(_CERTIFY / "bias-probes-2"), load_thresholds(
        _CERTIFY / "bias-thresholds.json"
    )


def test_a_fair_judge_clears_the_quickstart_battery_in_one_concurrent_run() -> None:
    judge = _judge()
    probes, thresholds = _quickstart()
    clients = _clients()
    account = _Account(clients, probes, lambda _p: "PASS")
    seen: list[int] = []

    run = run_agentcore_battery(
        judge,
        _calib(judge),
        thresholds,
        probes,
        clients=clients,
        confirm=_approve,
        max_concurrency=3,
        progress=lambda scored, _total: seen.append(scored),
    )

    assert run.record.verdict == "CLEAR"
    assert run.record.model == _ARN
    assert run.record.judge_fingerprint == judge.fingerprint
    assert run.thresholds_key == f"bedrock/{_MODEL}"
    # Every trial input and every placebo repeat is a call of its own.
    assert account.calls == len(_inputs(probes))
    assert sorted(run.session_ids) == sorted(account.sessions)
    assert len(set(run.session_ids)) == account.calls
    assert 2 <= account.peak <= 3
    assert seen == list(range(1, account.calls + 1))


def test_a_length_biased_judge_is_biased_even_when_the_recording_pass_could_not_clear() -> None:
    judge = _judge()
    probes, thresholds = _quickstart()
    # Only the length probe: the battery lacks two must-pass biases, so it can never be CLEAR,
    # but a measured must-pass exceedance is still BIASED.
    length = [p for p in probes if p.bias == "length"]
    clients = _clients()
    _Account(clients, length, lambda p: "FAIL" if p.payload.get("variant") == "padded" else "PASS")

    run = run_agentcore_battery(
        judge, _calib(judge), thresholds, length, clients=clients, confirm=_approve
    )

    assert run.record.verdict == "BIASED"
    assert run.record.failed_biases == ("length",)


def test_a_call_naming_another_judge_version_is_refused() -> None:
    judge = _judge()
    probe = ProbeInput("length", "p-1", {"answer": "ok"})
    runner = AgentCoreProbeRunner(judge, [(probe, "PASS")])

    with pytest.raises(ProbeExecutionError, match="pinned to"):
        runner.run("another-model", judge.prompt_version, probe)


def test_an_input_asked_for_out_of_the_recorded_order_is_refused() -> None:
    judge = _judge()
    first = ProbeInput("length", "p-1", {"answer": "ok"})
    second = ProbeInput("length", "p-2", {"answer": "ok"})
    runner = AgentCoreProbeRunner(judge, [(first, "PASS"), (second, "FAIL")])

    with pytest.raises(ProbeExecutionError, match="p-2"):
        runner.run(judge.model, judge.prompt_version, second)


_ROW: dict[BiasName, float] = {
    "length": 0.35,
    "self_preference": 0.35,
    "sycophancy": 0.35,
    "compassion_fade": 0.35,
}


def test_the_evaluator_arn_entry_wins_over_the_model_entry() -> None:
    judge = _judge()
    tight: dict[BiasName, float] = {**_ROW, "length": 0.10}
    thresholds = BiasThresholds(by_model={_ARN: tight, f"bedrock/{_MODEL}": _ROW})

    keyed, key = bias_thresholds_for(judge, thresholds)

    assert key == _ARN
    assert keyed.threshold_for(_ARN, "length") == 0.10


def test_the_model_entry_is_used_when_there_is_no_arn_entry() -> None:
    judge = _judge()

    keyed, key = bias_thresholds_for(judge, BiasThresholds(by_model={f"bedrock/{_MODEL}": _ROW}))

    assert key == f"bedrock/{_MODEL}"
    assert keyed.threshold_for(_ARN, "length") == 0.35


def test_a_thresholds_file_with_neither_entry_is_refused_naming_both_with_no_call() -> None:
    judge = _judge()
    probes, _ = _quickstart()
    clients = _clients()
    account = _Account(clients, probes, lambda _p: "PASS")

    with pytest.raises(MissingBiasThresholdError) as raised:
        run_agentcore_battery(
            judge,
            _calib(judge),
            BiasThresholds(by_model={"bedrock/some-other-model": _ROW}),
            probes,
            clients=clients,
            confirm=_approve,
        )

    assert _ARN in str(raised.value)
    assert f"bedrock/{_MODEL}" in str(raised.value)
    assert account.calls == 0


def test_a_judge_with_no_pinned_model_is_refused_naming_its_arn_only() -> None:
    judge = replace(_judge(), pinned_model=None)

    with pytest.raises(MissingBiasThresholdError) as raised:
        bias_thresholds_for(judge, BiasThresholds(by_model={f"bedrock/{_MODEL}": _ROW}))

    assert _ARN in str(raised.value)
    assert "bedrock/" not in str(raised.value)


def test_an_arn_entry_missing_a_must_pass_bias_is_refused_with_no_call() -> None:
    judge = _judge()
    probes, _ = _quickstart()
    clients = _clients()
    account = _Account(clients, probes, lambda _p: "PASS")
    partial: dict[BiasName, float] = {k: v for k, v in _ROW.items() if k != "sycophancy"}

    with pytest.raises(MissingBiasThresholdError):
        run_agentcore_battery(
            judge,
            _calib(judge),
            BiasThresholds(by_model={_ARN: partial, f"bedrock/{_MODEL}": _ROW}),
            probes,
            clients=clients,
            confirm=_approve,
        )

    assert account.calls == 0


def test_a_failed_evaluate_call_is_reported_as_a_failed_probe_run() -> None:
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError

    judge = _judge()
    probes, thresholds = _quickstart()
    clients = _clients()
    _Account(clients, probes, lambda _p: "PASS", error="ValidationException")

    with pytest.raises(ProbeExecutionError) as raised:
        run_agentcore_battery(
            judge, _calib(judge), thresholds, probes, clients=clients, confirm=_approve
        )

    assert isinstance(raised.value.__cause__, AgentCoreError)


def test_a_declined_estimate_sends_nothing_and_is_not_a_failed_probe_run() -> None:
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreRunDeclined

    judge = _judge()
    probes, thresholds = _quickstart()
    clients = _clients()
    account = _Account(clients, probes, lambda _p: "PASS")

    with pytest.raises(AgentCoreRunDeclined):
        run_agentcore_battery(
            judge, _calib(judge), thresholds, probes, clients=clients, confirm=lambda _e: False
        )

    assert account.calls == 0


def test_two_probe_inputs_sharing_an_id_are_refused_before_any_call() -> None:
    judge = _judge()
    probes, thresholds = _quickstart()
    length = next(p for p in probes if p.bias == "length")
    twin = replace(length.trials[0][1], probe_id=length.trials[0][0].probe_id)
    clash = replace(length, trials=((length.trials[0][0], twin), *length.trials[1:]))
    others = [p for p in probes if p.bias != "length"]
    clients = _clients()
    account = _Account(clients, [*others, length], lambda _p: "PASS")

    with pytest.raises(ProbeExecutionError, match="share the id"):
        run_agentcore_battery(
            judge, _calib(judge), thresholds, [*others, clash], clients=clients, confirm=_approve
        )

    assert account.calls == 0
