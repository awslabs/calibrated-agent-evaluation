# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The policy battery against a real agent calling its tools through a Gateway with a Policy engine.

Like every test here, these run only with ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE=1`` and as
the live-test role (see ``conftest.py``), and every client they build carries the run's billed-call
budget. They also need these variables, and are skipped, naming the unset ones, without them. No ARN
or account ID is written here.

- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_RUNTIME_ARN``: a runtime whose agent calls its tools
  through the Gateway and passes its trace context on those calls. Its ``DEFAULT`` endpoint is
  invoked once per trial.
- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_PAYLOAD_FORMAT``: the named payload format the runtime
  reads.
- ``CALIBRATED_AGENT_EVALUATION_POLICY_LIVE_MODE``: what the owner set the Gateway to before the
  run, and so which test runs: ``enforce`` (the Gateway enforcing a permit on ``read_rows`` only up
  to limit 100), ``log_only_policy`` (the Gateway enforcing, with that policy set to log-only), or
  ``log_only_gateway`` (the whole Gateway in log-only mode). The log-only ones run only against
  the side-effect-free echo target, with the owner's go-ahead in ``acceptance.md``.

The case file ``policy_live_cases.json`` asks for a small read the policy permits and a bulk read
it permits only under its condition, so the engine still lists ``read_rows`` in ``tools/list``
and the agent can attempt the call the policy should stop. No test asserts a verdict: an agent
that retries inside the condition makes calls the case counts as gaps.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.agentcore]

RUNTIME_ARN = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_RUNTIME_ARN"
PAYLOAD_FORMAT = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_PAYLOAD_FORMAT"
MODE = "CALIBRATED_AGENT_EVALUATION_POLICY_LIVE_MODE"
_MODES = ("enforce", "log_only_policy", "log_only_gateway")
_CASES = Path(__file__).with_name("policy_live_cases.json")
_QUALIFIER = "DEFAULT"
_EXPECTED_DENY = "read-bulk"


def _settings(mode: str) -> tuple[str, str]:
    values = {
        name: os.environ.get(name, "").strip() for name in (RUNTIME_ARN, PAYLOAD_FORMAT, MODE)
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        pytest.skip(f"set {', '.join(missing)} to run this live test")
    if values[MODE] not in _MODES:
        pytest.fail(f"{MODE}={values[MODE]!r}; set it to one of {', '.join(_MODES)}")
    if values[MODE] != mode:
        pytest.skip(f"the Gateway is set to {values[MODE]}, not {mode}")
    return values[RUNTIME_ARN], values[PAYLOAD_FORMAT]


@pytest.fixture(autouse=True)
def _no_cassette_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", raising=False)


def _battery(live_budget: Any, tmp_path: Path, runtime_arn: str, payload_format: str) -> Any:
    from calibrated_agent_evaluation.agentcore import (
        AgentCorePolicyDecisionSource,
        AwsClients,
        RuntimeProbeAdapter,
        RuntimeTarget,
        agent_reported_error_trial,
    )
    from calibrated_agent_evaluation.probe.policy import load_policy_cases
    from calibrated_agent_evaluation.probe.policy_battery import run_policy_battery

    target = RuntimeTarget(
        runtime_arn=runtime_arn, qualifier=_QUALIFIER, payload_format=payload_format
    )
    with (
        AwsClients.from_profile(region=target.region, handlers=[live_budget]) as clients,
        RuntimeProbeAdapter(target, clients=clients, log_dir=tmp_path / "runs") as adapter,
    ):
        source = AgentCorePolicyDecisionSource(clients, run_records=lambda: adapter.run_records)
        return run_policy_battery(
            adapter,
            target=adapter.target_label,
            authorized=True,
            cases=load_policy_cases(_CASES),
            decisions=source,
            agent_error=agent_reported_error_trial,
        )


def _expected_deny_decisions(result: Any) -> list[dict[str, Any]]:
    [finding] = [f for f in result.findings if f.payload["case_id"] == _EXPECTED_DENY]
    expected = {t for t, d in finding.payload["expect"].items() if d == "deny"}
    return [d for e in finding.payload["trials"] for d in e["decisions"] if d["tool"] in expected]


def _store(result: Any, tmp_path: Path) -> Any:
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.probe.policy import policy_labels

    store = LocalBackend(tmp_path / "store")
    labels = policy_labels(result.findings)
    for trace in result.traces:
        store.put_trace(trace)
    for label in labels:
        store.put_score(label)
    return store, labels


def test_a_live_enforcing_gateway_denies_the_call_the_case_expects_denied_and_labels_it(
    live_budget: Any, tmp_path: Path
) -> None:
    runtime_arn, payload_format = _settings("enforce")

    result = _battery(live_budget, tmp_path, runtime_arn, payload_format)

    decisions = _expected_deny_decisions(result)
    denied = [d for d in decisions if d["class"] == "denied"]
    tools = sorted({d["tool"] for d in decisions})
    assert denied, (
        f"case {_EXPECTED_DENY!r} held no deny on a tool it expects denied (its decisions named "
        f"{tools or 'no tool'}), so the run did not exercise the denied path"
    )
    assert all(d["ran"] is False and d["mode"] == "enforce" for d in denied)
    store, labels = _store(result, tmp_path)
    denied_trials = [
        e["probe_trace_id"] for f in result.findings for e in f.payload["trials"] if e["denied"]
    ]
    assert labels
    assert sorted(lb.trace_id for lb in labels) == sorted(denied_trials)
    for trace_id in denied_trials:
        assert [s.kind for s in store.get_scores(trace_id)] == ["policy-engine"]


def test_a_live_log_only_policy_is_recorded_as_flipping_a_call_that_ran(
    live_budget: Any, tmp_path: Path
) -> None:
    runtime_arn, payload_format = _settings("log_only_policy")

    result = _battery(live_budget, tmp_path, runtime_arn, payload_format)

    flipped = [d for d in _expected_deny_decisions(result) if d["log_only_flipping_policies"]]
    assert flipped, (
        f"case {_EXPECTED_DENY!r} held no decision a log-only policy would flip, so the run did "
        "not exercise the log-only path"
    )
    assert all(d["ran"] is True for d in flipped)


def test_a_live_log_only_gateway_records_no_per_call_decision(
    live_budget: Any, tmp_path: Path
) -> None:
    from calibrated_agent_evaluation.probe import PolicyDecisionsMissing

    runtime_arn, payload_format = _settings("log_only_gateway")

    with pytest.raises(PolicyDecisionsMissing, match="log-only mode"):
        _battery(live_budget, tmp_path, runtime_arn, payload_format)
