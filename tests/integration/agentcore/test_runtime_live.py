# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runtime executor and the runtime probe adapter against a real SigV4 AgentCore runtime in the
integration account.

Like every test here, they run only with ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE=1`` and as the
live-test role (see ``conftest.py``), and every client they build carries the run's billed-call
budget. They also need these variables, and are skipped, naming the unset ones, without them. No ARN
or account ID is written here.

- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_RUNTIME_ARN``: a runtime that accepts AWS IAM (SigV4)
  calls. Its ``DEFAULT`` endpoint, the one the live-test role names, is invoked once by each test, a
  billed call.
- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_PAYLOAD_FORMAT``: the named payload format the runtime
  reads, such as ``agentcore_sse``.
- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_QUESTION``: the question sent.
- ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_EXPECTED_AGENTS``: the executor's test only; the agents
  that question drives, comma-separated. The trace's span agents must be exactly these.

The executor's test needs the AgentCore span reader,
``calibrated_agent_evaluation.agentcore.AgentCoreSpanReader``, too, and is skipped, naming it, if
the install cannot load it. The probe adapter reads no spans, so its test needs no reader; it checks
the reply, trace ID and session stop that the stubbed probe tests assume against a real runtime. The
refusal of an OAuth-only runtime has no live test: the stubbed tests cover it, and its check against
a real runtime is run by hand.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.agentcore]

RUNTIME_ARN = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_RUNTIME_ARN"
PAYLOAD_FORMAT = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_PAYLOAD_FORMAT"
QUESTION = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_QUESTION"
EXPECTED_AGENTS = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_EXPECTED_AGENTS"

# The endpoint the live-test role allows invoking.
_QUALIFIER = "DEFAULT"


def _required(*names: str) -> list[str]:
    """The values of ``names``; the test is skipped, naming the unset ones, unless all are set."""
    values = [os.environ.get(name, "").strip() for name in names]
    missing = [name for name, value in zip(names, values, strict=True) if not value]
    if missing:
        pytest.skip(f"set {', '.join(missing)} to run this live test")
    return values


@pytest.fixture(autouse=True)
def _no_cassette_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cassette mode left in the shell must not turn these cassette-less runs into replays."""
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", raising=False)


def test_a_live_sigv4_runtime_run_returns_an_owned_trace_with_the_expected_agent_set(
    live_budget: Any, tmp_path: Path
) -> None:
    runtime_arn, payload_format, question, expected = _required(
        RUNTIME_ARN, PAYLOAD_FORMAT, QUESTION, EXPECTED_AGENTS
    )
    import calibrated_agent_evaluation.agentcore as agentcore
    from calibrated_agent_evaluation.agentcore import (
        AgentCoreRuntimeExecutor,
        AgentCoreRuntimeTask,
        AwsClients,
        RuntimeTarget,
    )
    from calibrated_agent_evaluation.agentcore.run_manifest import RUN_RECORD_STORE
    from calibrated_agent_evaluation.schema.validate import validate

    reader_type = getattr(agentcore, "AgentCoreSpanReader", None)
    if reader_type is None:
        pytest.skip(
            "needs the AgentCore span reader, "
            "calibrated_agent_evaluation.agentcore.AgentCoreSpanReader"
        )
    expected_agents = {name.strip() for name in expected.split(",") if name.strip()}
    target = RuntimeTarget(
        runtime_arn=runtime_arn, qualifier=_QUALIFIER, payload_format=payload_format
    )
    used_before = live_budget.used

    # One set of clients for the reader and the executor, so one budget covers both.
    with (
        AwsClients.from_profile(region=target.region, handlers=[live_budget]) as clients,
        AgentCoreRuntimeExecutor(
            target, span_reader=reader_type(clients), clients=clients, log_dir=tmp_path
        ) as executor,
    ):
        [trace] = executor.run(AgentCoreRuntimeTask(prompt=question))

    validate(trace)
    assert trace.ownership == "owned"
    assert trace.input == {"prompt": question}
    assert trace.output["text"]
    assert {span.agent for span in trace.spans} == expected_agents
    [record] = executor.run_records
    assert record.trace_id == trace.trace_id
    # The trace names the runtime session the agent ran under, the one the run record kept; the
    # span reader only returns a tree whose spans all carry that session.
    assert trace.session_id is not None
    assert trace.session_id == record.session_id
    assert record.runtime_version
    assert record.runtime_version_source == "endpoint"
    assert record.session_stop == "stopped"
    assert record.error_code is None
    # The run manifest in the log directory holds this one run, keyed by the trace's ID.
    assert [stored.trace_id for stored in RUN_RECORD_STORE.load(tmp_path, fmt="json")] == [
        trace.trace_id
    ]
    assert live_budget.used > used_before, "the invocation must count against the budget"


# The most attempts the live probe's one prompt may send, and so the most billed calls it may make.
_PROBE_ATTEMPTS = 3


def test_a_live_sigv4_runtime_probe_returns_a_probed_trace_named_after_its_run(
    live_budget: Any, tmp_path: Path
) -> None:
    runtime_arn, payload_format, question = _required(RUNTIME_ARN, PAYLOAD_FORMAT, QUESTION)
    from calibrated_agent_evaluation.agentcore import (
        AwsClients,
        BilledCallBudget,
        RuntimeProbeAdapter,
        RuntimeTarget,
    )
    from calibrated_agent_evaluation.agentcore.run_manifest import RUN_RECORD_STORE
    from calibrated_agent_evaluation.probe import endpoint_fingerprint, probe_with
    from calibrated_agent_evaluation.schema.validate import validate

    target = RuntimeTarget(
        runtime_arn=runtime_arn, qualifier=_QUALIFIER, payload_format=payload_format
    )
    # This test's own cap, under the run's: its one prompt bills at most its attempt limit.
    prompt_budget = BilledCallBudget(_PROBE_ATTEMPTS)

    with (
        AwsClients.from_profile(
            region=target.region, handlers=[live_budget, prompt_budget]
        ) as clients,
        RuntimeProbeAdapter(
            target, clients=clients, log_dir=tmp_path, attempt_limit=_PROBE_ATTEMPTS
        ) as adapter,
    ):
        report = probe_with(
            adapter,
            target=adapter.target_label,
            authorized=True,
            seeds=[question],
            payload_format=payload_format,
        )

    [trace] = report.traces
    validate(trace)
    assert trace.ownership == "probed"
    assert trace.spans == []
    assert trace.input == {"prompt": question}
    assert trace.output["text"]
    fingerprint = endpoint_fingerprint(f"{runtime_arn}@{_QUALIFIER}")
    assert trace.endpoint_fingerprint == fingerprint
    [record] = adapter.run_records
    assert trace.trace_id == f"probe-{fingerprint}-{record.w3c_trace_id}"
    assert record.trace_id == trace.trace_id
    assert record.runtime_version
    assert record.session_stop == "stopped"
    assert record.error_code is None
    # The run manifest in the log directory holds this one prompt's run, keyed by the trace's ID.
    assert [stored.trace_id for stored in RUN_RECORD_STORE.load(tmp_path, fmt="json")] == [
        trace.trace_id
    ]
    assert 1 <= prompt_budget.used <= _PROBE_ATTEMPTS
