# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live integration legs for the flagship proof.

Opt-in (``@pytest.mark.integration``): these are the two acceptance legs that keep the hermetic
frozen-fixture proof honest.

  * Both worlds: with a running self-hosted Langfuse stack (``just langfuse-up``), the proof
    runs against BOTH the local and Langfuse worlds and asserts field-for-field parity.
  * Live Bedrock: with AWS creds (``AWS_PROFILE=<your-profile>``), the headline traces re-execute
    against a live model through the shipped executors -- the model-plane sign-off leg.

Both are MANDATORY for the M1 sign-off; they skip cleanly here when the
stack/creds are absent so the default hermetic suite stays offline.

Run: ``just test-integration tests/integration/test_flagship_live.py`` (Langfuse env preset), or
     ``AWS_PROFILE=<your-profile> uv run pytest -m integration
     tests/integration/test_flagship_live.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from _bedrock import inspect_bedrock_model

from calibrated_agent_evaluation.backends import TraceStoreBackend, get_backend
from calibrated_agent_evaluation.flagship import FlagshipConfig, World, prove
from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError

_REPO = Path(__file__).resolve().parent.parent.parent
_FIXTURES = _REPO / "fixtures"


@pytest.mark.integration
def test_both_worlds_parity_over_running_langfuse(tmp_path: Path) -> None:
    """The OrderBot proof runs both worlds and asserts field-for-field parity."""
    from _flagship_helpers import orderbot_plan

    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-dev")
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-calibrated-agent-evaluation-dev")

    from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend

    # Skip cleanly if the stack is not reachable (the leg is opt-in; MANDATORY only at sign-off).
    try:
        LangfuseBackend.from_credentials(pk, sk, host, timeout_seconds=5)
    except CalibratedAgentEvaluationDataError as exc:  # pragma: no cover -- environment gate
        pytest.skip(f"Langfuse stack not reachable at {host}: {exc}")

    def factory(world: World) -> TraceStoreBackend:
        if world == "local":
            return get_backend(None, local_root=tmp_path / "backend")
        return LangfuseBackend.from_credentials(pk, sk, host, timeout_seconds=15)

    config = FlagshipConfig(
        agents=("orderbot",),
        worlds=("local", "langfuse"),
        model_plane="cassette",
        fixtures_root=str(_FIXTURES),
    )
    report = prove(config, [orderbot_plan()], backend_factory=factory)
    parity = next(p for p in report.parity if p.agent == "orderbot")
    assert parity.equal, f"both-worlds divergence at {parity.divergence} (a sign-off blocker)"
    assert report.verdict == "proven", report.reasons


@pytest.mark.integration
def test_live_bedrock_reexecutes_headline_traces(tmp_path: Path) -> None:
    """The headline traces re-execute against a live model (the sign-off leg).

    Runs the shipped InspectExecutor against live Bedrock (no cassette) for ORD-1001; the returned
    Trace is validated as the model-plane parity evidence. Skips without AWS creds.
    """
    if not (os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID")):
        pytest.skip("live-Bedrock leg needs AWS credentials (AWS_PROFILE=<your-profile>)")
    pytest.importorskip("inspect_ai")
    pytest.importorskip("boto3")

    from corpus.agents.orderbot_inspect import orderbot_fail

    from calibrated_agent_evaluation.execute import InspectExecutor
    from calibrated_agent_evaluation.schema.validate import validate

    model = inspect_bedrock_model()
    with InspectExecutor(model=model, log_dir=tmp_path) as executor:
        traces = executor.run(orderbot_fail)
    assert traces, "live-Bedrock re-execution produced no trace"
    for trace in traces:
        validate(trace)  # the model-plane parity evidence: a valid normalized Trace
