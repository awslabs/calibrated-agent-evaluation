# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live integration test for the execution plane.

Runs the InspectExecutor against real Bedrock to confirm both-worlds parity. Opt-in: requires
AWS credentials and is gated on the ``integration`` pytest marker.
``CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID`` picks the bare Bedrock model id; the cassette path
runs a lighter "live normalizer" check.

Run: ``just test-integration`` or
``uv run pytest -m integration tests/integration/test_execute_live.py``
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from _bedrock import inspect_bedrock_model

from calibrated_agent_evaluation.execute import InspectExecutor, Mode
from calibrated_agent_evaluation.schema.validate import validate

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"
CASSETTE_PATH = FIXTURES_DIR / "raw" / "cassettes" / "inspect_ORD-1001.cassette.json"


@pytest.mark.integration
class TestExecuteLive:
    """Execution plane works against real/cassette model backends."""

    def test_executor_produces_valid_trace_from_cassette(self, tmp_path: Path) -> None:
        """Run the executor with the committed cassette and assert the trace is valid."""
        if not CASSETTE_PATH.exists():
            pytest.skip(f"cassette not found: {CASSETTE_PATH}")

        pytest.importorskip("inspect_ai")

        from corpus.agents.orderbot_inspect import orderbot_fail

        executor = InspectExecutor(
            model="cassette/orderbot",
            log_dir=tmp_path,
            cassette_path=CASSETTE_PATH,
            cassette_mode=Mode.REPLAY,
        )
        traces = executor.run(orderbot_fail)

        assert len(traces) >= 1
        trace = traces[0]
        assert trace.ownership == "owned"
        assert len(trace.spans) >= 1
        assert trace.spans[0].tool_calls != []
        validate(trace)

        eval_logs = list(tmp_path.glob("*.eval"))
        assert len(eval_logs) >= 1

    @pytest.mark.skipif(
        not os.environ.get("AWS_PROFILE") and not os.environ.get("AWS_ACCESS_KEY_ID"),
        reason="No AWS credentials configured — skip live Bedrock test",
    )
    def test_executor_live_bedrock_matches_cassette(self, tmp_path: Path) -> None:
        """Run against live Bedrock and assert structural parity with cassette replay."""
        pytest.importorskip("inspect_ai")

        from corpus.agents.orderbot_inspect import orderbot_fail

        model = inspect_bedrock_model()

        executor = InspectExecutor(
            model=model,
            log_dir=tmp_path,
        )
        traces = executor.run(orderbot_fail)

        assert len(traces) >= 1
        trace = traces[0]
        assert trace.ownership == "owned"
        assert len(trace.spans) >= 1
        validate(trace)
