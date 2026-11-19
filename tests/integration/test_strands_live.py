# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live Bedrock record/replay for StrandsExecutor.

Marked `integration` (not run in the default hermetic suite): it records the ClaimsCrew Graph
against live Bedrock into a fresh cassette, then replays it offline through StrandsExecutor and
asserts the same owned Trace — keeping the executor honest against Bedrock/Strands SDK drift. Run
with `just test-integration -k strands` (needs the `[strands]` extra and
`AWS_PROFILE=<your-profile>`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _claims_task(trace_id: str = "CLM-4471") -> Any:
    from corpus.agents.claimscrew_strands import build_claims_graph
    from corpus.generate.scenarios import load_scenario

    from calibrated_agent_evaluation.execute import StrandsTask

    scenario = load_scenario(trace_id)
    ti = scenario.trace_input
    crew_input = (
        f"Adjudicate claim {ti.get('claim_id', trace_id)} "
        f"for ${ti.get('amount_requested_usd', '')}. Narrative: {ti.get('narrative', '')}"
    )
    return StrandsTask(
        build=lambda model: build_claims_graph(model, scenario),
        input=crew_input,
        mode="legacy",
        trace_id=trace_id,
        taxonomy_cell=dict(scenario.taxonomy_cell),
    )


def test_record_then_replay_reproduces_the_trace(tmp_path: Path) -> None:
    pytest.importorskip("strands")
    from calibrated_agent_evaluation.execute import Mode, StrandsExecutor
    from calibrated_agent_evaluation.schema.validate import validate

    cassette = tmp_path / "live.cassette.json"

    # RECORD against live Bedrock.
    with StrandsExecutor(cassette_path=cassette, cassette_mode=Mode.RECORD) as ex:
        recorded = ex.run(_claims_task())[0]
    assert recorded.ownership == "owned"
    assert cassette.exists()
    assert cassette.stat().st_size > 0
    validate(recorded)

    # REPLAY the just-recorded cassette offline — same structural trace, no live call.
    with StrandsExecutor(cassette_path=cassette, cassette_mode=Mode.REPLAY) as ex:
        replayed = ex.run(_claims_task())[0]
    assert {s.agent for s in replayed.spans} == {s.agent for s in recorded.spans}
    assert replayed.ownership == "owned"
