# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live proof: ``calibrated-agent-evaluation generate --run --multi-turn`` against real
Bedrock.

Holds one short conversation (two agent turns) between the real ``strands_evals`` ``ActorSimulator``
and a real one-agent Strands topology on Bedrock, through the shipped CLI. What only a live run can
show: the simulator's user turns are real model output, each agent turn is a genuine Strands run
over the history so far, and the stored conversation is the final turn's normalized trace with its
frame and taxonomy cell intact.

The agent's model calls go through a cassette, so the run leaves a recording of what Bedrock
returned. The simulator's own calls are not cassetted (it builds its own agent), so its turns differ
run to run and the cassette cannot drive a later offline conversation; the offline coverage is the
stub-executor suite in ``tests/test_synth_multiturn.py``.

Run: ``AWS_PROFILE=<your-profile> uv run pytest -m integration -k multiturn_live``
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_OK

pytestmark = pytest.mark.integration

_TESTS = Path(__file__).resolve().parent.parent
_MAX_TURNS = 2


def _require_bedrock() -> None:
    pytest.importorskip("strands")
    pytest.importorskip("strands_evals")
    if not (os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID")):
        pytest.skip(
            "no AWS credentials configured (set AWS_PROFILE to a profile with Bedrock access)"
        )


def test_live_a_multi_turn_run_stores_the_conversation_the_agent_actually_held(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _require_bedrock()
    # `cli_agents` lives in tests/, which pytest puts on sys.path; the CLI imports it by name.
    if str(_TESTS) not in sys.path:
        sys.path.insert(0, str(_TESTS))

    # A one-row frame, the shape `calibrated-agent-evaluation define --out` writes, so the live run
    # is one conversation and a handful of model calls rather than a whole pairwise grid.
    frame_file = tmp_path / "frame.json"
    frame_file.write_text(
        json.dumps(
            {
                "frame": "discovery",
                "coverage_strength": 2,
                "rows": [
                    {
                        "cell": {"feature": "refund", "persona": "first_time"},
                        "frame": "discovery",
                        "weight": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cassette = tmp_path / "conversation.cassette.json"
    root = tmp_path / "store"

    code = main(
        [
            "--root",
            str(root),
            "generate",
            "--run",
            "--executor",
            "strands",
            "--multi-turn",
            "--max-turns",
            str(_MAX_TURNS),
            "--agent",
            "cli_agents:support_agent",
            "--frame",
            str(frame_file),
            "--cassette",
            str(cassette),
        ]
    )
    captured = capsys.readouterr()
    assert code == EXIT_OK, captured.err
    assert f"up to {_MAX_TURNS} agent turn(s)" in captured.out
    assert "a persona per taxonomy cell" in captured.out

    with LocalBackend(root=str(root)) as backend:
        traces = backend.query()
    assert len(traces) == 1
    [conversation] = traces
    assert conversation.frame == "discovery"
    assert conversation.taxonomy_cell == {"feature": "refund", "persona": "first_time"}
    assert conversation.ownership == "owned"
    assert conversation.spans, "a live Strands run emits at least the agent's own span"

    # The stored trace is the LAST turn and carries the whole transcript: the simulator's user
    # turns and the agent's own replies, in order.
    transcript = conversation.input["conversation"]
    assert [turn["role"] for turn in transcript] == ["user", "assistant"] * _MAX_TURNS, transcript
    assert transcript[0]["text"].startswith("A customer request exercising")
    # The simulator's second turn is real model output, not a repeat of the opening line.
    assert transcript[2]["text"] not in {transcript[0]["text"], ""}
    assert all(turn["text"].strip() for turn in transcript)

    # The agent's turns really went to Bedrock through the cassette, one entry per model call.
    recorded = json.loads(cassette.read_text(encoding="utf-8"))
    entries = recorded.get("entries", recorded)
    assert len(entries) >= _MAX_TURNS, recorded
