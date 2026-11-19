# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recording and replaying the simulated user's side of a conversation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.synth.conversation import (
    ConversationTurn,
    UserTurn,
    drive_conversation,
)
from calibrated_agent_evaluation.synth.conversation_cassette import (
    UserCassette,
    actor_cassette_path,
)
from calibrated_agent_evaluation.synth.errors import ConversationReplayDiverged, SynthError

OPENING = "I want a refund for order A-7781."


class SayYesTwice:
    """A simulated user who says "yes" twice, then stops with the goal met."""

    def __init__(self) -> None:
        self.calls = 0

    def act(self, agent_message: str) -> UserTurn:
        self.calls += 1
        if self.calls <= 2:
            return UserTurn(message="yes", stop=False)
        return UserTurn(message=None, stop=True, stop_reason="goal_completed")


def _agent(turn: ConversationTurn, history: list[dict[str, Any]]) -> str:
    return f"Shall I go ahead? (turn {turn.turn}, {len(history)} messages so far)"


def _never_build() -> Any:
    raise AssertionError("the real simulated user was built during replay")


def _converse(cassette: UserCassette, build: Any = SayYesTwice, agent: Any = _agent) -> Any:
    user = cassette.user_for("ORD-1001-sim", build)
    return drive_conversation(
        user, agent, conversation_id="ORD-1001-sim", opening=OPENING, max_turns=5
    )


def _record(path: Path) -> Any:
    cassette = UserCassette(path, "record")
    history, reason = _converse(cassette)
    cassette.save()
    return history, reason


def test_recorded_user_turns_replay_in_order_even_when_repeated(tmp_path: Path) -> None:
    path = tmp_path / "run.actor.json"
    recorded_history, recorded_reason = _record(path)
    doc = json.loads(path.read_text())
    assert [(t["turn"], t["message"]) for t in doc["turns"]] == [(2, "yes"), (3, "yes"), (4, None)]

    replayed_history, replayed_reason = _converse(UserCassette(path, "replay"), _never_build)
    assert replayed_history == recorded_history
    assert replayed_reason == recorded_reason == "goal_completed"


def test_replay_refuses_a_turn_it_never_recorded(tmp_path: Path) -> None:
    path = tmp_path / "run.actor.json"
    _record(path)
    doc = json.loads(path.read_text())
    doc["turns"] = [t for t in doc["turns"] if t["turn"] != 4]
    path.write_text(json.dumps(doc))
    with pytest.raises(
        ConversationReplayDiverged, match="no recorded simulated-user turn 4"
    ) as exc:
        _converse(UserCassette(path, "replay"), _never_build)
    assert (exc.value.scenario_id, exc.value.turn, exc.value.side) == ("ORD-1001-sim", 4, "user")


def test_replay_refuses_when_the_agent_says_something_else(tmp_path: Path) -> None:
    path = tmp_path / "run.actor.json"
    _record(path)

    def changed_agent(turn: ConversationTurn, history: list[dict[str, Any]]) -> str:
        return "Something new." if turn.turn == 2 else _agent(turn, history)

    with pytest.raises(ConversationReplayDiverged, match="diverged before turn 3") as exc:
        _converse(UserCassette(path, "replay"), _never_build, changed_agent)
    assert (exc.value.turn, exc.value.side) == (3, "agent")


def test_replay_never_builds_the_simulated_user(tmp_path: Path) -> None:
    path = tmp_path / "run.actor.json"
    _record(path)
    # Auto mode with a recording replays it, and does not build the real user either.
    history, _reason = _converse(UserCassette(path, "auto"), _never_build)
    assert [message["role"] for message in history][:3] == ["user", "assistant", "user"]
    # Replay with no recording at all is refused before any conversation starts.
    with pytest.raises(SynthError, match="does not exist"):
        UserCassette(tmp_path / "missing.actor.json", "replay")


def test_an_unreadable_recording_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "run.actor.json"
    path.write_text(json.dumps({"schema": "something-else", "turns": []}))
    with pytest.raises(SynthError, match="not a simulated-user recording"):
        UserCassette(path, "replay")
    path.write_text(
        json.dumps(
            {"schema": "calibrated-agent-evaluation-actor-cassette/1", "turns": [{"turn": 1}]}
        )
    )
    with pytest.raises(SynthError, match="not a simulated-user recording"):
        UserCassette(path, "replay")


def test_the_recording_sits_next_to_the_agent_cassette() -> None:
    assert actor_cassette_path("fixtures/raw/cassettes/orderbot_simulated.cassette.json") == Path(
        "fixtures/raw/cassettes/orderbot_simulated.actor.json"
    )
    assert actor_cassette_path("runs.json") == Path("runs.actor.json")


@pytest.mark.parametrize(
    ("turns", "complaint"),
    [
        (5, "must be a list"),
        (
            [
                {
                    "scenario_id": "ORD-1001-sim",
                    "turn": 2,
                    "agent_reply_sha256": "0" * 64,
                    "message": "yes",
                    "stop": False,
                    "stop_reason": "bored",
                }
            ],
            "invalid stop_reason",
        ),
        (
            [
                {
                    "scenario_id": "ORD-1001-sim",
                    "turn": 2,
                    "agent_reply_sha256": "0" * 64,
                    "message": 7,
                    "stop": "no",
                    "stop_reason": None,
                }
            ],
            "invalid message, stop",
        ),
        (
            [
                {
                    "scenario_id": "ORD-1001-sim",
                    "turn": 2,
                    "agent_reply_sha256": "not-a-digest",
                    "message": "yes",
                    "stop": False,
                    "stop_reason": None,
                }
            ],
            "invalid agent_reply_sha256",
        ),
    ],
)
def test_a_malformed_recording_is_refused_with_a_typed_error(
    tmp_path: Path, turns: Any, complaint: str
) -> None:
    path = tmp_path / "run.actor.json"
    path.write_text(
        json.dumps({"schema": "calibrated-agent-evaluation-actor-cassette/1", "turns": turns})
    )
    with pytest.raises(SynthError, match=complaint):
        UserCassette(path, "replay")


def test_a_recording_with_a_turn_twice_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "run.actor.json"
    _record(path)
    doc = json.loads(path.read_text())
    doc["turns"].append(dict(doc["turns"][0]))
    path.write_text(json.dumps(doc))
    with pytest.raises(SynthError, match="records turn 2 of 'ORD-1001-sim' twice"):
        UserCassette(path, "replay")


def test_auto_records_a_scenario_with_no_recording_and_record_replaces_one(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.actor.json"
    cassette = UserCassette(path, "auto")
    built = SayYesTwice()
    _converse(cassette, lambda: built)
    cassette.save()
    assert built.calls == 3
    assert len(json.loads(path.read_text())["turns"]) == 3

    class SayNo:
        def act(self, agent_message: str) -> UserTurn:
            return UserTurn(message=None, stop=True, stop_reason="goal_completed")

    again = UserCassette(path, "record")
    _converse(again, SayNo)
    again.save()
    assert [t["turn"] for t in json.loads(path.read_text())["turns"]] == [2]


def test_an_unwritable_recording_is_a_typed_error(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    cassette = UserCassette(blocker / "run.actor.json", "record")
    _converse(cassette)
    with pytest.raises(SynthError, match="cannot write the simulated-user recording"):
        cassette.save()


class SayThanksAndLeave:
    """A simulated user who stops with a last message instead of none."""

    def act(self, agent_message: str) -> UserTurn:
        return UserTurn(message="Thanks, that's all.", stop=True, stop_reason="goal_completed")


def test_a_goodbye_the_user_stops_with_is_recorded_and_replayed(tmp_path: Path) -> None:
    path = tmp_path / "run.actor.json"
    recording = UserCassette(path, "record")
    recorded = _converse(recording, SayThanksAndLeave)
    recording.save()
    [entry] = json.loads(path.read_text())["turns"]
    assert (entry["message"], entry["stop"]) == ("Thanks, that's all.", True)

    replayed = _converse(UserCassette(path, "replay"), _never_build)
    assert replayed == recorded
    history, reason = replayed
    assert reason == "goal_completed"
    assert [m["role"] for m in history] == ["user", "assistant", "user"]
