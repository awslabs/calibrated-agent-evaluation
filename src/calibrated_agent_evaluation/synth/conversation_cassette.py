# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record and replay the simulated user's turns, so a whole conversation can run offline.

The agent under test is replayed by its own executor cassette; this file records the other side.
Each simulated-user turn is keyed by the scenario id, the turn it starts (2 for the user's first
reply, since the opening message is turn 1's) and a SHA-256 of the agent reply it answers. A
customer who says "yes" on two turns is therefore recorded twice and replayed in order, and a replay
whose agent said something different from the recording is refused instead of answered with a
message written for another conversation.

The recording lives next to the agent cassette, ``<stem>.actor.json``, so a run keeps one
``--cassette`` flag. Modes follow the executor cassettes:

* ``replay``: every turn comes from the recording; the real simulated user is never built, so no
  model is called. A missing or diverged turn raises :class:`ConversationReplayDiverged`.
* ``record``: the real simulated user answers every turn, and the scenario's earlier recording is
  replaced.
* ``auto``: a scenario that has a recording replays it; one with none is recorded whole. A real
  simulated user is never built part way through a conversation, since it would not know the
  turns before it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

from calibrated_agent_evaluation._recordstore import atomic_write_text
from calibrated_agent_evaluation.execute.cassette import Mode
from calibrated_agent_evaluation.synth.conversation import SimulatedUser, StopReason, UserTurn
from calibrated_agent_evaluation.synth.errors import ConversationReplayDiverged, SynthError

__all__ = ["UserCassette", "actor_cassette_path", "reply_digest"]

SCHEMA = "calibrated-agent-evaluation-actor-cassette/1"
_AGENT_SUFFIX = ".cassette.json"


def actor_cassette_path(cassette_path: str | Path) -> Path:
    """The simulated-user recording that goes with an agent cassette.

    ``runs.cassette.json`` pairs with ``runs.actor.json``; any other name gets ``.actor.json`` in
    place of its suffix.
    """
    path = Path(cassette_path)
    name = path.name
    if name.endswith(_AGENT_SUFFIX):
        return path.with_name(name[: -len(_AGENT_SUFFIX)] + ".actor.json")
    return path.with_suffix(".actor.json")


def reply_digest(agent_reply: str) -> str:
    """The SHA-256 a recorded user turn is keyed by: the agent reply it answered."""
    return hashlib.sha256(agent_reply.encode("utf-8")).hexdigest()


class UserCassette:
    """The simulated-user recording for one run, shared by every conversation in it."""

    def __init__(self, path: str | Path, mode: Mode | str = Mode.AUTO) -> None:
        """Open the recording at ``path`` (it need not exist yet).

        Raises:
            SynthError: ``mode`` is not a cassette mode, replay was asked for with no file, or the
                file is not a readable recording.
        """
        try:
            self._mode = Mode(mode)
        except ValueError as exc:
            raise SynthError(f"{mode!r} is not a cassette mode") from exc
        self._path = Path(path)
        self._turns: dict[str, list[dict[str, Any]]] = self._load()
        self._dirty = False

    @property
    def path(self) -> Path:
        """Where the recording is read from and written to."""
        return self._path

    def user_for(self, scenario_id: str, build: Callable[[], SimulatedUser]) -> SimulatedUser:
        """The simulated user for one conversation: replayed, recorded, or both by mode.

        Args:
            scenario_id: The conversation's scenario id.
            build: Builds the real simulated user; called only when this conversation records.

        Returns:
            A simulated user that replays or records this conversation's turns.
        """
        recorded = self._turns.get(scenario_id, [])
        replaying = self._mode is Mode.REPLAY or (self._mode is Mode.AUTO and bool(recorded))
        if replaying:
            return _ReplayingUser(scenario_id, list(recorded))
        self._turns[scenario_id] = []
        self._dirty = True
        return _RecordingUser(scenario_id, build(), self._turns[scenario_id])

    def save(self) -> None:
        """Write the recording if any conversation recorded; atomic.

        Call it only after every conversation of the run has finished: a run that fails stores
        nothing, and a partly recorded conversation saved anyway would later replay and diverge.

        Raises:
            SynthError: The file cannot be written.
        """
        if not self._dirty:
            return
        doc = {"schema": SCHEMA, "turns": [t for turns in self._turns.values() for t in turns]}
        text = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self._path, text)
        except OSError as exc:
            raise SynthError(
                f"cannot write the simulated-user recording {self._path}: {exc.strerror or exc}"
            ) from exc
        self._dirty = False

    def _load(self) -> dict[str, list[dict[str, Any]]]:
        if not self._path.exists():
            if self._mode is Mode.REPLAY:
                raise SynthError(
                    f"replay needs the simulated-user recording {self._path}, which does not "
                    "exist; record the conversations first"
                )
            return {}
        try:
            doc = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SynthError(
                f"cannot read the simulated-user recording {self._path}: {exc}"
            ) from exc
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            raise SynthError(
                f"{self._path} is not a simulated-user recording (schema {SCHEMA!r} expected)"
            )
        entries = doc.get("turns", [])
        if not isinstance(entries, list):
            raise SynthError(f"{self._path}: 'turns' must be a list of recorded turns")
        turns: dict[str, list[dict[str, Any]]] = {}
        seen: set[tuple[str, int]] = set()
        for entry in entries:
            _check_entry(entry, self._path)
            key = (entry["scenario_id"], entry["turn"])
            if key in seen:
                raise SynthError(
                    f"{self._path} records turn {entry['turn']} of {entry['scenario_id']!r} twice"
                )
            seen.add(key)
            turns.setdefault(entry["scenario_id"], []).append(entry)
        for entries in turns.values():
            entries.sort(key=lambda e: e["turn"])
        return turns


class _ReplayingUser:
    def __init__(self, scenario_id: str, recorded: list[dict[str, Any]]) -> None:
        self._scenario_id = scenario_id
        self._by_turn = {e["turn"]: e for e in recorded}
        self._turn = 1

    def act(self, agent_message: str) -> UserTurn:
        self._turn += 1
        entry = self._by_turn.get(self._turn)
        if entry is None:
            raise ConversationReplayDiverged(
                f"replay of {self._scenario_id!r} has no recorded simulated-user turn "
                f"{self._turn}; the conversation went further than the recording, so record it "
                "again",
                scenario_id=self._scenario_id,
                turn=self._turn,
                side="user",
            )
        if entry["agent_reply_sha256"] != reply_digest(agent_message):
            raise ConversationReplayDiverged(
                f"replay of {self._scenario_id!r} diverged before turn {self._turn}: the agent's "
                "reply differs from the one the recorded simulated-user turn answered, so record "
                "the conversation again",
                scenario_id=self._scenario_id,
                turn=self._turn,
                side="agent",
            )
        return UserTurn(
            message=entry["message"], stop=entry["stop"], stop_reason=entry["stop_reason"]
        )


class _RecordingUser:
    def __init__(self, scenario_id: str, user: SimulatedUser, sink: list[dict[str, Any]]) -> None:
        self._scenario_id = scenario_id
        self._user = user
        self._sink = sink
        self._turn = 1

    def act(self, agent_message: str) -> UserTurn:
        self._turn += 1
        move = self._user.act(agent_message)
        self._sink.append(
            {
                "scenario_id": self._scenario_id,
                "turn": self._turn,
                "agent_reply_sha256": reply_digest(agent_message),
                "message": move.message,
                "stop": move.stop,
                "stop_reason": move.stop_reason,
            }
        )
        return move


_ENTRY_KEYS = {"scenario_id", "turn", "agent_reply_sha256", "message", "stop", "stop_reason"}


_STOP_REASONS = (None, *get_args(StopReason))
_HEX = frozenset("0123456789abcdef")


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _check_entry(entry: Any, path: Path) -> None:
    """Refuse a recorded turn whose keys or value types are not a recording's."""
    if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
        raise SynthError(f"{path} has a turn that is not a simulated-user recording: {entry!r}")
    turn = entry["turn"]
    problems = [
        name
        for name, ok in (
            ("scenario_id", isinstance(entry["scenario_id"], str) and bool(entry["scenario_id"])),
            ("turn", isinstance(turn, int) and not isinstance(turn, bool) and turn >= 2),
            ("agent_reply_sha256", _is_digest(entry["agent_reply_sha256"])),
            ("message", entry["message"] is None or isinstance(entry["message"], str)),
            ("stop", isinstance(entry["stop"], bool)),
            ("stop_reason", entry["stop_reason"] in _STOP_REASONS),
        )
        if not ok
    ]
    if problems:
        raise SynthError(
            f"{path} has a recorded turn with an invalid {', '.join(problems)}: {entry!r}"
        )
