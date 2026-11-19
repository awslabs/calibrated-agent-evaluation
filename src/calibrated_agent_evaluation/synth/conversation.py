# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The simulated-conversation engine: one loop for every simulated multi-turn run.

A **simulated user** (an LLM playing the customer) and the **agent under test** take turns until the
user stops or a turn limit is hit. The user is reached through the vendor-free
:class:`SimulatedUser` port, so the loop never imports a simulator SDK; the ``strands_evals``
adapter lives in ``calibrated_agent_evaluation.synth.borrow`` and the AgentCore one in
``calibrated_agent_evaluation.agentcore``. The agent is reached through a caller-supplied
``agent_turn`` that runs one turn on the caller's executor and returns its trace.

The opening message is the scenario's (or the synthesized input's), so turn 1 needs no user call.
``max_turns`` bounds agent turns; once an agent turn reaches it, the user is not asked again. A
failure on either side is raised as :class:`ConversationError` naming the conversation, the turn and
the side, and the agent side's end hook runs in every case, so a runtime session is never left open.

:func:`assemble_conversation_trace` turns a finished run into **one** trace that keeps every turn's
spans (re-keyed ``t<n>:<span_id>`` so ids stay unique across turns) and records the transcript,
goal, assertions and stop reason in its ``input``. A conversation whose every turn is outcome-only
(``probed``, no spans) stays ``probed``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from calibrated_agent_evaluation._conversation import (
    ConversationTurn,
    SimulatedUser,
    StopReason,
    UserTurn,
)
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.schema.types import Frame, Span, Trace
from calibrated_agent_evaluation.schema.validate import validate
from calibrated_agent_evaluation.synth.errors import ConversationError, SynthError

__all__ = [
    "ConversationHooks",
    "ConversationRun",
    "ConversationTurn",
    "SimulatedUser",
    "Simulator",
    "StopReason",
    "UserTurn",
    "assemble_conversation_trace",
    "conversation_trace_id",
    "drive_conversation",
    "message_text",
    "reply_text",
    "run_simulated_conversation",
]

Simulator = Literal["strands", "agentcore"]

_USER = "user"
_AGENT = "assistant"

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversationHooks:
    """How a caller's executor and simulated user plug into ``generate_and_run``'s conversations.

    Attributes:
        turn_task: Builds one agent turn's executor task from the input, the history so far and
            the :class:`ConversationTurn` (an executor that holds a session per conversation needs
            the turn); ``None`` uses ``generate_and_run``'s two-argument ``turn_task``.
        end: Called with the conversation's id once it ends, fails or is interrupted (the
            Runtime executor's ``end_conversation``).
        user_for: Builds the simulated user from the input and its persona; ``None`` uses the
            ``strands_evals`` actor.
        user_for_conversation: Like ``user_for``, but also given the conversation's id (a
            simulated-user recording is keyed by it); used in place of ``user_for`` when set.
        simulator: Which simulated user plays the customer, recorded on each conversation trace.
    """

    turn_task: Callable[[Any, list[dict[str, Any]], ConversationTurn], Any] | None = None
    end: Callable[[str], object] | None = None
    user_for: Callable[[Any, Any], SimulatedUser] | None = None
    user_for_conversation: Callable[[Any, Any, str], SimulatedUser] | None = None
    simulator: Simulator = "strands"


@dataclass(frozen=True)
class ConversationRun:
    """A finished conversation: the per-turn traces, the transcript and why it stopped."""

    conversation_id: str
    turn_traces: tuple[Trace, ...]
    transcript: tuple[dict[str, str], ...]
    stop_reason: StopReason
    history: tuple[dict[str, Any], ...] = field(default=())


AgentReply = Callable[[ConversationTurn, list[dict[str, Any]]], str]
AgentTurn = Callable[[ConversationTurn, list[dict[str, Any]]], Trace]


def drive_conversation(
    user: SimulatedUser,
    agent_reply: AgentReply,
    *,
    conversation_id: str,
    opening: str,
    max_turns: int,
    on_end: Callable[[], object] | None = None,
) -> tuple[list[dict[str, Any]], StopReason]:
    """Alternate the simulated user and the agent until one of the stop rules fires.

    Args:
        user: The simulated user.
        agent_reply: Runs one agent turn on the history so far (Strands message shape,
            ``{"role", "content": [{"text"}]}``) and returns the agent's reply text.
        conversation_id: Names the conversation in errors and in each :class:`ConversationTurn`.
        opening: The first customer message.
        max_turns: The most agent turns, at least 1.
        on_end: Called once when the conversation ends, fails or is interrupted. When the
            conversation failed, an error from the hook is logged and added to the failure's
            notes, and the failure is raised; after a finished conversation the hook's error
            is raised.

    Returns:
        The history (both sides, in order) and the stop reason. A user who stops with a last
        message has it as the history's final entry, with no agent reply after it.

    Raises:
        SynthError: ``max_turns`` is below 1 or ``opening`` is empty.
        ConversationError: The user or the agent failed; the cause is chained.
    """
    if max_turns < 1:
        raise SynthError(f"a conversation needs max_turns of at least 1, got {max_turns}")
    if not opening.strip():
        raise SynthError(f"conversation {conversation_id!r} has an empty opening message")
    history: list[dict[str, Any]] = [_message(_USER, opening)]
    try:
        result = _converse_loop(user, agent_reply, conversation_id, max_turns, history)
    except BaseException as primary:
        _end_after_failure(on_end, conversation_id, primary)
        raise
    _end_after_success(on_end, conversation_id)
    return result


def _converse_loop(
    user: SimulatedUser,
    agent_reply: AgentReply,
    conversation_id: str,
    max_turns: int,
    history: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], StopReason]:
    turn = 0
    while True:
        turn += 1
        reply = _agent_step(agent_reply, conversation_id, turn, history)
        history.append(_message(_AGENT, reply))
        if turn >= max_turns:
            return history, "max_turns"
        move = _user_step(user, conversation_id, turn + 1, reply)
        text = (move.message or "").strip()
        if move.stop or not text:
            if text:
                # A customer who says goodbye as they stop said it; no agent turn answers it.
                history.append(_message(_USER, str(move.message)))
            return history, _stop_reason(move)
        history.append(_message(_USER, str(move.message)))


def _end_after_failure(
    on_end: Callable[[], object] | None, conversation_id: str, primary: BaseException
) -> None:
    """Run the end hook while ``primary`` propagates, without letting the hook's error replace it.

    The conversation's own failure is what the user must see; a hook that also fails (a session
    stop that cannot reach AWS) is logged and attached to it as a note.
    """
    if on_end is None:
        return
    try:
        on_end()
    except Exception as hook_exc:  # noqa: BLE001 - reported on the primary error, which is re-raised
        note = (
            f"ending conversation {conversation_id!r} after the failure also failed: "
            f"{type(hook_exc).__name__}: {hook_exc}"
        )
        _log.warning("%s", note)
        primary.add_note(note)


def _end_after_success(on_end: Callable[[], object] | None, conversation_id: str) -> None:
    """Run the end hook after a finished conversation, without throwing the conversation away.

    A hook that fails here (a session stop that cannot build its AWS client) is logged as a
    warning: the conversation finished and is kept, the executor tries the stop once more when it
    closes, and a runtime session left open ends at its idle timeout.
    """
    if on_end is None:
        return
    try:
        on_end()
    except Exception as hook_exc:  # noqa: BLE001 - the finished conversation is kept; logged
        _log.warning(
            "conversation %r finished, but ending it failed (%s: %s); the conversation is kept, "
            "and its runtime session is stopped on close or ends at its idle timeout",
            conversation_id,
            type(hook_exc).__name__,
            hook_exc,
        )


def run_simulated_conversation(
    user: SimulatedUser,
    agent_turn: AgentTurn,
    *,
    conversation_id: str,
    opening: str,
    max_turns: int,
    on_end: Callable[[], object] | None = None,
) -> ConversationRun:
    """Hold one conversation whose agent turns each return a trace; keep every turn's trace.

    Args:
        user: The simulated user.
        agent_turn: Runs one agent turn and returns its trace; the reply is read off the trace's
            text output.
        conversation_id: Names the conversation.
        opening: The first customer message.
        max_turns: The most agent turns, at least 1.
        on_end: Called once when the conversation ends, fails or is interrupted.

    Returns:
        The finished run.

    Raises:
        SynthError: ``max_turns`` is below 1 or ``opening`` is empty.
        ConversationError: The user or the agent failed, or an agent turn carried no text.
    """
    traces: list[Trace] = []

    def reply(turn: ConversationTurn, history: list[dict[str, Any]]) -> str:
        trace = agent_turn(turn, history)
        traces.append(trace)
        return reply_text(trace)

    history, stop_reason = drive_conversation(
        user,
        reply,
        conversation_id=conversation_id,
        opening=opening,
        max_turns=max_turns,
        on_end=on_end,
    )
    transcript = tuple({"role": m["role"], "text": message_text(m)} for m in history)
    return ConversationRun(
        conversation_id=conversation_id,
        turn_traces=tuple(traces),
        transcript=transcript,
        stop_reason=stop_reason,
        history=tuple(history),
    )


def assemble_conversation_trace(
    run: ConversationRun,
    *,
    scenario_id: str,
    goal: str,
    assertions: Sequence[str],
    simulator: Simulator,
    frame: Frame,
    taxonomy_cell: Mapping[str, str],
    dataset: Mapping[str, str] | None = None,
) -> Trace:
    """Turn a finished run into one trace that keeps every turn's spans.

    Args:
        run: The finished conversation.
        scenario_id: The scenario (or synthesized input) the conversation came from.
        goal: The simulated user's goal.
        assertions: The scenario's assertions, in order.
        simulator: Which simulated user played the customer.
        frame: The trace's frame.
        taxonomy_cell: The trace's taxonomy cell.
        dataset: ``{"dataset_id", "version"}`` of the published slice, or ``None``.

    Returns:
        The conversation trace, validated.

    Raises:
        ConversationError: The run has no agent turn.
        SchemaError: The assembled trace breaks a schema rule.
    """
    if not run.turn_traces:
        raise ConversationError(
            f"conversation {run.conversation_id!r} has no agent turn, so there is nothing to store",
            scenario_id=scenario_id,
            turn=0,
            side="agent",
        )
    first = run.turn_traces[0]
    trace_id = conversation_trace_id(scenario_id, first.captured_at, run.transcript)
    spans = _turn_spans(run.turn_traces, trace_id)
    agent_turns = sum(1 for m in run.transcript if m["role"] == _AGENT)
    conversation_input: dict[str, Any] = {
        "conversation": [dict(m) for m in run.transcript],
        "scenario_id": scenario_id,
        "goal": goal,
        "assertions": list(assertions),
        "stop_reason": run.stop_reason,
        "turns": agent_turns,
        "simulator": simulator,
    }
    if dataset is not None:
        conversation_input["dataset"] = {
            "dataset_id": dataset["dataset_id"],
            "version": dataset["version"],
        }
    probed = all(t.ownership == "probed" for t in run.turn_traces)
    trace = Trace(
        trace_id=trace_id,
        frame=frame,
        ownership="probed" if probed else "owned",
        input=conversation_input,
        # The agent's last reply, not a customer's goodbye that may follow it.
        output={"text": next(m["text"] for m in reversed(run.transcript) if m["role"] == _AGENT)},
        taxonomy_cell=dict(taxonomy_cell),
        spans=spans,
        captured_at=first.captured_at,
        endpoint_fingerprint=first.endpoint_fingerprint if probed else None,
    )
    validate(trace)
    return trace


def conversation_trace_id(
    scenario_id: str, started_at: str | None, transcript: Sequence[Mapping[str, str]]
) -> str:
    """A conversation's trace id: ``sim-<scenario_id>-<12 hex>`` from its start and transcript.

    Content-addressed, so replaying a recorded conversation stores the same id, and a conversation
    that went differently gets a different one.
    """
    doc = json.dumps(
        {"started_at": started_at, "transcript": [dict(m) for m in transcript]},
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(doc.encode("utf-8")).hexdigest()[:12]
    return f"sim-{scenario_id}-{digest}"


def reply_text(trace: Trace) -> str:
    """The agent's reply for one turn, read off its trace's text output.

    The normalized output is a dict whose shape depends on what the agent emitted, so every string
    value is joined in key order rather than one key being assumed. An output with no text is
    refused: the simulated user would otherwise answer an empty message.

    Raises:
        SynthError: The trace carries no text output.
    """
    texts = [str(value) for _key, value in sorted(trace.output.items()) if isinstance(value, str)]
    text = "\n".join(t for t in texts if t.strip())
    if not text.strip():
        raise SynthError(
            f"the agent's turn on trace {trace.trace_id!r} carried no text output, so the "
            "conversation cannot continue"
        )
    return text


def message_text(message: Mapping[str, Any]) -> str:
    """The text of one history message, whatever content blocks it was built from."""
    blocks = message.get("content") or []
    return "\n".join(
        str(block["text"]) for block in blocks if isinstance(block, dict) and "text" in block
    )


def _message(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "content": [{"text": text}]}


def _agent_step(
    agent_reply: AgentReply, conversation_id: str, turn: int, history: list[dict[str, Any]]
) -> str:
    try:
        return agent_reply(ConversationTurn(conversation_id, turn), list(history))
    except ConversationError:
        raise
    except Exception as exc:  # noqa: BLE001 - any agent or simulator failure becomes a typed ConversationError
        detail = str(exc) if isinstance(exc, SynthError) else f"{type(exc).__name__}: {exc}"
        raise ConversationError(
            f"agent run failed on turn {turn} of conversation {conversation_id!r}: {detail}",
            scenario_id=conversation_id,
            turn=turn,
            side="agent",
        ) from _root(exc)


def _user_step(user: SimulatedUser, conversation_id: str, turn: int, reply: str) -> UserTurn:
    try:
        return user.act(reply)
    except ConversationError:
        raise
    except Exception as exc:  # noqa: BLE001 - any agent or simulator failure becomes a typed ConversationError
        detail = str(exc) if isinstance(exc, SynthError) else f"{type(exc).__name__}: {exc}"
        raise ConversationError(
            f"input simulator failed before turn {turn} of conversation {conversation_id!r}: "
            f"{detail}",
            scenario_id=conversation_id,
            turn=turn,
            side="user",
        ) from _root(exc)


def _root(exc: Exception) -> BaseException:
    """The vendor or executor error behind ``exc``.

    An adapter's ``SynthError`` is a wrapper whose message the conversation error already quotes,
    so the chain points past it.
    """
    if isinstance(exc, SynthError) and exc.__cause__ is not None:
        return exc.__cause__
    return exc


def _stop_reason(move: UserTurn) -> StopReason:
    if move.stop and move.stop_reason is not None:
        return move.stop_reason
    return "actor_stopped"


def _turn_spans(turn_traces: Sequence[Trace], trace_id: str) -> list[Span]:
    """Every turn's spans, in turn order, re-keyed ``t<n>:`` and with seq kept increasing."""
    spans: list[Span] = []
    offset = 0
    for number, trace in enumerate(turn_traces, start=1):
        prefix = f"t{number}:"
        ordered = sorted(trace.spans, key=lambda s: s.seq)
        for span in ordered:
            spans.append(
                replace(
                    span,
                    span_id=prefix + span.span_id,
                    parent=None if span.parent is None else prefix + span.parent,
                    trace_id=trace_id,
                    seq=offset + span.seq,
                )
            )
        if ordered:
            offset = spans[-1].seq + 1
    return spans
