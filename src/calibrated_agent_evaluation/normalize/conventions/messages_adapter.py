# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``gen_ai/messages`` convention adapter.

The structured OTel GenAI convention emitted by Claude Code, current OpenLLMetry, and the
``Exgentic/agent-llm-traces`` dataset: message content rides as JSON strings in the span
attributes ``gen_ai.input.messages`` / ``gen_ai.output.messages``, each a list of
``{role, parts:[...]}`` where a tool invocation is an assistant ``tool_call`` part
(``{id, name, arguments}``) and its result is a later ``tool_call_response`` part
(``{id, result | response}``); paired by shared ``id`` (via :mod:`pairing`).

**Collapse-first tree policy.** A dataset session is a flat list of ``operation.name="chat"``
spans: the ``Exgentic/agent-llm-traces`` rows carry no ``parent_span_id`` at all, and a dataset
that does populate it may still point every span at one *out-of-session* parent, with no
``invoke_agent``/``execute_tool`` boundary spans. So this adapter always collapses the session
into a **one-span** ``owned`` Trace (a single root ``Span``, ``parent=null``, ``seq=0``) holding all
reconstructed ``tool_calls`` in call order; it has no multi-span branch. Under auto-detection a
trace that does carry boundary spans never reaches it: ``detect`` routes those to
``gen_ai/latest``, which builds the tree. Declaring this convention for such a trace
collapses it. It never emits N root-less sibling spans off a missing parent (that would
fail ``validate()``).

**Cross-span de-duplication (verified).** A harness that resends full conversation history each
turn repeats the same tool call across many spans (real session: 175 mentions, 38 unique ids). The
adapter keys calls and results by ``id`` across the whole session (last content wins), so each tool
invocation is reconstructed once, in first-seen call order.
"""

from __future__ import annotations

from typing import Any

from calibrated_agent_evaluation.normalize import content as content_mod
from calibrated_agent_evaluation.normalize import otlp_json
from calibrated_agent_evaluation.normalize.conventions import registry
from calibrated_agent_evaluation.normalize.conventions.pairing import (
    CONTENT_ID_PREFIX,
    RawToolCall,
    content_call_id,
    pair,
    split_content_call_id,
    trace_id_from,
)
from calibrated_agent_evaluation.normalize.errors import NormalizationError
from calibrated_agent_evaluation.normalize.sources import SessionSpans
from calibrated_agent_evaluation.normalize.tree import AgentNode, Intermediate

_ATTR_INPUT_MESSAGES = "gen_ai.input.messages"
_ATTR_OUTPUT_MESSAGES = "gen_ai.output.messages"
_ATTR_AGENT_NAME = "gen_ai.agent.name"
_ATTR_SYSTEM = "gen_ai.system"


def to_intermediate(session: SessionSpans) -> Intermediate:
    """Reconstruct the ``gen_ai/messages`` session into the vendor-free intermediate."""
    spans = sorted(session.spans, key=_start_time)
    calls, results_by_id, first_user_text, last_assistant_text = _walk_messages(spans)

    trace_id = session.trace_id or trace_id_from(spans) or "gen-ai-messages-trace"
    trace_input = {"message": first_user_text} if first_user_text else {}
    trace_output = {"answer": last_assistant_text} if last_assistant_text else {}

    tool_calls = pair(calls, results_by_id)
    # Ownership gate reads recovered content: no messages AND no tool calls → degrade to probed.
    if not tool_calls and not first_user_text and not last_assistant_text:
        return Intermediate(
            trace_id=trace_id,
            frame=session.frame,
            taxonomy_cell=dict(session.taxonomy_cell),
            trace_input=trace_input,
            trace_output=trace_output,
            nodes=[],
            degrade_reason=(
                "no gen_ai.input/output.messages content and no tool calls recovered from the "
                "gen_ai/messages session"
            ),
        )

    # Collapse-first: a trace that carries ONLY this convention's message attributes has no
    # agent/tool boundary spans to build a tree from, so the run is a single agent; a one-span tree
    # holding all reconstructed tool calls in call order.
    #
    # The stronger claim this comment used to make ("this convention has no boundary spans
    # (verified)") is false for mixed traces, which are the common case: a Traceloop-instrumented
    # crew emits `invoke_agent` boundary spans AND `gen_ai.*.messages`. Those no longer arrive here
    # at all; `detect`'s precedence rule routes them to `gen_ai/latest`, which builds the real tree.
    # So this collapse is correct for what still reaches it, and is not a statement about
    # the convention in general.
    agent = _agent_name(spans)
    started, ended = otlp_json.session_window(spans)
    node = AgentNode(
        agent=agent,
        role="specialist",
        parent=None,
        input=trace_input,
        output=trace_output,
        tool_calls=tool_calls,
        order=0,
        started_at=started,
        ended_at=ended,
        usage=otlp_json.session_usage(spans),
    )
    return Intermediate(
        trace_id=trace_id,
        frame=session.frame,
        taxonomy_cell=dict(session.taxonomy_cell),
        trace_input=trace_input,
        trace_output=trace_output,
        nodes=[node],
    )


def _walk_messages(
    spans: list[dict[str, Any]],
) -> tuple[list[RawToolCall], dict[str, Any], str, str]:
    """Collect deduped tool calls + results-by-id + terminal user/assistant text across the session.

    A history-resending harness repeats calls across spans; we key by ``id`` (first-seen call order,
    last content wins) so each invocation is reconstructed once. Terminal input is the
    earliest user text; terminal output is the latest assistant text.

    A call with **no** ``id`` gets a content-derived id (:func:`.pairing.content_call_id`)
    so it travels that same dedup path. The guard used to be ``if key is None or key not in
    seen_call_ids``, which appended unconditionally whenever the id was missing -- so every resend
    of the history counted the same call again, inflating the tool-call count by the number of
    turns, and each duplicate carried ``result={}`` because an id-less call joins to nothing.

    Occurrences are counted **per span and maxed**, never summed. That is what separates a resent
    history from a genuine repeat: a resending harness shows the same call once per span (max 1),
    while an agent that really called ``search`` twice in one turn shows it twice within a single
    span (max 2). Summing inflates the first; collapsing to one erases the second.
    """
    calls: list[RawToolCall] = []
    seen_call_ids: set[str] = set()
    results_by_id: dict[str, Any] = {}
    first_user_text = ""
    last_assistant_text = ""
    max_occurrences: dict[str, int] = {}

    for span in spans:
        attrs = otlp_json.attributes(span)
        # Ordered, not a set: an id-less RESULT is joined to its call by position within the span,
        # which is the only linkage a response carries (it has no name and no arguments).
        span_keys: list[str] = []
        span_results: list[Any] = []
        # Input messages carry the resent history: earliest user text is the terminal input, and
        # tool_call_responses (results) live here in a history-resending harness.
        for role, part in _iter_parts(attrs.get(_ATTR_INPUT_MESSAGES)):
            _absorb(part, results_by_id, calls, seen_call_ids, span_keys, span_results)
            if role == "user" and not first_user_text:
                first_user_text = _text_of(part) or first_user_text
        # Output messages carry this turn's assistant reply: its tool_calls and terminal text.
        for role, part in _iter_parts(attrs.get(_ATTR_OUTPUT_MESSAGES)):
            _absorb(part, results_by_id, calls, seen_call_ids, span_keys, span_results)
            if role == "assistant":
                txt = _text_of(part)
                if txt:
                    last_assistant_text = txt
        for key in span_keys:
            base, occurrence = split_content_call_id(key)
            max_occurrences[base] = max(max_occurrences.get(base, 0), occurrence)
        # An id-less response answers the id-less call at the same index in this span. Later spans
        # resend the history WITH the result, so a non-empty result overwrites an earlier absence.
        for key, raw_result in zip(span_keys, span_results, strict=False):
            if raw_result not in (None, "", {}):
                results_by_id[key] = raw_result

    kept = _drop_unwitnessed_repeats(calls, max_occurrences)
    return kept, results_by_id, first_user_text, last_assistant_text


def _drop_unwitnessed_repeats(
    calls: list[RawToolCall], max_occurrences: dict[str, int]
) -> list[RawToolCall]:
    """Keep only the occurrences some single span actually witnessed, in first-seen order."""
    kept: list[RawToolCall] = []
    for call in calls:
        cid = call.id or ""
        if not cid.startswith(CONTENT_ID_PREFIX):
            kept.append(call)
            continue
        base, occurrence = split_content_call_id(cid)
        if occurrence <= max_occurrences.get(base, 0):
            kept.append(call)
    return kept


def _absorb(
    part: dict[str, Any],
    results_by_id: dict[str, Any],
    calls: list[RawToolCall],
    seen_call_ids: set[str],
    span_keys: list[str],
    span_results: list[Any],
) -> None:
    """Record a tool_call (deduped by id) or a tool_call_response (result under result|response).

    ``span_keys`` accumulates this span's synthetic call keys in order, so the caller can max
    occurrence counts across spans rather than summing them, and can join id-less results by
    position. ``span_results`` accumulates this span's id-less results in the order they appear.
    """
    ptype = part.get("type")
    if ptype == "tool_call":
        cid = part.get("id")
        name = str(part.get("name") or "tool")
        args = part.get("arguments")
        if cid is not None:
            key = str(cid)
        else:
            base = split_content_call_id(content_call_id(name, args, 1))[0]
            # Count this span's sightings first, so a second sighting WITHIN one span becomes
            # occurrence 2 and is kept as a genuine repeat rather than deduped away.
            occurrence = sum(1 for k in span_keys if split_content_call_id(k)[0] == base) + 1
            key = f"{base}#{occurrence}"
            span_keys.append(key)
        if key not in seen_call_ids:
            seen_call_ids.add(key)
            calls.append(RawToolCall(id=key, name=name, args=args))
    elif ptype == "tool_call_response":
        cid = part.get("id")
        # Result rides under "result" OR "response" (the latest-OpenLLMetry dialect).
        raw_result = part.get("result", part.get("response"))
        if cid is not None:
            results_by_id[str(cid)] = raw_result
        else:
            # An id-less response was previously dropped outright, so an id-less call could never
            # recover a result even when the source carried one. Held for positional joining.
            span_results.append(raw_result)


def _iter_parts(raw: Any) -> list[tuple[str, dict[str, Any]]]:
    """Yield (role, part) via the shared reader, re-raising its error as a typed one.

    The parsing itself lives in
    :func:`calibrated_agent_evaluation.normalize.content.iter_message_parts` because the Strands
    latest-experimental convention carries the identical payload; two copies of "what a part is" is
    how the two conventions would drift apart.
    """
    try:
        return content_mod.iter_message_parts(raw)
    except ValueError as exc:
        raise NormalizationError(f"gen_ai/messages content is not valid JSON: {exc}") from exc


def _text_of(part: dict[str, Any]) -> str:
    return content_mod.part_text(part)


def _agent_name(spans: list[dict[str, Any]]) -> str:
    for span in spans:
        attrs = otlp_json.attributes(span)
        name = attrs.get(_ATTR_AGENT_NAME) or attrs.get(_ATTR_SYSTEM)
        if name:
            return str(name)
    return "agent"


def _start_time(span: dict[str, Any]) -> int:
    """The span's start as integer nanoseconds, 0 when absent or unreadable.

    Dataset rows carry an ISO ``start_time`` and OTLP a ``startTimeUnixNano`` that is an int or a
    decimal string. Comparing those raw raised ``TypeError`` on a mixed session and ordered decimal
    strings lexicographically (``"10"`` before ``"9"``).
    """
    return otlp_json.start_nanos(span) or 0


registry.register_builtin("gen_ai/messages", to_intermediate)
