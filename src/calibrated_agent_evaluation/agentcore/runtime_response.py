# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read an AgentCore Runtime invocation's response: decode it, refuse agent errors, find the answer.

Pure and vendor-free. It works on the body's raw lines, as ``StreamingBody.iter_lines`` yields
them, and decodes each line as UTF-8 only once it is whole: decoding first and splitting with
``str.splitlines`` would also break lines at characters such as U+2028, which JSON strings can
hold raw. The media type decides how the body is read; parameters such as ``charset`` are ignored.

- ``text/event-stream`` is read as server-sent events. One byte order mark (U+FEFF) at the very
  start of the body is dropped, as the browser algorithm drops it. ``data:`` is read with or
  without one space after the colon, and an event's ``data:`` lines are joined with a newline into
  one value; ``event:``, ``id:``, ``retry:`` and ``:`` comment lines are ignored. A value is
  decoded as JSON only when it starts with ``{``, ``[`` or ``"`` and decodes cleanly as standard
  JSON: no NaN or infinite number (a ``NaN``, ``Infinity`` or ``-Infinity`` token, or a literal
  such as ``1e999``) and at most 256 levels deep. Any other value, such as ``42`` or a lone ``{``,
  stays text, so no part of the answer is lost. A value of exactly ``[DONE]``, the end of an
  OpenAI-style stream, is skipped. A last event with no closing blank line is kept, where the
  browser algorithm would drop it, so a stream cut off at its last line keeps the answer's end.
- ``application/json`` is decoded as one standard JSON document, by the same rule. A leading byte
  order mark is not dropped, so such a body is refused.

Refusals raise :class:`~calibrated_agent_evaluation.agentcore.runtime_errors.RuntimeInvocationError`
naming the session, and the content type where it matters, never quoting the body or the agent's
error text, which can carry the payload or user data.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any, TypeAlias

from calibrated_agent_evaluation._finite import (
    MAX_JSON_DEPTH,
    NestedTooDeepError,
    NonFiniteNumberError,
    refuse_json_constant,
    require_finite,
)
from calibrated_agent_evaluation.agentcore.runtime_errors import (
    AgentReportedError,
    RuntimeInvocationError,
)

__all__ = ["DecodedValue", "check_errors", "decode", "final_text", "is_event_stream"]

DecodedValue: TypeAlias = str | int | float | bool | list[Any] | dict[str, Any] | None
"""One value read from a response: a JSON value, or the text of an event that is not JSON."""

_EVENT_STREAM = "text/event-stream"
_JSON = "application/json"

# The keys an object's answer is read from, in rank order: the first that holds a string wins.
# Event streams also carry it under ``data``, ranked last.
_TEXT_KEYS: tuple[str, ...] = ("response", "content", "message", "text", "output", "answer")
_EVENT_STREAM_TEXT_KEYS: tuple[str, ...] = (*_TEXT_KEYS, "data")

_DONE = "[DONE]"
_BYTE_ORDER_MARK = "\ufeff"
_JSON_STARTS = ("{", "[", '"')

# An error type is named only when it looks like an exception class name, so free agent text
# placed in ``error_type`` can't reach the message.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,99}")


def decode(
    content_type: str, lines: Iterable[bytes], *, session_id: str
) -> tuple[DecodedValue, ...]:
    """Decode a response body from its raw lines, by its media type.

    Args:
        content_type: The response's content type, with or without parameters.
        lines: The body's lines without their line breaks, as ``StreamingBody.iter_lines``
            yields them.
        session_id: The run's runtime session, named in every refusal.

    Returns:
        The values in body order: one per event-stream event that carries data, a ``[DONE]``
        marker skipped, or the one JSON document.

    Raises:
        RuntimeInvocationError: The media type is neither ``text/event-stream`` nor
            ``application/json``; the body is not valid UTF-8, or is empty (nothing but line
            breaks and, for an event stream, its leading byte order mark); a JSON body is not one
            standard JSON document (a leading byte order mark, a NaN or infinite number, or
            nesting past 256 levels included), the message saying which; or an event stream has
            no ``data:`` field.
    """
    media_type = _media_type(content_type)
    if media_type not in (_EVENT_STREAM, _JSON):
        raise RuntimeInvocationError(
            f"the runtime answered session {session_id} with content type {content_type!r}, "
            f"which Calibrated Agent Evaluation can't read: it reads {_EVENT_STREAM} and {_JSON}. "
            "Make the agent "
            "answer with one of them"
        )
    text_lines = _utf8_lines(lines, content_type, session_id)
    if media_type == _EVENT_STREAM:
        text_lines = _without_byte_order_mark(text_lines)
    if not any(text_lines):
        raise RuntimeInvocationError(
            f"the runtime sent an empty response body for session {session_id}, so there is no "
            "answer to read. Check that the agent returns its answer in the response"
        )
    if media_type == _JSON:
        return (_json_document(text_lines, content_type, session_id),)
    return _event_stream_values(text_lines, session_id)


def check_errors(
    values: Iterable[DecodedValue], *, session_id: str, trace_id: str | None = None
) -> None:
    """Refuse a response in which the agent reported an error.

    An object is an error report when its ``error`` key holds anything but null, false, an empty
    string, an empty array or an empty object, or sits next to a non-empty string ``error_type``,
    as in an error event whose exception had no text: ``{"error": "", "error_type":
    "TimeoutError"}``. ``{"response": "...", "error": null}`` is an answer.

    Args:
        values: The decoded values, as :func:`decode` returns them.
        session_id: The run's runtime session, named in the refusal.
        trace_id: The run's trace ID, carried on the refusal when given.

    Raises:
        AgentReportedError: A value reports an error. The message names the ``error_type`` when
            it is an identifier of at most 100 characters, and never quotes the error; the error
            code is that type, else ``"agent_error"``.
    """
    for value in values:
        if not isinstance(value, dict) or "error" not in value:
            continue
        error_type = value.get("error_type")
        typed = isinstance(error_type, str) and error_type != ""
        if not typed and _is_empty(value["error"]):
            continue
        named = isinstance(error_type, str) and _IDENTIFIER.fullmatch(error_type) is not None
        shown = f"an error of type {error_type}" if named else "an error"
        raise AgentReportedError(
            f"the agent reported {shown} in its response to session {session_id}, so the run "
            "has no answer to keep. The error text is not shown because it can carry the "
            "agent's data; look the session up in the agent's own logs",
            error_code=error_type if named and isinstance(error_type, str) else "agent_error",
            trace_id=trace_id,
        )


def final_text(values: Iterable[DecodedValue], content_type: str, *, session_id: str) -> str:
    """The agent's final answer: every value's text, joined in order.

    A string value is taken whole. An object gives the value of the first key, in the order
    ``response``, ``content``, ``message``, ``text``, ``output``, ``answer`` (then ``data``, for
    event streams), that holds a string; a key holding anything else is passed over. Other values
    add nothing.

    Args:
        values: The decoded values, as :func:`decode` returns them.
        content_type: The response's content type, which decides whether ``data`` counts.
        session_id: The run's runtime session, named in the refusal.

    Returns:
        The joined text.

    Raises:
        RuntimeInvocationError: The joined text is empty.
    """
    keys = _EVENT_STREAM_TEXT_KEYS if _media_type(content_type) == _EVENT_STREAM else _TEXT_KEYS
    parts: list[str] = []
    for value in values:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            parts.extend(_first_text(value, keys))
    text = "".join(parts)
    if not text:
        raise RuntimeInvocationError(
            f"no answer could be read from the response to session {session_id}: no value is "
            f"text, or an object holding text under {', '.join(keys)}. Check that the payload "
            "format matches the agent"
        )
    return text


def is_event_stream(content_type: str) -> bool:
    """Whether a response's content type is an event stream (parameters such as charset aside)."""
    return _media_type(content_type) == _EVENT_STREAM


def _media_type(content_type: str) -> str:
    # Media types are case-insensitive, and parameters such as charset don't change the reading.
    return content_type.partition(";")[0].strip().lower()


def _utf8_lines(lines: Iterable[bytes], content_type: str, session_id: str) -> list[str]:
    # Every line is checked, comments included: a body that isn't UTF-8 is refused whole, never
    # read in part.
    try:
        return [line.decode("utf-8") for line in lines]
    except UnicodeDecodeError as exc:
        raise RuntimeInvocationError(
            f"the {content_type!r} response to session {session_id} is not valid UTF-8. Check "
            "the encoding the agent writes its response in"
        ) from exc


def _without_byte_order_mark(lines: list[str]) -> list[str]:
    # Kept, it would make the first line an unknown field and lose its data. JSON bodies keep it.
    if lines and lines[0].startswith(_BYTE_ORDER_MARK):
        return [lines[0].removeprefix(_BYTE_ORDER_MARK), *lines[1:]]
    return lines


def _json_document(lines: list[str], content_type: str, session_id: str) -> DecodedValue:
    response = f"the {content_type!r} response to session {session_id}"
    try:
        return _standard_json("\n".join(lines), where="response body")
    except (NestedTooDeepError, RecursionError) as exc:
        # The decoder's own RecursionError comes only from nesting far past the limit, and on
        # some Python versions before the depth check can run, so it gets the same answer.
        raise RuntimeInvocationError(
            f"{response} nests more than {MAX_JSON_DEPTH} levels deep, deeper than Calibrated "
            "Agent Evaluation reads "
            "as JSON. Check the agent's response for that session"
        ) from exc
    except NonFiniteNumberError as exc:
        raise RuntimeInvocationError(
            f"{response} holds a number JSON cannot represent (NaN or an infinity), so it is not "
            "standard JSON. Check the agent's response for that session"
        ) from exc
    except ValueError as exc:
        raise RuntimeInvocationError(
            f"{response} does not decode as one JSON document; it may have been cut off. Check "
            "the agent's response for that session"
        ) from exc


def _standard_json(text: str, *, where: str) -> DecodedValue:
    # Python's json reads NaN and the infinities: the hook refuses their tokens, and the check a
    # literal such as 1e999, which decodes to an infinity without one, and nesting past the limit.
    decoded = json.loads(text, parse_constant=refuse_json_constant)
    require_finite(decoded, where=where)
    return decoded


def _event_stream_values(lines: list[str], session_id: str) -> tuple[DecodedValue, ...]:
    values: list[DecodedValue] = []
    data: list[str] = []
    saw_data = False
    for line in lines:
        if not line:
            _dispatch(data, values)
            data = []
            continue
        if line.startswith(":"):
            continue
        name, _, field_value = line.partition(":")
        if name != "data":
            continue
        saw_data = True
        data.append(field_value.removeprefix(" "))
    # Keep a last event the stream didn't close with a blank line.
    _dispatch(data, values)
    if not saw_data:
        raise RuntimeInvocationError(
            f"the event stream for session {session_id} has no data: field, so it carries no "
            "answer. Check that the agent streams its answer as data: lines"
        )
    return tuple(values)


def _dispatch(data: list[str], values: list[DecodedValue]) -> None:
    if not data:
        return
    value = "\n".join(data)
    if value == _DONE:
        return
    values.append(_event_value(value))


def _event_value(value: str) -> DecodedValue:
    if not value.startswith(_JSON_STARTS):
        return value
    try:
        return _standard_json(value, where="response value")
    except (ValueError, RecursionError):
        # A token that only looks like JSON, such as a lone "{", is part of the answer, and so is
        # a value that is not standard JSON or nests too deep.
        return value


def _is_empty(error: object) -> bool:
    # Identity, not equality: 0 == False in Python, but a zero is not a false error.
    if error is None or error is False:
        return True
    return isinstance(error, str | list | dict) and len(error) == 0


def _first_text(value: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str):
            return [candidate]
    return []
