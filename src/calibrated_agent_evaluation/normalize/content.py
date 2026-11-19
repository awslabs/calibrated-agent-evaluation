# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared content-coercion helpers for the adapters (vendor-free).

``gen_ai.*`` content rides the wire as a JSON string (an event attribute or a span attribute).
The normalized schema stores ``input``/``output``/tool ``args``/``result`` as objects, so these
helpers decode a wire string into a dict; tolerating the shapes the real corpus emits: a bare
JSON object, a JSON-encoded ``[{"text": "..."}]`` choice list wrapping an inner JSON object, or
plain prose (wrapped as ``{"answer": ...}`` so it stays a dict). Never raises on bad content;
an undecodable payload becomes a text-wrapped dict, so the adapter degrades gracefully rather
than dropping a step (a parse failure here is not a source-read failure; the source-read
boundary is handled elsewhere, this covers per-field best-effort decode).
"""

from __future__ import annotations

import ast
import json
from typing import Any


def _loads(text: str) -> Any:
    """Decode a wire string to a Python object, tolerating JSON and Python-repr dict shapes.

    Inspect's ``ToolEvent.result`` arrives as a Python ``repr`` string (single quotes, ``True``/
    ``None``), while OTel content is JSON; so try JSON first, then ``ast.literal_eval`` for the
    repr shape. Returns ``None`` when neither parses (the caller then text-wraps it).

    Pathologically nested input (``"[" * 100000``, a hostile tool result) exhausts the parser's
    recursion; that is unparseable content like any other and is text-wrapped too.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, RecursionError, MemoryError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
        return None


def as_object(value: Any, *, text_key: str = "answer") -> dict[str, Any]:
    """Coerce a recovered content value into a dict for the normalized schema.

    Handles: an already-decoded dict (returned as-is); a JSON-string object; a Strands
    ``gen_ai.choice`` message that is a JSON list ``[{"text": "<inner JSON or prose>"}]`` (the
    inner text is itself decoded when it is a JSON object); a vendor rich-content object (e.g.
    Inspect's ``ContentText``/``ContentImage``; see :func:`_coerce_vendor`); and plain prose
    (wrapped under ``text_key``). An empty/None value returns ``{}``. The result is always a
    JSON-serializable dict of plain types; no vendor object leaks past this boundary.
    """
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return _from_choice_list(value, text_key=text_key)
    if isinstance(value, str):
        decoded = _loads(value)
        if isinstance(decoded, dict):
            return decoded
        if isinstance(decoded, list):
            return _from_choice_list(decoded, text_key=text_key)
        return {text_key: value}
    if isinstance(value, (int, float, bool)):
        return {text_key: value}
    return _coerce_vendor(value, text_key=text_key)


def _from_choice_list(items: list[Any], *, text_key: str) -> dict[str, Any]:
    """Decode a choice/content list to a dict.

    A ``[{"text": "..."}]`` message list joins its text (inner JSON object decoded when present).
    A list of vendor rich-content objects (Inspect ``ContentText``/``ContentImage``/...; a
    multimodal tool result) is coerced element-wise so it is preserved, never silently dropped to
    ``{}`` (the silent-drop anti-pattern).
    """
    texts = [item["text"] for item in items if isinstance(item, dict) and "text" in item]
    # "\n", not "": a multi-block assistant message is separate blocks, and concatenating them
    # rendered "...ordersHere is..." in the judge prompt and the review UI -- two sentences fused
    # into a word that appears nowhere in the source.
    joined = "\n".join(t for t in texts if isinstance(t, str))
    if joined:
        inner = _loads(joined)
        if isinstance(inner, dict):
            return inner
        return {text_key: joined}
    # No plain ``{"text": ...}`` entries: coerce each element (vendor Content objects, etc.) so a
    # multimodal / rich-content list survives as a JSON-serializable list rather than vanishing.
    coerced = [_coerce_element(item) for item in items]
    coerced = [c for c in coerced if c not in (None, "", {})]
    if not coerced:
        return {}
    return {text_key: coerced}


def _coerce_vendor(value: Any, *, text_key: str) -> dict[str, Any]:
    """Coerce a single vendor rich-content object to a JSON-serializable dict (no vendor import).

    Inspect's ``ContentText`` etc. are pydantic models: prefer the human text (``.text``) so a
    ``ContentText`` reads like prose, else fall back to ``model_dump()`` (a plain dict, preserving
    image/audio/document payloads). Anything else is stringified; the result never holds a vendor
    object, so it stays JSON-serializable and no SDK type crosses the normalized boundary.
    """
    coerced = _coerce_element(value)
    if isinstance(coerced, dict):
        return coerced
    return {text_key: coerced}


def iter_message_parts(raw: Any) -> list[tuple[str, dict[str, Any]]]:
    """Yield ``(role, part)`` for every part of a latest-convention messages list.

    The payload is ``[{role, parts:[{type, ...}]}]``, JSON-encoded on the wire. Shared by every
    reader of ``gen_ai.input.messages`` / ``gen_ai.output.messages`` so the ``gen_ai/messages``
    convention and the Strands latest-experimental convention cannot come to disagree about what
    a part is. ``[]`` for an absent/blank payload; a malformed one raises :class:`ValueError`,
    because content that is present but unreadable is a source defect and must never read as
    "this run had no content"; the adapters wrap it as a typed normalization error.
    """
    if not raw:
        return []
    try:
        messages = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"gen_ai messages content is not valid JSON: {exc}") from exc
    out: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(messages, list):
        return out
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        for part in message.get("parts") or []:
            if isinstance(part, dict):
                out.append((role, part))
    return out


def part_text(part: dict[str, Any]) -> str:
    """The prose of a ``{"type": "text", "content": ...}`` part; ``""`` for any other part type."""
    return str(part.get("content") or "") if part.get("type") == "text" else ""


def messages_text(raw: Any) -> str:
    """Join the text parts of a latest-convention messages payload, in wire order.

    For a reader that needs the payload as one blob of prose rather than as structured parts; the
    Strands adapter's per-agent handoff content, whose legacy counterpart is also a single string.
    Tool-call parts contribute nothing here; they are recovered as tool calls instead.
    """
    parts = [part for _, part in iter_message_parts(raw)]
    return "\n".join(text for part in parts if (text := part_text(part)))


def _coerce_element(value: Any) -> Any:
    """Coerce one content element to a plain JSON-serializable value (list + scalar paths)."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return value
    text = getattr(value, "text", None)  # ContentText → its prose
    if isinstance(text, str):
        return text
    model_dump = getattr(value, "model_dump", None)  # any pydantic Content → a plain dict
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return str(value)
