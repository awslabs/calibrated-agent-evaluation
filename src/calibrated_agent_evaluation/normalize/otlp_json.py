# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stdlib OTLP-JSON reading core, shared by the OTel and Strands adapters.

Frozen OTLP-JSON is read with the **stdlib** ``json`` module (zero runtime dependency; the
``[otel]`` extra is needed only to decode *binary* ``.otlp`` protobuf). This module knows the
canonical OTLP-JSON wire shape a production OTLP/HTTP exporter emits
(``resourceSpans → scopeSpans → spans``, ``KeyValue``/``AnyValue`` attributes, ``Span.events``)
and the ``gen_ai.*`` semantic-convention names, so the two adapters share one place that knows
how content rides on the wire; never two divergent readers.

Two correctness rules this core enforces (the anti-patterns it exists to prevent):
  * it walks **every** ``ResourceSpans → ScopeSpans → Span`` container, never only the first
    (the silent-truncation failure mode); and
  * it reads content from span **events** as well as span **attributes** (the silent-drop
    failure mode), so content is never dropped for sitting in the placement the reader did not
    expect. There are three placements in practice, not two; legacy events, latest-convention
    span attributes, and the latest convention's single details *event*; so
    :func:`latest_content` tries them rather than trusting a declared mode.
"""

from __future__ import annotations

import base64
import binascii
import string
from datetime import UTC, datetime
from typing import Any

from calibrated_agent_evaluation.normalize._depth import MAX_CONTENT_DEPTH, TOO_DEEP_NOTE
from calibrated_agent_evaluation.schema.types import TokenUsage

# gen_ai semantic-convention attribute/event names (verified against the captured corpus).
ATTR_OP = "gen_ai.operation.name"
ATTR_AGENT_NAME = "gen_ai.agent.name"
ATTR_TOOL_NAME = "gen_ai.tool.name"
ATTR_SYSTEM = "gen_ai.system"
# Content on attributes (the ATTRIBUTES content-location variant).
ATTR_TOOL_INPUT = "gen_ai.tool.input"
ATTR_TOOL_OUTPUT = "gen_ai.tool.output"
ATTR_INPUT_CONTENT = "gen_ai.input.content"
ATTR_OUTPUT_CONTENT = "gen_ai.output.content"
# The latest-experimental GenAI convention: structured `{role, parts}` message lists, and tool
# call I/O, on span ATTRIBUTES. A pipeline that keeps attributes but drops span events leaves
# these as the only surviving placement, which is why every content reader has to try them.
ATTR_INPUT_MESSAGES = "gen_ai.input.messages"
ATTR_OUTPUT_MESSAGES = "gen_ai.output.messages"
ATTR_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
ATTR_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
# Event names carrying content (the EVENTS content-location variant).
EVENT_USER_MESSAGE = "gen_ai.user.message"
EVENT_ASSISTANT_MESSAGE = "gen_ai.assistant.message"
EVENT_TOOL_MESSAGE = "gen_ai.tool.message"
EVENT_CHOICE = "gen_ai.choice"
# Under the latest convention the same `gen_ai.*.messages` payload rides on ONE event's
# attributes instead of the span's, unless the emitter was also told to write span attributes
# only. Knowing both placements is what makes those two emitter settings interchangeable here.
EVENT_INFERENCE_DETAILS = "gen_ai.client.inference.operation.details"


def scalar(any_value: dict[str, Any], _depth: int = 0) -> Any:
    """Decode an OTLP ``AnyValue`` object to its Python value (``{"stringValue": ...}``).

    Recursive, per the OTLP/JSON encoding: an ``arrayValue`` becomes a list and a ``kvlistValue`` a
    dict, each element decoded the same way. A ``bytesValue`` stays the base64 string OTLP/JSON
    carries it as, which is also what the protobuf path's ``MessageToDict`` produces.
    """
    if not isinstance(any_value, dict) or not any_value:
        return None
    if _depth > MAX_CONTENT_DEPTH:
        return TOO_DEEP_NOTE
    key, val = next(iter(any_value.items()))
    if key == "intValue":
        # OTLP encodes int64 as a decimal string; keep it an int for the normalized schema.
        try:
            return int(val)
        except (TypeError, ValueError):
            return val
    if key == "boolValue":
        return bool(val)
    if key == "doubleValue":
        return float(val)
    if key == "arrayValue":
        values = val.get("values", []) if isinstance(val, dict) else []
        return [scalar(item, _depth + 1) for item in values or []]
    if key == "kvlistValue":
        values = val.get("values", []) if isinstance(val, dict) else []
        return {
            kv["key"]: scalar(kv.get("value", {}), _depth + 1)
            for kv in values or []
            if isinstance(kv, dict) and isinstance(kv.get("key"), str)
        }
    return val


def attributes(carrier: dict[str, Any]) -> dict[str, Any]:
    """Return the ``{key: scalar}`` map for a span/event/resource carrier.

    Tolerates both attribute shapes so ONE reader serves every source (no divergent
    readers): the OTLP-JSON **wire** form (``attributes`` is a ``KeyValue`` list
    ``[{"key","value":{"stringValue":...}}]`` from an OTLP exporter) AND an **already-decoded dict**
    (``attributes`` is ``{key: value}``, the shape a parquet/HF dataset row carries).
    A dict is returned as-is; a list is decoded via :func:`scalar`.
    """
    raw = carrier.get("attributes")
    if isinstance(raw, dict):
        # Already-decoded dataset shape: keys are attribute names, values plain scalars.
        return {k: v for k, v in raw.items() if isinstance(k, str)}
    out: dict[str, Any] = {}
    for kv in raw or []:
        key = kv.get("key")
        if isinstance(key, str):
            out[key] = scalar(kv.get("value", {}))
    return out


def iter_spans(otlp: dict[str, Any]) -> list[dict[str, Any]]:
    """Walk EVERY ``ResourceSpans → ScopeSpans → Span`` and return all spans.

    Reading only the first container silently truncates a trace whose spans are split across
    resources/scopes; the exact failure mode this exhaustive walk guards against.
    """
    spans: list[dict[str, Any]] = []
    for resource_spans in otlp.get("resourceSpans", []) or []:
        for scope_spans in resource_spans.get("scopeSpans", []) or []:
            spans.extend(scope_spans.get("spans", []) or [])
    return spans


def resource_attributes(otlp: dict[str, Any]) -> dict[str, Any]:
    """Merge every ``ResourceSpans.resource`` attribute map (later containers win on a clash)."""
    out: dict[str, Any] = {}
    for resource_spans in otlp.get("resourceSpans", []) or []:
        out.update(attributes(resource_spans.get("resource", {})))
    return out


def event_attributes(span: dict[str, Any], event_name: str) -> list[dict[str, Any]]:
    """Return the attribute maps of every span event with ``event_name`` (in wire order)."""
    return [
        attributes(event)
        for event in span.get("events", []) or []
        if event.get("name") == event_name
    ]


def latest_content(span: dict[str, Any], key: str) -> Any:
    """Read a latest-convention content attribute wherever the emitter put it.

    The same ``gen_ai.*.messages`` / ``gen_ai.tool.call.*`` payload rides either on the span's own
    attributes or on the attributes of the single ``gen_ai.client.inference.operation.details``
    event, depending on one emitter flag. Trying both is what keeps that flag from being a
    difference the adapters can see, and is the same never-drop rule this module already applies
    to the legacy events-vs-attributes split. ``None`` when neither placement carries ``key``.
    """
    value = attributes(span).get(key)
    if value not in (None, ""):
        return value
    for event in event_attributes(span, EVENT_INFERENCE_DETAILS):
        if event.get(key) not in (None, ""):
            return event[key]
    return None


def trace_id_hex(raw: str) -> str:
    """Render an OTLP-JSON ``traceId``/``spanId`` as stable lowercase hex.

    The wire carries the id in one of two encodings and nothing on the span says which: protobuf
    JSON renders the 16 raw bytes as base64, while ``MessageToDict`` and most collectors render them
    as 32 hex characters. So the encoding has to be *detected*, and hex detected FIRST.

    Detecting base64 first does not work, and that was the bug. ``base64.b64decode`` without
    ``validate=True`` silently discards non-alphabet characters instead of raising, and a
    32-character hex string is itself valid base64 -- so `d18aa7f7a211a5f9ae6b86ed58a27bd0` decoded
    to 24 junk bytes and re-hexed to `775f1a6bb7fb6b6d756b97fd69ee9bf3a79de7c6b6edb774`. A different
    value, a different length, matching nothing in the source: every downstream reference to that
    trace id pointed at a trace that existed in no other system, so a user could not
    cross-reference the library's output against their own Jaeger/Tempo/Honeycomb. The base64 branch
    was unreachable in practice AND corrupted the common case.

    Raises no error: an id in neither encoding is returned unchanged, because an unrecognized id is
    still a usable opaque key, and refusing a whole trace over its id encoding would be the wrong
    trade. A caller that wants the user told announces it (``guidance.synthesized_header_hint``).
    """
    if _is_hex_id(raw):
        return raw.lower()
    try:
        decoded = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        return raw
    # A trace id is 16 bytes and a span id 8; anything else decoded from something that merely
    # happened to be base64-shaped, so the raw string is the more faithful answer.
    return decoded.hex() if len(decoded) in (8, 16) else raw


def _is_hex_id(raw: str) -> bool:
    """True for a 32- or 16-character all-hex id (a 16-byte trace id or 8-byte span id)."""
    return len(raw) in (32, 16) and all(c in string.hexdigits for c in raw)


def status_code(span: dict[str, Any]) -> str | None:
    """Return the span's OTLP status code as its enum name (``"ERROR"``/``"OK"``/``"UNSET"``).

    Tolerates the two wire spellings for the same field: ``MessageToDict`` renders the enum as a
    string (``{"code": "ERROR"}`` / ``STATUS_CODE_ERROR``) while a hand-built or protobuf-numeric
    payload carries the integer (``2`` is ERROR, ``1`` OK). Returns ``None`` when the span carries
    no status at all, which is not the same claim as ``UNSET``.
    """
    status = span.get("status")
    if not isinstance(status, dict) or not status:
        return None
    code = status.get("code")
    if isinstance(code, bool) or code is None:
        return None
    if isinstance(code, int):
        return {0: "UNSET", 1: "OK", 2: "ERROR"}.get(code)
    if isinstance(code, str) and code:
        return code.removeprefix("STATUS_CODE_").upper()
    return None


def _iso_from_nanos(raw: Any) -> str | None:
    """Render an OTLP ``*TimeUnixNano`` field as a UTC ISO-8601 timestamp; ``None`` if unusable."""
    if raw in (None, "", 0, "0"):
        return None
    try:
        nanos = int(raw)
    except (TypeError, ValueError):
        return None
    if nanos <= 0:
        return None
    # Whole microseconds: `datetime` cannot hold nanosecond precision, and silently truncating is
    # honest here because the field is a display/ordering timestamp, not a duration measurement.
    try:
        moment = datetime.fromtimestamp(nanos // 1_000_000_000, tz=UTC)
    except (OverflowError, ValueError, OSError):
        return None  # out of datetime's range: no usable timestamp, same as an absent one
    return moment.replace(microsecond=(nanos % 1_000_000_000) // 1_000).isoformat()


def _nanos_of(span: dict[str, Any], nanos_key: str, iso_key: str) -> int | None:
    """A span time as integer nanoseconds: the OTLP ``*TimeUnixNano`` field, else a dataset ISO one.

    A dataset row carries ``start_time`` / ``end_time`` as ISO-8601 strings where OTLP has
    ``startTimeUnixNano`` (an int or a decimal string). A naive ISO time is read as UTC.
    """
    raw = span.get(nanos_key)
    if raw not in (None, ""):
        try:
            nanos = int(raw)
        except (TypeError, ValueError):
            nanos = 0
        if nanos > 0:
            return nanos
    iso = span.get(iso_key)
    if not isinstance(iso, str) or not iso:
        return None
    try:
        moment = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    since = moment - _EPOCH
    nanos = (since.days * 86_400 + since.seconds) * 1_000_000_000 + since.microseconds * 1_000
    return nanos if nanos > 0 else None


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def start_nanos(span: dict[str, Any]) -> int | None:
    """The span's start in nanoseconds since the epoch, ``None`` when it records none."""
    return _nanos_of(span, "startTimeUnixNano", "start_time")


def end_nanos(span: dict[str, Any]) -> int | None:
    """The span's end in nanoseconds since the epoch, ``None`` when it records none."""
    return _nanos_of(span, "endTimeUnixNano", "end_time")


def started_at(span: dict[str, Any]) -> str | None:
    """The span's start as a UTC ISO-8601 string (``None`` when the wire carries no start time)."""
    return _iso_from_nanos(start_nanos(span))


def ended_at(span: dict[str, Any]) -> str | None:
    """The span's end as a UTC ISO-8601 string (``None`` when the wire carries no end time)."""
    return _iso_from_nanos(end_nanos(span))


def session_window(spans: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """The earliest start and the latest end across ``spans``, each ``None`` when none records it.

    For a convention that collapses a session into one span: that span ran from the session's first
    start to its last end.
    """
    starts = [n for span in spans if (n := start_nanos(span)) is not None]
    ends = [n for span in spans if (n := end_nanos(span)) is not None]
    return (
        _iso_from_nanos(min(starts)) if starts else None,
        _iso_from_nanos(max(ends)) if ends else None,
    )


#: Current OTel GenAI spelling first, the earlier one second. Both are read for the reason every
#: other reader here tolerates two wire shapes: a trace is produced by whatever instrumentation the
#: team already runs, and refusing the older spelling would report "no usage" for a span that
#: plainly carries it.
USAGE_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("input_tokens", ("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens")),
    ("output_tokens", ("gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens")),
    ("total_tokens", ("gen_ai.usage.total_tokens",)),
)


def usage(
    span: dict[str, Any], keys: tuple[tuple[str, tuple[str, ...]], ...] = USAGE_KEYS
) -> TokenUsage | None:
    """Per-step token counts off a span's attributes, or ``None`` when it reports none.

    Nothing is derived: a missing ``total_tokens`` stays missing rather than becoming the sum of the
    other two, because a provider that counts cached or reasoning tokens reports a total the pair
    does not add up to, and a computed total is indistinguishable from a reported one once stored.
    ``keys`` maps each :class:`TokenUsage` field to the attribute names that carry it, in
    preference order; a convention with its own spelling passes its own.
    """
    attrs = attributes(span)
    found: dict[str, int] = {}
    for field_name, names in keys:
        for key in names:
            value = attrs.get(key)
            # `bool` is an `int` subclass; a boolean here is a mis-typed attribute, not a count.
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                found[field_name] = value
                break
            if isinstance(value, str) and value.lstrip("-").isdigit():
                found[field_name] = int(value)
                break
    return TokenUsage(**found) if found else None


def session_usage(
    spans: list[dict[str, Any]], keys: tuple[tuple[str, tuple[str, ...]], ...] = USAGE_KEYS
) -> TokenUsage | None:
    """The token counts of ``spans`` summed, for a convention that collapses them into one span.

    ``None`` when no span reports usage. A field is summed only when every span that reports usage
    reports that field; otherwise it stays ``None``, since a sum over some of the calls would be
    stored as if it covered all of them.
    """
    reports = [found for span in spans if (found := usage(span, keys)) is not None]
    if not reports:
        return None
    totals: dict[str, int] = {}
    for field_name, _ in keys:
        values = [getattr(report, field_name) for report in reports]
        if all(value is not None for value in values):
            totals[field_name] = sum(values)
    return TokenUsage(**totals)
