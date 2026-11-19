# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared tool_call ↔ tool-result pairing by id, used by every convention adapter.

Across all the OTel GenAI conventions reconstructed here, a model emits a tool *call* (with an
``id``, name, arguments) in one message/attribute and the tool *result* arrives separately (a
later ``tool_call_response`` message part, a ``tool``-role message with a ``tool_call_id``, or a
dotted-index prompt entry). This module is the one place that joins them **by shared id**, so an
unpaired call is reconstructed with an empty ``result`` rather than mis-paired to some other
call's result. Vendor-free: it operates on already-parsed intermediate
tuples, not on any SDK type.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from calibrated_agent_evaluation.normalize import content as content_mod
from calibrated_agent_evaluation.normalize import otlp_json
from calibrated_agent_evaluation.normalize.tree import ToolInvocation

CONTENT_ID_PREFIX = "_calibrated_agent_evaluation_content:"


@dataclass(frozen=True)
class RawToolCall:
    """A model-emitted tool call before its result is joined (convention-agnostic)."""

    id: str | None
    name: str
    args: Any  # raw (str/dict); coerced to a dict by `pair()`


def pair(
    calls: list[RawToolCall],
    results_by_id: dict[str, Any],
    *,
    ordered_results: list[Any] | None = None,
) -> list[ToolInvocation]:
    """Join each call to its result by ``id``; unpaired calls get an empty result.

    ``results_by_id`` maps a tool-call id to its raw result payload. A call whose ``id`` is absent
    from that map (or whose ``id`` is ``None``) reconstructs with ``result={}``; never the result
    of a different id. ``ordered_results`` is an optional positional fallback for a convention that
    emits results *without* ids (used only when a call has no id AND a positional result exists at
    the same index); it is deliberately last-resort so an id match always wins.
    """
    invocations: list[ToolInvocation] = []
    for i, call in enumerate(calls):
        raw_result: Any = None
        if call.id is not None and call.id in results_by_id:
            raw_result = results_by_id[call.id]
        elif call.id is None and ordered_results is not None and i < len(ordered_results):
            raw_result = ordered_results[i]
        invocations.append(
            ToolInvocation(
                name=call.name,
                args=content_mod.as_object(call.args),
                result=content_mod.as_object(raw_result),
            )
        )
    return invocations


def content_call_id(name: str, args: Any, occurrence: int) -> str:
    """A stable internal id for an id-less tool call, derived from its own content.

    ``id`` is often provider-assigned, and a harness that proxies or re-synthesizes messages drops
    it, so identity has to come from the call itself. ``(name, canonical args)`` is that identity;
    ``occurrence`` separates a genuine repeat within one span from a resend across spans. The value
    never leaves normalization: ``ToolInvocation`` has no id field, so it only feeds dedup and
    :func:`pair`.
    """
    canonical = json.dumps(content_mod.as_object(args), sort_keys=True, default=str)
    digest = hashlib.sha1(f"{name}\x00{canonical}".encode(), usedforsecurity=False).hexdigest()[:16]
    return f"{CONTENT_ID_PREFIX}{digest}#{occurrence}"


def split_content_call_id(key: str) -> tuple[str, int]:
    """Split a content id into its content base and its 1-based occurrence index."""
    base, _, occurrence = key.rpartition("#")
    return base, int(occurrence)


def keyed_span_calls(entries: Iterable[tuple[str | None, str, Any]]) -> list[RawToolCall]:
    """One span's ``(id, name, args)`` entries as calls, an id-less one keyed by its content.

    Repeats of the same id-less call within the span are numbered 1, 2, ... so a genuine repeat
    survives the cross-span dedup, while the same call resent on a later span maps to the same key
    and is dropped there.
    """
    calls: list[RawToolCall] = []
    occurrences: dict[str, int] = {}
    for cid, name, args in entries:
        if cid is None:
            base = split_content_call_id(content_call_id(name, args, 1))[0]
            occurrences[base] = occurrences.get(base, 0) + 1
            cid = f"{base}#{occurrences[base]}"
        calls.append(RawToolCall(id=cid, name=name, args=args))
    return calls


def trace_id_from(spans: list[dict[str, Any]]) -> str | None:
    """The first span's wire ``traceId`` as hex, the form the gen_ai and Strands paths use.

    Without the hex rendering a protobuf-JSON export kept its base64 id here while the same run
    read through the gen_ai path came back as hex: two ids for one run.
    """
    for span in spans:
        raw = span.get("traceId") or span.get("trace_id")
        if isinstance(raw, str) and raw:
            return otlp_json.trace_id_hex(raw)
    return None
