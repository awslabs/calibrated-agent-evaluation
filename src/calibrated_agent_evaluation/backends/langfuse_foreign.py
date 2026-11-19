# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map a Langfuse trace that Calibrated Agent Evaluation did not write into canonical OTLP-JSON.

``LangfuseBackend`` reconstructs its own traces losslessly from the
``calibrated_agent_evaluation_trace`` metadata envelope it writes. A trace written by anything else
(the Langfuse SDK used directly, an OpenTelemetry exporter pointed at Langfuse's OTLP endpoint, an
integration) has no envelope, so it is read the way a file is: mapped to the OTLP-JSON span shape
and run through ``normalize()``'s convention detection by the ingest path. This module does the
mapping and nothing else; it returns plain dicts, so no Langfuse type leaves ``backends/``.

**Verified against the live OSS server 3.202.1 with ``langfuse==3.15.0``** (a trace written with the
plain SDK, and one written by a plain OTel exporter to ``/api/public/otel``):

* ``api.trace.get(id)`` returns the trace with ``observations``; each carries ``id``, ``type``
  (``AGENT`` / ``TOOL`` / ``GENERATION`` / ``SPAN`` / ``CHAIN`` / ``EVENT`` / ...), ``name``,
  ``parent_observation_id``, ``start_time`` / ``end_time``, ``input`` / ``output``, ``model``,
  ``level`` and ``metadata``.
* Spans that arrived over OTLP keep their non-content attributes in ``metadata["attributes"]``
  (``gen_ai.operation.name``, ``gen_ai.agent.name``, ``gen_ai.tool.name``), the resource in
  ``metadata["resourceAttributes"]`` and the instrumentation scope in ``metadata["scope"]``.
  The content attributes (``gen_ai.input.messages``, ``gen_ai.tool.call.arguments`` and the like)
  are moved into ``input`` / ``output`` and are no longer in the attributes.
* The SDK's ``start_as_current_observation(trace_context=...)`` gives the outermost observation a
  ``parent_observation_id`` naming a span that is not in the trace, so a parent outside the
  observation set marks a root rather than an error.

The mapping, per observation:

* its recorded OTel attributes, when it has them, so the trace's own convention is what
  detection sees;
* otherwise a ``gen_ai`` operation from the Langfuse type: ``AGENT`` is ``invoke_agent`` (named by
  the observation), ``TOOL`` is ``execute_tool``, ``GENERATION`` is ``chat``, and anything else
  carries no operation and is an intermediate span the tree walks through;
* when no observation in the trace is an agent by either route, the outermost observations are
  the agent boundaries. A Langfuse trace is one run, and its outermost observation is that run;
  without this a trace built only from spans and generations (``@observe`` style) would have no
  agent at all;
* the observation's ``input`` / ``output`` back on the standard content keys,
  ``gen_ai.input.content`` / ``gen_ai.output.content`` for an agent and
  ``gen_ai.tool.call.arguments`` / ``gen_ai.tool.call.result`` for a tool, because that is where
  Langfuse took them from and where the content-placement registry reads;
* ``level == "ERROR"`` as an OTLP error status.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from calibrated_agent_evaluation.schema.errors import BackendError

_OP = "gen_ai.operation.name"
_AGENT_NAME = "gen_ai.agent.name"
_TOOL_NAME = "gen_ai.tool.name"
_MODEL = "gen_ai.request.model"
_AGENT_INPUT = "gen_ai.input.content"
_AGENT_OUTPUT = "gen_ai.output.content"
_TOOL_ARGS = "gen_ai.tool.call.arguments"
_TOOL_RESULT = "gen_ai.tool.call.result"

# Langfuse observation type -> the gen_ai operation it corresponds to. Types missing here (SPAN,
# CHAIN, EVENT, RETRIEVER, ...) carry no operation: the tree walks through them.
_TYPE_TO_OP = {"AGENT": "invoke_agent", "TOOL": "execute_tool", "GENERATION": "chat"}


def foreign_trace_to_otlp(raw: Any) -> dict[str, Any]:
    """Map one Langfuse trace (``api.trace.get`` result) to a canonical OTLP-JSON document.

    Raises :class:`BackendError` when the trace has nothing to map (no observations), which the
    ingest path reports as a per-trace skip.
    """
    trace_id = str(_get(raw, "id") or "")
    observations = list(_get(raw, "observations") or [])
    if not observations:
        raise BackendError(
            f"Langfuse trace {trace_id!r} has no observations; there is no span to normalize"
        )
    ids = {str(_get(o, "id")) for o in observations}
    recorded = [_recorded_attributes(o) for o in observations]
    has_agent = any(
        attrs.get(_OP) == "invoke_agent" or (_OP not in attrs and _type(o) == "AGENT")
        for o, attrs in zip(observations, recorded, strict=True)
    )

    spans: list[dict[str, Any]] = []
    for obs, attrs in zip(observations, recorded, strict=True):
        parent = _get(obs, "parent_observation_id")
        is_root = parent is None or str(parent) not in ids
        op = attrs.get(_OP) or _TYPE_TO_OP.get(_type(obs))
        if not has_agent and is_root:
            op = "invoke_agent"  # the run itself; see the module docstring
        merged = dict(attrs)
        name = str(_get(obs, "name") or _type(obs).lower() or "observation")
        if op is not None:
            merged.setdefault(_OP, op)
        if op == "invoke_agent":
            merged.setdefault(_AGENT_NAME, name)
            _put_content(merged, _AGENT_INPUT, _get(obs, "input"))
            _put_content(merged, _AGENT_OUTPUT, _get(obs, "output"))
        elif op == "execute_tool":
            merged.setdefault(_TOOL_NAME, name)
            _put_content(merged, _TOOL_ARGS, _get(obs, "input"))
            _put_content(merged, _TOOL_RESULT, _get(obs, "output"))
        elif op == "chat" and _get(obs, "model"):
            merged.setdefault(_MODEL, str(_get(obs, "model")))
        span: dict[str, Any] = {
            "traceId": trace_id,
            "spanId": str(_get(obs, "id")),
            "name": name,
            "attributes": [_key_value(k, v) for k, v in merged.items()],
        }
        if not is_root:
            span["parentSpanId"] = str(parent)
        start, end = _nanos(_get(obs, "start_time")), _nanos(_get(obs, "end_time"))
        if start is not None:
            span["startTimeUnixNano"] = str(start)
        if end is not None:
            span["endTimeUnixNano"] = str(end)
        if _get(obs, "level") == "ERROR":
            span["status"] = {"code": "STATUS_CODE_ERROR", "message": _get(obs, "status_message")}
        spans.append(span)

    meta = _metadata(observations[0])
    scope = meta.get("scope") if isinstance(meta.get("scope"), dict) else {}
    resource = meta.get("resourceAttributes")
    resource_attrs = resource if isinstance(resource, dict) else {}
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [_key_value(k, v) for k, v in resource_attrs.items()]},
                "scopeSpans": [
                    {
                        # The scope name rides through so a Strands app exporting to Langfuse is
                        # still routed to the Strands adapter, which reads its nested tools.
                        "scope": {"name": str((scope or {}).get("name") or "langfuse")},
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _type(obs: Any) -> str:
    return str(_get(obs, "type") or "").upper()


def _metadata(obs: Any) -> dict[str, Any]:
    meta = _get(obs, "metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            return {}
    return meta if isinstance(meta, dict) else {}


def _recorded_attributes(obs: Any) -> dict[str, Any]:
    """The OTel span attributes Langfuse kept for an observation that arrived over OTLP."""
    attrs = _metadata(obs).get("attributes")
    if not isinstance(attrs, dict):
        return {}
    return {str(k): v for k, v in attrs.items() if v is not None}


def _put_content(attrs: dict[str, Any], key: str, value: Any) -> None:
    if value is None or value in ("", [], {}) or key in attrs:
        return
    attrs[key] = value if isinstance(value, str) else json.dumps(value, default=str)


def _key_value(key: str, value: Any) -> dict[str, Any]:
    """Encode one attribute in the OTLP-JSON ``KeyValue`` wire form."""
    if isinstance(value, bool):
        any_value: dict[str, Any] = {"boolValue": value}
    elif isinstance(value, int):
        any_value = {"intValue": str(value)}
    elif isinstance(value, float):
        any_value = {"doubleValue": value}
    elif isinstance(value, str):
        any_value = {"stringValue": value}
    else:
        any_value = {"stringValue": json.dumps(value, default=str)}
    return {"key": key, "value": any_value}


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _nanos(value: Any) -> int | None:
    """Unix nanoseconds for a datetime or ISO string, in integer arithmetic; naive reads as UTC."""
    if isinstance(value, str) and value:
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return (value - _EPOCH) // timedelta(microseconds=1) * 1000
