# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where agent and tool content lives on a span, per convention family.

The GenAI semantic conventions do not agree on one attribute key per content field, and real
instrumentors add their own. A reader that knows a single placement builds a perfectly correct
multi-agent tree in which every ``input``, ``output``, ``args`` and ``result`` is ``{}``, passes
both
ingest gates as ``owned``/``ingested``, and hands a judge or a human labeler nothing to read. That
is
the defect this module removes.

The registry exists rather than a longer key list in the reader because the third family of keys is
one vendor's. ``traceloop.entity.input`` and ``gen_ai.task.input`` are Traceloop's own names, and
naming a vendor inside the convention-neutral
:mod:`~calibrated_agent_evaluation.normalize.otel_adapter` is exactly the coupling the
constitution's vendor-isolation rule prevents. So the standard ``gen_ai.*`` families live here as
built-ins, a vendor convention registers its own placements (``conventions/openllmetry_adapter``
contributes Traceloop's), and the reader iterates whatever is registered without ever naming one.

Lookup is exhaustive and ordered, never gating: the standard family is tried first, then each
registered family in registration order, and the first non-empty value wins. Reads go through
:func:`~calibrated_agent_evaluation.normalize.otlp_json.latest_content`, so each key is also checked
on the ``gen_ai.client.inference.operation.details`` event; the same never-drop rule the rest of the
module applies to the events-vs-attributes split.

Shared-mutable-state contract mirrors
:mod:`~calibrated_agent_evaluation.normalize.conventions.registry`: built-ins are written once at
import and read-only thereafter except explicit :func:`register_content_placements` calls,
re-registering a built-in family is refused, and an unknown role raises a typed error naming the
known ones rather than returning nothing.
"""

from __future__ import annotations

from typing import Any, Literal

from calibrated_agent_evaluation.normalize import otlp_json
from calibrated_agent_evaluation.normalize.errors import NormalizationError

#: The four content fields a reader has to recover. Roles, not keys; the whole point is that the
#: keys differ per family while the role a value plays does not.
ContentRole = Literal["agent_input", "agent_output", "tool_args", "tool_result"]

_ROLES: frozenset[str] = frozenset({"agent_input", "agent_output", "tool_args", "tool_result"})

#: The family tried first, always. Held explicitly rather than relying on dict insertion order so
#: that "standard keys win" is a stated property and not an accident of import sequence.
STANDARD_FAMILY = "gen_ai"

_PLACEMENTS: dict[str, dict[str, tuple[str, ...]]] = {}
_BUILTIN_FAMILIES: set[str] = set()


def register_builtin(family: str, placements: dict[str, tuple[str, ...]]) -> None:
    """Register a shipped placement family at import time (internal)."""
    _validate_roles(family, placements)
    _PLACEMENTS[family] = dict(placements)
    _BUILTIN_FAMILIES.add(family)


def register_content_placements(family: str, placements: dict[str, tuple[str, ...]]) -> None:
    """Register the attribute keys ``family`` uses for one or more content roles.

    Refuses to shadow a built-in family or to silently replace an already-registered custom one, so
    a registration clash is loud. Roles are validated up front: a typo'd role would otherwise
    register keys that no reader ever consults, which is silent content loss wearing the costume of
    a successful registration.
    """
    if family in _BUILTIN_FAMILIES:
        raise NormalizationError(
            f"cannot register content placements for {family!r}: it is a built-in family and may "
            f"not be overridden (built-ins: {sorted(_BUILTIN_FAMILIES)})"
        )
    if family in _PLACEMENTS:
        raise NormalizationError(
            f"cannot register content placements for {family!r}: a custom family is already "
            f"registered under that name; choose a distinct name"
        )
    _validate_roles(family, placements)
    _PLACEMENTS[family] = dict(placements)


def _validate_roles(family: str, placements: dict[str, tuple[str, ...]]) -> None:
    unknown = sorted(set(placements) - _ROLES)
    if unknown:
        raise NormalizationError(
            f"content placements for {family!r} name unknown role(s) {unknown}; "
            f"known roles are {sorted(_ROLES)}"
        )


def _ensure_vendor_families() -> None:
    """Import the shipped convention modules so their placement families are registered.

    The registry owns this rather than its callers. The generic OTel reader reaches a
    ``gen_ai/latest`` trace without ever importing the convention package; that is the whole point
    of the ``gen_ai/latest`` fast path; so leaving the import to the caller means a
    Traceloop-shaped
    trace read under ``gen_ai/latest`` silently finds no Traceloop placements registered. Which is
    the original bug, reintroduced one layer down.

    The import is deferred to call time, not module scope, because the convention modules import
    this one. By the time any read happens this module is fully initialized, so the cycle does not
    bite. Idempotent: Python's module cache makes the repeat calls free.
    """
    from calibrated_agent_evaluation.normalize.conventions import (  # noqa: F401 - imported for the side effect
        openllmetry_adapter,
    )


def registered_families() -> tuple[str, ...]:
    """The registered placement families, standard first, then custom in registration order."""
    _ensure_vendor_families()
    return (STANDARD_FAMILY, *(f for f in _PLACEMENTS if f != STANDARD_FAMILY))


def placement_keys(role: ContentRole) -> tuple[str, ...]:
    """Every attribute key that may carry ``role``, in the order a reader should try them.

    Duplicates are collapsed keeping first position, so a family that also lists a standard key does
    not make the reader read it twice.
    """
    if role not in _ROLES:
        raise NormalizationError(f"unknown content role {role!r}; known roles are {sorted(_ROLES)}")
    keys: list[str] = []
    for family in registered_families():
        for key in _PLACEMENTS.get(family, {}).get(role, ()):
            if key not in keys:
                keys.append(key)
    return tuple(keys)


def standard_key(role: ContentRole) -> str:
    """The canonical ``gen_ai.*`` key for ``role``: what the span-event readers look for."""
    keys = _PLACEMENTS[STANDARD_FAMILY].get(role, ())
    if not keys:
        raise NormalizationError(f"no standard key registered for content role {role!r}")
    return keys[0]


def read(span: dict[str, Any], role: ContentRole) -> Any:
    """Return the first non-empty value any registered placement holds for ``role``.

    ``None`` when no placement carries it, which the caller distinguishes from a recovered-but-empty
    value. Empty string and ``None`` are both treated as absent so that an emitter writing ``""``
    does not stop the search at the first key and mask a real value further down.
    """
    for key in placement_keys(role):
        value = otlp_json.latest_content(span, key)
        if value not in (None, ""):
            return value
    return None


# --- built-in families ------------------------------------------------------------------------
#
# Order within a role matters: the key the current convention prefers comes first, the older
# spelling after it. Both are read, so an emitter switching spellings is not a difference any
# adapter can see.

register_builtin(
    STANDARD_FAMILY,
    {
        # `gen_ai.input.content` is the legacy single-blob spelling; `gen_ai.input.messages` is the
        # latest convention's structured `{role, parts}` list. A trace may carry either.
        "agent_input": (otlp_json.ATTR_INPUT_CONTENT, otlp_json.ATTR_INPUT_MESSAGES),
        "agent_output": (otlp_json.ATTR_OUTPUT_CONTENT, otlp_json.ATTR_OUTPUT_MESSAGES),
        # `gen_ai.tool.input`/`output` is the legacy spelling; `gen_ai.tool.call.arguments`/`result`
        # is the latest convention's. Both are already defined in otlp_json and were already read
        # by the Strands adapter; the generic reader is what never learned them.
        "tool_args": (otlp_json.ATTR_TOOL_INPUT, otlp_json.ATTR_TOOL_CALL_ARGUMENTS),
        "tool_result": (otlp_json.ATTR_TOOL_OUTPUT, otlp_json.ATTR_TOOL_CALL_RESULT),
    },
)
