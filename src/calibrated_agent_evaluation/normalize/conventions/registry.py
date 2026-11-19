# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convention adapter registry.

Maps a convention ``kind`` → a :class:`ConventionAdapter` (a callable turning one session's
:class:`~calibrated_agent_evaluation.normalize.sources.SessionSpans` into the vendor-free
:class:`~calibrated_agent_evaluation.normalize.tree.Intermediate`). The three built-ins
(``gen_ai/messages``, ``openllmetry``, ``openinference``) are pre-registered at import;
``register_convention(kind, fn)`` lets a user add one for a format Calibrated Agent Evaluation
doesn't ship; dispatched to without editing core.

Shared-mutable-state contract: the built-ins are written once at import and the map
is read-only thereafter *except* explicit ``register_convention`` calls; a user re-registering a
built-in kind is refused (a silent override of a shipped adapter is a footgun, not a feature). An
unknown kind raises a typed error naming what is registered; never a silent no-op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from calibrated_agent_evaluation.normalize.errors import NormalizationError

if TYPE_CHECKING:
    from calibrated_agent_evaluation.normalize.sources import SessionSpans
    from calibrated_agent_evaluation.normalize.tree import Intermediate


class ConventionAdapter(Protocol):
    """Turns one session's raw spans + headers into the vendor-free ``Intermediate``."""

    def __call__(self, session: SessionSpans) -> Intermediate:
        """Turn ``session`` into the vendor-free intermediate form."""
        ...


_REGISTRY: dict[str, ConventionAdapter] = {}
# Built-in kinds may not be overridden via register_convention (protects shipped adapters).
_BUILTIN_KINDS: frozenset[str] = frozenset({"gen_ai/messages", "openllmetry", "openinference"})


def register_builtin(kind: str, adapter: ConventionAdapter) -> None:
    """Register a shipped adapter at import time (internal: used by the built-in modules)."""
    _REGISTRY[kind] = adapter


def register_convention(kind: str, adapter: ConventionAdapter) -> None:
    """Register a user convention adapter under ``kind``.

    Refuses to shadow a built-in kind or to silently replace an already-registered custom kind;
    both raise a typed :class:`NormalizationError`, so a registration clash is loud, not silent.
    """
    if kind in _BUILTIN_KINDS:
        raise NormalizationError(
            f"cannot register convention {kind!r}: it is a built-in adapter and may not be "
            f"overridden (built-ins: {sorted(_BUILTIN_KINDS)})"
        )
    if kind in _REGISTRY:
        raise NormalizationError(
            f"cannot register convention {kind!r}: a custom adapter is already registered under "
            f"that kind; unregister or choose a distinct kind"
        )
    _REGISTRY[kind] = adapter


def get_adapter(kind: str) -> ConventionAdapter:
    """Return the adapter for ``kind``, else raise naming what's registered."""
    try:
        return _REGISTRY[kind]
    except KeyError as exc:
        raise NormalizationError(
            f"no convention adapter registered for {kind!r}; "
            f"registered conventions are {sorted(_REGISTRY)}"
        ) from exc


def registered_kinds() -> frozenset[str]:
    """The currently-registered convention kinds (built-in + user)."""
    return frozenset(_REGISTRY)
