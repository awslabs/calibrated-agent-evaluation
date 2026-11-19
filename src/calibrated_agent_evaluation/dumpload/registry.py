# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dump/load codec registry.

The dump/load layer owns the round-trip *mechanism*; each artifact type registers a
``(to_row, from_row)`` codec pair once, and then participates in every subsequent ``dump`` /
``load`` automatically. Lookup of an unregistered type raises :class:`UnregisteredArtifactError`
rather than silently skipping it; a silent skip would make the no-lock-in
round-trip lie. Re-registering an already-registered type is a typed error, not silent
last-wins, so two features cannot quietly clobber each other's codec.

A ``Row`` is a flat ``dict[str, str]``; the CSV-friendly shape both the CSV and JSON
writers serialize (CSV columns flatten one level; nested objects become JSON-encoded cells).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibrated_agent_evaluation.schema.errors import UnregisteredArtifactError

Row = dict[str, str]
ToRow = Callable[[Any], Row]
FromRow = Callable[[Row], Any]


class Codec:
    """A registered ``(to_row, from_row)`` pair plus the stable ``name`` used for filenames."""

    __slots__ = ("from_row", "name", "to_row")

    def __init__(self, name: str, to_row: ToRow, from_row: FromRow) -> None:
        self.name = name
        self.to_row = to_row
        self.from_row = from_row


# Module-level registry: artifact type -> Codec. The core registers Trace/Span/Label (see io.py);
# downstream features register their own types against this same map.
#
# Concurrency contract: the registry is a write-once-at-startup, read-mostly map. Codecs are
# registered during import/initialization (``register_core_codecs`` + downstream module import),
# before any ``dump``/``load`` runs, and are never mutated afterward; reads of a steady-state dict
# are safe across threads. ``register()`` itself is NOT thread-safe; do not call it concurrently
# from worker threads. If a future feature needs dynamic mid-run registration from multiple
# threads, guard ``_REGISTRY`` with a lock then (premature here; it would add cost for no caller).
_REGISTRY: dict[type, Codec] = {}


def register(
    artifact_type: type,
    to_row: ToRow,
    from_row: FromRow,
    *,
    name: str | None = None,
) -> None:
    """Register a codec for ``artifact_type``.

    ``name`` is the stable identifier used for the dump filename (``<name>.csv`` /
    ``<name>.json``); it defaults to the type's lowercased class name. Re-registering an
    already-registered ``artifact_type`` raises :class:`ValueError` (not
    silent last-wins). Registering a *distinct* type under an already-used ``name`` is likewise
    rejected: two codecs sharing a filename would clobber each other's ``<name>.json``/``.csv`` on
    dump (one silently overwrites the other), breaking the lossless round-trip.
    """
    if artifact_type in _REGISTRY:
        raise ValueError(
            f"artifact type {artifact_type.__name__!r} already has a registered codec; "
            "re-registration is rejected (use a distinct type)"
        )
    resolved_name = name or artifact_type.__name__.lower()
    clashing = next((c for c in _REGISTRY.values() if c.name == resolved_name), None)
    if clashing is not None:
        raise ValueError(
            f"codec name {resolved_name!r} is already registered for another type; "
            "two codecs cannot share a dump filename (they would clobber each other); "
            "use a distinct name"
        )
    _REGISTRY[artifact_type] = Codec(resolved_name, to_row, from_row)


def codec_for(artifact_type: type) -> Codec:
    """Return the codec registered for ``artifact_type`` or raise ``UnregisteredArtifactError``."""
    try:
        return _REGISTRY[artifact_type]
    except KeyError:
        raise UnregisteredArtifactError(
            f"no codec registered for artifact type {artifact_type.__name__!r} "
            "(register it via dumpload.register before dump/load)"
        ) from None


def registered_types() -> list[type]:
    """Return the artifact types currently registered, in registration order."""
    return list(_REGISTRY)


def is_registered(artifact_type: type) -> bool:
    """Return whether ``artifact_type`` has a registered codec."""
    return artifact_type in _REGISTRY


def _reset_registry() -> None:
    """Clear the registry. Test-only helper (each test starts from a known registry state)."""
    _REGISTRY.clear()
